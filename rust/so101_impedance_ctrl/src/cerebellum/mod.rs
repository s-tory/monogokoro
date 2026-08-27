//! The cerebellum: a GPU-resident adaptive feedforward layer that learns to cancel standing loads
//! before the reflex has to react to them.
//!
//! # What it is for
//!
//! The impedance law in `control.rs` is a spinal reflex. It only ever responds to an error that
//! has *already happened*, so under a standing load it settles at a permanent offset: a joint
//! carrying `g` PWM worth of gravity sits exactly `g / K` counts below its target, and the only
//! way a pure feedback law can shrink that droop is to raise `K` -- to trade away the compliance
//! this whole fork exists to provide. Predicting the load decouples the two:
//!
//! ```text
//! pwm = K*(x_t - x) + D*(v_t - v)  +  ff(sensory state)
//!       \___________ ____________/    \_______ _______/
//!                   v                         v
//!             reflex, 400 Hz on the      cerebellum, GPU,
//!             isolated core              off the critical path
//! ```
//!
//! # Threading, and why the control loop never waits for the GPU
//!
//! This runs on its own thread and communicates with the RT loop through two seqlocks -- the same
//! non-blocking pattern `shm.rs` uses across the process boundary, for the same reason. The
//! control loop publishes a sensory snapshot and reads whatever feedforward is currently
//! available; it never takes a lock, never blocks, and never notices if this thread is slow or
//! gone.
//!
//! That is not caution, it is forced. On this hardware Vulkan reports a single queue family with a
//! single queue, shared with graphics, so a submission queues behind the desktop compositor's
//! work. The GPU's kernel-side service path -- driver workqueues, the DRM scheduler, completion
//! interrupts -- runs on housekeeping cores by construction, because steering interrupts away from
//! the RT core is exactly what `irqaffinity=` is for. There is no configuration in which waiting
//! on a GPU fence inside a `SCHED_FIFO` tick is a bounded operation, so the design does not try.
//!
//! Nothing is lost by the decoupling. The load being predicted is quasi-static -- that is the
//! premise of the velocity gate below -- so a feedforward that is a few milliseconds old is a
//! feedforward that is still correct. Biology arranges it the same way: the cerebellum is not in
//! the stretch reflex's arc either.
//!
//! # Safety properties
//!
//! A learned term that can push a compliant arm around is the one genuinely new hazard this module
//! introduces, so it is bounded four separate ways, and none of them depend on the network having
//! learned anything sensible:
//!
//! 1. **Zero at rest.** Purkinje weights start at zero, so an untrained cerebellum contributes
//!    exactly nothing and enabling it cannot change how the arm behaves until it has learned.
//! 2. **Clamped.** [`CerebellumConfig::ff_max`] caps the per-joint duty far below `--pwm-max`.
//! 3. **Slew-limited.** [`CerebellumConfig::ff_slew`] bounds how fast the applied feedforward may
//!    change, so neither a learning transient nor a stale-to-fresh transition can step the
//!    command.
//! 4. **Fail-safe.** The feedforward is zeroed on watchdog timeout, on blind ticks, and if this
//!    thread stops publishing. A learned term must not be the one part of the controller that
//!    survives its own fail-safe.

pub mod gpu;
pub mod net;

use std::cell::UnsafeCell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::shm::NUM_MOTORS;

pub use net::{SensoryState, MF_DIM, NUM_OUTPUTS};

/// Which implementation of the network to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Backend {
    /// Vulkan compute on the integrated GPU. The intended configuration.
    Gpu,
    /// The `net.rs` reference implementation, on a CPU thread. Exists so the daemon can be brought
    /// up on a machine with no working Vulkan driver, and so the shaders have something to be
    /// checked against. Not a silent fallback: asking for `gpu` and not getting it disables the
    /// cerebellum rather than quietly switching, because "the feedforward is running" and "the
    /// feedforward is running somewhere else at a different speed" are things an operator has to
    /// be able to tell apart.
    Cpu,
    /// No cerebellum. The daemon is a pure reflex, exactly as it was before this module existed.
    Off,
}

#[derive(Clone, Debug)]
pub struct CerebellumConfig {
    pub backend: Backend,
    /// Granule cells. The expansion is the entire reason a single linear readout suffices, so this
    /// is the knob that trades GPU memory and bandwidth for how finely the feedforward can vary
    /// with pose.
    pub gc_dim: usize,
    /// Seed for the fixed granule connectivity. Saved weights are bound to it -- they are
    /// meaningless against a different random projection -- so changing it invalidates a weights
    /// file, and loading refuses rather than reinterpreting.
    pub seed: u64,
    /// How often this thread steps the network. Independent of `--loop-hz`: the reflex needs
    /// 400 Hz because it closes a mechanical loop, and this does not close any loop faster than
    /// the load changes.
    pub hz: f64,
    /// Hebbian step size. Because the granule code is normalised to unit length, this is the
    /// fraction of the remaining error corrected per step -- a time constant of `1 / rate` steps,
    /// or `1 / (rate * hz)` seconds, independent of how the layer is sized.
    pub rate: f32,
    /// Heterosynaptic decay coefficient -- the "modified" half of the learning rule, and what
    /// keeps a Hebbian product from growing without bound under a permanently-signed error.
    ///
    /// It buys that boundedness with a residual. At equilibrium the Hebbian and decay terms cancel
    /// at `w = cf / leak`, so a little feedback duty has to remain to hold the weights up:
    /// `residual ~= load * leak / (leak + sum_j g_j)`, and `sum_j g_j` is about the square root of
    /// the active-cell count -- roughly 18 at the defaults. So even a fairly aggressive leak costs
    /// a fraction of a percent of the load.
    ///
    /// The decay is proportional to the eligibility trace rather than applied to every weight, so
    /// a synapse forgets only while it is being used. That keeps poses the arm is not currently
    /// visiting intact, and it means the plasticity dispatch still touches only the ~2% of the
    /// layer that is active.
    pub leak: f32,
    /// Fraction of granule cells the Golgi threshold aims to leave active.
    pub sparsity: f32,
    /// Integrator gain for that threshold.
    pub golgi_gain: f32,
    /// Eligibility-trace time constant, covering the delay between a pose and the climbing-fibre
    /// signal it eventually produces.
    pub trace_tau_s: f32,
    /// Low-pass on the climbing-fibre signal, so plasticity integrates load rather than noise.
    pub cf_tau_s: f32,
    /// Per-joint cap on the applied feedforward duty.
    pub ff_max: f32,
    /// Cap on how fast the applied feedforward may change, in duty per second.
    pub ff_slew: f32,
    /// Learn only below this joint speed (counts/s): a moving joint's duty is inertia and damping,
    /// neither of which is a function of pose.
    pub vel_gate: f32,
    /// Learn only within this tracking error (counts). This is what separates droop from contact:
    /// gravity droop settles small, an arm leaning on the table holds a large error that never
    /// closes, and both look identical to the velocity gate.
    pub error_gate: f32,
    /// Joints whose feedforward is applied and whose climbing fibre is live. The gripper is absent
    /// by default -- see [`DEFAULT_JOINTS`].
    pub joints: Vec<usize>,
    /// Discard the feedforward if the cerebellum thread has not published within this long.
    pub staleness_ms: u64,
    /// Optional CPU core for this thread. Deliberately *not* the RT core.
    pub cpu_core: Option<usize>,
    /// Optional `SCHED_FIFO` priority. 0 leaves it at normal scheduling, which is the default:
    /// this thread has no hard deadline, and the isolation that matters is already provided by the
    /// RT core being `isolcpus`'d away from it.
    pub priority: i32,
    /// Where to persist Purkinje weights across runs.
    pub weights_path: Option<PathBuf>,
}

/// Joints that get a feedforward by default: the five arm joints, not the gripper.
///
/// Excluding the gripper is a safety property, not an oversight. A gripper holding an object shows
/// exactly the signature this module exists to cancel -- a large, motionless, standing duty -- but
/// that duty *is the grasp*, not a load to compensate. Learning it would make the gripper squeeze
/// harder and harder at the same commanded position, and keep squeezing after the object was gone.
/// The arm joints face a weaker version of the same hazard whenever they rest against something,
/// which is what [`CerebellumConfig::error_gate`] is for; on the gripper, contact is the normal
/// case rather than the exception, so no gate can separate the two and it is left out entirely.
pub const DEFAULT_JOINTS: &str = "0,1,2,3,4";

impl Default for CerebellumConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Off,
            gc_dim: 16384,
            seed: 0x5041_524B_494E_4A45, // "PARKINJE"
            hz: 200.0,
            rate: 0.01,
            leak: 0.05,
            sparsity: 0.02,
            golgi_gain: 0.05,
            trace_tau_s: 0.15,
            cf_tau_s: 0.1,
            ff_max: 300.0,
            ff_slew: 500.0,
            vel_gate: 80.0,
            error_gate: 200.0,
            joints: vec![0, 1, 2, 3, 4],
            staleness_ms: 200,
            cpu_core: None,
            priority: 0,
            weights_path: None,
        }
    }
}

/// A single-slot seqlock cell, for handing a `Copy` snapshot between two threads without either
/// of them taking a lock.
///
/// The writer is the party that must never block -- on the sensory side that is the `SCHED_FIFO`
/// control loop, and a mutex it shares with a normal-priority thread is a textbook priority
/// inversion. A seqlock has no such failure mode: the writer never waits, and a reader that
/// catches a write in progress retries or gives up. Same discipline as `shm::seqlock_write`, which
/// documents the fences.
struct SeqCell<T: Copy> {
    seq: AtomicU32,
    value: UnsafeCell<T>,
}

// SAFETY: all access goes through the seqlock protocol below, which is what provides the
// synchronisation; `T: Copy` rules out any interior pointer whose ownership could be duplicated by
// a torn read being discarded.
unsafe impl<T: Copy + Send> Sync for SeqCell<T> {}

impl<T: Copy> SeqCell<T> {
    fn new(value: T) -> Self {
        Self {
            seq: AtomicU32::new(0),
            value: UnsafeCell::new(value),
        }
    }

    fn write(&self, value: T) {
        let s = self.seq.load(Ordering::Relaxed);
        self.seq.store(s.wrapping_add(1), Ordering::Release);
        fence(Ordering::SeqCst);
        // SAFETY: the odd sequence number published above tells every reader that what it reads
        // now must be discarded, so this is the only party touching the cell in a way anyone will
        // act on.
        unsafe { *self.value.get() = value };
        fence(Ordering::SeqCst);
        self.seq.store(s.wrapping_add(2), Ordering::Release);
    }

    fn read(&self, max_retries: u32) -> Option<T> {
        for _ in 0..max_retries {
            let s1 = self.seq.load(Ordering::Acquire);
            if !s1.is_multiple_of(2) {
                continue;
            }
            fence(Ordering::SeqCst);
            // SAFETY: `T: Copy`, so this is a bitwise read; if the sequence check below fails the
            // value is thrown away without ever being observed by the caller.
            let value = unsafe { *self.value.get() };
            fence(Ordering::SeqCst);
            if self.seq.load(Ordering::Acquire) == s1 {
                return Some(value);
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct SensoryPacket {
    state: SensoryState,
    /// The control loop's own `fresh && !blind`. Learning and prediction both stop when it is
    /// false: a snapshot taken while the daemon could not see the arm describes an arm it could
    /// not see.
    safe: bool,
    timestamp_ns: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct OutputPacket {
    ff: [f32; NUM_MOTORS],
    timestamp_ns: u64,
    theta: f32,
    active_frac: f32,
    learning: bool,
}

/// Shared state between the control loop and the cerebellum thread.
struct Exchange {
    sensory: SeqCell<SensoryPacket>,
    output: SeqCell<OutputPacket>,
    running: AtomicBool,
    steps: AtomicU64,
    learn_steps: AtomicU64,
    step_ns_total: AtomicU64,
    step_ns_max: AtomicU64,
    errors: AtomicU64,
    /// Set once the backend has failed unrecoverably, so the control loop can stop trusting the
    /// last published feedforward even before it goes stale.
    faulted: AtomicBool,
}

/// Telemetry bits published to Python alongside the feedforward. Mirrored in
/// `shm_client.py`'s `CEREBELLUM_*` constants.
pub const CEREBELLUM_ACTIVE: u32 = 1 << 0;
pub const CEREBELLUM_LEARNING: u32 = 1 << 1;
pub const CEREBELLUM_STALE: u32 = 1 << 2;
pub const CEREBELLUM_FAULTED: u32 = 1 << 3;

/// The control loop's handle on the cerebellum.
pub struct Cerebellum {
    exchange: Arc<Exchange>,
    thread: Option<JoinHandle<()>>,
    cfg: CerebellumConfig,
    joint_mask: [bool; NUM_MOTORS],
    staleness_ns: u64,
    /// Slew-limiter state: the feedforward actually applied last tick.
    applied: [f32; NUM_MOTORS],
    pub device_label: String,
}

impl Cerebellum {
    /// Starts the cerebellum thread, or returns `Err` with a reason the caller should log before
    /// continuing without one.
    pub fn start(cfg: CerebellumConfig) -> Result<Self, String> {
        if cfg.backend == Backend::Off {
            return Err("disabled (--cerebellum-backend off)".to_string());
        }
        if !(0.0..1.0).contains(&cfg.sparsity) || cfg.sparsity <= 0.0 {
            return Err(format!(
                "--cerebellum-sparsity must be in (0, 1), got {}",
                cfg.sparsity
            ));
        }

        let params = net::GranuleParams::generate(cfg.gc_dim, cfg.seed);
        let mut backend = match cfg.backend {
            Backend::Gpu => Net::Gpu(Box::new(gpu::GpuNet::new(&params, true).map_err(|e| {
                format!(
                    "{e} -- run with `--cerebellum-backend cpu` to use the reference \
                         implementation instead, or `off` to disable the feedforward"
                )
            })?)),
            Backend::Cpu => Net::Cpu(Box::new(net::CpuNet::new(params))),
            Backend::Off => unreachable!("handled above"),
        };
        let device_label = backend.label();

        if let Some(path) = &cfg.weights_path {
            match load_weights(path, cfg.gc_dim, cfg.seed) {
                Ok(Some(w)) => match backend.write_weights(&w) {
                    Ok(()) => log::info!(
                        "cerebellum: loaded Purkinje weights from {}",
                        path.display()
                    ),
                    Err(e) => log::warn!("cerebellum: ignoring {}: {e}", path.display()),
                },
                Ok(None) => log::info!(
                    "cerebellum: no weights file at {} yet -- starting from zero, which means no \
                     feedforward until it has learned some",
                    path.display()
                ),
                Err(e) => log::warn!(
                    "cerebellum: could not load {}: {e} -- starting from zero",
                    path.display()
                ),
            }
        }

        let mut joint_mask = [false; NUM_MOTORS];
        for &j in &cfg.joints {
            if j >= NUM_MOTORS {
                return Err(format!(
                    "--cerebellum-joints lists joint {j}, which is out of range (0..{NUM_MOTORS})"
                ));
            }
            joint_mask[j] = true;
        }

        let exchange = Arc::new(Exchange {
            sensory: SeqCell::new(SensoryPacket::default()),
            output: SeqCell::new(OutputPacket::default()),
            running: AtomicBool::new(true),
            steps: AtomicU64::new(0),
            learn_steps: AtomicU64::new(0),
            step_ns_total: AtomicU64::new(0),
            step_ns_max: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            faulted: AtomicBool::new(false),
        });

        let thread_exchange = Arc::clone(&exchange);
        let thread_cfg = cfg.clone();
        let thread_mask = joint_mask;
        let thread = std::thread::Builder::new()
            .name("cerebellum".to_string())
            .spawn(move || run(thread_exchange, thread_cfg, thread_mask, backend))
            .map_err(|e| format!("could not spawn the cerebellum thread: {e}"))?;

        Ok(Self {
            exchange,
            thread: Some(thread),
            staleness_ns: cfg.staleness_ms * 1_000_000,
            cfg,
            joint_mask,
            applied: [0.0; NUM_MOTORS],
            device_label,
        })
    }

    /// Hands the cerebellum this tick's sensory snapshot. Called from the RT loop; never blocks.
    pub fn publish(&self, state: SensoryState, safe: bool, now_ns: u64) {
        self.exchange.sensory.write(SensoryPacket {
            state,
            safe,
            timestamp_ns: now_ns,
        });
    }

    /// The feedforward duty to add to this tick's command, already clamped, slew-limited, masked
    /// to the configured joints, and zeroed if unsafe or stale.
    ///
    /// Returns the duty and the telemetry flags describing why it is what it is.
    pub fn feedforward(&mut self, now_ns: u64, dt_s: f32, safe: bool) -> ([f32; NUM_MOTORS], u32) {
        let mut flags = 0u32;
        let faulted = self.exchange.faulted.load(Ordering::Relaxed);
        if faulted {
            flags |= CEREBELLUM_FAULTED;
        }

        // A retry-exhausted read is not an error, just a collision with the writer; fall through
        // to the zero target below, which the slew limiter walks to gently rather than stepping.
        let packet = self.exchange.output.read(4);
        let mut target = [0f32; NUM_MOTORS];
        if let Some(p) = packet {
            let stale = now_ns.saturating_sub(p.timestamp_ns) > self.staleness_ns;
            if stale {
                flags |= CEREBELLUM_STALE;
            }
            if p.learning {
                flags |= CEREBELLUM_LEARNING;
            }
            if safe && !stale && !faulted {
                flags |= CEREBELLUM_ACTIVE;
                for (i, t) in target.iter_mut().enumerate() {
                    if self.joint_mask[i] {
                        *t = p.ff[i].clamp(-self.cfg.ff_max, self.cfg.ff_max);
                    }
                }
            }
        }

        // Slew limit even on the way to zero. Dropping a held feedforward instantaneously is a
        // step input into a compliant joint, which is a visible drop of the arm; ramping it means
        // a fail-safe looks like the arm sagging rather than being kicked.
        let max_step = (self.cfg.ff_slew * dt_s).max(0.0);
        for (applied, &want) in self.applied.iter_mut().zip(target.iter()) {
            *applied += (want - *applied).clamp(-max_step, max_step);
        }
        (self.applied, flags)
    }

    /// One-line summary for the periodic log, resetting the counters it reports.
    pub fn summarise(&self) -> String {
        let e = &self.exchange;
        let steps = e.steps.swap(0, Ordering::Relaxed);
        let learn_steps = e.learn_steps.swap(0, Ordering::Relaxed);
        let total = e.step_ns_total.swap(0, Ordering::Relaxed);
        let max = e.step_ns_max.swap(0, Ordering::Relaxed);
        let errors = e.errors.swap(0, Ordering::Relaxed);
        let out = e.output.read(4).unwrap_or_default();
        let mean_us = total.checked_div(steps).unwrap_or(0) / 1000;
        format!(
            "cerebellum [{}]: {steps} steps ({learn_steps} with plasticity), {mean_us} us mean / \
             {} us max per step, {errors} errors; granule activity {:.2}% (theta {:.3}); ff [{}]",
            self.device_label,
            max / 1000,
            out.active_frac * 100.0,
            out.theta,
            self.applied
                .iter()
                .map(|v| format!("{v:.0}"))
                .collect::<Vec<_>>()
                .join(", "),
        )
    }

    /// Stops the thread and lets it persist its weights. Called on a clean shutdown only.
    pub fn stop(mut self) {
        self.exchange.running.store(false, Ordering::Release);
        if let Some(t) = self.thread.take() {
            if t.join().is_err() {
                log::error!("cerebellum thread panicked; its weights were not saved");
            }
        }
    }
}

impl Drop for Cerebellum {
    fn drop(&mut self) {
        self.exchange.running.store(false, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// The two interchangeable implementations of the network.
enum Net {
    Gpu(Box<gpu::GpuNet>),
    Cpu(Box<net::CpuNet>),
}

impl Net {
    fn label(&self) -> String {
        match self {
            Net::Gpu(g) => format!("gpu: {}", g.device_name),
            Net::Cpu(_) => "cpu reference".to_string(),
        }
    }

    fn step(
        &mut self,
        mf: &[f32; MF_DIM],
        theta: f32,
        trace_decay: f32,
        cf: Option<&[f32; NUM_OUTPUTS]>,
        rate: f32,
        leak: f32,
    ) -> Result<([f32; NUM_OUTPUTS], f32), String> {
        match self {
            Net::Gpu(g) => g.step(mf, theta, trace_decay, cf, rate, leak),
            Net::Cpu(c) => Ok(c.step(mf, theta, trace_decay, cf, rate, leak)),
        }
    }

    fn read_weights(&self) -> Vec<f32> {
        match self {
            Net::Gpu(g) => g.read_weights(),
            Net::Cpu(c) => c.weights.clone(),
        }
    }

    fn write_weights(&mut self, w: &[f32]) -> Result<(), String> {
        match self {
            Net::Gpu(g) => g.write_weights(w),
            Net::Cpu(c) => {
                if w.len() != c.weights.len() {
                    return Err(format!(
                        "weight vector has {} entries, this network needs {}",
                        w.len(),
                        c.weights.len()
                    ));
                }
                c.weights.copy_from_slice(w);
                Ok(())
            }
        }
    }
}

/// The cerebellum thread's body.
fn run(
    exchange: Arc<Exchange>,
    cfg: CerebellumConfig,
    joint_mask: [bool; NUM_MOTORS],
    mut backend: Net,
) {
    if let Some(core) = cfg.cpu_core {
        crate::rt::apply_rt_settings("cerebellum", core, cfg.priority);
    }

    let period = Duration::from_secs_f64(1.0 / cfg.hz.max(1.0));
    let staleness_ns = cfg.staleness_ms * 1_000_000;
    let dt_s = period.as_secs_f32();
    let trace_decay = decay_for(cfg.trace_tau_s, dt_s);
    let cf_alpha = if cfg.cf_tau_s > 0.0 {
        dt_s / (cfg.cf_tau_s + dt_s)
    } else {
        1.0
    };

    let mut theta = 0.0f32;
    let mut cf_lp = [0f32; NUM_OUTPUTS];
    let mut last_sensory = SensoryPacket::default();

    while exchange.running.load(Ordering::Acquire) {
        let started = Instant::now();

        if let Some(p) = exchange.sensory.read(4) {
            last_sensory = p;
        }
        // The control loop republishes every tick, so a snapshot older than the staleness window
        // means the loop has stopped or is wedged. Treat that exactly like an unsafe tick: keep
        // predicting from the last pose (harmless, and the control loop is not applying it anyway)
        // but do not learn from a sensory picture that is no longer describing the arm.
        let sensory_fresh = last_sensory.timestamp_ns != 0
            && monotonic_ns().saturating_sub(last_sensory.timestamp_ns) <= staleness_ns;
        let state = last_sensory.state;
        let mf = net::encode_mossy_fibres(&state);

        // The climbing fibre. Filtered unconditionally so that a gate opening mid-motion does not
        // hand plasticity a value left over from before the arm started moving.
        let mut any_learning = false;
        let mut cf = [0f32; NUM_OUTPUTS];
        for i in 0..NUM_OUTPUTS {
            cf_lp[i] += cf_alpha * (state.fb_pwm[i] - cf_lp[i]);
            let gated = !joint_mask[i]
                || !last_sensory.safe
                || !sensory_fresh
                || state.present_vel[i].abs() > cfg.vel_gate
                || state.pos_error[i].abs() > cfg.error_gate;
            if !gated {
                cf[i] = cf_lp[i];
                any_learning = true;
            }
        }

        let result = backend.step(
            &mf,
            theta,
            trace_decay,
            any_learning.then_some(&cf),
            cfg.rate,
            cfg.leak,
        );

        match result {
            Ok((ff, active_frac)) => {
                // Golgi feedback: nudge the global threshold until the measured fraction of active
                // granule cells matches the target. Doing it as an integrator on the measurement,
                // rather than as a k-winners-take-all sort or a fixed threshold derived from an
                // assumed distribution, is both what Golgi cells actually do and the only version
                // that costs a single atomic counter on the GPU.
                theta += cfg.golgi_gain * (active_frac - cfg.sparsity);
                // tanh outputs live in (-1, 1), so a threshold outside that range means "all off"
                // or "all on" and the integrator has nothing left to act on.
                theta = theta.clamp(-1.0, 1.0);

                let mut out = [0f32; NUM_MOTORS];
                out[..NUM_OUTPUTS].copy_from_slice(&ff);
                exchange.output.write(OutputPacket {
                    ff: out,
                    timestamp_ns: monotonic_ns(),
                    theta,
                    active_frac,
                    learning: any_learning,
                });
                exchange.steps.fetch_add(1, Ordering::Relaxed);
                if any_learning {
                    exchange.learn_steps.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(e) => {
                // A failed submission is not recoverable in any way this thread can attempt: the
                // device is gone or wedged. Latch the fault so the control loop stops applying the
                // last feedforward, and stop stepping rather than logging once per tick forever.
                log::error!(
                    "cerebellum: {e} -- feedforward disabled for the rest of this run; the arm \
                     keeps running on the reflex alone"
                );
                exchange.errors.fetch_add(1, Ordering::Relaxed);
                exchange.faulted.store(true, Ordering::Release);
                break;
            }
        }

        let elapsed = started.elapsed();
        exchange
            .step_ns_total
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        exchange
            .step_ns_max
            .fetch_max(elapsed.as_nanos() as u64, Ordering::Relaxed);
        if elapsed < period {
            std::thread::sleep(period - elapsed);
        }
    }

    if let Some(path) = &cfg.weights_path {
        let weights = backend.read_weights();
        match save_weights(path, &weights, cfg.gc_dim, cfg.seed) {
            Ok(()) => log::info!("cerebellum: saved Purkinje weights to {}", path.display()),
            Err(e) => log::error!(
                "cerebellum: could not save weights to {}: {e} -- this run's learning is lost",
                path.display()
            ),
        }
    }
}

/// Per-step decay factor for a first-order filter with time constant `tau_s`.
fn decay_for(tau_s: f32, dt_s: f32) -> f32 {
    if tau_s <= 0.0 {
        0.0
    } else {
        (-dt_s / tau_s).exp()
    }
}

fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `clock_gettime` writes only into the `timespec` we hand it.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

/// Magic + version for the weights file.
const WEIGHTS_MAGIC: &[u8; 8] = b"SO101CB1";

/// Persists Purkinje weights.
///
/// Binary rather than text because there are `NUM_OUTPUTS * gc_dim` of them -- ~98k at the default
/// -- and nobody is going to read them. The header carries `gc_dim` and the seed because a weight
/// vector is only meaningful against the exact random projection that produced it: loading one
/// against a different granule code would not be degraded, it would be arbitrary, and it would be
/// arbitrary *on a real arm*. [`load_weights`] refuses rather than reinterpreting.
pub fn save_weights(path: &Path, weights: &[f32], gc_dim: usize, seed: u64) -> std::io::Result<()> {
    let mut buf = Vec::with_capacity(32 + weights.len() * 4);
    buf.extend_from_slice(WEIGHTS_MAGIC);
    buf.extend_from_slice(&(gc_dim as u32).to_le_bytes());
    buf.extend_from_slice(&(NUM_OUTPUTS as u32).to_le_bytes());
    buf.extend_from_slice(&seed.to_le_bytes());
    for w in weights {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::write(path, buf)
}

/// Loads Purkinje weights, returning `Ok(None)` if the file simply does not exist yet.
pub fn load_weights(path: &Path, gc_dim: usize, seed: u64) -> Result<Option<Vec<f32>>, String> {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.to_string()),
    };
    if bytes.len() < 24 || &bytes[..8] != WEIGHTS_MAGIC {
        return Err("not a cerebellum weights file (bad magic)".to_string());
    }
    let stored_gc = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let stored_outputs = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let stored_seed = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
    if stored_gc != gc_dim || stored_outputs != NUM_OUTPUTS || stored_seed != seed {
        return Err(format!(
            "file holds gc_dim={stored_gc} outputs={stored_outputs} seed={stored_seed:#x}, this \
             run is gc_dim={gc_dim} outputs={NUM_OUTPUTS} seed={seed:#x} -- these weights only \
             mean anything against the granule code that produced them"
        ));
    }
    let payload = &bytes[24..];
    let expected = NUM_OUTPUTS * gc_dim;
    if payload.len() != expected * 4 {
        return Err(format!(
            "file holds {} weights, expected {expected}",
            payload.len() / 4
        ));
    }
    Ok(Some(
        payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    ))
}

/// Parses `--cerebellum-joints`, e.g. `"0,1,2,3,4"`.
pub fn parse_joints(spec: &str) -> Result<Vec<usize>, String> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("`{s}` is not a joint index: {e}"))
                .and_then(|j| {
                    if j < NUM_MOTORS {
                        Ok(j)
                    } else {
                        Err(format!("joint {j} is out of range (0..{NUM_MOTORS})"))
                    }
                })
        })
        .collect()
}
