//! Entry point for the SO101 impedance-control RT daemon.
//!
//! This process exclusively owns the SO101's single half-duplex serial bus (all 6 STS3215
//! servos, including the gripper) and exposes control to Python only through the shared-memory
//! protocol in `shm.rs`. See `README.md` for build/run instructions and PREEMPT_RT environment
//! prerequisites, and the top-level plan doc for the overall architecture.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Parser;
use shared_memory::ShmemConf;

use nix::unistd::{Gid, Uid};
use so101_impedance_ctrl::cerebellum::{self, Backend, Cerebellum, CerebellumConfig, SensoryState};
use so101_impedance_ctrl::control::{
    apply_soft_limits, apply_startup_config, finite_difference_velocity, impedance_pwm,
    input_is_fresh, log_homing_offsets, poll_and_apply_commands, wrapped_delta, MovingAverage,
};
use so101_impedance_ctrl::feetech::{self, FeetechBus};
use so101_impedance_ctrl::leader::LeaderGripper;
use so101_impedance_ctrl::rt;
use so101_impedance_ctrl::shm::{self, ShmLayout, NUM_MOTORS};

/// Motor IDs, matching `src/lerobot/robots/so_follower/so_follower.py`'s `Motor(id, "sts3215", ...)`
/// assignments: shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4, wrist_roll=5,
/// gripper=6.
const MOTOR_IDS: [u8; NUM_MOTORS] = [1, 2, 3, 4, 5, 6];

/// Index of the gripper within the per-motor arrays. The leader's force feedback is driven
/// from this joint's tracking error.
const GRIPPER_INDEX: usize = NUM_MOTORS - 1;

#[derive(Parser, Debug)]
#[command(
    name = "so101_impedance_ctrl",
    about = "PREEMPT_RT impedance controller for the SO101 arm"
)]
struct Cli {
    /// Serial device path for the SO101's single UART, e.g. /dev/ttyACM0.
    #[arg(long)]
    port: String,

    #[arg(long, default_value_t = 1_000_000)]
    baud: u32,

    /// CPU core index to pin this process's control-loop thread to (see README.md for isolcpus=
    /// kernel setup).
    #[arg(long, default_value_t = 0)]
    cpu_core: usize,

    /// SCHED_FIFO priority, 1-99.
    #[arg(long, default_value_t = 80)]
    priority: i32,

    /// POSIX shared-memory segment name Python attaches to.
    #[arg(long)]
    shm_name: String,

    /// Control loop rate. 400 Hz is what this arm sustains with SYNC_READ: three bus transactions
    /// per tick at ~256 us each is ~0.8 ms against a 2.5 ms period, so there is real headroom for
    /// the occasional stalled reply. Higher is available but pointless -- the limit on how the arm
    /// feels is the open-loop PWM and the gearbox friction, not the loop rate, and 400 Hz is
    /// already far past the arm's mechanical bandwidth and ACT's ~30 Hz camera rate.
    #[arg(long, default_value_t = 400.0)]
    loop_hz: f64,

    /// Fixed sample count for the per-motor Present_Current moving average.
    #[arg(long, default_value_t = 32)]
    current_avg_window: usize,

    /// Sample count for the velocity moving average that feeds the D term (1 = unfiltered).
    ///
    /// Raising `loop_hz` makes the raw finite difference *worse*, not better: position is quantised
    /// to whole encoder ticks, so one LSB of jitter reads as `1 / dt` ticks/s -- 400 ticks/s at
    /// 400 Hz. Fed straight into the D term that is 400*D PWM of pure noise, which chatters audibly
    /// well before D is large enough to damp anything.
    ///
    /// Averaging N consecutive finite differences telescopes exactly to `(p[t] - p[t-N]) / (N*dt)`,
    /// i.e. the N-tick difference, so this divides quantisation noise by N while costing N/2 ticks
    /// of lag -- at the defaults, noise drops to 50 ticks/s for 10 ms of lag. Doing it as a running
    /// average of wrapped per-tick deltas (rather than differencing against a saved older sample)
    /// also keeps it correct across the 4095/0 encoder wrap for free.
    #[arg(long, default_value_t = 8)]
    vel_filter_window: usize,

    /// Sample one motor's `Present_Current` every Nth tick, round-robin (1 = every tick).
    ///
    /// The impedance law only needs `Present_Position` at full rate; current is averaged and
    /// consumed by ACT as an observation at camera rate (~30 Hz), so it can be sampled far slower.
    /// One motor per tick keeps the per-tick cost flat, which matters more than the raw saving:
    /// reading all six on one tick made that tick twice as expensive as the rest, and those fat
    /// ticks accounted for every single overrun observed at 300 Hz.
    ///
    /// A given motor is therefore refreshed every `NUM_MOTORS * N` ticks, and its averaging window
    /// spans `current_avg_window * NUM_MOTORS * N / loop_hz` seconds -- at the defaults and 300 Hz,
    /// 32 * 6 * 1 / 300 = 0.64 s.
    #[arg(long, default_value_t = 1)]
    current_read_divisor: u64,

    /// Serial port of the *leader* arm, to give its gripper force feedback. Omit for follower-only
    /// operation, which is exactly the previous behaviour.
    ///
    /// Supplying this alone changes nothing about how the arm feels: `--force-feedback-gain`
    /// defaults to 0, so the leader gripper is read and commanded to zero duty. That is the
    /// measurement mode -- it costs the two extra bus transactions the real thing costs, so the
    /// per-second timing summary answers "can this machine afford bilateral" before any force is
    /// ever put into the operator's hand.
    #[arg(long)]
    leader_port: Option<String>,

    /// Leader gripper's motor ID on `--leader-port`. The other five leader servos are left
    /// untouched and backdrivable.
    #[arg(long, default_value_t = 6)]
    leader_gripper_id: u8,

    /// Leader duty per count of follower gripper tracking error. 0 disables force feedback.
    ///
    /// Signed, and the sign has to be measured rather than assumed: which encoder direction means
    /// "closed" is a property of each gripper's calibration, and the two arms need not agree. Start
    /// near zero and raise until the trigger resists a blocked follower; if it *assists* the
    /// squeeze instead, negate it. Assisting is the dangerous polarity -- it is positive feedback
    /// through the operator's hand -- which is why the default renders nothing at all.
    #[arg(long, default_value_t = 0.0)]
    force_feedback_gain: f32,

    /// Leader duty per count/s of trigger velocity, always opposing motion. Without it the operator
    /// feels a bare spring, which rings; this is what makes contact feel like contact.
    #[arg(long, default_value_t = 0.2)]
    force_feedback_damping: f32,

    /// Hard cap on leader duty, deliberately far below `--pwm-max`.
    ///
    /// The follower's limit protects a gearbox; this one is what a human hand is holding. It bounds
    /// how hard a wrong sign, a bad gain or an unstable loop can push before the operator simply
    /// overpowers it.
    #[arg(long, default_value_t = 250.0)]
    leader_pwm_max: f32,

    /// If no fresh input is received within this window, PWM output is zeroed (fail-safe).
    #[arg(long, default_value_t = 75)]
    watchdog_timeout_ms: u64,

    #[arg(long, default_value_t = 1000.0)]
    pwm_max: f32,

    /// Bit position of the direction flag in the PWM command register.
    ///
    /// Defaults to the **measured** value, 10 -- not the 11 that `feetech.py`'s `OperatingMode`
    /// docstring states. See `feetech::PWM_SIGN_BIT` for the measurement.
    ///
    /// If you need to re-derive it on other hardware, the test is whether flipping `--invert-pwm`
    /// reverses the *sign* of `--probe-direction`'s reported delta. A wrong bit still changes
    /// behaviour -- it gets read as extra magnitude -- so "the joint moved differently" is not
    /// enough; the delta has to actually change sign.
    #[arg(long, default_value_t = feetech::PWM_SIGN_BIT)]
    pwm_sign_bit: u32,

    /// Whether the PWM direction bit must be flipped for positive duty to raise
    /// `Present_Position`.
    ///
    /// **Defaults to true because that is what the SO101's servos actually do.** Measured with
    /// `check_so101_impedance.py --probe-direction wrist_roll` once `--pwm-sign-bit` was correct:
    /// +150 PWM toward +position moved the joint +130 ticks with this flag set, versus -227 with
    /// it clear. The two settings are coupled -- an incorrect sign bit makes this flag look
    /// ineffective, because it changes magnitude instead of direction.
    ///
    /// Get this backwards and the impedance law becomes positive feedback -- nudge a joint and the
    /// controller pushes it further from the target, so it accelerates into a stop instead of
    /// springing back. Since that is a hardware-damaging default to get wrong, it ships as the
    /// measured value rather than as something to discover. Pass `--invert-pwm false` if your
    /// servos are wired the other way; re-run the probe to check.
    #[arg(long, default_value_t = true)]
    invert_pwm: bool,

    /// Soft position limits in raw encoder ticks. PWM that would drive a joint further past these
    /// is zeroed; motion back toward the middle is always allowed.
    #[arg(long, default_value_t = 100.0)]
    pos_min: f32,

    #[arg(long, default_value_t = 3995.0)]
    pos_max: f32,

    /// Zero all PWM after this many consecutive failed position reads.
    ///
    /// Without a fresh position the impedance law is computing against a stale error, so it keeps
    /// driving hard while blind -- exactly the wrong thing when a servo has stalled or the bus has
    /// gone bad. Better to drop torque and let the arm go limp.
    #[arg(long, default_value_t = 5)]
    max_blind_ticks: u32,

    /// Serial round-trip timeout per register transaction.
    ///
    /// A transaction costs ~256 us in practice (16 bytes of wire time at 1 Mbaud plus the USB
    /// round trip), so this is ~20x the normal case -- generous, but low enough to *bound* a
    /// stalled transaction instead of letting it blow the tick budget. The previous 20 ms meant a
    /// single dropped reply cost six control periods at 300 Hz; capping it at 5 ms turns the same
    /// event into a bounded blip that the blind-tick fail-safe then handles.
    #[arg(long, default_value_t = 5)]
    serial_timeout_ms: u64,

    /// Read all six positions in one SYNC_READ instead of one READ per motor.
    ///
    /// On by default now that it has been validated against real servos: positions tracked
    /// hand-movement cleanly with no comms errors. That check is worth repeating on unfamiliar
    /// hardware before trusting it, because the failure mode is quiet -- this is a protocol-0-only
    /// instruction whose reply framing the daemon parses by hand, and a misparse yields
    /// plausible-looking but wrong positions rather than an error. The impedance law then chases
    /// an error that never converges and drives the joint continuously, which looks like a runaway
    /// and is *not* fixed by flipping `--invert-pwm` (that only reverses which way it runs).
    ///
    /// To validate safely, run monitor mode: Python writes nothing, so the watchdog holds PWM at
    /// zero and the arm stays limp while you move it by hand and watch the positions. Pass
    /// `--sync-read false` to fall back to per-motor reads (and drop `--loop-hz` to ~300, since
    /// that path costs eight transactions per tick instead of three).
    #[arg(long, default_value_t = true)]
    sync_read: bool,

    // ---- cerebellum -------------------------------------------------------------------------
    /// Where the adaptive feedforward runs: `gpu` (Vulkan compute on the iGPU), `cpu` (the
    /// reference implementation, for a host with no working Vulkan driver), or `off`.
    ///
    /// Defaults to `off`, so this daemon behaves exactly as it did before the cerebellum existed
    /// until someone asks for it. Turning it on is safe with the arm already holding a position:
    /// the Purkinje weights start at zero, so an untrained network adds precisely nothing.
    #[arg(long, value_enum, default_value_t = Backend::Off)]
    cerebellum_backend: Backend,

    /// Granule cells. The expansion layer is why a single learned linear readout suffices, so this
    /// trades GPU time for how finely the feedforward can vary with pose.
    ///
    /// Measured on the reference machine (Arc 140V, Mesa ANV): a step costs ~174 us at 1024 and
    /// ~190 us at 4096, i.e. below ~4k the cost is *entirely* submission latency and the compute
    /// is free. 16384 costs ~307 us and 65536 ~748 us.
    #[arg(long, default_value_t = 16384)]
    cerebellum_gc_dim: usize,

    /// Seed for the fixed granule connectivity. A saved weights file is bound to it -- the weights
    /// mean nothing against a different random projection -- so changing this invalidates one.
    #[arg(long, default_value_t = 0x5041_524B_494E_4A45)]
    cerebellum_seed: u64,

    /// How often the cerebellum thread steps. Deliberately unrelated to `--loop-hz`: the reflex
    /// needs 400 Hz because it closes a mechanical loop, and this closes nothing faster than the
    /// load changes.
    #[arg(long, default_value_t = 200.0)]
    cerebellum_hz: f64,

    /// Hebbian step size. The granule code is normalised to unit length, so this is the fraction
    /// of the remaining error corrected per step: a time constant of `1 / rate` steps, or
    /// `1 / (rate * cerebellum_hz)` seconds, independent of how the layer is sized.
    #[arg(long, default_value_t = 0.01)]
    cerebellum_rate: f32,

    /// Heterosynaptic decay -- the "modified" half of the Hebbian rule, and what stops a
    /// permanently-signed error growing the weights without bound. Costs a small steady-state
    /// residual in exchange (a fraction of a percent of the load at this default).
    #[arg(long, default_value_t = 0.05)]
    cerebellum_leak: f32,

    /// Fraction of granule cells the Golgi inhibition aims to leave active.
    #[arg(long, default_value_t = 0.02)]
    cerebellum_sparsity: f32,

    /// Integrator gain on the Golgi threshold.
    #[arg(long, default_value_t = 0.05)]
    cerebellum_golgi_gain: f32,

    /// Eligibility-trace time constant. Covers the delay between a pose and the climbing-fibre
    /// signal it eventually produces -- serial round trips, the velocity filter's group delay, and
    /// the arm's own mechanical response all sit in between.
    #[arg(long, default_value_t = 0.15)]
    cerebellum_trace_tau_s: f32,

    /// Low-pass on the climbing-fibre signal, so plasticity integrates load rather than the D
    /// term's quantisation noise.
    #[arg(long, default_value_t = 0.1)]
    cerebellum_cf_tau_s: f32,

    /// Per-joint cap on the feedforward duty. Independent of `--pwm-max` and far below it: this
    /// one bounds how hard a *learned* term can push an arm nobody is commanding.
    #[arg(long, default_value_t = 300.0)]
    cerebellum_ff_max: f32,

    /// Cap on how fast the applied feedforward may change, in duty per second. Applies in both
    /// directions, including on the way back to zero -- dropping a held feedforward instantly is a
    /// step input into a compliant joint.
    #[arg(long, default_value_t = 500.0)]
    cerebellum_ff_slew: f32,

    /// Learn only below this joint speed, in counts/s. A moving joint's duty is inertia and
    /// damping, neither of which is a function of pose, so fitting them would attribute them to
    /// whatever pose the arm was passing through.
    #[arg(long, default_value_t = 80.0)]
    cerebellum_vel_gate: f32,

    /// Learn only within this tracking error, in counts.
    ///
    /// This is what separates droop from contact. Gravity droop settles small, at `duty / K`; an
    /// arm resting on the table holds a large standing error that never closes. Both look
    /// identical to the velocity gate, and only this one refuses to learn the second.
    #[arg(long, default_value_t = 200.0)]
    cerebellum_error_gate: f32,

    /// Joints that get a feedforward and a live climbing fibre.
    ///
    /// The gripper (5) is absent by default and should stay that way: a gripper holding an object
    /// shows exactly the signature this module cancels -- a large, motionless, standing duty -- but
    /// that duty *is* the grasp. Learning it makes the gripper squeeze harder at the same commanded
    /// position, and keep squeezing after the object is gone.
    #[arg(long, default_value_t = cerebellum::DEFAULT_JOINTS.to_string())]
    cerebellum_joints: String,

    /// Discard the feedforward if the cerebellum thread has not published within this long.
    #[arg(long, default_value_t = 200)]
    cerebellum_staleness_ms: u64,

    /// CPU core for the cerebellum thread. Must NOT be `--cpu-core`.
    ///
    /// A housekeeping core is the right answer, and a second *isolated* core is not worth taking:
    /// the GPU exposes a single queue shared with graphics, and its completion interrupts and
    /// driver workqueues are steered onto housekeeping cores by the very `irqaffinity=` setting
    /// that protects the RT core -- so there is nothing about a fence wait that core isolation can
    /// make deterministic. The isolation that matters (this thread can never preempt the control
    /// loop) already comes from the control loop's core being isolated from everything else.
    #[arg(long)]
    cerebellum_cpu_core: Option<usize>,

    /// SCHED_FIFO priority for the cerebellum thread; 0 leaves it at normal scheduling.
    ///
    /// Normal scheduling is the default because this thread has no deadline. If you do raise it,
    /// keep it below `--priority`, so that a mistake in core assignment cannot let the cerebellum
    /// outrank the reflex.
    #[arg(long, default_value_t = 0)]
    cerebellum_priority: i32,

    /// File to persist the learned Purkinje weights to, loaded at startup and written on a clean
    /// shutdown (Ctrl-C, SIGTERM, or the Shutdown command). Omit to start from zero every run.
    #[arg(long)]
    cerebellum_weights: Option<std::path::PathBuf>,
}

/// Set by the signal handler; the control loop leaves at the top of the next tick.
///
/// A daemon that only exits through Python's Shutdown command would lose a session's learning to
/// every Ctrl-C, and would leave the leader gripper energised on the way out. Both are reasons to
/// have a clean exit path that does not depend on anything else still running.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

extern "C" fn on_shutdown_signal(_sig: libc::c_int) {
    // Only an atomic store, which is async-signal-safe; everything else happens on the control
    // loop's own thread.
    if SHUTDOWN_REQUESTED.swap(true, Ordering::Relaxed) {
        // A second signal means the first one did not get us out -- most likely the loop is stuck
        // on a serial transaction. Leave immediately rather than appear hung.
        // SAFETY: `_exit` is async-signal-safe by definition; it is the only correct way out here.
        unsafe { libc::_exit(130) };
    }
}

fn install_shutdown_handler() {
    // SAFETY: installing a handler that does nothing but an atomic store. `signal()` on
    // Linux/glibc gives BSD semantics -- the handler stays installed and syscalls restart -- which
    // is what we want.
    unsafe {
        libc::signal(
            libc::SIGINT,
            on_shutdown_signal as *const () as libc::sighandler_t,
        );
        libc::signal(
            libc::SIGTERM,
            on_shutdown_signal as *const () as libc::sighandler_t,
        );
    }
}

/// Reads one register from every motor, either as one READ per motor (default, well-understood)
/// or a single SYNC_READ (opt-in, faster). Values are returned in `MOTOR_IDS` order.
fn read_register_all(
    bus: &mut FeetechBus,
    reg: (u8, u8),
    sync_read: bool,
) -> std::io::Result<Vec<i32>> {
    if sync_read {
        bus.sync_read(reg, &MOTOR_IDS)
    } else {
        MOTOR_IDS
            .iter()
            .map(|&id| bus.read_register(id, reg))
            .collect()
    }
}

/// A segment whose telemetry timestamp is older than this is treated as abandoned. The control
/// loop rewrites it every tick (>=200 Hz), so a live daemon is never anywhere near this stale.
const SHM_LIVENESS_TIMEOUT: Duration = Duration::from_secs(1);

/// What an already-existing segment with our name turned out to be.
enum ExistingShm {
    /// Ours, and something is actively writing telemetry to it.
    LiveDaemon { age: Duration },
    /// Ours, but abandoned -- a previous daemon was killed before it could clean up.
    Stale { age: Duration },
    /// Not ours (bad magic, wrong size, unreadable). Never auto-delete this.
    Foreign(String),
}

/// Inspects an existing `/dev/shm` segment without disturbing it.
fn inspect_existing_shm(shm_name: &str) -> ExistingShm {
    let existing = match ShmemConf::new().os_id(shm_name).open() {
        Ok(s) => s,
        Err(e) => return ExistingShm::Foreign(format!("cannot open it ({e})")),
    };
    if existing.len() < std::mem::size_of::<ShmLayout>() {
        return ExistingShm::Foreign(format!(
            "it is {} bytes, smaller than our {} byte layout",
            existing.len(),
            std::mem::size_of::<ShmLayout>()
        ));
    }
    // SAFETY: size checked above; we only read POD fields and never hand out the reference.
    let layout: &ShmLayout = unsafe { &*(existing.as_ptr() as *const ShmLayout) };
    if layout.magic != shm::SHM_MAGIC {
        return ExistingShm::Foreign(format!(
            "its magic is 0x{:08x}, not ours (0x{:08x})",
            layout.magic,
            shm::SHM_MAGIC
        ));
    }

    // Read the telemetry timestamp through the seqlock so a mid-write snapshot can't fool us.
    let written_ns = shm::seqlock_read(
        &layout.output.seq,
        &layout.output.data,
        |d| d.timestamp_mono_ns,
        8,
    )
    .unwrap_or(0);
    let age = Duration::from_nanos(monotonic_ns().saturating_sub(written_ns));

    if age <= SHM_LIVENESS_TIMEOUT {
        ExistingShm::LiveDaemon { age }
    } else {
        ExistingShm::Stale { age }
    }
}

/// Creates the shared-memory segment, reclaiming one left behind by a killed daemon.
///
/// A daemon that is SIGKILLed (or panics past its cleanup) leaves `/dev/shm/<name>` in place, and
/// the next launch then dies on "already exists" -- with the arm wired up and the operator having
/// no idea a stale file is the problem. Reclaiming it automatically is safe *only* after proving
/// nothing is using it, which the telemetry timestamp settles: a live daemon rewrites it every
/// tick. Anything we can't positively identify as our own abandoned segment is left alone.
fn create_shm_reclaiming_stale(shm_name: &str) -> shared_memory::Shmem {
    let conf = || {
        ShmemConf::new()
            .size(std::mem::size_of::<ShmLayout>())
            .os_id(shm_name)
    };

    if let Ok(shmem) = conf().create() {
        return shmem;
    }

    match inspect_existing_shm(shm_name) {
        ExistingShm::LiveDaemon { age } => panic!(
            "shared memory segment '{shm_name}' is already in use -- its telemetry was updated \
             {age:?} ago, so another so101_impedance_ctrl is running and driving the arm. Refusing \
             to start a second one on the same servos. Stop the other daemon, or pass a different \
             --shm-name."
        ),
        ExistingShm::Foreign(why) => panic!(
            "shared memory segment '{shm_name}' already exists but {why}, so it is not ours to \
             remove. Inspect /dev/shm/{shm_name} and delete it yourself if it is junk, or pass a \
             different --shm-name."
        ),
        ExistingShm::Stale { age } => {
            log::warn!(
                "reclaiming stale shared memory segment '{shm_name}' (last written {age:?} ago -- \
                 a previous daemon exited without cleaning up)"
            );
            let path = std::path::Path::new("/dev/shm").join(shm_name);
            if let Err(e) = std::fs::remove_file(&path) {
                panic!("failed to remove stale segment {}: {e}", path.display());
            }
            conf().create().unwrap_or_else(|e| {
                panic!("failed to create '{shm_name}' after reclaiming it: {e}")
            })
        }
    }
}

/// Hands the shared-memory segment back to the user who invoked `sudo`.
///
/// The segment is created with the *effective* uid, so running the daemon under `sudo` leaves
/// `/dev/shm/<name>` owned by root mode 0600 -- and the Python robot, which runs unprivileged,
/// then can't attach to it. When `SUDO_UID`/`SUDO_GID` are present we chown it to the invoking
/// user so the normal workflow keeps working.
///
/// Prefer avoiding `sudo` altogether (`setcap cap_sys_nice+ep` on this binary, see README) -- then
/// the segment is created with the right owner in the first place and none of this is needed.
fn chown_shm_to_sudo_user(shm_name: &str) {
    let (Ok(uid), Ok(gid)) = (std::env::var("SUDO_UID"), std::env::var("SUDO_GID")) else {
        return; // not running under sudo -- the segment already belongs to the right user
    };
    let (Ok(uid), Ok(gid)) = (uid.parse::<u32>(), gid.parse::<u32>()) else {
        log::warn!("SUDO_UID/SUDO_GID are set but unparsable; leaving shm ownership as root");
        return;
    };

    let path = std::path::Path::new("/dev/shm").join(shm_name);
    match nix::unistd::chown(&path, Some(Uid::from_raw(uid)), Some(Gid::from_raw(gid))) {
        Ok(()) => log::info!(
            "chowned {} to uid={uid} gid={gid} (invoking sudo user)",
            path.display()
        ),
        Err(e) => log::warn!(
            "failed to chown {} to uid={uid} gid={gid}: {e} -- the Python robot may not be able to \
             attach. Run this binary without sudo (see `setcap cap_sys_nice+ep` in the README), or \
             chown the file manually.",
            path.display()
        ),
    }
}

fn monotonic_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `ts` is a valid, appropriately-sized out-param for clock_gettime.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

fn main() {
    env_logger::init();
    let args = Cli::parse();

    assert!(
        args.current_read_divisor >= 1,
        "--current-read-divisor must be >= 1 (1 = read Present_Current every tick)"
    );

    let shmem = create_shm_reclaiming_stale(&args.shm_name);
    // SAFETY: the segment is exactly sized for `ShmLayout` and only this process writes the
    // header/`init_in_place` fields; Python only attaches after this daemon has started.
    let layout: &mut ShmLayout = unsafe { &mut *(shmem.as_ptr() as *mut ShmLayout) };
    layout.init_in_place();
    log::info!(
        "shared memory segment '{}' created ({} bytes), layout_version={}",
        args.shm_name,
        std::mem::size_of::<ShmLayout>(),
        shm::LAYOUT_VERSION
    );
    chown_shm_to_sudo_user(&args.shm_name);

    // Echo the settings that actually shape the control law. Twice during bring-up a "fix" looked
    // ineffective simply because the running process predated it, so make what is live impossible
    // to guess wrong.
    log::info!(
        "config: invert_pwm={} pwm_sign_bit={} loop_hz={} pwm_max={} sync_read={} current_read_divisor={} \
         vel_filter_window={} pos_limits=[{}, {}] max_blind_ticks={} watchdog_ms={} \
         serial_timeout_ms={} leader_port={:?} force_feedback_gain={} force_feedback_damping={} \
         leader_pwm_max={}",
        args.invert_pwm,
        args.pwm_sign_bit,
        args.loop_hz,
        args.pwm_max,
        args.sync_read,
        args.current_read_divisor,
        args.vel_filter_window,
        args.pos_min,
        args.pos_max,
        args.max_blind_ticks,
        args.watchdog_timeout_ms,
        args.serial_timeout_ms,
        args.leader_port,
        args.force_feedback_gain,
        args.force_feedback_damping,
        args.leader_pwm_max,
    );
    log::info!(
        "cerebellum config: backend={:?} gc_dim={} seed={:#x} hz={} rate={} leak={} sparsity={} \
         trace_tau_s={} cf_tau_s={} ff_max={} ff_slew={} vel_gate={} error_gate={} joints={} \
         staleness_ms={} cpu_core={:?} priority={} weights={:?}",
        args.cerebellum_backend,
        args.cerebellum_gc_dim,
        args.cerebellum_seed,
        args.cerebellum_hz,
        args.cerebellum_rate,
        args.cerebellum_leak,
        args.cerebellum_sparsity,
        args.cerebellum_trace_tau_s,
        args.cerebellum_cf_tau_s,
        args.cerebellum_ff_max,
        args.cerebellum_ff_slew,
        args.cerebellum_vel_gate,
        args.cerebellum_error_gate,
        args.cerebellum_joints,
        args.cerebellum_staleness_ms,
        args.cerebellum_cpu_core,
        args.cerebellum_priority,
        args.cerebellum_weights,
    );

    install_shutdown_handler();
    rt::apply_rt_settings("control loop", args.cpu_core, args.priority);

    // The cerebellum is an enhancement, never a prerequisite: no Vulkan driver, a headless host or
    // a GPU in a bad state must all still leave the arm running. So a failure here degrades to the
    // pure reflex with a loud warning, exactly as the leader gripper does.
    let mut cerebellum = match build_cerebellum_config(&args) {
        Err(e) => {
            log::error!("cerebellum configuration rejected: {e} -- running without one");
            None
        }
        Ok(cfg) => match Cerebellum::start(cfg) {
            Ok(c) => {
                log::info!(
                    "cerebellum running on {} ({} granule cells, {} Hz)",
                    c.device_label,
                    args.cerebellum_gc_dim,
                    args.cerebellum_hz
                );
                Some(c)
            }
            Err(e) => {
                if args.cerebellum_backend == Backend::Off {
                    log::info!("cerebellum: {e}");
                } else {
                    log::error!(
                        "cerebellum unavailable: {e} -- continuing on the reflex alone; expect a \
                         loaded joint to droop by holding_duty / K"
                    );
                }
                None
            }
        },
    };

    let mut bus = FeetechBus::open(
        &args.port,
        args.baud,
        Duration::from_millis(args.serial_timeout_ms),
    )
    .expect("failed to open SO101 serial port");

    apply_startup_config(&mut bus, &MOTOR_IDS);
    log_homing_offsets(&mut bus, &MOTOR_IDS);

    // Force feedback is an enhancement to teleoperation, never a prerequisite for it: if the
    // leader's port is missing or its gripper will not go into PWM mode, the follower must still
    // run. So this degrades to follower-only with a loud warning rather than aborting.
    let mut leader = match &args.leader_port {
        None => None,
        Some(port) => match LeaderGripper::open(
            port,
            args.baud,
            Duration::from_millis(args.serial_timeout_ms),
            args.leader_gripper_id,
            args.force_feedback_gain,
            args.force_feedback_damping,
            args.leader_pwm_max,
            args.invert_pwm,
            args.pwm_sign_bit,
            args.vel_filter_window,
        ) {
            Ok(l) => Some(l),
            Err(e) => {
                log::error!(
                    "leader gripper on {port} unavailable: {e} -- continuing without force \
                     feedback; the follower is unaffected"
                );
                None
            }
        },
    };
    if leader.is_some() && args.force_feedback_gain == 0.0 {
        log::info!(
            "leader attached in measurement mode (--force-feedback-gain 0): the trigger is read \
             and held at zero duty, so the timing summary shows bilateral's real cost with no \
             force rendered"
        );
    }

    let mut current_avgs: Vec<MovingAverage> = (0..NUM_MOTORS)
        .map(|_| MovingAverage::new(args.current_avg_window))
        .collect();
    let mut vel_avgs: Vec<MovingAverage> = (0..NUM_MOTORS)
        .map(|_| MovingAverage::new(args.vel_filter_window.max(1)))
        .collect();
    let mut prev_pos = [0f32; NUM_MOTORS];
    let mut tick: u64 = 0;
    let mut blind_ticks: u32 = 0;
    let mut current_latches: [CommsFailureLatch; NUM_MOTORS] = Default::default();
    let mut write_latch = CommsFailureLatch::default();
    let mut timing = LoopTiming::default();

    let loop_period = Duration::from_secs_f64(1.0 / args.loop_hz);
    let watchdog_timeout_ns = args.watchdog_timeout_ms * 1_000_000;

    // Seeded to all-zero (zero targets/gains -> zero PWM) so the very first tick, before Python
    // has written anything, is a safe no-op rather than reading uninitialized-looking data.
    #[allow(clippy::type_complexity)]
    let mut last_good_input: (
        [f32; NUM_MOTORS],
        [f32; NUM_MOTORS],
        [f32; NUM_MOTORS],
        [f32; NUM_MOTORS],
        u64,
    ) = (
        [0.0; NUM_MOTORS],
        [0.0; NUM_MOTORS],
        [0.0; NUM_MOTORS],
        [0.0; NUM_MOTORS],
        0,
    );

    log::info!("entering control loop at {} Hz", args.loop_hz);
    loop {
        let loop_start = std::time::Instant::now();

        let command_applied = poll_and_apply_commands(&mut bus, &layout.command);

        // On an unstable (possibly-torn) read, `seqlock_read` returns `None` -- reuse the last
        // known-good snapshot rather than ever act on torn data (see shm.rs's doc comment; this
        // was a real bug caught by tests/shm_layout_tests.rs's concurrent stress test).
        if let Some(snapshot) = shm::seqlock_read(
            &layout.input.seq,
            &layout.input.data,
            |d| {
                (
                    d.target_pos,
                    d.target_vel,
                    d.k_gain,
                    d.d_gain,
                    d.timestamp_mono_ns,
                )
            },
            8,
        ) {
            last_good_input = snapshot;
        }
        let (target_pos, target_vel, k_gain, d_gain, input_ts) = last_good_input;

        let now_ns = monotonic_ns();
        let fresh = input_is_fresh(now_ns, input_ts, watchdog_timeout_ns);

        let mut present_pos = prev_pos; // default: hold last known-good on a failed read
        let mut present_current = [0f32; NUM_MOTORS];
        let mut comms_error = false;

        match read_register_all(&mut bus, feetech::REG_PRESENT_POSITION, args.sync_read) {
            Ok(values) => {
                for i in 0..NUM_MOTORS {
                    present_pos[i] = values[i] as f32;
                }
                if blind_ticks > 0 {
                    log::warn!(
                        "read Present_Position recovered after {blind_ticks} failed tick(s)"
                    );
                }
                blind_ticks = 0;
            }
            Err(e) => {
                blind_ticks += 1;
                // Edge-triggered, for the reason spelled out on `CommsFailureLatch`.
                // `blind_ticks` is already the length of the current run, so this path needs no
                // latch of its own to recognise the first failure.
                if blind_ticks == 1 {
                    log::warn!(
                        "read Present_Position failed: {e} (holding last known positions; \
                         subsequent failures are counted in the per-second loop timing summary)"
                    );
                }
                comms_error = true;
            }
        }

        // Sample one motor's current per tick, round-robin, instead of all six on every Nth tick.
        //
        // Batching them made every Nth tick cost roughly twice a normal one, and that single fat
        // tick was the *entire* overrun population: at 300 Hz with a divisor of 10, exactly 10% of
        // ticks overran. Spreading the work keeps per-tick cost flat -- one extra transaction,
        // always -- which removes the periodic spike rather than just making it rarer. It also
        // samples each motor *more* often than a divisor of 10 did (every NUM_MOTORS ticks), and
        // current only feeds a moving average consumed by ACT at camera rate, so staggering the
        // six samples in time is of no consequence to it.
        if tick.is_multiple_of(args.current_read_divisor) {
            let i = (tick / args.current_read_divisor) as usize % NUM_MOTORS;
            match bus.read_register(MOTOR_IDS[i], feetech::REG_PRESENT_CURRENT) {
                Ok(value) => {
                    if let Some(n) = current_latches[i].recover() {
                        log::warn!(
                            "read Present_Current recovered for motor {} after {n} failed read(s)",
                            MOTOR_IDS[i]
                        );
                    }
                    current_avgs[i].push(value as f32);
                }
                Err(e) => {
                    if current_latches[i].fail() {
                        log::warn!(
                            "read Present_Current failed for motor {}: {e} (keeping last average; \
                             subsequent failures are counted in the per-second loop timing \
                             summary)",
                            MOTOR_IDS[i]
                        );
                    }
                    comms_error = true;
                    // Deliberately do NOT push a placeholder -- feeding a fake 0 into the moving
                    // average would corrupt the value ACT consumes as an observation.
                }
            }
        }
        // Every tick republishes all six averages; only the freshly-sampled one changed.
        for i in 0..NUM_MOTORS {
            present_current[i] = current_avgs[i].average();
        }
        tick = tick.wrapping_add(1);

        let dt_s = loop_period.as_secs_f32();
        let mut present_vel = [0f32; NUM_MOTORS];
        for i in 0..NUM_MOTORS {
            // Filtered, because the raw difference carries `1 / dt` ticks/s of quantisation noise
            // -- see `--vel-filter-window`. On a tick whose read failed, `present_pos` still holds
            // the previous sample, so this pushes a genuine zero rather than a fabricated spike.
            present_vel[i] = vel_avgs[i].push(finite_difference_velocity(
                prev_pos[i],
                present_pos[i],
                dt_s,
            ));
        }
        prev_pos = present_pos;

        // All 6 motors -- including the gripper -- go through the same impedance law. A rigidly
        // position-controlled gripper has no compliance: it keeps commanding full force toward
        // its target position regardless of contact, which crushes fragile objects before it can
        // ever "feel" them. Running the gripper under K/D like the arm joints lets a low K yield
        // near the target instead of continuing to squeeze.
        // Driving hard against a stale position is how a stalled servo or a flaky bus turns into a
        // runaway, so treat "cannot see where the arm is" as a reason to drop torque entirely.
        let blind = blind_ticks >= args.max_blind_ticks;
        if blind && blind_ticks == args.max_blind_ticks {
            log::error!(
                "no valid position read for {blind_ticks} consecutive ticks -- zeroing PWM until \
                 telemetry recovers"
            );
        }

        let safe = fresh && !blind;

        // The cerebellum's contribution, already clamped, slew-limited, masked to the configured
        // joints, and zeroed if this tick is unsafe or the thread has stopped publishing. Read
        // rather than computed: the control loop never waits on the GPU (see cerebellum/mod.rs).
        let (ff_pwm, cerebellum_flags) = match cerebellum.as_mut() {
            Some(c) => c.feedforward(now_ns, dt_s, safe),
            None => ([0f32; NUM_MOTORS], 0),
        };

        let mut pos_error = [0f32; NUM_MOTORS];
        for i in 0..NUM_MOTORS {
            pos_error[i] = wrapped_delta(target_pos[i], present_pos[i]);
        }

        // The feedback term is kept separate from the total because it is the climbing-fibre
        // signal: what the reflex is still having to supply is exactly what the feedforward has
        // not yet learned to supply. Handing the cerebellum the *total* instead would make its
        // teaching signal never reach zero, and it would learn until it saturated.
        let mut fb_pwm = [0f32; NUM_MOTORS];
        let mut pwm_cmd = [0f32; NUM_MOTORS];
        let mut sync_values: Vec<(u8, u32)> = Vec::with_capacity(NUM_MOTORS);
        for i in 0..NUM_MOTORS {
            pwm_cmd[i] = if safe {
                fb_pwm[i] = impedance_pwm(
                    k_gain[i],
                    d_gain[i],
                    target_pos[i],
                    present_pos[i],
                    target_vel[i],
                    present_vel[i],
                    args.pwm_max,
                );
                // Re-clamped after the sum: each term is bounded on its own, and the total has to
                // be too. Soft limits are applied last so they still veto a feedforward that would
                // drive a joint further past its limit.
                let total = (fb_pwm[i] + ff_pwm[i]).clamp(-args.pwm_max, args.pwm_max);
                apply_soft_limits(total, present_pos[i], args.pos_min, args.pos_max)
            } else {
                0.0 // watchdog / blind: fail-safe zero output, independent of Python
            };
            // `pwm_cmd` is in the position-increasing convention; the hardware's direction bit is
            // only applied here, so --invert-pwm cannot skew the control law or the soft limits.
            let commanded = if args.invert_pwm {
                -pwm_cmd[i]
            } else {
                pwm_cmd[i]
            };
            let raw = feetech::encode_sign_magnitude(commanded as i32, args.pwm_sign_bit);
            sync_values.push((MOTOR_IDS[i], raw as u32));
        }
        match bus.sync_write(feetech::REG_GOAL_PWM, &sync_values) {
            Ok(()) => {
                if let Some(n) = write_latch.recover() {
                    log::warn!("sync_write Goal_PWM recovered after {n} failed tick(s)");
                }
            }
            Err(e) => {
                if write_latch.fail() {
                    log::warn!(
                        "sync_write Goal_PWM failed: {e} (subsequent failures are counted in the \
                         per-second loop timing summary)"
                    );
                }
                comms_error = true;
            }
        }

        // The leader is driven from the follower's *gripper* tracking error: ~0 while the gripper
        // closes freely, growing once an object stops it. `fresh && !blind` gates it on the same
        // condition the follower's own output is gated on -- rendering force from an error the
        // follower is no longer acting on would push the operator's hand for no reason.
        if let Some(l) = leader.as_mut() {
            l.tick(pos_error[GRIPPER_INDEX], safe, dt_s);
        }

        // Hand the cerebellum this tick's sensory picture. A seqlock write: no lock, no blocking,
        // and no way for a slow or dead cerebellum thread to hold up the control loop.
        if let Some(c) = cerebellum.as_ref() {
            c.publish(
                SensoryState {
                    present_pos,
                    present_vel,
                    pos_error,
                    present_current,
                    fb_pwm,
                },
                safe,
                now_ns,
            );
        }

        let mut fault_flags = 0u32;
        if !fresh {
            fault_flags |= shm::FAULT_WATCHDOG_TIMEOUT;
        }
        if comms_error {
            fault_flags |= shm::FAULT_COMMS_ERROR;
        }
        if leader.as_ref().is_some_and(|l| l.comms_error) {
            fault_flags |= shm::FAULT_LEADER_COMMS_ERROR;
        }

        shm::seqlock_write(&layout.output.seq, &mut layout.output.data, |o| {
            o.timestamp_mono_ns = now_ns;
            o.present_pos = present_pos;
            o.present_vel = present_vel;
            o.present_current_avg = present_current;
            o.pwm_cmd_debug = pwm_cmd;
            o.ff_pwm_debug = ff_pwm;
            o.cerebellum_flags = cerebellum_flags;
            o.fault_flags = fault_flags;
            o.leader_gripper_pos = leader.as_ref().map_or(0.0, |l| l.present_pos);
            o.leader_gripper_vel = leader.as_ref().map_or(0.0, |l| l.present_vel);
            o.leader_gripper_pwm = leader.as_ref().map_or(0.0, |l| l.pwm_cmd);
        });

        let shutdown_command =
            command_applied && layout.command.cmd_kind == shm::CommandKind::Shutdown as u32;
        if shutdown_command || SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
            if shutdown_command {
                log::info!("received Shutdown command, exiting control loop");
            } else {
                log::info!("received SIGINT/SIGTERM, exiting control loop");
            }
            if let Some(l) = leader.as_mut() {
                l.release();
            }
            break;
        }

        let elapsed = loop_start.elapsed();
        timing.record(
            elapsed,
            loop_period,
            comms_error,
            leader.as_ref().map(|l| l.last_tick),
        );
        if let Some(summary) = timing.take_summary_if_due(loop_period) {
            log::info!("{summary}");
            if let Some(c) = cerebellum.as_ref() {
                log::info!("{}", c.summarise());
            }
        }
        if elapsed < loop_period {
            std::thread::sleep(loop_period - elapsed);
        }
    }

    // Joins the cerebellum thread, which persists its weights on the way out. Deliberately after
    // the loop rather than in a `Drop`, so that a session's learning is only saved on an exit we
    // actually reached -- and so the log line about it lands before the process goes.
    if let Some(c) = cerebellum.take() {
        c.stop();
    }
}

/// Turns the parsed CLI into a [`CerebellumConfig`], failing on anything that would otherwise
/// only show up as strange behaviour on a real arm.
fn build_cerebellum_config(args: &Cli) -> Result<CerebellumConfig, String> {
    let joints = cerebellum::parse_joints(&args.cerebellum_joints)
        .map_err(|e| format!("--cerebellum-joints: {e}"))?;

    // Sharing the RT core would put a thread that blocks on a GPU fence onto the one core the
    // whole design exists to keep free of anything that blocks. Worth refusing rather than
    // warning about: the symptom would be control-loop overruns that look like a bus problem.
    if args.cerebellum_cpu_core == Some(args.cpu_core) && args.cerebellum_backend != Backend::Off {
        return Err(format!(
            "--cerebellum-cpu-core {} is the control loop's own core -- pick a housekeeping core",
            args.cpu_core
        ));
    }
    if args.cerebellum_priority != 0 && args.cerebellum_priority >= args.priority {
        return Err(format!(
            "--cerebellum-priority {} is not below --priority {} -- the reflex must always be able \
             to preempt the cerebellum",
            args.cerebellum_priority, args.priority
        ));
    }

    Ok(CerebellumConfig {
        backend: args.cerebellum_backend,
        gc_dim: args.cerebellum_gc_dim,
        seed: args.cerebellum_seed,
        hz: args.cerebellum_hz,
        rate: args.cerebellum_rate,
        leak: args.cerebellum_leak,
        sparsity: args.cerebellum_sparsity,
        golgi_gain: args.cerebellum_golgi_gain,
        trace_tau_s: args.cerebellum_trace_tau_s,
        cf_tau_s: args.cerebellum_cf_tau_s,
        ff_max: args.cerebellum_ff_max,
        ff_slew: args.cerebellum_ff_slew,
        vel_gate: args.cerebellum_vel_gate,
        error_gate: args.cerebellum_error_gate,
        joints,
        staleness_ms: args.cerebellum_staleness_ms,
        cpu_core: args.cerebellum_cpu_core,
        priority: args.cerebellum_priority,
        weights_path: args.cerebellum_weights.clone(),
    })
}

/// Edge-triggered gate for a repeatedly-failing bus transaction.
///
/// Same reasoning as `LoopTiming` below, applied to the failure path: when the bus is dead or
/// unplugged *every* tick fails, so a per-tick `warn!` is ~800 lines a second at 400 Hz. Those
/// writes are not free -- stderr is usually a pipe, and if whatever is on the far end reads
/// slowly (or, as in a piped smoke test, not until the process exits) the blocking write throttles
/// the control loop itself. The observable symptom is the per-second timing summary vanishing
/// entirely, which is precisely the line that would have explained what was wrong.
///
/// The *count* still reaches the operator, via `LoopTiming`'s `comms_errors`. This only decides
/// when the individual error text is worth a line: once entering a run of failures, and once on
/// leaving it. The `blind_ticks` fail-safe ERROR is left alone -- it is already edge-triggered.
#[derive(Default)]
struct CommsFailureLatch {
    failing: bool,
    consecutive: u64,
}

impl CommsFailureLatch {
    /// Records a failure. Returns true only for the first of a consecutive run, i.e. when the
    /// error text is new information rather than a repeat.
    fn fail(&mut self) -> bool {
        self.consecutive += 1;
        !std::mem::replace(&mut self.failing, true)
    }

    /// Records a success. Returns `Some(n)` -- the length of the run that just ended -- only when
    /// this success is the recovery from a run of failures.
    fn recover(&mut self) -> Option<u64> {
        std::mem::replace(&mut self.failing, false).then(|| std::mem::take(&mut self.consecutive))
    }
}

/// Rolling loop-timing statistics, summarised once a second.
///
/// Logging every overrun individually is self-defeating: at 1 kHz a saturated bus emits a thousand
/// lines a second, and writing them costs more time than it reports. A periodic min/mean/max plus
/// an overrun count says strictly more while staying off the hot path.
#[derive(Default)]
struct LoopTiming {
    ticks: u64,
    overruns: u64,
    /// Ticks where a serial read or write failed. Reported alongside the timings because the two
    /// are usually the same event: a transaction that had to wait on the timeout shows up as an
    /// outlier `max`, and correlating them says whether a spike was the bus or the scheduler.
    comms_errors: u64,
    total: Duration,
    min: Option<Duration>,
    max: Duration,
    /// Time spent on the leader's bus, summed. Reported separately because the whole question
    /// about bilateral is whether the second arm fits in the period, and a combined mean cannot
    /// answer it.
    leader_total: Duration,
    leader_ticks: u64,
}

impl LoopTiming {
    fn record(
        &mut self,
        elapsed: Duration,
        period: Duration,
        comms_error: bool,
        leader: Option<Duration>,
    ) {
        self.ticks += 1;
        if let Some(d) = leader {
            self.leader_total += d;
            self.leader_ticks += 1;
        }
        if comms_error {
            self.comms_errors += 1;
        }
        self.total += elapsed;
        self.max = self.max.max(elapsed);
        self.min = Some(self.min.map_or(elapsed, |m: Duration| m.min(elapsed)));
        if elapsed > period {
            self.overruns += 1;
        }
    }

    /// Returns a summary and resets once roughly a second's worth of ticks have accumulated.
    fn take_summary_if_due(&mut self, period: Duration) -> Option<String> {
        let ticks_per_second = (1.0 / period.as_secs_f64()).round() as u64;
        if self.ticks < ticks_per_second.max(1) {
            return None;
        }
        let mean = self.total / self.ticks as u32;
        let leader = if self.leader_ticks > 0 {
            format!(
                ", leader {:?}/tick",
                self.leader_total / self.leader_ticks as u32
            )
        } else {
            String::new()
        };
        let summary = format!(
            "loop timing over {} ticks: min {:?} / mean {:?} / max {:?} (period {:?}); \
             {} overruns, {} comms errors{}",
            self.ticks,
            self.min.unwrap_or_default(),
            mean,
            self.max,
            period,
            self.overruns,
            self.comms_errors,
            leader,
        );
        *self = Self::default();
        Some(summary)
    }
}
