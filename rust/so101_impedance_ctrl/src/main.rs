//! Entry point for the SO101 impedance-control RT daemon.
//!
//! This process exclusively owns the SO101's single half-duplex serial bus (all 6 STS3215
//! servos, including the gripper) and exposes control to Python only through the shared-memory
//! protocol in `shm.rs`. See `README.md` for build/run instructions and PREEMPT_RT environment
//! prerequisites, and the top-level plan doc for the overall architecture.

use std::time::Duration;

use clap::Parser;
use shared_memory::ShmemConf;

use nix::unistd::{Gid, Uid};
use so101_impedance_ctrl::control::{
    apply_soft_limits, apply_startup_config, finite_difference_velocity, impedance_pwm,
    input_is_fresh, log_homing_offsets, poll_and_apply_commands, MovingAverage,
};
use so101_impedance_ctrl::feetech::{self, FeetechBus};
use so101_impedance_ctrl::rt;
use so101_impedance_ctrl::shm::{self, ShmLayout, NUM_MOTORS};

/// Motor IDs, matching `src/lerobot/robots/so_follower/so_follower.py`'s `Motor(id, "sts3215", ...)`
/// assignments: shoulder_pan=1, shoulder_lift=2, elbow_flex=3, wrist_flex=4, wrist_roll=5,
/// gripper=6.
const MOTOR_IDS: [u8; NUM_MOTORS] = [1, 2, 3, 4, 5, 6];

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
         pos_limits=[{}, {}] max_blind_ticks={} watchdog_ms={} serial_timeout_ms={}",
        args.invert_pwm,
        args.pwm_sign_bit,
        args.loop_hz,
        args.pwm_max,
        args.sync_read,
        args.current_read_divisor,
        args.pos_min,
        args.pos_max,
        args.max_blind_ticks,
        args.watchdog_timeout_ms,
        args.serial_timeout_ms,
    );

    rt::apply_rt_settings(args.cpu_core, args.priority);

    let mut bus = FeetechBus::open(
        &args.port,
        args.baud,
        Duration::from_millis(args.serial_timeout_ms),
    )
    .expect("failed to open SO101 serial port");

    apply_startup_config(&mut bus, &MOTOR_IDS);
    log_homing_offsets(&mut bus, &MOTOR_IDS);

    let mut current_avgs: Vec<MovingAverage> = (0..NUM_MOTORS)
        .map(|_| MovingAverage::new(args.current_avg_window))
        .collect();
    let mut prev_pos = [0f32; NUM_MOTORS];
    let mut tick: u64 = 0;
    let mut blind_ticks: u32 = 0;
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
                blind_ticks = 0;
            }
            Err(e) => {
                blind_ticks += 1;
                log::warn!(
                    "read Present_Position failed: {e} (holding last known positions, blind for \
                     {blind_ticks} tick(s))"
                );
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
                    current_avgs[i].push(value as f32);
                }
                Err(e) => {
                    log::warn!(
                        "read Present_Current failed for motor {}: {e} (keeping last average)",
                        MOTOR_IDS[i]
                    );
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
            present_vel[i] = finite_difference_velocity(prev_pos[i], present_pos[i], dt_s);
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

        let mut pwm_cmd = [0f32; NUM_MOTORS];
        let mut sync_values: Vec<(u8, u32)> = Vec::with_capacity(NUM_MOTORS);
        for i in 0..NUM_MOTORS {
            pwm_cmd[i] = if fresh && !blind {
                let raw_pwm = impedance_pwm(
                    k_gain[i],
                    d_gain[i],
                    target_pos[i],
                    present_pos[i],
                    target_vel[i],
                    present_vel[i],
                    args.pwm_max,
                );
                apply_soft_limits(raw_pwm, present_pos[i], args.pos_min, args.pos_max)
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
        if let Err(e) = bus.sync_write(feetech::REG_GOAL_PWM, &sync_values) {
            log::warn!("sync_write Goal_PWM failed: {e}");
            comms_error = true;
        }

        let mut fault_flags = 0u32;
        if !fresh {
            fault_flags |= shm::FAULT_WATCHDOG_TIMEOUT;
        }
        if comms_error {
            fault_flags |= shm::FAULT_COMMS_ERROR;
        }

        shm::seqlock_write(&layout.output.seq, &mut layout.output.data, |o| {
            o.timestamp_mono_ns = now_ns;
            o.present_pos = present_pos;
            o.present_vel = present_vel;
            o.present_current_avg = present_current;
            o.pwm_cmd_debug = pwm_cmd;
            o.fault_flags = fault_flags;
        });

        if command_applied && layout.command.cmd_kind == shm::CommandKind::Shutdown as u32 {
            log::info!("received Shutdown command, exiting control loop");
            break;
        }

        let elapsed = loop_start.elapsed();
        timing.record(elapsed, loop_period, comms_error);
        if let Some(summary) = timing.take_summary_if_due(loop_period) {
            log::info!("{summary}");
        }
        if elapsed < loop_period {
            std::thread::sleep(loop_period - elapsed);
        }
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
}

impl LoopTiming {
    fn record(&mut self, elapsed: Duration, period: Duration, comms_error: bool) {
        self.ticks += 1;
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
        let summary = format!(
            "loop timing over {} ticks: min {:?} / mean {:?} / max {:?} (period {:?}); \
             {} overruns, {} comms errors",
            self.ticks,
            self.min.unwrap_or_default(),
            mean,
            self.max,
            period,
            self.overruns,
            self.comms_errors,
        );
        *self = Self::default();
        Some(summary)
    }
}
