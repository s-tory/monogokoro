//! Pure control-law math and the live command-channel handler, kept free of I/O so it can be
//! unit-tested without a serial port.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;

use crate::feetech::{self, FeetechBus};
use crate::shm::{CommandKind, CommandRegion};

/// Fixed-sample-count moving average of `Present_Current` for one motor (per the plan's decision:
/// fixed window size, not a time window).
pub struct MovingAverage {
    window: VecDeque<f32>,
    capacity: usize,
    sum: f32,
}

impl MovingAverage {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "moving average capacity must be positive");
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }

    /// Pushes a new sample and returns the updated average.
    pub fn push(&mut self, sample: f32) -> f32 {
        self.window.push_back(sample);
        self.sum += sample;
        if self.window.len() > self.capacity {
            if let Some(oldest) = self.window.pop_front() {
                self.sum -= oldest;
            }
        }
        self.average()
    }

    pub fn average(&self) -> f32 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f32
        }
    }
}

/// Encoder counts per revolution on the sts3215; `Present_Position` wraps 4095 -> 0.
pub const ENCODER_RESOLUTION: f32 = 4096.0;

/// Shortest-path difference between two positions on a wrapping encoder.
///
/// `Present_Position` is a 12-bit value, so it rolls over mid-travel -- and no `Homing_Offset`
/// makes that go away in general: it only moves *where* the seam sits, and a continuously-rotating
/// joint (the SO101 calibrates `wrist_roll` over the full [0, 4095]) has no seam-free placement at
/// all. Subtracting naively turns a one-count rollover into a ~4096-count error, which saturates
/// PWM and slams the joint. Wrapping the difference into +/- half a revolution keeps the error
/// continuous across the seam, so the controller always drives the short way round.
pub fn wrapped_delta(a: f32, b: f32) -> f32 {
    let half = ENCODER_RESOLUTION / 2.0;
    let mut d = a - b;
    while d > half {
        d -= ENCODER_RESOLUTION;
    }
    while d < -half {
        d += ENCODER_RESOLUTION;
    }
    d
}

/// Open-loop impedance control law:
/// `pwm = clamp(K * wrapped(target_pos - present_pos) + D * (target_vel - present_vel), +/-pwm_max)`.
///
/// This is the PWM open-loop approach chosen for SO101's STS3215 servos, which (unlike this
/// repo's Robstride/Damiao CAN actuators) have no host-streamable torque/current command
/// register -- an accepted, noisier-than-true-torque-control trade-off.
#[allow(clippy::too_many_arguments)]
pub fn impedance_pwm(
    k: f32,
    d: f32,
    target_pos: f32,
    present_pos: f32,
    target_vel: f32,
    present_vel: f32,
    pwm_max: f32,
) -> f32 {
    let raw = k * wrapped_delta(target_pos, present_pos) + d * (target_vel - present_vel);
    raw.clamp(-pwm_max, pwm_max)
}

/// PWM to render on a leader-side joint so the operator feels the follower's tracking error.
///
/// `follower_error` is the follower joint's own `target - present` gap, in follower encoder
/// counts: near zero while the follower moves freely, growing once something blocks it. Scaling
/// that into leader duty is what turns the trigger into a haptic display, with no force sensor
/// anywhere in the loop.
///
/// `feedback_gain` is signed on purpose. Which way "closed" runs depends on each gripper's own
/// calibration, and the two arms need not agree, so the correct sign is a property of a particular
/// pair of arms -- measured like `--invert-pwm` was, not derived. `damping` is separate and always
/// opposes leader motion, so it stays correct whichever sign the gain takes, and it is what keeps
/// the operator from feeling a bare spring.
pub fn force_feedback_pwm(
    feedback_gain: f32,
    damping: f32,
    follower_error: f32,
    leader_vel: f32,
    pwm_max: f32,
) -> f32 {
    let raw = feedback_gain * follower_error - damping * leader_vel;
    raw.clamp(-pwm_max, pwm_max)
}

/// The first motor in a freshly read batch whose position could not have got there, given the
/// last accepted batch and a budget in counts.
///
/// Returns the motor's index and its (wrapped) step, so the caller can name it. `budget` is the
/// caller's slew limit times the time since the last *accepted* sample -- not since the last tick
/// -- so a joint that really did move during a blind run is not rejected on the way back.
///
/// The comparison goes through [`wrapped_delta`] because a joint crossing the 4095/0 boundary
/// moves one count while its raw reading moves 4095, and rejecting that would make the check fire
/// hardest exactly where the encoder is most awkward.
pub fn first_implausible_step(values: &[i32], prev: &[f32], budget: f32) -> Option<(usize, f32)> {
    values
        .iter()
        .zip(prev)
        .enumerate()
        .find_map(|(i, (&value, &previous))| {
            let step = wrapped_delta(value as f32, previous);
            (step.abs() > budget).then_some((i, step))
        })
}

/// Zeroes any command that would drive a servo further past a soft position limit.
///
/// `pwm` and the limits are both in the *position-increasing* convention: positive `pwm` is
/// whatever moves `Present_Position` up, regardless of how the hardware's direction bit is wired
/// (that is applied later, at encode time). Motion back toward the middle is always allowed, so a
/// joint that is already past a limit can recover instead of being stuck there.
///
/// That last promise is what `last_in_range` is for. The limits are an interval on a line while
/// [`wrapped_delta`] treats position as a circle, and the two disagree about which way "back" is.
/// Measured on the arm: a joint sitting at 4051 with its target at 123 has every positive command
/// looking like "further past `pos_max`" to the raw rule, while the short way back to that target
/// is positive and runs through the 4095/0 seam. Judging by the raw sign parks the joint there for
/// as long as the target stands -- the exact opposite of recovery.
///
/// So once the joint is outside the limits the question becomes *does this move it toward where it
/// was last seen legally*, which the loop observed rather than inferred. `None` means it has not
/// been seen inside the limits at all since startup: there is no observation to appeal to, so this
/// falls back to the raw rule, which is right whenever the travel does not cross the seam and
/// declines to guess when it does.
pub fn apply_soft_limits(
    pwm: f32,
    present_pos: f32,
    pos_min: f32,
    pos_max: f32,
    last_in_range: Option<f32>,
) -> f32 {
    let past_min = present_pos <= pos_min;
    let past_max = present_pos >= pos_max;
    if !past_min && !past_max {
        return pwm;
    }
    match last_in_range {
        Some(back) => {
            let escape = wrapped_delta(back, present_pos);
            if escape == 0.0 || pwm.signum() == escape.signum() {
                pwm
            } else {
                0.0
            }
        }
        None => {
            if (past_min && pwm < 0.0) || (past_max && pwm > 0.0) {
                0.0
            } else {
                pwm
            }
        }
    }
}

/// Finite-difference velocity estimate between two successive position samples `dt_s` apart.
/// Preferred over the `Present_Velocity` register for this hardware, matching this repo's own
/// precedent (`JointVelocityProcessorStep` in `src/lerobot/rl/joint_observations_processor.py`).
/// The position difference is taken with [`wrapped_delta`] for the same reason the error term is:
/// a rollover between two samples would otherwise read as ~4096 counts of travel in one tick, i.e.
/// a colossal phantom velocity that the D term turns straight into a saturated command.
pub fn finite_difference_velocity(prev_pos: f32, curr_pos: f32, dt_s: f32) -> f32 {
    if dt_s <= 0.0 {
        0.0
    } else {
        wrapped_delta(curr_pos, prev_pos) / dt_s
    }
}

/// Watchdog check: `true` (fresh/safe) if the input region's timestamp has advanced within
/// `timeout_ns` of `now_ns`. Rust's watchdog is the primary hardware safety net and does not
/// depend on Python at all -- on staleness, the caller must zero the PWM output.
pub fn input_is_fresh(now_ns: u64, last_input_ns: u64, timeout_ns: u64) -> bool {
    now_ns.saturating_sub(last_input_ns) <= timeout_ns
}

/// Polls the live command channel and applies at most one pending command per call. Returns
/// `true` if a command was applied (caller may want to log/rate-limit around this).
pub fn poll_and_apply_commands(bus: &mut FeetechBus, cmd: &CommandRegion) -> bool {
    let cmd_seq = cmd.cmd_seq.load(Ordering::Acquire);
    let ack_seq = cmd.ack_seq.load(Ordering::Acquire);
    if cmd_seq == ack_seq {
        return false; // nothing new
    }

    let kind = cmd.cmd_kind;
    let motor_id = cmd.motor_id as u8;
    let payload = cmd.payload;
    let status = apply_command(bus, kind, motor_id, payload);

    cmd.status.store(status, Ordering::Release);
    cmd.ack_seq.store(cmd_seq, Ordering::Release);
    true
}

/// Writes the EPROM-resident `Operating_Mode` register, which the servo silently ignores unless
/// torque is off **and** the EPROM write-protect latch is cleared first.
///
/// Getting this wrong is not a loud failure: the write is dropped, the servo stays in whatever
/// mode it was in (POSITION, out of the box), and the daemon's PWM commands then land in
/// `Goal_Time` where they mean "travel time" instead of duty cycle. The arm drives to whatever
/// stale `Goal_Position` it still holds and sits there rigidly -- looking like a control-law bug
/// rather than a failed register write. Mirrors
/// `FeetechMotorsBus.disable_torque` + `SOFollower.configure`'s `torque_disabled()` on the Python
/// side, which do exactly this dance.
fn write_operating_mode(bus: &mut FeetechBus, motor_id: u8, mode: u32) -> std::io::Result<()> {
    bus.write_register(motor_id, feetech::REG_TORQUE_ENABLE, 0)?;
    bus.write_register(motor_id, feetech::REG_LOCK, 0)?; // unlock EPROM
    bus.write_register(motor_id, feetech::REG_OPERATING_MODE, mode)?;
    // An EPROM write is a flash commit, not a RAM poke -- the servo needs time to settle and can
    // NAK or answer stale until it finishes. Give it a beat before anyone reads the value back.
    std::thread::sleep(EPROM_COMMIT_DELAY);
    // Left unlocked and torque-off on purpose: the caller follows up with SetTorqueEnable(1),
    // which re-latches `Lock` -- same ordering as the Python `configure()` path.
    Ok(())
}

/// Settle time allowed after an EPROM write before reading the register back.
const EPROM_COMMIT_DELAY: std::time::Duration = std::time::Duration::from_millis(20);

/// How many times to attempt the write+verify cycle before giving up.
const OPERATING_MODE_ATTEMPTS: usize = 3;

/// Reads `Operating_Mode` back to confirm it actually took, since a rejected EPROM write is
/// otherwise silent.
fn verify_operating_mode(bus: &mut FeetechBus, motor_id: u8, expected: u32) -> std::io::Result<()> {
    let actual = bus.read_register(motor_id, feetech::REG_OPERATING_MODE)?;
    if actual as u32 != expected {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "motor {motor_id}: Operating_Mode readback is {actual}, expected {expected} -- \
                 the EPROM write was rejected"
            ),
        ));
    }
    Ok(())
}

/// Write + read-back the operating mode, retrying a few times.
///
/// Retries matter here because the failure modes overlap: a genuinely unreachable servo and one
/// that was merely busy committing flash both surface as an error on the first attempt. Retrying
/// separates them -- a wiring/ID problem fails every time, a timing hiccup clears.
fn set_operating_mode(bus: &mut FeetechBus, motor_id: u8, mode: u32) -> std::io::Result<()> {
    let mut last_err = None;
    for attempt in 1..=OPERATING_MODE_ATTEMPTS {
        match write_operating_mode(bus, motor_id, mode)
            .and_then(|_| verify_operating_mode(bus, motor_id, mode))
        {
            Ok(()) => {
                if attempt > 1 {
                    log::warn!("motor {motor_id}: Operating_Mode took {attempt} attempts");
                }
                return Ok(());
            }
            Err(e) => {
                log::warn!(
                    "motor {motor_id}: Operating_Mode attempt {attempt}/{OPERATING_MODE_ATTEMPTS} \
                     failed: {e}"
                );
                last_err = Some(e);
            }
        }
    }
    Err(last_err.expect("loop runs at least once"))
}

/// Puts one servo into PWM mode with torque enabled, the state the impedance and force-feedback
/// laws both assume. Split out so the leader's single gripper can reach it without going through
/// the shared-memory command channel, which only the follower's Python client drives.
pub fn prepare_pwm_motor(bus: &mut FeetechBus, motor_id: u8) -> std::io::Result<()> {
    set_operating_mode(bus, motor_id, feetech::OPERATING_MODE_PWM)?;
    bus.write_register(motor_id, feetech::REG_TORQUE_ENABLE, 1)
}

/// One-time hardware fixups applied to every servo at daemon startup, mirroring
/// `FeetechMotorsBus.configure_motors` on the Python side.
///
/// Both writes target EPROM, so they need the same torque-off + unlock dance as `Operating_Mode`.
/// Skipping them is not harmless:
///
/// * `Return_Delay_Time` ships at 250 (500 us). With six servos answering one SYNC_READ back to
///   back that is milliseconds of pure dead air per tick, which alone can blow a 1 kHz budget.
/// * `Phase` bit 4 left set makes `Present_Position` wrap/go negative instead of staying inside
///   `[0, resolution-1]`, which silently corrupts the impedance error and the velocity estimate.
///
/// These are persistent EPROM settings, so a servo previously set up through the Python stack may
/// already be correct -- this just makes the daemon self-sufficient rather than depending on it.
pub fn apply_startup_config(bus: &mut FeetechBus, motor_ids: &[u8]) {
    for &id in motor_ids {
        if let Err(e) = configure_one_motor(bus, id) {
            log::warn!(
                "motor {id}: startup configuration failed: {e} (continuing -- expect extra latency \
                 or bad position readings if this servo was not already configured)"
            );
        }
    }
}

/// Logs each servo's stored `Homing_Offset` so it is obvious from the daemon's own output whether
/// a calibration is in effect. An all-zero column here means positions are raw encoder counts and
/// any joint whose travel crosses 4095/0 will report a full-scale jump.
pub fn log_homing_offsets(bus: &mut FeetechBus, motor_ids: &[u8]) {
    for &id in motor_ids {
        match bus.read_register(id, feetech::REG_HOMING_OFFSET) {
            Ok(raw) => {
                let offset =
                    feetech::decode_sign_magnitude(raw as u16, feetech::HOMING_OFFSET_SIGN_BIT);
                log::info!("motor {id}: stored Homing_Offset = {offset}");
            }
            Err(e) => log::warn!("motor {id}: could not read Homing_Offset: {e}"),
        }
    }
}

fn configure_one_motor(bus: &mut FeetechBus, motor_id: u8) -> std::io::Result<()> {
    bus.write_register(motor_id, feetech::REG_TORQUE_ENABLE, 0)?;
    bus.write_register(motor_id, feetech::REG_LOCK, 0)?; // unlock EPROM

    bus.write_register(motor_id, feetech::REG_RETURN_DELAY_TIME, 0)?;
    std::thread::sleep(EPROM_COMMIT_DELAY);

    // Only rewrite Phase if the bit is actually set -- an EPROM write costs a flash cycle.
    let phase = bus.read_register(motor_id, feetech::REG_PHASE)? as u32;
    if phase & feetech::PHASE_ANGLE_FEEDBACK_BIT != 0 {
        log::info!("motor {motor_id}: clearing Phase bit 4 (angle feedback mode)");
        bus.write_register(
            motor_id,
            feetech::REG_PHASE,
            phase & !feetech::PHASE_ANGLE_FEEDBACK_BIT,
        )?;
        std::thread::sleep(EPROM_COMMIT_DELAY);
    }
    Ok(())
}

/// Writes a motor's calibration (homing offset + position limits) into its EPROM, mirroring
/// `FeetechMotorsBus.write_calibration` on the Python side.
///
/// Without the homing offset a joint whose travel straddles the 4095/0 encoder boundary reports a
/// position that jumps the full scale mid-motion. The impedance law reads that as an instantaneous
/// ~4095-tick error, saturates PWM, and slams the joint -- a runaway that `--invert-pwm` cannot
/// fix, because the discontinuity is in the measurement, not the drive direction.
///
/// `homing_offset` is sign-magnitude at bit 11, so it must be encoded rather than cast: a plain
/// `as u32` on a negative value saturates to 0 in Rust and silently writes the wrong offset.
fn write_calibration(
    bus: &mut FeetechBus,
    motor_id: u8,
    homing_offset: f32,
    range_min: u32,
    range_max: u32,
) -> std::io::Result<()> {
    bus.write_register(motor_id, feetech::REG_TORQUE_ENABLE, 0)?;
    bus.write_register(motor_id, feetech::REG_LOCK, 0)?; // unlock EPROM

    let wanted = homing_offset.round() as i32;
    let encoded = feetech::encode_sign_magnitude(wanted, feetech::HOMING_OFFSET_SIGN_BIT);
    bus.write_register(motor_id, feetech::REG_HOMING_OFFSET, encoded as u32)?;
    std::thread::sleep(EPROM_COMMIT_DELAY);
    bus.write_register(motor_id, feetech::REG_MIN_POSITION_LIMIT, range_min)?;
    std::thread::sleep(EPROM_COMMIT_DELAY);
    bus.write_register(motor_id, feetech::REG_MAX_POSITION_LIMIT, range_max)?;
    std::thread::sleep(EPROM_COMMIT_DELAY);

    // Read it back. A silently-dropped EPROM write here is not cosmetic: without the offset a
    // joint's travel can straddle the 4095/0 wrap, `Present_Position` jumps a full turn mid-motion
    // and the impedance law slams the joint. This is the third register where an unverified write
    // looked like a control bug, so verify rather than assume.
    let raw = bus.read_register(motor_id, feetech::REG_HOMING_OFFSET)?;
    let actual = feetech::decode_sign_magnitude(raw as u16, feetech::HOMING_OFFSET_SIGN_BIT);
    if actual != wanted {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "motor {motor_id}: Homing_Offset readback is {actual}, expected {wanted} -- the \
                 EPROM write was rejected, so positions will still wrap"
            ),
        ));
    }
    log::info!("motor {motor_id}: calibration written (homing_offset={actual}, range=[{range_min}, {range_max}])");
    Ok(())
}

fn apply_command(bus: &mut FeetechBus, kind: u32, motor_id: u8, payload: [f32; 4]) -> u32 {
    let result = if kind == CommandKind::SetOperatingMode as u32 {
        set_operating_mode(bus, motor_id, payload[0] as u32)
    } else if kind == CommandKind::SetTorqueEnable as u32 {
        let enable = payload[0] as u32;
        // Mirror Python's enable_torque/disable_torque, which also drive the EPROM `Lock` latch:
        // enabling re-locks, disabling unlocks so a following EPROM write can succeed.
        bus.write_register(motor_id, feetech::REG_TORQUE_ENABLE, enable)
            .and_then(|_| bus.write_register(motor_id, feetech::REG_LOCK, enable))
    } else if kind == CommandKind::SetPidCoefficients as u32 {
        // payload = [p, i, d, _]
        bus.write_register(motor_id, feetech::REG_P_COEFFICIENT, payload[0] as u32)
            .and_then(|_| {
                bus.write_register(motor_id, feetech::REG_I_COEFFICIENT, payload[1] as u32)
            })
            .and_then(|_| {
                bus.write_register(motor_id, feetech::REG_D_COEFFICIENT, payload[2] as u32)
            })
    } else if kind == CommandKind::SetCalibration as u32 {
        // payload = [homing_offset, range_min, range_max, _]
        write_calibration(
            bus,
            motor_id,
            payload[0],
            payload[1] as u32,
            payload[2] as u32,
        )
    } else {
        Ok(()) // CommandKind::None / Shutdown handled by the caller's main loop, not here
    };
    match result {
        Ok(()) => 0,
        Err(e) => {
            // Python only sees the nonzero status code, so log the reason here -- a rejected
            // EPROM write and a dead serial link are very different problems.
            log::error!("command kind={kind} motor={motor_id} failed: {e}");
            1
        }
    }
}
