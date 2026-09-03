//! Pure control-law math and the live command-channel handler, kept free of I/O so it can be
//! unit-tested without a serial port.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::Duration;

use crate::feetech::{self, FeetechBus};
use crate::shm::{CommandKind, CommandRegion, NUM_MOTORS};

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
/// makes that go away in general: it only moves *where* the seam sits, and a joint whose travel
/// covers the whole circle has no seam-free placement at all. Subtracting naively turns a
/// one-count rollover into a ~4096-count error, which saturates PWM and slams the joint. Wrapping
/// the difference into +/- half a revolution keeps the error continuous across the seam, so the
/// controller always drives the short way round.
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
/// Returns the motor's index and its (wrapped) step, so the caller can name it. `budget` is one
/// tick's worth of the caller's slew limit: a joint that really did move further than that, during
/// a blind run, is admitted by corroboration instead (see [`PositionGate`]) rather than by
/// relaxing this.
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

/// Admits position batches, holding back one that could not have happened until a second read
/// agrees with it.
///
/// The naive version of this scaled the budget by how long the loop had been blind, so that an arm
/// that really fell during a bus blackout would not have its recovery rejected. That is the right
/// goal and the wrong mechanism, and the arm demonstrated why: after a ~24-tick blackout the
/// budget had grown past a thousand counts, and a single misparsed reply putting `wrist_flex` at 0
/// -- while it was really at 1187 -- sailed straight through. The longer the loop had been blind,
/// the less it checked, which is exactly backwards.
///
/// Corroboration inverts that. A batch that fails the one-tick budget is not accepted and not
/// discarded: it is held, and the *next* read is measured against it. Two consecutive reads that
/// agree describe a joint that is really there, however far it travelled while nobody was looking;
/// a misparse has to be wrong the same way twice to get through. The cost is one extra tick of
/// staleness on a genuine excursion, against a check that never weakens.
pub struct PositionGate {
    /// One tick's worth of slew, in counts.
    budget: f32,
    /// A batch that failed against the last accepted one, waiting to see if the next read agrees.
    pending: Option<[f32; NUM_MOTORS]>,
}

/// Why a batch was not accepted, for the caller to log.
#[derive(Debug, Clone, Copy)]
pub struct GateRejection {
    pub motor: usize,
    pub step: f32,
    /// True once a batch is being held and the next read will be measured against it, so the
    /// caller can say whether it is waiting for corroboration or has just started.
    pub corroborating: bool,
}

impl PositionGate {
    pub fn new(budget: f32) -> Self {
        Self {
            budget,
            pending: None,
        }
    }

    /// Offers a freshly read batch. `Ok(())` means the caller may use `values`.
    ///
    /// `prev` is the last accepted batch. The first call after construction is accepted
    /// unconditionally: there is nothing to be implausible against yet.
    pub fn accept(&mut self, values: &[i32], prev: Option<&[f32]>) -> Result<(), GateRejection> {
        let Some(prev) = prev else {
            self.pending = None;
            return Ok(());
        };
        match first_implausible_step(values, prev, self.budget) {
            None => {
                self.pending = None;
                Ok(())
            }
            Some((motor, step)) => {
                // Does it agree with what we were holding? Then the joint is really there.
                if let Some(held) = self.pending {
                    if first_implausible_step(values, &held, self.budget).is_none() {
                        self.pending = None;
                        return Ok(());
                    }
                }
                let mut held = [0f32; NUM_MOTORS];
                for (h, &v) in held.iter_mut().zip(values.iter()) {
                    *h = v as f32;
                }
                let corroborating = self.pending.is_some();
                self.pending = Some(held);
                Err(GateRejection {
                    motor,
                    step,
                    corroborating,
                })
            }
        }
    }
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

/// A command the daemon has just carried out, handed back to the caller.
///
/// The channel's fields are returned by value rather than left to be re-read from shared memory:
/// Python is free to write the next command as soon as `ack_seq` moves, so a later look at
/// `cmd_kind` can describe a different command than the one that was applied.
///
/// It exists because two of these commands change what the loop must believe about the arm.
/// `SetOperatingMode` changes which frame `Present_Position` arrives in, and `SetCalibration`
/// rewrites the travel the daemon read at startup -- both silently, in the middle of a run, from
/// the other side of the shared memory.
#[derive(Clone, Copy, Debug)]
pub struct AppliedCommand {
    /// A [`CommandKind`] value.
    pub kind: u32,
    pub motor_id: u8,
    pub payload: [f32; 4],
    /// 0 on success, as reported back to Python.
    pub status: u32,
}

/// Polls the live command channel and applies at most one pending command per call. Returns what
/// was applied, or `None` if there was nothing new.
pub fn poll_and_apply_commands(
    bus: &mut FeetechBus,
    cmd: &CommandRegion,
) -> Option<AppliedCommand> {
    let cmd_seq = cmd.cmd_seq.load(Ordering::Acquire);
    let ack_seq = cmd.ack_seq.load(Ordering::Acquire);
    if cmd_seq == ack_seq {
        return None; // nothing new
    }

    let kind = cmd.cmd_kind;
    let motor_id = cmd.motor_id as u8;
    let payload = cmd.payload;
    let status = apply_command(bus, kind, motor_id, payload);

    cmd.status.store(status, Ordering::Release);
    cmd.ack_seq.store(cmd_seq, Ordering::Release);
    Some(AppliedCommand {
        kind,
        motor_id,
        payload,
        status,
    })
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
    // The only register write in the daemon that waits on a flash cycle, so the only one given a
    // budget the control loop would never tolerate. See `EEPROM_ACK_TIMEOUT`.
    bus.with_read_timeout(EEPROM_ACK_TIMEOUT, |bus| {
        bus.write_register(motor_id, feetech::REG_OPERATING_MODE, mode)
    })?;
    // An EPROM write is a flash commit, not a RAM poke -- the servo needs time to settle and can
    // NAK or answer stale until it finishes. Give it a beat before anyone reads the value back.
    std::thread::sleep(EPROM_COMMIT_DELAY);
    // Left unlocked and torque-off on purpose: the caller follows up with SetTorqueEnable(1),
    // which re-latches `Lock` -- same ordering as the Python `configure()` path.
    Ok(())
}

/// Settle time allowed after an EPROM write before reading the register back.
const EPROM_COMMIT_DELAY: Duration = Duration::from_millis(20);

/// Read timeout for the one register write in this daemon that commits flash.
///
/// **Measured**, SO-101 follower on `/dev/ttyACM0` (CH343, serial 5B42076600) at 1 Mbaud,
/// 2026-09-02, arm limp, 72 writes per case:
///
/// | write | median | p95 | max |
/// | --- | --- | --- | --- |
/// | `Torque_Enable` (RAM) | 0.23 ms | 0.32 ms | 0.36 ms |
/// | `Operating_Mode`, value unchanged | 0.41 ms | 0.49 ms | 0.56 ms |
/// | `Operating_Mode`, value changed | **20.36 ms** | **42.73 ms** | **43.52 ms** |
///
/// So this is not a slow or flaky bus: it is an erase/program cycle, and the servo skips it
/// entirely when the value already matches. The loop's `--serial-timeout-ms` (5 ms by default)
/// could never cover it, which made *every* genuine mode change time out, desynchronise the
/// stream (see `FeetechBus::transact`) and take the following motors in the burst down with it.
/// The symptom read as an intermittent bus fault; the tell was that writing the same mode twice
/// always worked the second time, because by then there was nothing left to commit.
///
/// 150 ms is ~3.5x the measured worst case. It costs nothing in normal operation -- modes are set
/// when a client attaches, never inside the control loop.
const EEPROM_ACK_TIMEOUT: Duration = Duration::from_millis(150);

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

/// Logs each servo's supply voltage and case temperature.
///
/// Neither feeds the control law, which is exactly why they were missing: the loop had no use for
/// them, so nothing read them, so the daemon had no vocabulary for "something physical is wrong"
/// at all. `fault_flags` names the watchdog, the two buses and overcurrent -- every one of them a
/// symptom. When this arm's bus started dropping a position read every four seconds, the only way
/// to reason about why was to read the corrupted bytes out of the log and infer from their shape.
///
/// One transaction per motor per register. The startup pass reads all six; the periodic one reads
/// a single motor, because *when* the rail sags is the question a startup number cannot answer --
/// the arm was idle then. See [`read_supply_and_temperature`].
pub fn log_supply_and_temperature(bus: &mut FeetechBus, motor_ids: &[u8]) {
    for &id in motor_ids {
        let volts = match bus.read_register(id, feetech::REG_PRESENT_VOLTAGE) {
            Ok(raw) => format!("{:.1} V (raw {raw})", raw as f32 / 10.0),
            Err(e) => format!("unreadable ({e})"),
        };
        let temp = match bus.read_register(id, feetech::REG_PRESENT_TEMPERATURE) {
            Ok(raw) => format!("{raw} C"),
            Err(e) => format!("unreadable ({e})"),
        };
        log::info!("motor {id}: supply {volts}, temperature {temp}");
    }
}

/// One motor's supply voltage (in 0.1 V units) and case temperature, for the per-second summary.
///
/// Read one motor at a time, round-robin, for the same reason the current reads are: six
/// registers on one tick makes that tick twice as expensive as the rest, and those fat ticks were
/// the entire overrun population when this loop was first measured. Two transactions once a second
/// is 0.16% of the bus against the control loop's 1200 per second.
///
/// The rail is shared, so any one motor answers the question this exists for: a supply that reads
/// fine with the arm limp and sags once two joints are holding their own weight is a supply that
/// is too small, and only a reading taken *under load* can tell those apart.
pub fn read_supply_and_temperature(bus: &mut FeetechBus, motor_id: u8) -> Option<(u32, u32)> {
    let volts = bus
        .read_register(motor_id, feetech::REG_PRESENT_VOLTAGE)
        .ok()?;
    let temp = bus
        .read_register(motor_id, feetech::REG_PRESENT_TEMPERATURE)
        .ok()?;
    Some((volts as u32, temp as u32))
}

/// Which frame a joint reports `Present_Position` in, which is a property of its `Operating_Mode`.
///
/// Measured on this arm 2026-09-03, all six joints, torque off, nothing driven: in
/// `Operating_Mode = 2` (PWM) the servo reports the raw encoder count, and in `Operating_Mode = 0`
/// (position) it reports that count with `Homing_Offset` subtracted, wrapped into `[0, 4096)`.
/// `Min/Max_Position_Limit` do not move with the mode -- they are always in the corrected frame.
///
/// | id | Homing_Offset | position | PWM  | PWM - position - offset |
/// |----|---------------|----------|------|-------------------------|
/// | 1  |          1739 |     2074 | 3814 | +1                      |
/// | 2  |         -1242 |      960 | 3814 | 0 (delta +2854)         |
/// | 3  |          1484 |     2937 |  325 | 0 (delta -2612)         |
/// | 4  |         -1917 |     2433 |  517 | +1                      |
/// | 5  |          1932 |     2043 | 3976 | +1                      |
/// | 6  |          1867 |     2200 | 4068 | +1                      |
///
/// The residual is one count or less everywhere -- encoder noise on a limp arm, and 200 times
/// inside `--travel-margin`. Two of the six offsets are negative, so the fold has to be modulo the
/// revolution rather than a saturating add.
///
/// The loop runs in PWM, so `Raw` is the frame the gate almost always faces; `Corrected` is the
/// state the arm starts and ends in, and the state a limp arm on the bench is in -- which is why
/// the first version of this check passed its bench test and would still have dropped the arm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionFrame {
    /// `Operating_Mode = 0`: the servo has already applied `Homing_Offset`.
    Corrected,
    /// `Operating_Mode = 2`: raw encoder counts, `Homing_Offset` not applied.
    Raw,
}

impl PositionFrame {
    /// The frame implied by an `Operating_Mode` value, or `None` for a mode nobody has measured.
    ///
    /// Modes 1 and 3 exist on this servo and neither has been on a bench here, so they get no
    /// answer rather than a plausible one. `None` switches that joint's travel check off, which
    /// costs a rejection that will not happen; guessing the frame wrong costs *every* read
    /// rejected, and that rides `--max-blind-ticks` into a torque drop on a loaded arm.
    pub fn from_operating_mode(mode: u32) -> Option<Self> {
        match mode {
            feetech::OPERATING_MODE_POSITION => Some(Self::Corrected),
            feetech::OPERATING_MODE_PWM => Some(Self::Raw),
            _ => None,
        }
    }

    fn other(self) -> Self {
        match self {
            Self::Corrected => Self::Raw,
            Self::Raw => Self::Corrected,
        }
    }
}

/// A joint's physically reachable band of positions, in counts, widened by a margin.
///
/// Held in the *corrected* frame -- the one `Min/Max_Position_Limit` are written in -- next to the
/// joint's own `Homing_Offset`, so that one envelope can answer for a position reported in either
/// [`PositionFrame`]. The first version of this stored the band alone and compared it against
/// whatever the loop happened to read, which is how it came to compare a PWM-frame position
/// against a position-mode envelope and would have rejected every joint on the arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TravelEnvelope {
    /// Low end of the calibrated travel, corrected frame, margin already subtracted.
    pub min: f32,
    /// High end of the calibrated travel, corrected frame, margin already added.
    pub max: f32,
    /// The joint's stored `Homing_Offset`, signed: `raw = corrected + homing_offset (mod 4096)`.
    pub homing_offset: f32,
}

/// Travel this wide, *once the margin is added to it*, is no constraint at all, so the joint gets
/// no envelope.
///
/// A servo reporting `0-4095` says "anywhere is reachable", and a check that admits everything is
/// worse than no check because it reads like protection. The margin is counted in here for the
/// same reason: the comparison wraps, so a 3900-count travel widened by 200 counts either side is
/// an arc that laps itself and silently admits the whole circle.
///
/// This used to name `wrist_roll` as the joint that genuinely turns freely. On this arm it does
/// not: measured 2026-09-03 with torque off, it reaches 113-3981 -- 340 deg -- and is stopped from
/// both directions by a printed corner, with the 11 deg it cannot reach lying on the encoder seam.
/// It reported `0-4095` because the calibration script assigned that without ever sweeping the
/// joint, which has since been fixed (`so_follower.py`) -- and until this arm is recalibrated its
/// registers still hold `0-4095`, which is what they read today. So a full-width travel means one
/// of two things, and the threshold treats them the same on purpose: the joint really does spin,
/// or nobody has measured it. Neither is an envelope this code can enforce.
const UNCONSTRAINED_TRAVEL: f32 = 4000.0;

impl TravelEnvelope {
    /// The envelope for one joint's calibration, or `None` when that travel constrains nothing.
    ///
    /// `margin` buys room for three things that are all real and none of them large: the stop
    /// itself is compliant, the calibration sweep can stop short of it (measured 45 counts on
    /// `elbow_flex`, 2026-09-02), and a joint can be pushed a little past its stop by hand.
    pub fn new(range_min: f32, range_max: f32, homing_offset: f32, margin: f32) -> Option<Self> {
        if range_max <= range_min || range_max - range_min + 2.0 * margin >= UNCONSTRAINED_TRAVEL {
            return None;
        }
        Some(Self {
            min: range_min - margin,
            max: range_max + margin,
            homing_offset,
        })
    }

    /// The envelope's two ends as the servo would report them in `frame`.
    ///
    /// `low > high` is not an error and not rare: folding this arm's six envelopes into the raw
    /// frame moved *every one of them* across the 4095/0 seam (2026-09-03). That is structural
    /// rather than bad luck -- a homing offset is chosen to put the seam outside the travel in the
    /// corrected frame, which is exactly what puts it inside the travel in the raw one.
    pub fn band(&self, frame: PositionFrame) -> (f32, f32) {
        let shift = match frame {
            PositionFrame::Corrected => 0.0,
            PositionFrame::Raw => self.homing_offset,
        };
        (
            (self.min + shift).rem_euclid(ENCODER_RESOLUTION),
            (self.max + shift).rem_euclid(ENCODER_RESOLUTION),
        )
    }

    /// Whether a position reported in `frame` lies inside the joint's travel.
    ///
    /// The envelope is folded and the value is not, which is the whole asymmetry: a position
    /// outside `[0, 4096)` is not a joint that wrapped, it is a bad read, and folding it would
    /// tuck it neatly inside the arc and hide it. [`first_outside_travel`] refuses those before it
    /// ever gets here.
    pub fn contains(&self, value: f32, frame: PositionFrame) -> bool {
        let (low, _) = self.band(frame);
        (value - low).rem_euclid(ENCODER_RESOLUTION) <= self.max - self.min
    }
}

/// Puts every follower servo down before the process exits: duty to zero first, then torque off.
///
/// The order is the whole point, and it is measured rather than assumed. On this arm, 2026-09-03,
/// motor 6: `Torque_Enable` was written to 0 and read back 0, then a single `Goal_PWM = 0` write
/// put it back to 1 and it stayed there. **Writing the duty re-arms the torque**, so torque has to
/// be cleared after the last duty write; clearing it first is undone by the next tick.
///
/// Without this the daemon exits leaving each servo holding the last duty it was handed. A joint
/// carrying the arm's weight at 250/1000 keeps driving at 250 after Ctrl-C, for as long as it has
/// power, because nothing else on the bus will ever tell it otherwise. The leader's gripper has
/// had this since it was written -- "the operator is never left holding a powered trigger" -- and
/// the same sentence is more true of the arm.
///
/// It also explains a state the arm was found in: torque enabled with duty 0 after a session had
/// ended, which brakes the joint (both bridge legs low) and reads by hand as a stiffness nobody
/// commanded. The client's `SetTorqueEnable(0)` had been sent and had worked; the loop's next duty
/// write took it back. While the loop runs, torque cannot be revoked from outside at all.
pub fn release_all(bus: &mut FeetechBus, motor_ids: &[u8]) {
    for &id in motor_ids {
        if let Err(e) = bus.write_register(id, feetech::REG_GOAL_PWM, 0) {
            log::warn!("motor {id}: failed to zero PWM on shutdown: {e}");
        }
    }
    for &id in motor_ids {
        if let Err(e) = bus.write_register(id, feetech::REG_TORQUE_ENABLE, 0) {
            log::warn!("motor {id}: failed to disable torque on shutdown: {e}");
        }
    }
    // Read back rather than assume: this is the last thing standing between a driven arm and an
    // unattended one, and a dropped write here is silent.
    let still_on: Vec<u8> = motor_ids
        .iter()
        .copied()
        .filter(|&id| !matches!(bus.read_register(id, feetech::REG_TORQUE_ENABLE), Ok(0)))
        .collect();
    if still_on.is_empty() {
        log::info!("all motors released: duty zeroed, torque off");
    } else {
        log::error!(
            "motors {still_on:?} are still energised after shutdown -- they hold whatever duty \
             they were last given; power the arm down or re-run and stop it again"
        );
    }
}

/// Each servo's stored `Homing_Offset`, logged as it is read.
///
/// Logged because it is otherwise invisible: an all-zero column here means positions are raw
/// encoder counts and any joint whose travel crosses 4095/0 will report a full-scale jump.
/// Returned because the travel gate cannot work without it -- the offset *is* the difference
/// between the frame the envelope is written in and the frame the loop reads in.
pub fn read_homing_offsets(bus: &mut FeetechBus, motor_ids: &[u8]) -> [Option<f32>; NUM_MOTORS] {
    let mut out = [None; NUM_MOTORS];
    for (slot, &id) in out.iter_mut().zip(motor_ids) {
        match bus.read_register(id, feetech::REG_HOMING_OFFSET) {
            Ok(raw) => {
                let offset =
                    feetech::decode_sign_magnitude(raw as u16, feetech::HOMING_OFFSET_SIGN_BIT);
                log::info!("motor {id}: stored Homing_Offset = {offset}");
                *slot = Some(offset as f32);
            }
            Err(e) => log::warn!("motor {id}: could not read Homing_Offset: {e}"),
        }
    }
    out
}

/// Each joint's position frame at startup, read from `Operating_Mode`.
///
/// Read once and then tracked rather than re-read: every mode change on this bus goes through the
/// daemon's own command channel (`SetOperatingMode`), so its belief stays exact as long as it
/// updates on the writes it performs -- and a stale belief here is not a missed rejection, it is
/// every joint rejected at once. Polling the register instead would cost six transactions per
/// sweep and still be wrong for however long the sweep takes.
pub fn read_position_frames(
    bus: &mut FeetechBus,
    motor_ids: &[u8],
) -> [Option<PositionFrame>; NUM_MOTORS] {
    let mut out = [None; NUM_MOTORS];
    for (slot, &id) in out.iter_mut().zip(motor_ids) {
        match bus.read_register(id, feetech::REG_OPERATING_MODE) {
            Ok(mode) => {
                *slot = PositionFrame::from_operating_mode(mode as u32);
                match slot {
                    Some(frame) => {
                        log::info!("motor {id}: Operating_Mode {mode}, positions in the {frame:?} frame")
                    }
                    None => log::warn!(
                        "motor {id}: Operating_Mode {mode} -- which frame that reports positions in \
                         has not been measured, so this joint's travel is not checked"
                    ),
                }
            }
            Err(e) => log::warn!(
                "motor {id}: could not read Operating_Mode ({e}); this joint's travel is not checked"
            ),
        }
    }
    out
}

/// Reads each joint's calibrated travel from the servos and folds it into an envelope.
///
/// The numbers come off the arm rather than out of the calibration JSON on purpose. They are the
/// same numbers -- `lerobot-calibrate` writes `range_min`/`range_max` into these registers as it
/// writes the homing offset -- but reading them here means the daemon cannot be driving one arm
/// while trusting a file that describes another, or a file that was recalibrated and not copied.
/// The Python client re-sends both as a `SetCalibration` command when it connects, and the caller
/// rebuilds the affected envelope from that payload rather than leaving this startup read to go
/// stale behind it.
pub fn read_travel_envelopes(
    bus: &mut FeetechBus,
    motor_ids: &[u8],
    homing_offsets: &[Option<f32>],
    margin: f32,
) -> [Option<TravelEnvelope>; NUM_MOTORS] {
    let mut out = [None; NUM_MOTORS];
    for ((slot, &id), offset) in out.iter_mut().zip(motor_ids).zip(homing_offsets) {
        let limits = bus
            .read_register(id, feetech::REG_MIN_POSITION_LIMIT)
            .and_then(|lo| {
                bus.read_register(id, feetech::REG_MAX_POSITION_LIMIT)
                    .map(|hi| (lo, hi))
            });
        match (limits, offset) {
            (Ok((lo, hi)), &Some(offset)) => {
                match TravelEnvelope::new(lo as f32, hi as f32, offset, margin) {
                    Some(env) => {
                        let (rlo, rhi) = env.band(PositionFrame::Raw);
                        log::info!(
                            "motor {id}: travel {lo}-{hi}, rejecting positions outside \
                             {:.0}-{:.0} in position mode / {rlo:.0}-{rhi:.0} in PWM \
                             (Homing_Offset {offset:+.0})",
                            env.min,
                            env.max
                        );
                        *slot = Some(env);
                    }
                    None => log::info!(
                        "motor {id}: travel {lo}-{hi} widened by {margin:.0} spans the circle -- \
                         no position envelope"
                    ),
                }
            }
            (Ok((lo, hi)), None) => log::warn!(
                "motor {id}: travel {lo}-{hi} is known but `Homing_Offset` is not, so the envelope \
                 cannot be put in the frame the loop reads; no envelope for this joint"
            ),
            (Err(e), _) => log::warn!(
                "motor {id}: could not read position limits ({e}); no envelope for this joint"
            ),
        }
    }
    out
}

/// Why a batch of positions was refused by [`first_outside_travel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TravelVerdict {
    /// Outside `[0, 4096)`, which the encoder cannot report at all once `Phase` bit 4 is clear --
    /// so it is a misparsed reply rather than a joint anywhere. Checked for every joint, envelope
    /// or no envelope, because it needs neither a calibration nor a frame to be certain of itself.
    OffEncoder,
    /// Inside the encoder's range but outside this joint's own travel.
    OutsideTravel {
        /// Whether *every* joint being checked is outside its travel in the frame the daemon
        /// believes it is in and inside it in the other frame.
        ///
        /// That is the signature of a mode switch the daemon did not see, because the client
        /// switches all six joints together -- and it is the failure the first version of this
        /// gate would have produced, so the daemon says it in its own log rather than leaving it
        /// to be rediscovered from the arm.
        ///
        /// It takes all of them because one joint proves nothing: on this arm a joint's two
        /// envelopes cover most of the circle between them, so a single bad reading lands inside
        /// the other frame's band about as often as not.
        every_checked_joint_fits_the_other_frame: bool,
    },
}

/// One motor's refused position, and why.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TravelReject {
    pub motor: usize,
    pub value: f32,
    pub verdict: TravelVerdict,
}

impl TravelReject {
    /// The middle of the daemon's warning line, so the reason travels with the rejection.
    pub fn reason(&self) -> &'static str {
        match self.verdict {
            TravelVerdict::OffEncoder => {
                "outside the encoder's own range 0-4095, so the reply was misparsed rather than \
                 the joint being anywhere"
            }
            TravelVerdict::OutsideTravel {
                every_checked_joint_fits_the_other_frame: false,
            } => "outside the travel it can physically reach",
            TravelVerdict::OutsideTravel {
                every_checked_joint_fits_the_other_frame: true,
            } => {
                "outside the travel it can physically reach -- and so is every other checked \
                 joint, each of them landing inside its travel in the other position frame. That \
                 is a mode switch this daemon did not see, i.e. a bug here rather than a bad read"
            }
        }
    }
}

/// The first motor in a batch reporting a position its joint cannot physically be at.
///
/// This is deliberately *not* part of [`PositionGate`], and the difference is the whole point of
/// it. The gate asks "could the joint have got there since last tick?", and answers a batch it
/// doubts by holding it and accepting it if the next read agrees. That is right for a joint that
/// really moved while the bus was quiet, and it is exactly the hole a *stable* wrong reading walks
/// through: agreeing with itself is all corroboration asks for.
///
/// Measured on this arm, 2026-09-02: `shoulder_pan` reported jumps of +-4083 counts -- a whole
/// turn -- three times in one 25-second hand sweep, while the joint's hard stops sit at 867 and
/// 3280 (repeatable to 3-6 counts) with 815 counts of clearance to the 4095/0 wrap on either side.
/// The joint could not have been where it said it was. A full-turn misreport is steady while it
/// lasts, so it corroborates itself, and the impedance law then answers ~4090 counts of error with
/// saturated duty in one direction. That is what drove a joint into its stop earlier that day:
/// motor 1 went from 29 C to 42 C in 36 seconds and pulled the shared rail from 4.6 V to 4.0 V.
///
/// So this check runs first and its verdict is final. There is nothing to corroborate: a position
/// outside the travel is not a joint that moved, it is a reading that is wrong, and the arm is
/// better off blind (the existing `max_blind_ticks` fail-safe drops torque) than driven towards a
/// place the joint has never been.
///
/// `frames` says which frame each joint is reporting in *now*. A joint whose frame is unknown is
/// not checked against its travel, and is still checked against the encoder's own range -- that
/// one needs no calibration and no frame to be sure.
pub fn first_outside_travel(
    values: &[i32],
    envelopes: &[Option<TravelEnvelope>],
    frames: &[Option<PositionFrame>],
) -> Option<TravelReject> {
    // Off the encoder first, and for every joint: it is the more certain of the two verdicts, and
    // reporting it as a travel violation would blame the calibration for a misparsed reply.
    if let Some((motor, value)) = values.iter().enumerate().find_map(|(motor, &value)| {
        let value = value as f32;
        (!(0.0..ENCODER_RESOLUTION).contains(&value)).then_some((motor, value))
    }) {
        return Some(TravelReject {
            motor,
            value,
            verdict: TravelVerdict::OffEncoder,
        });
    }

    let checked = || {
        values
            .iter()
            .zip(envelopes)
            .zip(frames)
            .enumerate()
            .filter_map(|(motor, ((&value, envelope), frame))| {
                Some((motor, value as f32, envelope.as_ref()?, (*frame)?))
            })
    };
    let (motor, value, ..) =
        checked().find(|(_, value, envelope, frame)| !envelope.contains(*value, *frame))?;
    let every_checked_joint_fits_the_other_frame = checked().all(|(_, value, envelope, frame)| {
        !envelope.contains(value, frame) && envelope.contains(value, frame.other())
    });
    Some(TravelReject {
        motor,
        value,
        verdict: TravelVerdict::OutsideTravel {
            every_checked_joint_fits_the_other_frame,
        },
    })
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
