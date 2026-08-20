//! Shared-memory IPC between this RT daemon and the Python-side `SO101ImpedanceFollower` robot
//! (`src/lerobot/robots/so101_impedance_follower/shm_client.py`). The layout below is
//! `#[repr(C)]` and must be mirrored byte-for-byte by the Python `ctypes.Structure` there --
//! bump [`LAYOUT_VERSION`] on any shape change, both sides assert equality on attach.
//!
//! `input`/`output` are high-rate regions (per ACT inference step / per RT tick respectively),
//! guarded by an independent seqlock each. `command` is a low-rate config/calibration channel
//! with an explicit request/ack sequence instead of a seqlock, since a command must be observed
//! exactly once, not merely "the latest stable snapshot".

use std::sync::atomic::{fence, AtomicU32, Ordering};

pub const LAYOUT_VERSION: u32 = 3;
pub const SHM_MAGIC: u32 = 0x534F_3130; // ASCII "SO10"
/// All 6 servos -- the 5 arm joints AND the gripper -- are impedance-controlled (K/D over PWM).
/// A rigid position-mode gripper crushes anything it grips before it can sense resistance;
/// running it under the same compliant impedance law as the arm is what makes gentle/adaptive
/// grasping of fragile objects possible at all. Matches so_follower.py's motor IDs 1..6.
pub const NUM_MOTORS: usize = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputData {
    pub timestamp_mono_ns: u64,
    /// rad; order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
    pub target_pos: [f32; NUM_MOTORS],
    /// rad/s; 0 if the caller doesn't supply a velocity target.
    pub target_vel: [f32; NUM_MOTORS],
    pub k_gain: [f32; NUM_MOTORS],
    pub d_gain: [f32; NUM_MOTORS],
}

#[repr(C)]
pub struct InputRegion {
    /// Seqlock counter: odd = writer in progress, even = stable. Kept as a sibling field (not
    /// nested inside `data`) so Rust's borrow checker can split `&region.seq` /
    /// `&mut region.data` as disjoint borrows.
    pub seq: AtomicU32,
    pub data: InputData,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputData {
    pub timestamp_mono_ns: u64,
    /// 5 arm joints + gripper.
    pub present_pos: [f32; NUM_MOTORS],
    pub present_vel: [f32; NUM_MOTORS],
    /// Pre-averaged in Rust (fixed-sample-count moving average), mA.
    pub present_current_avg: [f32; NUM_MOTORS],
    pub pwm_cmd_debug: [f32; NUM_MOTORS],
    pub fault_flags: u32,
    /// Leader-side gripper telemetry, present only when the daemon was given `--leader-port`;
    /// all zero otherwise. Exposed so the operator-facing tools can show whether the trigger is
    /// being driven, and so a recording pipeline can eventually label grip intent with it.
    pub leader_gripper_pos: f32,
    pub leader_gripper_vel: f32,
    pub leader_gripper_pwm: f32,
}

#[repr(C)]
pub struct OutputRegion {
    pub seq: AtomicU32,
    pub data: OutputData,
}

pub const FAULT_WATCHDOG_TIMEOUT: u32 = 1 << 0;
pub const FAULT_COMMS_ERROR: u32 = 1 << 1;
pub const FAULT_OVERCURRENT: u32 = 1 << 2;
/// The leader arm's bus failed this tick. Kept distinct from [`FAULT_COMMS_ERROR`] because the
/// consequences differ: the follower losing its bus stops the robot, whereas the leader losing
/// its bus only drops force feedback -- the follower keeps tracking normally.
pub const FAULT_LEADER_COMMS_ERROR: u32 = 1 << 3;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandKind {
    None = 0,
    SetOperatingMode = 1,
    SetPidCoefficients = 2,
    SetTorqueEnable = 3,
    SetCalibration = 4,
    Shutdown = 5,
}

#[repr(C)]
pub struct CommandRegion {
    /// Python increments this (to any value != `ack_seq`) to submit a new command.
    pub cmd_seq: AtomicU32,
    /// Rust sets this equal to the observed `cmd_seq` once the command has been executed.
    pub ack_seq: AtomicU32,
    /// 0 = ok, nonzero = an implementation-defined error code. Valid once `ack_seq == cmd_seq`.
    pub status: AtomicU32,
    /// A [`CommandKind`] value.
    pub cmd_kind: u32,
    pub motor_id: u32,
    /// Interpreted per `cmd_kind`, e.g. `[operating_mode, _, _, _]` or `[p, i, d, _]`.
    pub payload: [f32; 4],
}

#[repr(C)]
pub struct ShmLayout {
    pub magic: u32,
    pub layout_version: u32,
    pub input: InputRegion,
    pub output: OutputRegion,
    pub command: CommandRegion,
}

impl ShmLayout {
    /// Initializes a freshly-created shared-memory segment in place. Only the daemon (creator)
    /// calls this; Python only ever attaches to an already-initialized segment.
    pub fn init_in_place(&mut self) {
        self.magic = SHM_MAGIC;
        self.layout_version = LAYOUT_VERSION;
        self.input.seq.store(0, Ordering::Relaxed);
        self.input.data = InputData::default();
        self.output.seq.store(0, Ordering::Relaxed);
        self.output.data = OutputData::default();
        self.command.cmd_seq.store(0, Ordering::Relaxed);
        self.command.ack_seq.store(0, Ordering::Relaxed);
        self.command.status.store(0, Ordering::Relaxed);
        self.command.cmd_kind = CommandKind::None as u32;
        self.command.motor_id = 0;
        self.command.payload = [0.0; 4];
    }
}

/// Seqlock write: bumps `seq` to odd before writing, then to even after, with the memory fences
/// needed so a concurrent reader (in another process, via [`seqlock_read`]) never observes a
/// torn write.
pub fn seqlock_write<T>(seq: &AtomicU32, region: &mut T, write_fn: impl FnOnce(&mut T)) {
    let s = seq.load(Ordering::Relaxed);
    seq.store(s.wrapping_add(1), Ordering::Release); // now odd: writer in progress
    fence(Ordering::SeqCst);
    write_fn(region);
    fence(Ordering::SeqCst);
    seq.store(s.wrapping_add(2), Ordering::Release); // back to even: stable
}

/// Seqlock read: retries until it observes a stable (even, unchanged) sequence number
/// surrounding the read, guaranteeing `read_fn` never sees a torn write. `max_retries` bounds
/// worst-case latency under contention; an RT loop must not block indefinitely, so on exhaustion
/// this returns `None` rather than ever handing back a possibly-torn read -- callers (the RT
/// control loop, the Python shm client) must fall back to their last known-good snapshot instead
/// of trusting an unstable one. This matters: an earlier version returned a best-effort read on
/// exhaustion, which a concurrent stress test (`tests/shm_layout_tests.rs`) caught actually
/// leaking a torn value under heavy contention.
pub fn seqlock_read<T, R>(
    seq: &AtomicU32,
    region: &T,
    mut read_fn: impl FnMut(&T) -> R,
    max_retries: u32,
) -> Option<R> {
    for _ in 0..max_retries {
        let s1 = seq.load(Ordering::Acquire);
        if !s1.is_multiple_of(2) {
            continue; // writer in progress, retry
        }
        fence(Ordering::SeqCst);
        let result = read_fn(region);
        fence(Ordering::SeqCst);
        let s2 = seq.load(Ordering::Acquire);
        if s1 == s2 {
            return Some(result);
        }
    }
    None
}
