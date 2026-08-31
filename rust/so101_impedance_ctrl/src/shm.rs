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

pub const LAYOUT_VERSION: u32 = 6;
pub const SHM_MAGIC: u32 = 0x534F_3130; // ASCII "SO10"
/// All 6 servos -- the 5 arm joints AND the gripper -- are impedance-controlled (K/D over PWM).
/// A rigid position-mode gripper crushes anything it grips before it can sense resistance;
/// running it under the same compliant impedance law as the arm is what makes gentle/adaptive
/// grasping of fragile objects possible at all. Matches so_follower.py's motor IDs 1..6.
pub const NUM_MOTORS: usize = 6;

/// Context channels the policy layer can hand down to the cerebellum -- the pontine relay.
///
/// The cerebellum's mossy fibres otherwise carry proprioception only, so it cannot tell two
/// payloads apart: a gripper holding 20 g and one holding 200 g produce the same joint angles on
/// the way to the same place, and the load only becomes visible *after* it has already pulled the
/// arm down. These channels are what a granule cell can read to know which of several standing
/// loads it is looking at before the error appears.
///
/// **This is an identity, not a mass.** Biology hands the cerebellum the object, not the number:
/// grip force is scaled before lift-off from a memory indexed by which object this is, and the
/// weight-to-force map lives in the cerebellum rather than in cortex. Asking the policy for
/// "how many grams" would move the cerebellum's job up a layer and require it to learn a
/// calibration nothing in the loop can teach it. So anything separable will do -- a grasp flag, a
/// few bits of object identity -- and the Purkinje layer learns a feedforward per context.
///
/// Two, because two is what the experiment needs (empty vs. loaded, with one spare) and because
/// widening this reshuffles the entire granule code. Every cell draws `GC_FAN_IN` fibres from
/// `0..MF_DIM`, so changing the count changes every draw and invalidates every learned weight --
/// which the weights file's header now refuses rather than silently accepting.
pub const NUM_CONTEXT: usize = 2;

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
    /// Pontine context, nominally in `[-1, 1]`; see [`NUM_CONTEXT`]. All-zero means "no context",
    /// which is also what an older writer that never touches this field leaves behind -- and zero
    /// is the value at which these channels contribute nothing, so a policy that does not know
    /// about them degrades to the proprioception-only cerebellum rather than to garbage.
    pub context: [f32; NUM_CONTEXT],
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
    /// The cerebellum's contribution to `pwm_cmd_debug`, already clamped, slew-limited and gated.
    /// Zero when no cerebellum is running. Published separately because the whole bring-up
    /// question is how much of the holding duty the feedforward has taken over, and a total that
    /// mixes the two cannot answer it.
    pub ff_pwm_debug: [f32; NUM_MOTORS],
    /// `CEREBELLUM_*` bits from `cerebellum::mod`, describing why the feedforward is what it is.
    pub cerebellum_flags: u32,
    pub fault_flags: u32,
    /// Leader-side gripper telemetry, present only when the daemon was given `--leader-port`;
    /// all zero otherwise. Exposed so the operator-facing tools can show whether the trigger is
    /// being driven, and so a recording pipeline can eventually label grip intent with it.
    pub leader_gripper_pos: f32,
    pub leader_gripper_vel: f32,
    pub leader_gripper_pwm: f32,
    /// Supply voltage in 0.1 V units, as reported by the servo named by `health_motor_id`, with
    /// that servo's case temperature in C. Sampled round-robin once a second by the summary
    /// branch, not per tick -- see `control::read_supply_and_temperature` for why. Both zero until
    /// the first sample lands.
    ///
    /// Published rather than only logged because the failure this exists to catch is a *transient*
    /// one: a servo whose power stage shorts drags the shared rail down for hundreds of ms at a
    /// time, which starves every other joint of torque and inflates the duty they need to hold a
    /// pose. A startup reading cannot see that (the arm is idle then), and a number that only
    /// reaches the log cannot be lined up against the droop it caused. Landing it in the same
    /// telemetry snapshot as `pwm_cmd_debug` is what makes a holding-duty measurement able to say
    /// whether the rail was healthy while it was taken.
    pub supply_decivolts: u32,
    pub case_temp_c: u32,
    /// Servo ID the two fields above were read from, or 0 before the first sample.
    pub health_motor_id: u32,
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
/// At least one joint is outside `--pos-min`/`--pos-max`. Reported because the symptom is
/// otherwise indistinguishable from several unrelated ones: a joint the limits are holding at zero
/// PWM looks exactly like the watchdog, a blind run, or a gain that is simply too soft.
pub const FAULT_POS_LIMIT: u32 = 1 << 4;

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
