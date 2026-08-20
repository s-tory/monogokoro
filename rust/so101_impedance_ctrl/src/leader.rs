//! Force feedback on the leader arm's gripper, so the operator physically feels what the
//! follower's gripper is touching.
//!
//! # Why only the gripper
//!
//! Full-arm bilateral teleoperation is the eventual goal, but one degree of freedom is where the
//! payoff is concentrated and the risk is not. Grip force is the quantity a demonstrator cannot
//! currently express -- a torque-off leader is a position sensor and nothing else, so every
//! recorded demonstration squeezes with whatever stiffness the config happened to hold. It is also
//! the axis where a mistake breaks the object. And should this loop go unstable, a buzzing trigger
//! is a very different event from a buzzing shoulder joint with the operator's arm attached to it.
//!
//! # Position-position coupling, no force sensor
//!
//! The feedback is driven by the *follower's own tracking error*, not by a measured force. When
//! the follower's gripper is free it reaches its target and the error is ~0, so the trigger stays
//! slack. When an object blocks it, the commanded position runs ahead of the achieved one and that
//! gap grows in proportion to how hard the operator is asking the follower to squeeze -- which is
//! exactly the signal to render back into their hand. This is the classic position-position
//! bilateral architecture, and it needs no load cell and no current sensing.
//!
//! Deliberately *not* driven by the follower's `Present_Current`: that is averaged over ~0.5 s to
//! be usable as an ACT observation, which is an eternity for haptics.
//!
//! # One loop, both arms
//!
//! This shares the caller's control loop rather than running a thread of its own. The two arms are
//! on separate serial ports, so the half-duplex constraint does not couple them, and a single tick
//! keeps leader and follower samples in lockstep by construction. Two independent loops would let
//! their phase free-run, injecting up to a full period of variable delay into the coupling -- and
//! variable delay is precisely what destabilises a bilateral loop.

use std::time::Duration;

use crate::control::{finite_difference_velocity, force_feedback_pwm, MovingAverage};
use crate::feetech::{self, FeetechBus};

/// Runs the leader gripper as a haptic display driven by the follower gripper's tracking error.
pub struct LeaderGripper {
    bus: FeetechBus,
    motor_id: u8,
    feedback_gain: f32,
    damping: f32,
    pwm_max: f32,
    invert_pwm: bool,
    pwm_sign_bit: u32,
    vel_avg: MovingAverage,
    prev_pos: f32,
    /// Last successfully read position; also the fallback when a read fails.
    pub present_pos: f32,
    pub present_vel: f32,
    pub pwm_cmd: f32,
    pub comms_error: bool,
    /// Wall time the most recent `tick` spent on the leader's bus, for the timing summary.
    pub last_tick: Duration,
}

#[allow(clippy::too_many_arguments)]
impl LeaderGripper {
    /// Opens the leader's port and puts its gripper servo into PWM mode with torque enabled.
    ///
    /// Only the gripper is touched: the other five leader servos stay torque-off and backdrivable,
    /// so the operator moves the arm exactly as before and Python keeps reading their positions
    /// over its own connection to this same port.
    pub fn open(
        port: &str,
        baud: u32,
        timeout: Duration,
        motor_id: u8,
        feedback_gain: f32,
        damping: f32,
        pwm_max: f32,
        invert_pwm: bool,
        pwm_sign_bit: u32,
        vel_filter_window: usize,
    ) -> std::io::Result<Self> {
        let mut bus = FeetechBus::open(port, baud, timeout)?;
        crate::control::apply_startup_config(&mut bus, &[motor_id]);
        crate::control::prepare_pwm_motor(&mut bus, motor_id)?;

        let present_pos = bus.read_register(motor_id, feetech::REG_PRESENT_POSITION)? as f32;
        log::info!(
            "leader gripper (motor {motor_id} on {port}) ready at {present_pos} counts; \
             feedback_gain={feedback_gain} damping={damping} pwm_max={pwm_max}"
        );

        Ok(Self {
            bus,
            motor_id,
            feedback_gain,
            damping,
            pwm_max,
            invert_pwm,
            pwm_sign_bit,
            vel_avg: MovingAverage::new(vel_filter_window.max(1)),
            prev_pos: present_pos,
            present_pos,
            present_vel: 0.0,
            pwm_cmd: 0.0,
            comms_error: false,
            last_tick: Duration::ZERO,
        })
    }

    /// One leader-side tick: read the trigger, render the follower's error as resistance.
    ///
    /// `enabled` must be false whenever the follower's own output is being held at zero -- a stale
    /// watchdog or a blind tick means `follower_error` describes a situation the follower is no
    /// longer acting on, and pushing a human's hand around based on that is worse than going
    /// slack. A failed read of the trigger position zeroes the output for the same reason: the
    /// damping term would otherwise be computed against a stale velocity.
    pub fn tick(&mut self, follower_error: f32, enabled: bool, dt_s: f32) {
        let started = std::time::Instant::now();
        self.comms_error = false;

        match self
            .bus
            .read_register(self.motor_id, feetech::REG_PRESENT_POSITION)
        {
            Ok(v) => self.present_pos = v as f32,
            Err(e) => {
                log::warn!("leader gripper read Present_Position failed: {e}");
                self.comms_error = true;
            }
        }

        self.present_vel = self.vel_avg.push(finite_difference_velocity(
            self.prev_pos,
            self.present_pos,
            dt_s,
        ));
        self.prev_pos = self.present_pos;

        self.pwm_cmd = if enabled && !self.comms_error {
            force_feedback_pwm(
                self.feedback_gain,
                self.damping,
                follower_error,
                self.present_vel,
                self.pwm_max,
            )
        } else {
            0.0
        };

        let commanded = if self.invert_pwm {
            -self.pwm_cmd
        } else {
            self.pwm_cmd
        };
        let raw = feetech::encode_sign_magnitude(commanded as i32, self.pwm_sign_bit);
        if let Err(e) = self
            .bus
            .write_register(self.motor_id, feetech::REG_GOAL_PWM, raw as u32)
        {
            log::warn!("leader gripper write Goal_PWM failed: {e}");
            self.comms_error = true;
        }

        self.last_tick = started.elapsed();
    }

    /// Drops torque on the leader gripper, leaving the trigger free. Called on shutdown so the
    /// operator is never left holding a powered trigger after the daemon exits.
    pub fn release(&mut self) {
        self.pwm_cmd = 0.0;
        if let Err(e) = self
            .bus
            .write_register(self.motor_id, feetech::REG_GOAL_PWM, 0)
        {
            log::warn!("leader gripper: failed to zero PWM on shutdown: {e}");
        }
        if let Err(e) = self
            .bus
            .write_register(self.motor_id, feetech::REG_TORQUE_ENABLE, 0)
        {
            log::warn!("leader gripper: failed to disable torque on shutdown: {e}");
        }
    }
}
