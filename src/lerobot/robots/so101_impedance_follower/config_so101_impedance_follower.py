#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig

# Fixed order of all 6 impedance-controlled motors -- the 5 arm joints AND the gripper. The
# gripper is impedance-controlled too (not left in plain position mode): a rigid position-mode
# gripper keeps commanding full force toward its target regardless of contact, which crushes
# fragile objects before it can ever "feel" them via K/D compliance.
DEFAULT_IMPEDANCE_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class SO101ImpedanceFollowerConfig:
    """Configuration for the SO101 impedance-controlled follower.

    Unlike `SOFollowerConfig`, this robot does not open its own serial port: the
    `rust/so101_impedance_ctrl` RT daemon exclusively owns the SO101's single half-duplex UART
    (all 6 servos, including the gripper) to avoid two processes racing on one bus. This robot
    only ever talks to that daemon through a shared-memory segment; start the daemon first.
    """

    # Name of the POSIX shared-memory segment created by the Rust daemon (must match its
    # `--shm-name` argument).
    shm_name: str

    disable_torque_on_disconnect: bool = True

    # Same semantics as `SOFollowerConfig.max_relative_target`, applied to all 6 motors'
    # `.pos` targets (in normalized units -- degrees or range -100..100 -- not raw ticks).
    max_relative_target: float | dict[str, float] | None = None

    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Set to `True` for consistency with plain `SOFollower` datasets/policies.
    use_degrees: bool = True

    impedance_joints: tuple[str, ...] = DEFAULT_IMPEDANCE_JOINTS

    # Default per-motor spring (K) / damper (D) gains, in `impedance_joints` order (arm joints,
    # then gripper). Used as a fallback in `send_action` when a predicted/teleop action lacks
    # `.k`/`.d`, and by the recording pipeline's `ImpedanceGainDefaultsProcessorStep` to label
    # ground-truth K/D during data collection. The gripper's default is deliberately softer than
    # the arm joints' -- a low K lets it yield near the target instead of continuing to squeeze,
    # which is the whole point of impedance-controlling it: gentler grasping of fragile objects.
    #
    # NOTE on units: K and D operate on the *raw encoder tick* position/velocity error computed
    # inside the Rust control loop (see `rust/so101_impedance_ctrl/src/control.rs`), not on
    # degrees -- Rust never applies this robot's degree/range calibration math, only this Python
    # class does (at the `.pos` <-> raw-tick boundary). Tune these values in PWM-per-raw-tick
    # units, not PWM-per-degree.
    #
    # These K values follow a hand, not a formula. The formula came first and was wrong about the
    # part that matters.
    #
    # The measurement it starts from is still the same one: holding the arm outstretched (near
    # worst case for gravity) at K=1 makes the reported PWM read out directly as the duty each
    # joint needs to hold itself, since pwm = 1 * err. On 2026-09-16, on a 7.4 V rail, that is
    # 42 / 56.5 / 55 / 1 / 1 / 0 counts for pan / lift / elbow / wrist_flex / wrist_roll / gripper.
    # (At 5 V the same pose read 17 / 87 / 61 / ~0 / ~0 / 0, on the previous shoulder_lift servo
    # and with no stereo cameras on the wrist. Three conditions differ; do not difference them.)
    #
    # A pure PD law droops under a constant load by `err = holding_duty / K`, so K follows from the
    # droop you will accept -- and this file used to accept ~5 counts (0.4 deg), which is where the
    # old values came from. **That target was never checked against anyone's hand.** When it was,
    # blind, on 2026-09-16 -- the judge pushing the arm and saying only stiff / right / too soft,
    # with gains switched underneath a held pose so nothing gave the value away -- K=20 came back
    # かたい twice, K=12 ちょっと硬い twice, and K=7 and K=8 both こんなもんかなー. The droop that
    # a hand actually accepts here is ~7 counts (0.62 deg), not 5.
    #
    # So: lift K=8, every other joint scaled from the old set by the same 0.4. The ratios are kept
    # because they were never the thing under test -- and there is a live objection to them: the
    # same judge said **先端ほど固すぎる**, tip joints feel stiffer than the shoulder. That is
    # expected if the ratios are wrong, because the old ones were derived from each joint's
    # *gravity* load, while a hand feels the reaction to its own push, which is much the same at
    # every joint. Sweeping the ratio (rather than one scalar over all of them) is the open work.
    #
    # wrist_flex has been swept, alone, on 2026-09-17 (blind, the others at the values below). With
    # the cerebellum off, K=4 came back かたい three times out of three, and K=1.3 felt right but did
    # not return against gravity after a push (twice, and 1.6 twice the same); only 2.0 returned. K=0
    # was named as static friction, blind, twice. With the cerebellum on (GPU, learning, weights
    # from zero), every K from 1.0 to 2.0 returned, and 1.3 was the one called right, twice. So
    # wrist_flex is 1.3 **and that value assumes the cerebellum is running**: without it the wrist
    # stays wherever it is pushed upward. The derivation this sweep was testing -- K proportional to
    # the square of the distance from the joint to the fingertip, which put wrist_flex at 2.0 -- got
    # the direction and missed the level.
    #
    # shoulder_pan was swept the same way, cerebellum on, the same day. The derivation put it at 12.8
    # -- the shipped 4 too *soft* -- and the hand said the opposite: 8, 12 and 16 were all かたい,
    # 4 was ちょっとかたい four times out of four, 2 leaned soft (two of four called it soft), and 3
    # was called right twice. So pan is 3, and the r-squared rule is dead for pan.
    #
    # elbow_flex, same day, cerebellum on, discriminated worst of the three: 6 and even 9 drew only
    # ちょっとかたい, and 2 and 3 each drew one "about right" and one verdict on either side. The
    # feedforward was not settled across trials (duty -55 to -283 at |err| <= 3), which may be what
    # blurred it. 3 is the pick: the only value never called soft or plain stiff.
    #
    # wrist_roll, same day, cerebellum on: K=0 was already かたい and did not come back (left 323 and
    # 191 counts off), so the floor on this joint is static friction and no K goes below it. 1 and 2
    # felt like that floor and returned; 3.2 was かたすぎ and 5 うごかない. So wrist_roll is 1, the
    # lowest that returns.
    #
    # The gripper, same day, judged by working the jaw open and shut by hand (the cerebellum does not
    # drive it): K=0 かたい and did not come back, twice; 1 was "a little stiff but about right" and
    # returned, twice, word for word; 2 かたい; 4 かた！. So the gripper is 1. That is the feel of
    # back-driving the jaw, which stands in for grip force (K times how far short of its target the
    # jaw is stopped) but is not a measurement of it. Steps below 1 were not tried; the judge's view
    # was that they would not be told apart.
    #
    # shoulder_lift was swept alone the same day and the verdicts did not follow K (12 called about
    # right, 8 and 3 stiff, repeats split) while the feedforward varied from -89 to -293 at
    # |err| <= 2.5. That run measured the cerebellum's state, not K, so lift stays at 8.
    #
    # Conditions these were taken under, since none of them are the arm alone: 7.4 V rail, this
    # unit's friction and wiring, stereo cameras + bracket on the wrist, natural-rubber finger cots
    # on the gripper. Re-measure per arm, and re-measure after a supply change -- the duty a joint
    # needs scales with 1/V, so these numbers are wrong by half at 5 V.
    default_k: tuple[float, ...] = (3.0, 8.0, 3.0, 1.3, 1.0, 1.0)

    # D is bounded from above by velocity *noise*, not by stability. Position is quantised to whole
    # counts, so the filtered finite difference has a noise floor of about
    # `1 / (vel_filter_window * dt)` -- ~50 counts/s at the daemon's defaults -- which D turns
    # straight into PWM chatter. Keeping D near K/40 holds that under ~2% duty while still damping
    # a real 100 counts/s motion with a meaningful command. Scaled by the same 0.4 as K on
    # 2026-09-16 so that D/K stays at K/40 -- the bound above is a ratio, not an absolute. wrist_flex
    # Every joint but shoulder_lift followed its K on 2026-09-17 at the D/K it already had.
    default_d: tuple[float, ...] = (0.09, 0.2, 0.08, 0.039, 0.025, 0.03)

    # Defense-in-depth clamps applied in `send_action`, independent of (but should be kept
    # consistent with) whatever bound the Rust daemon itself enforces via `--pwm-max`. `d_max` is
    # the tighter of the two on purpose: an over-large K merely saturates PWM at a small error,
    # whereas an over-large D amplifies the velocity noise floor into full-duty chatter, which the
    # gearboxes pay for. Nothing above ~5 is usable given that noise floor.
    k_min: float = 0.0
    k_max: float = 500.0
    d_min: float = 0.0
    d_max: float = 5.0

    # The pontine context this arm declares to the cerebellum, nominally in `[-1, 1]` per channel.
    # Length must equal the daemon's `shm::NUM_CONTEXT` (mirrored as `shm_client.NUM_CONTEXT`).
    #
    # This is what the policy layer hands *down*: object identity, not mass -- see `NUM_CONTEXT`'s
    # comment for why the cerebellum is not told how heavy the thing is. All-zero is the neutral
    # "no context" value, and it is deliberately the default: an unconfigured arm degrades to the
    # proprioception-only cerebellum rather than pinning one arbitrary context.
    #
    # `send_action` uses this as the *fallback*, the same way `default_k`/`default_d` are used --
    # an action that carries its own `context.<i>` keys (a policy that emits them, or the recording
    # pipeline's `PontineContextProcessorStep`) wins. During teleop nothing emits them, so this is
    # the operator's declaration of which situation is being demonstrated.
    #
    # Unlike the Rust daemon's `--cerebellum-context`, this travels the real path: shared memory,
    # the pontine low-pass, and the mossy fibres, so it can differ per `lerobot-record` run without
    # restarting the daemon and without bypassing the filter.
    pontine_context: tuple[float, ...] = (0.0, 0.0)

    # Contexts to rotate through, one per kept episode, so a single `lerobot-record` run
    # interleaves them. Empty keeps `pontine_context` fixed for the whole run. Interleaving is not
    # a convenience: most granule cells draw no context fibre, so their weights are shared, and
    # teaching one context to convergence before the other leaves the first reading 98 where it
    # should read 40. When set, the first entry replaces `pontine_context` as the starting value.
    pontine_context_cycle: tuple[tuple[float, ...], ...] = ()

    # How long to wait for the shared-memory segment to appear (i.e. for the Rust daemon to have
    # started) before giving up in `connect()`.
    shm_attach_timeout_s: float = 5.0
    # How long to wait for the daemon to ack a configuration/calibration command.
    command_ack_timeout_s: float = 2.0


@RobotConfig.register_subclass("so101_follower_impedance")
@dataclass
class SO101ImpedanceFollowerRobotConfig(RobotConfig, SO101ImpedanceFollowerConfig):
    pass
