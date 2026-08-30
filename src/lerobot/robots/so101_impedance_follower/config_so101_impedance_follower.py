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
    # These K values are measured, not chosen. Holding the arm outstretched (near worst case for
    # gravity) at K=1 makes the daemon's reported PWM read out directly as the duty each joint
    # needs to hold itself, since pwm = 1 * err: 17 / 87 / 61 / ~0 / ~0 / 0 counts for pan / lift /
    # elbow / wrist_flex / wrist_roll / gripper. A pure PD law always droops under a constant load
    # by `err = holding_duty / K`, so K follows from the droop you will accept -- these target ~5
    # counts (0.4 deg) and are floored at a value that still gives each joint positioning
    # authority. Re-measure per arm: the numbers are specific to this unit's friction and wiring.
    default_k: tuple[float, ...] = (10.0, 20.0, 15.0, 10.0, 8.0, 5.0)

    # D is bounded from above by velocity *noise*, not by stability. Position is quantised to whole
    # counts, so the filtered finite difference has a noise floor of about
    # `1 / (vel_filter_window * dt)` -- ~50 counts/s at the daemon's defaults -- which D turns
    # straight into PWM chatter. Keeping D near K/40 holds that under ~2% duty while still damping
    # a real 100 counts/s motion with a meaningful command.
    default_d: tuple[float, ...] = (0.3, 0.5, 0.4, 0.3, 0.2, 0.15)

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
