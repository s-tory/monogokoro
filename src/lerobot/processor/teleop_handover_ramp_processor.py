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

import time
from dataclasses import dataclass, field

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("teleop_handover_ramp_processor")
@dataclass
class TeleopHandoverRampProcessorStep(RobotActionProcessorStep):
    """
    Moves each joint's target from where the follower is to where the leader is at a bounded rate,
    then passes the leader's `.pos` through unchanged.

    Teleoperation maps the leader's joint angles onto the follower's absolutely. Both arms normalize
    angles about the middle of their calibrated travel, which does not depend on the pose held at
    calibration, so the two agree wherever their mechanical stops agree (not verified joint by joint;
    on 2026-09-25 the calibrated widths matched within 80 counts except shoulder_pan, 2090 on the
    leader against 2782). What that leaves is the start: the follower is parked wherever it was and
    the leader is held wherever the hand is, and an absolute map jumps the follower across the gap
    in one tick. On 2026-09-24 that jump was read as a fault and answered with a clutch, which
    re-referenced every session. That made the gripper's grip force depend on how far the leader
    happened to be closed at connect (a closed leader held -104 duty in one session, -35 in the
    next), and made the leader's shape stop matching the follower's. Removed 2026-09-25.

    Per joint, the target starts at the follower's own pose and moves toward the live leader reading
    by at most `rate * dt` per call. Once it reaches the leader it latches and follows it exactly for
    the rest of the step's lifetime, so a later fast motion is not slowed. `reset()` starts over.

    `rate` is 10 deg/s for DEGREES joints and 10 %/s for the gripper's 0-100 range. Derivation: the
    median speed of each joint while moving, over four teleoperated episodes on 2026-09-25 (7.4 V,
    30 fps finite difference), ran from 5.3 deg/s (shoulder_pan, wrist_roll) to 15.8 deg/s
    (elbow_flex), gripper 5.7 %/s. 10 sits inside that band, so the handover is no faster than
    ordinary operation, and the largest leader/follower difference measured that day (24 deg) takes
    2.4 s.

    Attributes:
        joints: Motor names whose `.pos` is ramped. Other action keys pass through untouched.
        rate: Maximum target speed during the handover, in the action's own units per second.
    """

    joints: tuple[str, ...] = ()
    rate: float = 10.0
    _target: dict[str, float] | None = field(default=None, init=False, repr=False)
    _engaged: set[str] = field(default_factory=set, init=False, repr=False)
    _last_t: float | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        now = time.monotonic()
        if self._target is None:
            observation = self.transition.get(TransitionKey.OBSERVATION)
            missing = [f"{m}.pos" for m in self.joints if f"{m}.pos" not in (observation or {})]
            if missing:
                raise ValueError(f"cannot start the handover: observation is missing {missing}")
            self._target = {m: float(observation[f"{m}.pos"]) for m in self.joints}
            self._last_t = now
        max_step = self.rate * (now - self._last_t)
        self._last_t = now

        new_action = dict(action)
        for m in self.joints:
            leader = float(action[f"{m}.pos"])
            if m not in self._engaged:
                gap = leader - self._target[m]
                if abs(gap) <= max_step:
                    self._engaged.add(m)
                else:
                    self._target[m] += max_step if gap > 0 else -max_step
            new_action[f"{m}.pos"] = leader if m in self._engaged else self._target[m]
        return new_action

    def reset(self) -> None:
        self._target = None
        self._engaged = set()
        self._last_t = None

    def get_config(self) -> dict:
        return {"joints": list(self.joints), "rate": self.rate}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
