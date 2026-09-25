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

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction, TransitionKey

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("teleop_clutch_processor")
@dataclass
class TeleopClutchProcessorStep(RobotActionProcessorStep):
    """
    Turns a leader arm's absolute `.pos` readings into follower targets relative to where the
    follower was when teleoperation started.

    The first call anchors: it takes the follower's own `.pos` from the observation and this call's
    leader reading as the reference. Every later call targets `follower_ref + (leader - leader_ref)`.
    So the first commanded target is always the follower's current pose, however far the leader's
    home is from the follower's.

    Why a teleop step and not part of the robot: `record_loop()` writes this pipeline's *output* as
    the dataset's `action`. Until 2026-09-25 the clutch lived in `SO101ImpedanceFollower.send_action`,
    so the recorded `action` was the leader's reading before the clutch. It sat a constant
    offset away from `observation.state`, and that offset was re-drawn on every connect: on the first
    recorded episode it was -8.35 deg on shoulder_pan and -22.55 deg on wrist_roll. Here the recorded
    action is the target the follower was actually sent, in the same frame as the state. A policy's
    output also reaches the robot unchanged, since policies never go through this pipeline.

    Why the anchor is needed at all (2026-09-24): a leader folded into a pistol grip rests, by
    construction, away from the middle of several joints' travel, and normalized "0" is that middle.
    Without the anchor every session opened with the follower jumping to the leader's absolute pose.

    The anchor is held for the step's lifetime. `lerobot-record` and `lerobot-teleoperate` build the
    pipeline once per run, which is also once per `connect()`; `reset()` clears it.

    Attributes:
        joints: Motor names whose `.pos` is clutched. Other action keys pass through untouched.
    """

    joints: tuple[str, ...] = ()
    _leader_ref: dict[str, float] | None = field(default=None, init=False, repr=False)
    _follower_ref: dict[str, float] | None = field(default=None, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        if self._leader_ref is None:
            observation = self.transition.get(TransitionKey.OBSERVATION)
            missing = [f"{m}.pos" for m in self.joints if f"{m}.pos" not in (observation or {})]
            if missing:
                raise ValueError(f"cannot anchor the clutch: observation is missing {missing}")
            self._leader_ref = {m: float(action[f"{m}.pos"]) for m in self.joints}
            self._follower_ref = {m: float(observation[f"{m}.pos"]) for m in self.joints}

        new_action = dict(action)
        for m in self.joints:
            new_action[f"{m}.pos"] = self._follower_ref[m] + (float(action[f"{m}.pos"]) - self._leader_ref[m])
        return new_action

    def reset(self) -> None:
        self._leader_ref = None
        self._follower_ref = None

    def get_config(self) -> dict:
        return {"joints": list(self.joints)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
