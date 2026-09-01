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

from dataclasses import dataclass

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep

# The 5 arm joints AND the gripper -- all 6 motors are impedance-controlled on the SO101
# impedance follower (a rigid position-mode gripper can't grasp fragile objects gently).
DEFAULT_IMPEDANCE_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@ProcessorStepRegistry.register("impedance_gain_defaults_processor")
@dataclass
class ImpedanceGainDefaultsProcessorStep(RobotActionProcessorStep):
    """
    Fills in default per-motor spring (K) / damper (D) gains for an impedance-controlled robot's
    action whenever they're absent -- e.g. during plain position teleoperation, where the leader
    arm only ever supplies `.pos` values, never `.k`/`.d`.

    This step must run in the *teleop* action pipeline, upstream of `robot.send_action()`:
    `record_loop()` (`src/lerobot/scripts/lerobot_record.py`) writes the teleop action pipeline's
    *output* to the dataset, not whatever the robot actually receives after its own clipping --
    so filling in K/D here (not inside the robot's `send_action`) is what makes them show up as
    recorded ground truth for ACT to train on.

    `lerobot-record` inserts this automatically for robots that ask for it via
    `teleop_action_processor_steps()`, seeded from the robot's own configured gains; constructing
    it by hand is only needed to record gains that differ from the ones the robot will apply.

    Recording constant default gains this way is a known v1 limitation, not variable
    teleop-taught gains: ACT will initially learn to reproduce the configured default K/D rather
    than gains that vary with the demonstration.

    Attributes:
        impedance_joints: Names of the impedance-controlled motors, in order (arm joints, then
            gripper).
        default_k: Default spring constant per motor, same order as `impedance_joints`.
        default_d: Default damper constant per motor, same order as `impedance_joints`.
    """

    # Fallback only. `lerobot-record` overrides these from the robot's own config, because
    # recording gains the robot would not actually apply would teach the policy a K/D pair that
    # does not correspond to the compliance visible in the videos. These mirror
    # `SO101ImpedanceFollowerConfig`, which documents how they were measured.
    impedance_joints: tuple[str, ...] = DEFAULT_IMPEDANCE_JOINTS
    default_k: tuple[float, ...] = (10.0, 20.0, 15.0, 10.0, 8.0, 5.0)
    default_d: tuple[float, ...] = (0.3, 0.5, 0.4, 0.3, 0.2, 0.15)

    def __post_init__(self):
        if len(self.default_k) != len(self.impedance_joints) or len(self.default_d) != len(
            self.impedance_joints
        ):
            raise ValueError("default_k/default_d must have the same length as impedance_joints")

    def action(self, action: RobotAction) -> RobotAction:
        new_action = dict(action)
        for i, motor in enumerate(self.impedance_joints):
            new_action.setdefault(f"{motor}.k", self.default_k[i])
            new_action.setdefault(f"{motor}.d", self.default_d[i])
        return new_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for motor in self.impedance_joints:
            for suffix in ("k", "d"):
                key = f"{motor}.{suffix}"
                if key not in features[PipelineFeatureType.ACTION]:
                    features[PipelineFeatureType.ACTION][key] = PolicyFeature(
                        type=FeatureType.ACTION, shape=(1,)
                    )
        return features
