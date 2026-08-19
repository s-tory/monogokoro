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

import pytest

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    ImpedanceGainDefaultsProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

# All 6 impedance-controlled motors -- the 5 arm joints AND the gripper (a rigid position-mode
# gripper can't grasp fragile objects gently, so it's under the same K/D law as the arm).
_IMPEDANCE_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _base_action() -> dict:
    return {f"{motor}.pos": float(i) for i, motor in enumerate(_IMPEDANCE_JOINTS)}


def test_fills_missing_gains_with_configured_defaults():
    step = ImpedanceGainDefaultsProcessorStep(
        impedance_joints=_IMPEDANCE_JOINTS,
        default_k=(1.0, 2.0, 3.0, 4.0, 5.0, 0.5),
        default_d=(0.1, 0.2, 0.3, 0.4, 0.5, 0.05),
    )

    out = step.action(_base_action())

    for i, motor in enumerate(_IMPEDANCE_JOINTS):
        assert out[f"{motor}.k"] == step.default_k[i]
        assert out[f"{motor}.d"] == step.default_d[i]
    assert out["gripper.k"] == step.default_k[-1]
    assert out["gripper.d"] == step.default_d[-1]


def test_does_not_override_gains_already_present():
    step = ImpedanceGainDefaultsProcessorStep(impedance_joints=_IMPEDANCE_JOINTS)
    action = _base_action()
    action["shoulder_pan.k"] = 999.0
    action["shoulder_pan.d"] = -1.0

    out = step.action(action)

    assert out["shoulder_pan.k"] == 999.0
    assert out["shoulder_pan.d"] == -1.0
    # Untouched motors still get the configured defaults.
    assert out["wrist_roll.k"] == step.default_k[_IMPEDANCE_JOINTS.index("wrist_roll")]
    assert out["gripper.k"] == step.default_k[_IMPEDANCE_JOINTS.index("gripper")]


def test_action_does_not_mutate_input_dict():
    step = ImpedanceGainDefaultsProcessorStep(impedance_joints=_IMPEDANCE_JOINTS)
    action = _base_action()
    original_keys = set(action)

    step.action(action)

    assert set(action) == original_keys


def test_rejects_mismatched_gain_and_joint_lengths():
    with pytest.raises(ValueError):
        ImpedanceGainDefaultsProcessorStep(impedance_joints=_IMPEDANCE_JOINTS, default_k=(1.0, 2.0))


def test_transform_features_adds_gain_dims_without_clobbering_existing():
    step = ImpedanceGainDefaultsProcessorStep(impedance_joints=_IMPEDANCE_JOINTS)
    features = {
        PipelineFeatureType.ACTION: {
            f"{motor}.pos": PolicyFeature(type=FeatureType.ACTION, shape=(1,)) for motor in _IMPEDANCE_JOINTS
        }
    }
    # Pre-existing gain feature (e.g. from a policy's output_features) must be left untouched.
    features[PipelineFeatureType.ACTION]["shoulder_pan.k"] = PolicyFeature(
        type=FeatureType.ACTION, shape=(3,)
    )

    out = step.transform_features(features)
    action_features = out[PipelineFeatureType.ACTION]

    assert action_features["shoulder_pan.k"].shape == (3,)  # untouched
    for motor in _IMPEDANCE_JOINTS:
        if motor == "shoulder_pan":
            continue
        assert action_features[f"{motor}.k"] == PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        assert action_features[f"{motor}.d"] == PolicyFeature(type=FeatureType.ACTION, shape=(1,))
    assert "gripper.k" in action_features
    assert "gripper.d" in action_features


def test_composes_with_robot_processor_pipeline_like_record_loop():
    """Mirrors how `record_loop()` (lerobot_record.py) calls `teleop_action_processor((act, obs))`."""
    pipeline = RobotProcessorPipeline(
        steps=[ImpedanceGainDefaultsProcessorStep(impedance_joints=_IMPEDANCE_JOINTS)],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    out = pipeline((_base_action(), {}))

    for motor in _IMPEDANCE_JOINTS:
        assert f"{motor}.k" in out
        assert f"{motor}.d" in out
