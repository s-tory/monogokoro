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
    PontineContextProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

_IMPEDANCE_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _base_action() -> dict:
    return {f"{motor}.pos": float(i) for i, motor in enumerate(_IMPEDANCE_JOINTS)}


def test_fills_the_configured_context():
    step = PontineContextProcessorStep(context=(1.0, -1.0))

    out = step.action(_base_action())

    assert out["context.0"] == 1.0
    assert out["context.1"] == -1.0


def test_default_context_is_neutral():
    # An unconfigured step must be a no-op on the cerebellum: all-zero is the value that degrades
    # it to proprioception-only rather than pinning one arbitrary context.
    step = PontineContextProcessorStep()

    out = step.action(_base_action())

    assert [out["context.0"], out["context.1"]] == [0.0, 0.0]


def test_does_not_override_a_context_already_in_the_action():
    # Once a policy emits its own context, that is the cortex the configured constant stands in
    # for -- overwriting it would silently discard the thing being tested.
    step = PontineContextProcessorStep(context=(1.0, 1.0))

    action = _base_action()
    action["context.0"] = -0.25
    out = step.action(action)

    assert out["context.0"] == -0.25
    assert out["context.1"] == 1.0


def test_leaves_the_rest_of_the_action_untouched():
    step = PontineContextProcessorStep(context=(0.5, 0.5))

    action = _base_action()
    out = step.action(action)

    for key, value in action.items():
        assert out[key] == value
    # Input dict is not mutated in place.
    assert "context.0" not in action


@pytest.mark.parametrize("bad", [(1.5, 0.0), (0.0, -2.0)])
def test_rejects_out_of_range_context(bad):
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        PontineContextProcessorStep(context=bad)


def test_rejects_empty_context():
    with pytest.raises(ValueError, match="at least one channel"):
        PontineContextProcessorStep(context=())


def test_declares_the_context_action_features():
    # Without this the dataset would carry the values but never declare the columns, so
    # `build_dataset_frame` would drop them.
    step = PontineContextProcessorStep(context=(0.0, 0.0))
    features = {t: {} for t in PipelineFeatureType}

    out = step.transform_features(features)

    for i in range(2):
        assert out[PipelineFeatureType.ACTION][f"context.{i}"] == PolicyFeature(
            type=FeatureType.ACTION, shape=(1,)
        )


def test_works_inside_a_robot_action_pipeline():
    pipeline = RobotProcessorPipeline(
        steps=[PontineContextProcessorStep(context=(1.0, 0.0))],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    out = pipeline((_base_action(), {}))

    assert out["context.0"] == 1.0
    assert out["context.1"] == 0.0


def test_a_cycle_sets_the_starting_context():
    # The cycle is the source of truth once configured, so the first episode records its first
    # entry rather than a `context` the caller may not have bothered to keep in sync.
    step = PontineContextProcessorStep(context=(0.0, 0.0), cycle=((1.0, 0.0), (-1.0, 0.0)))

    assert step.context == (1.0, 0.0)
    assert step.cycle_index == 0
    assert step.action(_base_action())["context.0"] == 1.0


def test_cycle_context_advances_and_wraps():
    step = PontineContextProcessorStep(cycle=((1.0, 0.0), (-1.0, 0.0)))

    assert step.cycle_context() == (-1.0, 0.0)
    assert step.action(_base_action())["context.0"] == -1.0
    # Wrapping is what makes an odd number of episodes still alternate rather than stall.
    assert step.cycle_context() == (1.0, 0.0)
    assert step.cycle_index == 0


def test_cycle_context_is_a_no_op_without_a_cycle():
    # `record` calls this unconditionally on collected steps; a fixed context must stay fixed.
    step = PontineContextProcessorStep(context=(0.5, -0.5))

    assert step.cycle_context() == (0.5, -0.5)
    assert step.context == (0.5, -0.5)


def test_cycle_entries_are_validated():
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        PontineContextProcessorStep(cycle=((1.0, 0.0), (2.0, 0.0)))


def test_cycle_entries_must_agree_on_length():
    # A shorter entry would silently write fewer channels than the daemon reads.
    with pytest.raises(ValueError, match="same length"):
        PontineContextProcessorStep(cycle=((1.0, 0.0), (-1.0,)))
