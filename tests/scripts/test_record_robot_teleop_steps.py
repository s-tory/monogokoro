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

"""Covers the hook `lerobot-record` uses to let a robot add teleop-pipeline steps of its own."""

import pytest

pytest.importorskip("datasets", reason="lerobot.scripts.lerobot_record requires the `dataset` extra")

from lerobot.scripts.lerobot_record import _robot_teleop_action_steps  # noqa: E402


class _PlainRobot:
    """Stands in for every robot that has no extra teleop requirements -- i.e. almost all of them."""


class _HookedRobot:
    def teleop_action_processor_steps(self):
        return ["step-a", "step-b"]


def test_robot_without_the_hook_contributes_no_steps():
    # The hook is duck-typed, so the common case must be a robot that simply does not define it.
    assert _robot_teleop_action_steps(_PlainRobot()) == []


def test_robot_with_the_hook_contributes_its_steps():
    assert _robot_teleop_action_steps(_HookedRobot()) == ["step-a", "step-b"]


def test_returned_steps_are_a_fresh_list():
    # `record` splices these into a pipeline; handing back the robot's own container would let one
    # recording session mutate what the next one gets.
    robot = _HookedRobot()
    steps = _robot_teleop_action_steps(robot)
    steps.append("mutated")
    assert _robot_teleop_action_steps(robot) == ["step-a", "step-b"]
