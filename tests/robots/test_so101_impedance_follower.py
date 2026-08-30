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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode
from lerobot.processor import ImpedanceGainDefaultsProcessorStep, PontineContextProcessorStep
from lerobot.robots.so101_impedance_follower import (
    SO101ImpedanceFollower,
    SO101ImpedanceFollowerRobotConfig,
)
from lerobot.robots.so101_impedance_follower.shm_client import NUM_CONTEXT, CommandKind
from lerobot.robots.utils import make_robot_from_config

_PATCH_TARGET = "lerobot.robots.so101_impedance_follower.so101_impedance_follower.ImpedanceShmClient"


def _make_shm_mock() -> MagicMock:
    shm = MagicMock(name="ImpedanceShmClientMock")
    shm.is_output_fresh.return_value = True
    shm.send_command_and_wait.return_value = 0
    return shm


@pytest.fixture
def follower():
    shm_mock = _make_shm_mock()
    with patch(_PATCH_TARGET, return_value=shm_mock):
        cfg = SO101ImpedanceFollowerRobotConfig(shm_name="test_shm", id="test")
        robot = SO101ImpedanceFollower(cfg)
        # Deterministic calibration: full raw range [0, 4095] for every motor, no drive-mode flip,
        # so the degree/percent conversions below can be checked against a closed-form formula.
        robot.calibration = {
            motor: MotorCalibration(id=i + 1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095)
            for i, motor in enumerate(robot.impedance_joints)
        }
        robot.connect(calibrate=False)
        yield robot, shm_mock
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    robot, _shm_mock = follower
    assert robot.is_connected
    robot.disconnect()
    assert not robot.is_connected


def test_feature_shapes(follower):
    robot, _ = follower
    # 6 motors x (.pos + .current_avg) = 12
    assert len(robot.observation_features) == 12
    # 6 motors x (.pos + .k + .d) = 18 -- the gripper is impedance-controlled too -- plus the
    # pontine context channels, which are action columns because the policy layer declares them
    # downward rather than sensing them.
    assert len(robot.action_features) == 18 + NUM_CONTEXT
    for motor in robot.impedance_joints:
        assert f"{motor}.k" in robot.action_features
        assert f"{motor}.d" in robot.action_features
    assert "gripper.k" in robot.action_features
    assert "gripper.d" in robot.action_features
    for i in range(NUM_CONTEXT):
        assert f"context.{i}" in robot.action_features


def test_get_observation_converts_raw_ticks_to_normalized_units(follower):
    robot, shm_mock = follower
    raw_positions = [0.0, 4095.0, 2047.5, 1000.0, 3000.0, 2048.0]
    currents = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    shm_mock.read_output.return_value = {
        "timestamp_mono_ns": 0,
        "present_pos": raw_positions,
        "present_vel": [0.0] * 6,
        "present_current_avg": currents,
        "fault_flags": 0,
    }

    obs = robot.get_observation()

    # DEGREES conversion for arm joints: (raw - mid) * 360 / max_res, mid=2047.5, max_res=4095.
    for i, motor in enumerate(robot.impedance_joints[:-1]):
        expected = (raw_positions[i] - 2047.5) * 360 / 4095
        assert obs[f"{motor}.pos"] == pytest.approx(expected)

    # RANGE_0_100 conversion for the gripper: (raw - min) / (max - min) * 100.
    expected_gripper = (raw_positions[5] - 0) / (4095 - 0) * 100
    assert obs["gripper.pos"] == pytest.approx(expected_gripper)

    for i, motor in enumerate(robot.impedance_joints):
        assert obs[f"{motor}.current_avg"] == pytest.approx(currents[i])


def test_send_action_requires_all_positions(follower):
    robot, _ = follower
    incomplete = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints if motor != "gripper"}
    with pytest.raises(ValueError, match="missing required keys"):
        robot.send_action(incomplete)


def test_send_action_clamps_gains_and_converts_positions(follower):
    robot, shm_mock = follower
    shm_mock.read_output.return_value = {
        "timestamp_mono_ns": 0,
        "present_pos": [2047.5] * 6,
        "present_vel": [0.0] * 6,
        "present_current_avg": [0.0] * 6,
        "fault_flags": 0,
    }

    action = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints}
    action["gripper.pos"] = 50.0
    # Out-of-range gains on shoulder_pan should clamp; other motors omit .k/.d entirely and must
    # fall back to the configured defaults.
    action["shoulder_pan.k"] = 10_000.0  # above k_max
    action["shoulder_pan.d"] = -5.0  # below d_min

    sent = robot.send_action(action)

    assert sent["shoulder_pan.k"] == robot.config.k_max
    assert sent["shoulder_pan.d"] == robot.config.d_min
    for i, motor in enumerate(robot.impedance_joints):
        if motor == "shoulder_pan":
            continue
        assert sent[f"{motor}.k"] == robot.config.default_k[i]
        assert sent[f"{motor}.d"] == robot.config.default_d[i]

    shm_mock.write_input.assert_called_once()
    _, kwargs = shm_mock.write_input.call_args
    # 0.0 degrees at mid=2047.5, max_res=4095 -> raw tick == mid, for the 5 arm joints.
    assert kwargs["target_pos"][:5] == pytest.approx([2047.5] * 5)
    # Gripper: 50.0 (RANGE_0_100) at range [0, 4095] -> raw tick == (50/100)*4095.
    assert kwargs["target_pos"][5] == pytest.approx((50.0 / 100) * 4095)
    assert kwargs["k_gain"][0] == robot.config.k_max
    assert kwargs["d_gain"][0] == robot.config.d_min
    # Gripper still gets its own (softer) default gain, same defense-in-depth path as arm joints.
    assert kwargs["k_gain"][5] == robot.config.default_k[5]
    assert kwargs["d_gain"][5] == robot.config.default_d[5]


def test_configure_sets_pwm_and_torque_enable_for_all_motors(follower):
    robot, shm_mock = follower
    shm_mock.reset_mock()

    robot.configure()

    timeout = robot.config.command_ack_timeout_s
    # All 6 motors -- including the gripper -- run in PWM mode under the impedance law; there is
    # no separate position-mode branch for the gripper anymore.
    for motor in robot.impedance_joints:
        shm_mock.send_command_and_wait.assert_any_call(
            CommandKind.SET_OPERATING_MODE, robot._motor_id(motor), [float(OperatingMode.PWM.value)], timeout
        )
        shm_mock.send_command_and_wait.assert_any_call(
            CommandKind.SET_TORQUE_ENABLE, robot._motor_id(motor), [1.0], timeout
        )


def test_make_robot_from_config_dispatches_impedance_follower():
    cfg = SO101ImpedanceFollowerRobotConfig(shm_name="test_shm_dispatch", id="test_dispatch")
    robot = make_robot_from_config(cfg)
    assert isinstance(robot, SO101ImpedanceFollower)


def test_teleop_action_processor_steps_seeds_gains_from_the_robots_own_config():
    # The gains recorded into a dataset have to be the gains the arm actually applied while the
    # demonstration happened -- otherwise the policy learns a K/D pair that does not correspond to
    # the compliance visible in the videos. So the step is built from this robot's config, not from
    # the processor step's own fallback defaults.
    config = SO101ImpedanceFollowerRobotConfig(
        shm_name="unused",
        default_k=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        default_d=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
    )
    robot = SO101ImpedanceFollower(config)

    step, _ = robot.teleop_action_processor_steps()

    assert isinstance(step, ImpedanceGainDefaultsProcessorStep)
    assert step.impedance_joints == robot.impedance_joints
    assert step.default_k == config.default_k
    assert step.default_d == config.default_d


def test_teleop_action_processor_steps_fill_every_recorded_action_dimension():
    # A position-only leader arm supplies 6 keys; the dataset's action feature is 18 + NUM_CONTEXT
    # wide. The gap is exactly what these steps close, and they have to close it here rather than
    # in `send_action`, which runs after the frame is written.
    robot = SO101ImpedanceFollower(SO101ImpedanceFollowerRobotConfig(shm_name="unused"))
    steps = robot.teleop_action_processor_steps()

    filled = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints}
    for step in steps:
        filled = step.action(filled)

    assert set(filled) == set(robot.action_features)
    assert len(filled) == 18 + NUM_CONTEXT


def test_teleop_action_processor_steps_do_not_override_supplied_gains():
    # During policy rollout the action already carries predicted K/D; the step must leave those be.
    robot = SO101ImpedanceFollower(SO101ImpedanceFollowerRobotConfig(shm_name="unused"))
    step, _ = robot.teleop_action_processor_steps()

    action = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints}
    action["gripper.k"] = 99.0
    filled = step.action(action)

    assert filled["gripper.k"] == 99.0
    assert filled["shoulder_lift.k"] == robot.config.default_k[1]


def _telemetry_for(robot) -> dict:
    return {
        "timestamp_mono_ns": 0,
        "present_pos": [2047.5] * len(robot.impedance_joints),
        "present_vel": [0.0] * len(robot.impedance_joints),
        "present_current_avg": [0.0] * len(robot.impedance_joints),
        "fault_flags": 0,
    }


def test_send_action_defaults_context_from_config(follower):
    # Nothing in a teleop action carries a context, so the arm falls back to the one the operator
    # configured for this run -- the same defaulting path K/D take.
    robot, shm_mock = follower
    robot.config.pontine_context = (1.0, -1.0)
    shm_mock.read_output.return_value = _telemetry_for(robot)

    sent = robot.send_action({f"{motor}.pos": 0.0 for motor in robot.impedance_joints})

    _, kwargs = shm_mock.write_input.call_args
    assert kwargs["context"] == [1.0, -1.0]
    assert [sent[f"context.{i}"] for i in range(NUM_CONTEXT)] == [1.0, -1.0]


def test_send_action_context_from_the_action_wins_over_config(follower):
    # A policy that has learned to emit the context is the cortex the config default stands in for,
    # so its declaration must override the operator's.
    robot, shm_mock = follower
    robot.config.pontine_context = (0.0, 0.0)
    shm_mock.read_output.return_value = _telemetry_for(robot)

    action = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints}
    action["context.0"] = 0.7

    robot.send_action(action)

    _, kwargs = shm_mock.write_input.call_args
    # Channel 0 comes from the action; channel 1 was not predicted and still defaults.
    assert kwargs["context"] == [pytest.approx(0.7), 0.0]


def test_send_action_clamps_context_to_the_documented_range(follower):
    # The granule code was sized for [-1, 1]; a policy is free to overshoot it, and an overshoot
    # would be a step in the feedforward the reflex has to absorb.
    robot, shm_mock = follower
    shm_mock.read_output.return_value = _telemetry_for(robot)

    action = {f"{motor}.pos": 0.0 for motor in robot.impedance_joints}
    action["context.0"] = 42.0
    action["context.1"] = -42.0

    robot.send_action(action)

    _, kwargs = shm_mock.write_input.call_args
    assert kwargs["context"] == [1.0, -1.0]


def test_context_length_mismatch_fails_at_construction():
    # A wrong length would otherwise surface as a ctypes error mid-episode, after the run was set up.
    with pytest.raises(ValueError, match="pontine_context"):
        SO101ImpedanceFollower(
            SO101ImpedanceFollowerRobotConfig(shm_name="unused", pontine_context=(1.0,) * (NUM_CONTEXT + 1))
        )


def test_teleop_action_processor_steps_seed_context_from_the_robots_own_config():
    # Same contract as the gains: the dataset must be labeled with the context that was actually
    # declared to the cerebellum while the demonstration happened.
    config = SO101ImpedanceFollowerRobotConfig(shm_name="unused", pontine_context=(1.0, 0.0))
    robot = SO101ImpedanceFollower(config)

    _, context_step = robot.teleop_action_processor_steps()

    assert isinstance(context_step, PontineContextProcessorStep)
    assert context_step.context == (1.0, 0.0)
