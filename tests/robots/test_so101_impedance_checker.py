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

from lerobot.motors.feetech import OperatingMode
from lerobot.robots.so101_impedance_follower.checker import (
    MOTOR_NAMES,
    SO101ImpedanceChecker,
    format_state_table,
)
from lerobot.robots.so101_impedance_follower.shm_client import (
    FAULT_COMMS_ERROR,
    FAULT_POS_LIMIT,
    FAULT_WATCHDOG_TIMEOUT,
    CommandKind,
    ImpedanceShmClientError,
)

_CLIENT_PATCH_TARGET = "lerobot.robots.so101_impedance_follower.checker.ImpedanceShmClient"


def _telemetry(positions=None, currents=None, fault_flags=0, pwm_cmd=None, ff_pwm=None) -> dict:
    positions = positions or list(range(6))
    currents = currents or [0.0] * 6
    pwm_cmd = pwm_cmd if pwm_cmd is not None else [0.0] * 6
    # `read_state` reads this unconditionally, so a fake without it fails with a bare KeyError
    # rather than telling you the fake has fallen behind the client.
    ff_pwm = ff_pwm if ff_pwm is not None else [0.0] * 6
    return {
        "timestamp_mono_ns": 0,
        "present_pos": positions,
        "present_vel": [0.0] * 6,
        "present_current_avg": currents,
        "pwm_cmd": pwm_cmd,
        "ff_pwm": ff_pwm,
        "fault_flags": fault_flags,
    }


def _make_client_mock() -> MagicMock:
    client = MagicMock(name="ImpedanceShmClientMock")
    client.read_output.return_value = _telemetry()
    client.send_command_and_wait.return_value = 0
    return client


@pytest.fixture
def checker():
    client_mock = _make_client_mock()
    with patch(_CLIENT_PATCH_TARGET, return_value=client_mock):
        c = SO101ImpedanceChecker(shm_name="test_shm")
        yield c, client_mock
        c.close()


def test_read_state_maps_telemetry_by_motor_name(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(
        positions=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        currents=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )

    state = c.read_state()

    assert set(state.keys()) == set(MOTOR_NAMES)
    for i, motor in enumerate(MOTOR_NAMES):
        assert state[motor]["present_pos"] == pytest.approx(10.0 * (i + 1))
        assert state[motor]["present_current_avg"] == pytest.approx(float(i + 1))


def test_move_to_holds_unspecified_motors_at_current_position(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(positions=[100.0] * 6)

    c.move_to({"gripper": 2048.0}, k=15.0, d=1.5)

    client_mock.write_input.assert_called_once()
    _, kwargs = client_mock.write_input.call_args
    assert kwargs["target_pos"][:5] == pytest.approx([100.0] * 5)  # arm joints stay put
    assert kwargs["target_pos"][5] == pytest.approx(2048.0)  # gripper moves
    assert kwargs["k_gain"] == pytest.approx([15.0] * 6)
    assert kwargs["d_gain"] == pytest.approx([1.5] * 6)


def test_move_to_supports_per_motor_gains(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry()

    c.move_to({}, k={"gripper": 5.0}, d={"gripper": 0.5})

    _, kwargs = client_mock.write_input.call_args
    assert kwargs["k_gain"][5] == pytest.approx(5.0)
    assert kwargs["k_gain"][0] == pytest.approx(0.0)  # unspecified motors default to 0.0
    assert kwargs["d_gain"][5] == pytest.approx(0.5)


def test_hold_current_position_targets_the_freshly_read_position(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(positions=[7.0] * 6)

    c.hold_current_position(k=10.0, d=1.0)

    _, kwargs = client_mock.write_input.call_args
    assert kwargs["target_pos"] == pytest.approx([7.0] * 6)
    assert kwargs["k_gain"] == pytest.approx([10.0] * 6)


def test_set_pwm_mode_sends_operating_mode_for_all_motors(checker):
    c, client_mock = checker
    client_mock.reset_mock()

    c.set_pwm_mode()

    for i, _motor in enumerate(MOTOR_NAMES):
        client_mock.send_command_and_wait.assert_any_call(
            CommandKind.SET_OPERATING_MODE, i + 1, [float(OperatingMode.PWM.value)], c.command_timeout_s
        )


def test_enable_and_disable_torque(checker):
    c, client_mock = checker
    client_mock.reset_mock()

    c.enable_torque()
    for i, _motor in enumerate(MOTOR_NAMES):
        client_mock.send_command_and_wait.assert_any_call(
            CommandKind.SET_TORQUE_ENABLE, i + 1, [1.0], c.command_timeout_s
        )

    client_mock.reset_mock()
    c.disable_torque()
    for i, _motor in enumerate(MOTOR_NAMES):
        client_mock.send_command_and_wait.assert_any_call(
            CommandKind.SET_TORQUE_ENABLE, i + 1, [0.0], c.command_timeout_s
        )


def test_describe_faults_reports_known_flags(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(fault_flags=FAULT_WATCHDOG_TIMEOUT | FAULT_COMMS_ERROR)

    faults = c.describe_faults()

    assert any("watchdog_timeout" in f for f in faults)
    assert any("comms_error" in f for f in faults)


def test_describe_faults_reports_a_joint_past_its_soft_limits(checker):
    """The symptom is otherwise ambiguous: a joint the limits hold at zero PWM looks exactly like
    the watchdog, a blind run, or a gain that is simply too soft."""
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(fault_flags=FAULT_POS_LIMIT)

    faults = c.describe_faults()

    assert len(faults) == 1
    assert "pos_limit" in faults[0]


def test_describe_faults_empty_when_no_faults(checker):
    c, client_mock = checker
    client_mock.read_output.return_value = _telemetry(fault_flags=0)

    assert c.describe_faults() == []


def test_close_without_started_daemon_never_touches_subprocess(checker):
    c, client_mock = checker
    with patch("lerobot.robots.so101_impedance_follower.checker.subprocess.Popen") as popen_mock:
        c.close()
        popen_mock.assert_not_called()
    client_mock.close.assert_called_once()


def test_start_daemon_spawns_subprocess_and_terminates_on_close():
    client_mock = _make_client_mock()
    process_mock = MagicMock()
    process_mock.poll.return_value = None  # still running

    with (
        patch(_CLIENT_PATCH_TARGET, return_value=client_mock),
        patch(
            "lerobot.robots.so101_impedance_follower.checker.subprocess.Popen", return_value=process_mock
        ) as popen_mock,
    ):
        c = SO101ImpedanceChecker(
            shm_name="test_shm_daemon", port="/dev/ttyFAKE", start_daemon=True, binary_path="fake_binary"
        )
        args = popen_mock.call_args.args[0]
        assert args[0] == "fake_binary"
        assert "--port" in args and "/dev/ttyFAKE" in args
        assert "--shm-name" in args and "test_shm_daemon" in args

        c.close()

    # Graceful shutdown command sent before falling back to process termination.
    client_mock.send_command_and_wait.assert_any_call(CommandKind.SHUTDOWN, 0, [0.0], timeout_s=3.0)
    process_mock.terminate.assert_called_once()


def test_daemon_is_terminated_even_if_shm_attach_fails():
    process_mock = MagicMock()
    process_mock.poll.return_value = None

    with (
        patch(_CLIENT_PATCH_TARGET, side_effect=ImpedanceShmClientError("boom")),
        patch("lerobot.robots.so101_impedance_follower.checker.subprocess.Popen", return_value=process_mock),
        pytest.raises(ImpedanceShmClientError),
    ):
        SO101ImpedanceChecker(shm_name="test_shm_fail", port="/dev/ttyFAKE", start_daemon=True)

    process_mock.terminate.assert_called_once()


def test_start_daemon_requires_port():
    with pytest.raises(ValueError, match="port"):
        SO101ImpedanceChecker(shm_name="test_shm_no_port", start_daemon=True)


def test_format_state_table_contains_all_motors_and_faults():
    state = {
        motor: {"present_pos": 1.0, "present_vel": 0.0, "present_current_avg": 2.0} for motor in MOTOR_NAMES
    }
    table = format_state_table(state, faults=["watchdog_timeout (no fresh input from Python)"])

    for motor in MOTOR_NAMES:
        assert motor in table
    assert "FAULTS" in table


def _moving_client(client_mock, motor_index: int, per_call_delta: float):
    """Makes read_output() report `motor_index` drifting by `per_call_delta` on each call."""
    pos = [2048.0] * 6
    calls = {"n": 0}

    def read_output(*_a, **_k):
        calls["n"] += 1
        p = list(pos)
        p[motor_index] = 2048.0 + per_call_delta * calls["n"]
        return _telemetry(positions=p)

    client_mock.read_output.side_effect = read_output


def test_probe_direction_reports_correct_when_joint_moves_toward_target(checker):
    c, client_mock = checker
    # wrist_roll (index 4) moves in the +position direction, i.e. the way we commanded.
    _moving_client(client_mock, 4, per_call_delta=+30.0)

    r = c.probe_direction("wrist_roll", duration_s=0.05)

    assert r["moved"] is True
    assert r["inverted"] is False
    assert r["delta"] > 0


def test_probe_direction_reports_inverted_when_joint_moves_away(checker):
    c, client_mock = checker
    _moving_client(client_mock, 4, per_call_delta=-30.0)

    r = c.probe_direction("wrist_roll", duration_s=0.05)

    assert r["moved"] is True
    assert r["inverted"] is True
    assert r["delta"] < 0


def test_probe_direction_is_inconclusive_when_joint_barely_moves(checker):
    c, client_mock = checker
    _moving_client(client_mock, 4, per_call_delta=+0.5)  # below the 10-tick threshold

    r = c.probe_direction("wrist_roll", duration_s=0.05)

    assert r["moved"] is False
    assert r["inverted"] is False


def test_probe_direction_drops_torque_even_when_aborting(checker):
    c, client_mock = checker
    # Huge jump per sample so the abort guard trips immediately.
    _moving_client(client_mock, 4, per_call_delta=+5000.0)

    c.probe_direction("wrist_roll", duration_s=5.0, abort_ticks=100.0)

    # The final command must be the all-zero-gain release, regardless of how the probe ended.
    _, kwargs = client_mock.write_input.call_args
    assert kwargs["k_gain"] == [0.0] * 6
    assert kwargs["d_gain"] == [0.0] * 6


def test_probe_direction_rejects_unknown_motor(checker):
    c, _ = checker
    with pytest.raises(ValueError, match="Unknown motor"):
        c.probe_direction("not_a_joint")
