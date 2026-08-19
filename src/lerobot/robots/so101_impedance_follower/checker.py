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

"""
A lightweight, standalone Python wrapper around the `rust/so101_impedance_ctrl` daemon, for
quickly sanity-checking that the daemon and the SO101's servos are wired up and talking to each
other -- independent of the full `SO101ImpedanceFollower` `Robot` class (no calibration file,
camera config, or dataset pipeline required).

Positions here are always raw encoder ticks (0-4095), not the calibrated degrees/percent units
`SO101ImpedanceFollower` uses: this tool is meant for bring-up/smoke-testing, including *before* a
calibration exists at all.

Typical usage, attaching to an already-running daemon:

    from lerobot.robots.so101_impedance_follower.checker import SO101ImpedanceChecker

    with SO101ImpedanceChecker(shm_name="so101_impedance") as checker:
        print(checker.read_state())
        checker.hold_current_position(k=20.0, d=1.0)

Or letting the checker launch the daemon itself:

    with SO101ImpedanceChecker(
        shm_name="so101_impedance_check", port="/dev/ttyACM0", start_daemon=True
    ) as checker:
        print(checker.read_state())

See `examples/check_so101_impedance.py` for a runnable CLI built on top of this class.
"""

import contextlib
import logging
import subprocess
import time

from lerobot.motors.feetech import OperatingMode

from .shm_client import (
    FAULT_COMMS_ERROR,
    FAULT_OVERCURRENT,
    FAULT_WATCHDOG_TIMEOUT,
    CommandKind,
    ImpedanceShmClient,
    ImpedanceShmClientError,
)

logger = logging.getLogger(__name__)

MOTOR_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
_MOTOR_IDS = {name: i + 1 for i, name in enumerate(MOTOR_NAMES)}


def _motor_id(name: str) -> int:
    try:
        return _MOTOR_IDS[name]
    except KeyError as e:
        raise ValueError(f"Unknown motor {name!r}; expected one of {MOTOR_NAMES}") from e


class SO101ImpedanceChecker:
    """Standalone helper for exercising the `so101_impedance_ctrl` daemon by hand.

    If `start_daemon=True`, this also launches the Rust binary as a subprocess and terminates it
    on `close()`; if `False` (default), it only attaches to a segment created by a daemon started
    elsewhere and never touches its lifecycle.
    """

    def __init__(
        self,
        shm_name: str,
        *,
        port: str | None = None,
        start_daemon: bool = False,
        binary_path: str = "so101_impedance_ctrl",
        daemon_args: dict[str, str | int | float] | None = None,
        attach_timeout_s: float = 10.0,
        command_timeout_s: float = 2.0,
    ):
        self.shm_name = shm_name
        self.command_timeout_s = command_timeout_s
        self._process: subprocess.Popen | None = None

        if start_daemon:
            if not port:
                raise ValueError("`port` is required when start_daemon=True")
            self._process = self._spawn_daemon(binary_path, port, shm_name, daemon_args or {})

        try:
            self.client = ImpedanceShmClient(shm_name, attach_timeout_s=attach_timeout_s)
        except ImpedanceShmClientError:
            self._terminate_daemon()
            raise

    def _spawn_daemon(
        self, binary_path: str, port: str, shm_name: str, daemon_args: dict[str, str | int | float]
    ) -> subprocess.Popen:
        args = [binary_path, "--port", port, "--shm-name", shm_name]
        for key, value in daemon_args.items():
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        logger.info("starting so101_impedance_ctrl: %s", " ".join(args))
        # `binary_path` and `daemon_args` are caller-supplied local configuration (a CLI flag or a
        # Python call site), not untrusted remote input; shell=False (the default) avoids shell
        # injection regardless.
        return subprocess.Popen(args)  # nosec B603 B607

    def _terminate_daemon(self, timeout_s: float = 3.0) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                logger.warning("so101_impedance_ctrl did not exit in time, killing it")
                self._process.kill()
                self._process.wait()
        self._process = None

    def shutdown_daemon(self, timeout_s: float = 3.0) -> None:
        """Asks the daemon (that this checker started) to exit gracefully via the command
        channel, falling back to SIGTERM/SIGKILL if it doesn't. No-op if this checker attached to
        a daemon it didn't start."""
        if self._process is None:
            return
        # The daemon may exit before acking -- fall through to process wait/terminate either way.
        with contextlib.suppress(ImpedanceShmClientError):
            self.client.send_command_and_wait(CommandKind.SHUTDOWN, 0, [0.0], timeout_s=timeout_s)
        self._terminate_daemon(timeout_s=timeout_s)

    def read_state(self) -> dict[str, dict[str, float]]:
        """Returns raw-tick telemetry per motor: `present_pos`, `present_vel`,
        `present_current_avg`, and the `pwm_cmd` the daemon last applied."""
        telemetry = self.client.read_output()
        return {
            motor: {
                "present_pos": telemetry["present_pos"][i],
                "present_vel": telemetry["present_vel"][i],
                "present_current_avg": telemetry["present_current_avg"][i],
                "pwm_cmd": telemetry["pwm_cmd"][i],
            }
            for i, motor in enumerate(MOTOR_NAMES)
        }

    @property
    def fault_flags(self) -> int:
        return self.client.read_output()["fault_flags"]

    def describe_faults(self) -> list[str]:
        flags = self.fault_flags
        faults = []
        if flags & FAULT_WATCHDOG_TIMEOUT:
            faults.append("watchdog_timeout (no fresh input from Python -- PWM held at zero)")
        if flags & FAULT_COMMS_ERROR:
            faults.append("comms_error (a register read/write to a servo failed)")
        if flags & FAULT_OVERCURRENT:
            faults.append("overcurrent")
        return faults

    def move_to(
        self,
        targets: dict[str, float],
        k: float | dict[str, float] = 20.0,
        d: float | dict[str, float] = 1.0,
        target_vel: dict[str, float] | None = None,
    ) -> None:
        """Commands the named motors (raw ticks, 0-4095) toward `targets` under impedance
        control. Motors not named in `targets` are commanded to hold their current (freshly-read)
        position, so `move_to({"gripper": 2048})` moves only the gripper without disturbing the
        arm's current pose."""
        state = self.read_state()
        target_pos = [targets.get(motor, state[motor]["present_pos"]) for motor in MOTOR_NAMES]
        k_gain = [k.get(motor, 0.0) if isinstance(k, dict) else k for motor in MOTOR_NAMES]
        d_gain = [d.get(motor, 0.0) if isinstance(d, dict) else d for motor in MOTOR_NAMES]
        vel = [(target_vel or {}).get(motor, 0.0) if target_vel is not None else 0.0 for motor in MOTOR_NAMES]
        self.client.write_input(target_pos=target_pos, k_gain=k_gain, d_gain=d_gain, target_vel=vel)

    def probe_direction(
        self,
        motor: str,
        probe_pwm_frac: float = 0.15,
        probe_ticks: float = 300.0,
        duration_s: float = 0.6,
        pwm_max: float = 1000.0,
        abort_ticks: float = 400.0,
    ) -> dict:
        """Measures which way positive PWM actually moves one joint.

        Determining the drive direction by turning K up until the arm reacts is a bad trade: if the
        sign is wrong, that *is* positive feedback, and the joint accelerates into a mechanical stop
        and stalls at full duty. This instead nudges a single joint with a small, bounded command
        and just watches the sign of the resulting motion -- every other motor is left at zero gain,
        the duty is a fraction of maximum, it runs for well under a second, and it bails out early
        if the joint travels further than `abort_ticks`.

        Returns the measured displacement and a verdict; `inverted=True` means the daemon needs
        (or needs to drop) `--invert-pwm`.
        """
        if motor not in MOTOR_NAMES:
            raise ValueError(f"Unknown motor {motor!r}; expected one of {MOTOR_NAMES}")

        start = self.read_state()[motor]["present_pos"]
        # Aim `probe_ticks` in the +position direction; K is sized so the initial command is
        # `probe_pwm_frac` of full scale.
        target = start + probe_ticks
        k = probe_pwm_frac * pwm_max / probe_ticks

        samples: list[float] = []
        deadline = time.monotonic() + duration_s
        try:
            while time.monotonic() < deadline:
                pos = self.read_state()[motor]["present_pos"]
                samples.append(pos)
                if abs(pos - start) > abort_ticks:
                    logger.warning(
                        "probe aborted: %s moved %.0f ticks (> %.0f)", motor, pos - start, abort_ticks
                    )
                    break
                self.move_to({motor: target}, k={motor: k}, d={motor: 0.0})
                time.sleep(0.02)
        finally:
            # Always drop torque, including on abort or Ctrl+C.
            self.move_to({}, k=0.0, d=0.0)

        end = samples[-1] if samples else start
        delta = end - start
        # We commanded motion toward +position, so a negative delta means the drive is inverted.
        moved = abs(delta) >= 10.0
        return {
            "motor": motor,
            "start": start,
            "end": end,
            "delta": delta,
            "commanded_pwm": probe_pwm_frac * pwm_max,
            "moved": moved,
            "inverted": moved and delta < 0,
        }

    def hold_current_position(
        self, k: float | dict[str, float] = 20.0, d: float | dict[str, float] = 1.0
    ) -> None:
        """Commands every motor to hold wherever it currently is -- a good first test that the
        impedance loop is stable (not drifting/oscillating) before commanding real motion."""
        self.move_to({}, k=k, d=d)

    def write_calibration(self, calibration: dict) -> None:
        """Pushes homing offsets and position limits into the servos' EPROM.

        This tool works in raw ticks and needs no calibration to *display* positions -- but the
        homing offset is not cosmetic. It is what stops a joint's travel from straddling the
        4095/0 encoder wrap, and without it `Present_Position` jumps a full turn mid-motion. Any
        closed-loop test then sees an instantaneous ~4095-tick error and slams the joint, so write
        the calibration before `--hold` or `--probe-direction`.

        `calibration` is the dict decoded from a LeRobot calibration JSON: motor name ->
        `{"homing_offset": int, "range_min": int, "range_max": int, ...}`.
        """
        for motor, calib in calibration.items():
            if motor not in MOTOR_NAMES:
                logger.warning("skipping unknown motor %r in calibration", motor)
                continue
            self.client.send_command_and_wait(
                CommandKind.SET_CALIBRATION,
                _motor_id(motor),
                [
                    float(calib["homing_offset"]),
                    float(calib["range_min"]),
                    float(calib["range_max"]),
                ],
                self.command_timeout_s,
            )
        logger.info("wrote calibration for %d motors", len(calibration))

    def set_pwm_mode(self) -> None:
        """Switches all 6 motors into `Operating_Mode = PWM`, matching
        `SO101ImpedanceFollower.configure()` -- required once before `move_to`/
        `hold_current_position` will have any effect."""
        for motor in MOTOR_NAMES:
            self.client.send_command_and_wait(
                CommandKind.SET_OPERATING_MODE,
                _motor_id(motor),
                [float(OperatingMode.PWM.value)],
                self.command_timeout_s,
            )

    def enable_torque(self) -> None:
        for motor in MOTOR_NAMES:
            self.client.send_command_and_wait(
                CommandKind.SET_TORQUE_ENABLE, _motor_id(motor), [1.0], self.command_timeout_s
            )

    def disable_torque(self) -> None:
        for motor in MOTOR_NAMES:
            self.client.send_command_and_wait(
                CommandKind.SET_TORQUE_ENABLE, _motor_id(motor), [0.0], self.command_timeout_s
            )

    def close(self, disable_torque: bool = True) -> None:
        try:
            if disable_torque:
                self.disable_torque()
        except ImpedanceShmClientError as e:
            logger.warning("failed to disable torque during close(): %s", e)
        finally:
            self.shutdown_daemon()
            self.client.close()

    def __enter__(self) -> "SO101ImpedanceChecker":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def format_state_table(
    state: dict[str, dict[str, float]],
    faults: list[str],
    targets: dict[str, float] | None = None,
    pwm_max: float = 1000.0,
) -> str:
    """Renders `read_state()`'s output as a fixed-width table for terminal printing.

    Passing `targets` adds the columns that actually diagnose a misbehaving impedance loop --
    position error and the PWM the daemon derived from it. Those two separate the failure modes
    that look identical from the outside: a joint that drifts away with **near-zero PWM** is simply
    too soft to hold itself (raise K), whereas one that drifts away with **saturated PWM** is being
    driven the wrong way (positive feedback -- flip `--invert-pwm`).
    """
    show_target = targets is not None
    header = f"{'motor':<14}{'pos':>9}{'vel':>9}{'cur_avg':>9}"
    if show_target:
        header += f"{'target':>9}{'err':>9}{'pwm':>9}{'%max':>7}"
    lines = [header, "-" * len(header)]
    for motor in MOTOR_NAMES:
        s = state[motor]
        row = f"{motor:<14}{s['present_pos']:>9.1f}{s['present_vel']:>9.1f}{s['present_current_avg']:>9.1f}"
        if show_target:
            target = targets.get(motor, float("nan"))
            err = target - s["present_pos"]
            pwm = s.get("pwm_cmd", float("nan"))
            pct = abs(pwm) / pwm_max * 100 if pwm_max else float("nan")
            row += f"{target:>9.1f}{err:>9.1f}{pwm:>9.1f}{pct:>6.0f}%"
        lines.append(row)
    if faults:
        lines.append("FAULTS: " + "; ".join(faults))
    return "\n".join(lines)
