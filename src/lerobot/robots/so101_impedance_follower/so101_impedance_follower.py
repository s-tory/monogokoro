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

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import MotorNormMode
from lerobot.motors.feetech import OperatingMode
from lerobot.motors.feetech.tables import MODEL_RESOLUTION
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_impedance_follower import SO101ImpedanceFollowerRobotConfig
from .shm_client import CommandKind, ImpedanceShmClient, ImpedanceShmClientError

logger = logging.getLogger(__name__)

GRIPPER_MOTOR = "gripper"

# Matches the Motor IDs assigned in src/lerobot/robots/so_follower/so_follower.py, which the
# rust/so101_impedance_ctrl daemon also hard-codes -- keep all three in sync.
_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}

_STS3215_RESOLUTION = MODEL_RESOLUTION["sts3215"]


class SO101ImpedanceFollower(Robot):
    """
    SO101 follower whose 5 arm joints AND gripper are all driven by an impedance controller
    (`K * delta_pos + D * delta_vel`) running in a separate `rust/so101_impedance_ctrl` RT process
    pinned to a PREEMPT_RT-isolated CPU core. The gripper is impedance-controlled too, not left in
    plain position mode: a rigidly position-controlled gripper keeps commanding full force toward
    its target regardless of contact, crushing fragile objects before it can ever "feel" them --
    running it under the same compliant K/D law as the arm (with a softer default K, see
    `config.default_k`) is what makes gentle, adaptive grasping possible at all.

    Unlike `SOFollower`, this class never opens the SO101's serial port itself: the Rust daemon
    exclusively owns the bus (all 6 servos) to avoid two processes racing on one half-duplex UART.
    This robot only talks to the daemon through shared memory -- start the daemon first (see
    `rust/so101_impedance_ctrl/README.md`) before calling `connect()`.

    Units: `.pos` values in observations/actions are in this robot's normalized units (degrees if
    `config.use_degrees`, else range -100..100; range 0..100 for the gripper), matching plain
    `SOFollower` datasets. `.k`/`.d` gains, however, operate on the *raw encoder tick* position
    error computed inside the Rust control loop -- Rust never applies this class's degree/range
    calibration math, only Python does, at the `.pos` <-> raw-tick boundary in `get_observation`/
    `send_action`. `.current_avg` values are raw `Present_Current` register units (unconverted),
    matching this repo's existing `MotorCurrentProcessorStep` convention.
    """

    config_class = SO101ImpedanceFollowerRobotConfig
    name = "so101_follower_impedance"

    def __init__(self, config: SO101ImpedanceFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._shm_client: ImpedanceShmClient | None = None

    @property
    def impedance_joints(self) -> tuple[str, ...]:
        """All 6 impedance-controlled motors (5 arm joints + gripper), in shared-memory order."""
        return tuple(self.config.impedance_joints)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.impedance_joints}

    @property
    def _current_ft(self) -> dict[str, type]:
        return {f"{motor}.current_avg": float for motor in self.impedance_joints}

    @property
    def _gains_ft(self) -> dict[str, type]:
        gains_ft: dict[str, type] = {}
        for motor in self.impedance_joints:
            gains_ft[f"{motor}.k"] = float
            gains_ft[f"{motor}.d"] = float
        return gains_ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            if getattr(self.cameras[cam], "use_rgb", True):
                features[cam] = (self.cameras[cam].height, self.cameras[cam].width, 3)
            if getattr(self.cameras[cam], "use_depth", False):
                features[f"{cam}_depth"] = (self.cameras[cam].height, self.cameras[cam].width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._current_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._motors_ft, **self._gains_ft}

    @property
    def is_connected(self) -> bool:
        return self._shm_client is not None and all(cam.is_connected for cam in self.cameras.values())

    def _shm(self) -> ImpedanceShmClient:
        if self._shm_client is None:
            raise ConnectionError(f"{self} is not connected.")
        return self._shm_client

    @staticmethod
    def _motor_id(name: str) -> int:
        try:
            return _MOTOR_IDS[name]
        except KeyError as e:
            raise ValueError(
                f"Unknown motor name {name!r}; expected one of {list(_MOTOR_IDS)} (the Rust "
                "daemon and this robot must agree on Feetech motor IDs)."
            ) from e

    def _norm_mode(self, motor: str) -> MotorNormMode:
        if motor == GRIPPER_MOTOR:
            return MotorNormMode.RANGE_0_100
        return MotorNormMode.DEGREES if self.config.use_degrees else MotorNormMode.RANGE_M100_100

    def _raw_to_normalized(self, motor: str, raw: float) -> float:
        """Converts a raw encoder-tick position (as read by the Rust daemon) to this robot's
        normalized `.pos` units, mirroring `SerialMotorsBus._normalize`."""
        calib = self.calibration[motor]
        min_, max_ = calib.range_min, calib.range_max
        if max_ == min_:
            raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")
        bounded = min(max_, max(min_, raw))
        mode = self._norm_mode(motor)
        if mode is MotorNormMode.RANGE_M100_100:
            norm = (((bounded - min_) / (max_ - min_)) * 200) - 100
            return -norm if calib.drive_mode else norm
        if mode is MotorNormMode.RANGE_0_100:
            norm = ((bounded - min_) / (max_ - min_)) * 100
            return 100 - norm if calib.drive_mode else norm
        mid = (min_ + max_) / 2
        max_res = _STS3215_RESOLUTION - 1
        return (raw - mid) * 360 / max_res

    def _normalized_to_raw(self, motor: str, value: float) -> float:
        """Inverse of `_raw_to_normalized`, mirroring `SerialMotorsBus._unnormalize`."""
        calib = self.calibration[motor]
        min_, max_ = calib.range_min, calib.range_max
        if max_ == min_:
            raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")
        mode = self._norm_mode(motor)
        if mode is MotorNormMode.RANGE_M100_100:
            value = -value if calib.drive_mode else value
            bounded = min(100.0, max(-100.0, value))
            return ((bounded + 100) / 200) * (max_ - min_) + min_
        if mode is MotorNormMode.RANGE_0_100:
            value = 100 - value if calib.drive_mode else value
            bounded = min(100.0, max(0.0, value))
            return (bounded / 100) * (max_ - min_) + min_
        mid = (min_ + max_) / 2
        max_res = _STS3215_RESOLUTION - 1
        return (value * max_res / 360) + mid

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        try:
            self._shm_client = ImpedanceShmClient(
                self.config.shm_name, attach_timeout_s=self.config.shm_attach_timeout_s
            )
        except ImpedanceShmClientError as e:
            raise ConnectionError(
                f"{self} could not attach to the impedance daemon's shared memory: {e}"
            ) from e

        if not self.is_calibrated and calibrate:
            logger.info(
                "No calibration found for %s -- an impedance-controlled SO101 cannot calibrate "
                "itself interactively yet, see `calibrate()`.",
                self,
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return bool(self.calibration) and all(motor in self.calibration for motor in self.impedance_joints)

    def calibrate(self) -> None:
        # Interactive calibration (homing offsets, moving through the full range of motion while
        # torque is disabled, per `SOFollower.calibrate()`) is out of scope for v1: it requires a
        # live, low-latency read-while-moving loop that the low-rate command channel isn't
        # designed for, and it's identical hardware/math to plain `SOFollower`'s calibration.
        #
        # Instead: run calibration once using the plain `so101_follower` robot type against these
        # same servos (with this Rust daemon stopped, so the two don't race for the serial port),
        # then either point `config.calibration_dir` at that calibration's directory, or copy its
        # `<id>.json` into this robot's own calibration directory (`self.calibration_dir`).
        # `SOFollower.name` is "so_follower", so plain-follower calibrations land in a *different*
        # directory than this robot's -- spell out the exact copy so this isn't a scavenger hunt.
        plain_dir = HF_LEROBOT_CALIBRATION / ROBOTS / "so_follower"
        raise NotImplementedError(
            f"{self} has no calibration for id={self.id!r} in {self.calibration_dir}.\n"
            "Interactive calibration isn't supported on this robot type (see the comment in "
            "calibrate()). Calibrate once with the plain `so101_follower` type instead -- stop the "
            "so101_impedance_ctrl daemon first, or the two will fight over the serial port:\n\n"
            f"  lerobot-calibrate --robot.type=so101_follower --robot.port=<PORT> --robot.id={self.id}\n\n"
            "then copy the result into this robot's calibration directory:\n\n"
            f"  mkdir -p {self.calibration_dir}\n"
            f"  cp {plain_dir / f'{self.id}.json'} {self.calibration_fpath}\n\n"
            f"(or point --robot.calibration_dir at {plain_dir} instead of copying)."
        )

    def configure(self) -> None:
        shm = self._shm()
        timeout = self.config.command_ack_timeout_s

        for motor in self.impedance_joints:
            motor_id = self._motor_id(motor)

            # Push the calibration into the servo *before* anything reads a position from it, the
            # same thing `FeetechMotorsBus.write_calibration` does for the stock follower. The
            # homing offset is what keeps a joint's travel from straddling the 4095/0 encoder wrap;
            # without it `Present_Position` jumps a full turn mid-motion, the impedance law reads
            # that as an instantaneous ~4095-tick error, saturates PWM and slams the joint. That
            # runaway is immune to `--invert-pwm`, because the discontinuity is in the measurement
            # rather than the drive direction.
            calib = self.calibration[motor]
            shm.send_command_and_wait(
                CommandKind.SET_CALIBRATION,
                motor_id,
                [float(calib.homing_offset), float(calib.range_min), float(calib.range_max)],
                timeout,
            )

            # All 6 motors -- including the gripper -- run in PWM mode under the impedance law.
            shm.send_command_and_wait(
                CommandKind.SET_OPERATING_MODE,
                motor_id,
                [float(OperatingMode.PWM.value)],
                timeout,
            )
            shm.send_command_and_wait(CommandKind.SET_TORQUE_ENABLE, motor_id, [1.0], timeout)

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        shm = self._shm()
        start = time.perf_counter()
        telemetry = shm.read_output()
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        if not shm.is_output_fresh():
            logger.warning(
                f"{self} impedance daemon telemetry looks stale -- is so101_impedance_ctrl still running?"
            )

        obs_dict: RobotObservation = {}
        for i, motor in enumerate(self.impedance_joints):
            obs_dict[f"{motor}.pos"] = self._raw_to_normalized(motor, telemetry["present_pos"][i])
            obs_dict[f"{motor}.current_avg"] = telemetry["present_current_avg"][i]

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command all 6 motors (arm joints + gripper) toward a target configuration under
        impedance control.

        The relative position magnitude may be clipped (`config.max_relative_target`); K/D are
        clamped to `[k_min, k_max]`/`[d_min, d_max]`; K/D default to `config.default_k`/
        `default_d` per motor when absent from `action` (defense-in-depth -- the primary source
        of truth for recorded K/D during teleop is `ImpedanceGainDefaultsProcessorStep`, injected
        upstream in the teleop action pipeline, not here).

        Returns the action actually sent (post-clipping/defaulting), matching `SOFollower`'s
        contract.
        """
        shm = self._shm()

        missing = [f"{m}.pos" for m in self.impedance_joints if f"{m}.pos" not in action]
        if missing:
            raise ValueError(f"{self} action is missing required keys: {missing}")

        goal_pos = {m: action[f"{m}.pos"] for m in self.impedance_joints}
        if self.config.max_relative_target is not None:
            telemetry = shm.read_output()
            present_by_name = {
                m: self._raw_to_normalized(m, telemetry["present_pos"][i])
                for i, m in enumerate(self.impedance_joints)
            }
            goal_present_pos = {m: (v, present_by_name[m]) for m, v in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        target_pos: list[float] = []
        target_vel: list[float] = []
        k_gain: list[float] = []
        d_gain: list[float] = []
        sent: RobotAction = {}
        for i, motor in enumerate(self.impedance_joints):
            pos = goal_pos[motor]
            k = float(action.get(f"{motor}.k", self.config.default_k[i]))
            d = float(action.get(f"{motor}.d", self.config.default_d[i]))
            k = min(max(k, self.config.k_min), self.config.k_max)
            d = min(max(d, self.config.d_min), self.config.d_max)

            target_pos.append(self._normalized_to_raw(motor, pos))
            target_vel.append(float(action.get(f"{motor}.vel", 0.0)))
            k_gain.append(k)
            d_gain.append(d)
            sent[f"{motor}.pos"] = pos
            sent[f"{motor}.k"] = k
            sent[f"{motor}.d"] = d

        shm.write_input(target_pos=target_pos, k_gain=k_gain, d_gain=d_gain, target_vel=target_vel)
        return sent

    @check_if_not_connected
    def disconnect(self):
        if self.config.disable_torque_on_disconnect:
            for motor in self.impedance_joints:
                try:
                    self._shm().send_command_and_wait(
                        CommandKind.SET_TORQUE_ENABLE, self._motor_id(motor), [0.0]
                    )
                except ImpedanceShmClientError as e:
                    logger.warning(f"{self} failed to disable torque on {motor} during disconnect: {e}")

        if self._shm_client is not None:
            self._shm_client.close()
            self._shm_client = None

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
