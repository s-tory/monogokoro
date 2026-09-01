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
from typing import TYPE_CHECKING

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import MotorNormMode
from lerobot.motors.feetech import OperatingMode
from lerobot.motors.feetech.tables import MODEL_RESOLUTION
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_so101_impedance_follower import SO101ImpedanceFollowerRobotConfig
from .shm_client import NUM_CONTEXT, CommandKind, ImpedanceShmClient, ImpedanceShmClientError

if TYPE_CHECKING:
    from lerobot.processor import ProcessorStep

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
    matching this repo's existing `MotorCurrentProcessorStep` convention. The telemetry columns are
    likewise unconverted: `.pwm_cmd`/`.ff_pwm` are in the Rust loop's duty units, and
    `supply_decivolts` is in 0.1 V steps, as the name says.
    """

    config_class = SO101ImpedanceFollowerRobotConfig
    name = "so101_follower_impedance"

    def __init__(self, config: SO101ImpedanceFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self._shm_client: ImpedanceShmClient | None = None
        # Checked here rather than at the shared-memory write: a mismatched length would otherwise
        # surface as a ctypes assignment error mid-episode, long after the run was configured.
        if len(config.pontine_context) != NUM_CONTEXT:
            raise ValueError(
                f"pontine_context must have {NUM_CONTEXT} channels (the daemon's NUM_CONTEXT), "
                f"got {len(config.pontine_context)}: {config.pontine_context}"
            )

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
    def _telemetry_ft(self) -> dict[str, type]:
        """The daemon's own view of the loop, recorded because none of it can be recovered later.

        `pwm_cmd - ff_pwm` is the feedback share of the duty, which is the quantity the climbing
        fibre is derived from. Keeping both columns lets a frame's salience be judged after the
        fact -- which frames the cerebellum could not predict -- instead of that judgement having
        to be right before the first episode is recorded.

        The rail voltage travels with the current for the reason `shm.rs` gives for publishing it:
        a current taken without a concurrent rail reading cannot be told apart from a stiff
        mechanism. The rail is shared, so the reading means the same thing whichever servo the
        round-robin health poll happened to take it from, and it needs no companion column.

        `case_temp_c` is deliberately *not* here, for the opposite reason. It belongs to the one
        servo named by `health_motor_id`, so recording it truthfully means recording that id too --
        and that id is a bare round-robin counter, which is exactly the kind of periodic column a
        policy can learn a spurious correlation from. Nothing asks a temperature question today.
        When something does, the column to add is a per-joint latch that is interpretable on its
        own, not a value plus an index explaining who it belongs to.
        """
        ft: dict[str, type] = {}
        for motor in self.impedance_joints:
            ft[f"{motor}.pwm_cmd"] = float
            ft[f"{motor}.ff_pwm"] = float
        ft["supply_decivolts"] = float
        ft["cerebellum_flags"] = float
        return ft

    @property
    def _gains_ft(self) -> dict[str, type]:
        gains_ft: dict[str, type] = {}
        for motor in self.impedance_joints:
            gains_ft[f"{motor}.k"] = float
            gains_ft[f"{motor}.d"] = float
        return gains_ft

    @property
    def _context_ft(self) -> dict[str, type]:
        # Action columns, not observation: the context is declared downward by the policy layer,
        # never sensed. Recording it as an action is what lets an imitation policy learn to emit
        # it -- see `PontineContextProcessorStep`.
        return {f"context.{i}": float for i in range(NUM_CONTEXT)}

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
        return {**self._motors_ft, **self._current_ft, **self._telemetry_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._motors_ft, **self._gains_ft, **self._context_ft}

    def teleop_action_processor_steps(self) -> list["ProcessorStep"]:
        """Extra teleop-pipeline steps this robot needs when driven by a plain position teleoperator.

        `lerobot-record` calls this (duck-typed, so it stays robot-agnostic) and prepends the
        result to the teleop action pipeline. That placement is the whole point: `record_loop()`
        writes the teleop pipeline's *output* to the dataset, not what the robot actually receives,
        so K/D filled in by `send_action` would drive the arm correctly but be absent from the
        recorded action columns -- leaving the dataset's `.k`/`.d` features declared but never
        populated.

        The gains come from this robot's own config, so a dataset is always labeled with the gains
        that were actually applied while it was demonstrated.
        """
        from lerobot.processor import ImpedanceGainDefaultsProcessorStep, PontineContextProcessorStep

        return [
            ImpedanceGainDefaultsProcessorStep(
                impedance_joints=self.impedance_joints,
                default_k=tuple(self.config.default_k),
                default_d=tuple(self.config.default_d),
            ),
            # Same reasoning as the gains, for the same reason: a leader arm cannot supply a
            # context, and one filled in by `send_action` would reach the arm but never the
            # dataset -- leaving the `context.<i>` action columns declared but never populated.
            PontineContextProcessorStep(
                context=tuple(self.config.pontine_context),
                cycle=tuple(tuple(entry) for entry in self.config.pontine_context_cycle),
            ),
        ]

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
            obs_dict[f"{motor}.pwm_cmd"] = telemetry["pwm_cmd"][i]
            obs_dict[f"{motor}.ff_pwm"] = telemetry["ff_pwm"][i]

        obs_dict["supply_decivolts"] = float(telemetry["supply_decivolts"])
        obs_dict["cerebellum_flags"] = float(telemetry["cerebellum_flags"])

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

        The pontine context channels (`context.<i>`) follow the same rule: clamped to `[-1, 1]`,
        defaulted from `config.pontine_context`, with `PontineContextProcessorStep` as the
        recording-time source of truth.

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

        # The pontine context, defaulted from config exactly as K/D are. A plain position
        # teleoperator never supplies it; a policy that has learned to emit it does, and then its
        # value wins. Clamped for the same defense-in-depth reason as K/D: the daemon documents
        # `[-1, 1]` as the range the granule code was sized for, and a policy is free to overshoot.
        context: list[float] = []
        for i in range(NUM_CONTEXT):
            c = float(action.get(f"context.{i}", self.config.pontine_context[i]))
            c = min(max(c, -1.0), 1.0)
            context.append(c)
            sent[f"context.{i}"] = c

        shm.write_input(
            target_pos=target_pos,
            k_gain=k_gain,
            d_gain=d_gain,
            target_vel=target_vel,
            context=context,
        )
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
