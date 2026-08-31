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

"""Python client for the shared-memory protocol exposed by the `rust/so101_impedance_ctrl` RT
daemon.

The `ctypes.Structure` layout below must mirror `rust/so101_impedance_ctrl/src/shm.rs` field-for-
field (same types, same order). Both `ctypes.Structure` (default, no `_pack_`) and Rust's
`#[repr(C)]` (default, not `packed`) follow the target platform's standard C struct-layout ABI, so
as long as the field lists agree, the byte layout agrees too -- there's no need to hand-compute
padding. `LAYOUT_VERSION` is asserted on attach as the actual safety net against the two drifting
out of sync.
"""

import ctypes
import mmap
import os
import time

# Linux/glibc maps POSIX shared-memory objects to files here, which is what the Rust daemon's
# `shm_open` creates and what this client opens directly.
SHM_DIR = "/dev/shm"  # nosec B108

# All 6 servos -- the 5 arm joints AND the gripper -- are impedance-controlled. A rigid
# position-mode gripper crushes fragile objects before it can sense resistance; running it under
# the same compliant K/D law as the arm is what makes gentle grasping possible.
NUM_MOTORS = 6

# Pontine context channels: what the policy layer hands down to the cerebellum so it can tell two
# standing loads apart before either has pulled the arm out of position. Mirrors shm::NUM_CONTEXT.
#
# These carry an identity, not a mass -- see the Rust side for why the policy is not asked how
# heavy the object is. Two practical notes for whoever fills them in:
#   * Swing every channel to +/-1. Contexts are separated by the granule cells that draw a context
#     fibre, so what buys separation is how many fibres differ, and `[0, 0]` vs `[1, 0]` moves only
#     one of the 32.
#   * Interleave them while the cerebellum is learning. Most granule cells draw no context fibre,
#     so their weights are shared; converging on one context drags those weights with it. A policy
#     that picks things up and puts them down interleaves by itself.
NUM_CONTEXT = 2

LAYOUT_VERSION = 6
SHM_MAGIC = 0x534F3130  # ASCII "SO10", matches shm::SHM_MAGIC in shm.rs

FAULT_WATCHDOG_TIMEOUT = 1 << 0
FAULT_COMMS_ERROR = 1 << 1
FAULT_OVERCURRENT = 1 << 2
# The *leader* arm's bus failed, so force feedback is dropped. Deliberately distinct from
# FAULT_COMMS_ERROR: this one does not mean the robot stopped tracking.
FAULT_LEADER_COMMS_ERROR = 1 << 3
# At least one joint is outside the daemon's `--pos-min`/`--pos-max`. Worth its own bit because a
# joint the limits are holding at zero PWM looks exactly like the watchdog, a blind run, or a gain
# that is simply too soft.
FAULT_POS_LIMIT = 1 << 4

# `cerebellum_flags` bits, mirroring `cerebellum::mod`'s CEREBELLUM_* constants. They describe why
# the feedforward is what it is, which is otherwise unanswerable from the outside: a zero
# feedforward could mean "not learned yet", "gated off", "gone stale" or "the GPU died", and those
# call for very different responses.
CEREBELLUM_ACTIVE = 1 << 0  # a feedforward is being applied this tick
CEREBELLUM_LEARNING = 1 << 1  # at least one joint's climbing fibre passed the gates
CEREBELLUM_STALE = 1 << 2  # the cerebellum thread has stopped publishing; output discarded
CEREBELLUM_FAULTED = 1 << 3  # its backend failed unrecoverably; no feedforward for this run


class CommandKind:
    """Mirrors `shm::CommandKind` in shm.rs."""

    NONE = 0
    SET_OPERATING_MODE = 1
    SET_PID_COEFFICIENTS = 2
    SET_TORQUE_ENABLE = 3
    SET_CALIBRATION = 4
    SHUTDOWN = 5


class InputData(ctypes.Structure):
    _fields_ = [
        ("timestamp_mono_ns", ctypes.c_uint64),
        ("target_pos", ctypes.c_float * NUM_MOTORS),
        ("target_vel", ctypes.c_float * NUM_MOTORS),
        ("k_gain", ctypes.c_float * NUM_MOTORS),
        ("d_gain", ctypes.c_float * NUM_MOTORS),
        # All-zero means "no context", which is both the neutral value and what a caller that
        # never touches this field leaves behind -- so a policy that does not know about the
        # pontine channel degrades to the proprioception-only cerebellum rather than to garbage.
        ("context", ctypes.c_float * NUM_CONTEXT),
    ]


class InputRegion(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint32),
        ("data", InputData),
    ]


class OutputData(ctypes.Structure):
    _fields_ = [
        ("timestamp_mono_ns", ctypes.c_uint64),
        ("present_pos", ctypes.c_float * NUM_MOTORS),
        ("present_vel", ctypes.c_float * NUM_MOTORS),
        ("present_current_avg", ctypes.c_float * NUM_MOTORS),
        ("pwm_cmd_debug", ctypes.c_float * NUM_MOTORS),
        # The cerebellum's share of pwm_cmd_debug, already clamped, slew-limited and gated; zero
        # when no cerebellum is running. Separate from the total because the bring-up question is
        # how much of the holding duty the feedforward has taken over, and a sum cannot answer it.
        ("ff_pwm_debug", ctypes.c_float * NUM_MOTORS),
        ("cerebellum_flags", ctypes.c_uint32),
        ("fault_flags", ctypes.c_uint32),
        # Leader-side gripper, populated only when the daemon runs with `--leader-port`; zero
        # otherwise. Distinct from the follower arrays above -- this is the *operator's* trigger.
        ("leader_gripper_pos", ctypes.c_float),
        ("leader_gripper_vel", ctypes.c_float),
        ("leader_gripper_pwm", ctypes.c_float),
        # Supply voltage (0.1 V units) and case temperature (C) of the servo named by
        # health_motor_id, sampled round-robin once a second rather than per tick; all zero until
        # the first sample lands. Carried in the same snapshot as pwm_cmd_debug on purpose: a
        # servo whose power stage shorts pulls the shared rail down for hundreds of ms, which
        # inflates the duty every other joint needs to hold a pose, and a holding-duty number
        # taken without a concurrent rail reading cannot be told apart from a stiff mechanism.
        ("supply_decivolts", ctypes.c_uint32),
        ("case_temp_c", ctypes.c_uint32),
        ("health_motor_id", ctypes.c_uint32),
    ]


class OutputRegion(ctypes.Structure):
    _fields_ = [
        ("seq", ctypes.c_uint32),
        ("data", OutputData),
    ]


class CommandRegion(ctypes.Structure):
    _fields_ = [
        ("cmd_seq", ctypes.c_uint32),
        ("ack_seq", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
        ("cmd_kind", ctypes.c_uint32),
        ("motor_id", ctypes.c_uint32),
        ("payload", ctypes.c_float * 4),
    ]


class ShmLayout(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32),
        ("layout_version", ctypes.c_uint32),
        ("input", InputRegion),
        ("output", OutputRegion),
        ("command", CommandRegion),
    ]


class ImpedanceShmClientError(RuntimeError):
    """Raised when the shared-memory segment can't be attached to, or looks stale/incompatible."""


class ImpedanceShmClient:
    """Attaches to the shared-memory segment created by the `rust/so101_impedance_ctrl` RT daemon
    and provides seqlock-safe reads/writes plus the low-rate command/ack channel.

    The daemon must already be running -- it creates the segment; this client only ever attaches,
    retrying for `attach_timeout_s` in case Python starts slightly before the daemon.

    The segment is opened as a plain file under `/dev/shm` and `mmap`ed directly, deliberately
    *not* through `multiprocessing.shared_memory.SharedMemory`. Before Python 3.13 that class
    registers every segment with the `resource_tracker` even when merely attaching to someone
    else's (`create=False`), and the tracker then **unlinks it at interpreter exit** -- so simply
    running and quitting a Python tool would delete the running daemon's segment out from under it
    (visible as the `resource_tracker: There appear to be N leaked shared_memory objects`
    warning). Opening the file ourselves keeps ownership entirely with the daemon: nothing here
    ever unlinks.
    """

    def __init__(self, shm_name: str, attach_timeout_s: float = 5.0):
        self.shm_name = shm_name
        self._mmap: mmap.mmap | None = None
        self._layout: ShmLayout | None = None
        self._last_output: dict | None = None

        path = os.path.join(SHM_DIR, shm_name)
        deadline = time.monotonic() + attach_timeout_s
        last_error: Exception | None = None
        fd: int | None = None
        while time.monotonic() < deadline:
            try:
                fd = os.open(path, os.O_RDWR)
                break
            except FileNotFoundError as e:
                last_error = e
                time.sleep(0.1)
        if fd is None:
            raise ImpedanceShmClientError(
                f"Could not attach to shared memory segment '{shm_name}' ({path}) within "
                f"{attach_timeout_s}s. Is the `so101_impedance_ctrl` Rust daemon running with "
                f"--shm-name {shm_name}?"
            ) from last_error

        try:
            size = os.fstat(fd).st_size
            expected_size = ctypes.sizeof(ShmLayout)
            if size < expected_size:
                raise ImpedanceShmClientError(
                    f"Shared memory segment '{shm_name}' is {size} bytes, expected at least "
                    f"{expected_size}. This usually means the Python `ShmLayout` mirror "
                    f"(shm_client.py) and the Rust daemon's `ShmLayout` (shm.rs) have drifted out "
                    f"of sync -- rebuild/restart both together."
                )
            self._mmap = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            # `mmap` keeps the mapping alive independently of the descriptor.
            os.close(fd)

        layout = ShmLayout.from_buffer(self._mmap)
        magic, layout_version = layout.magic, layout.layout_version  # read before releasing `layout`
        if magic != SHM_MAGIC:
            del layout  # drop the exported buffer reference before closing the mmap below
            self._close_shm()
            raise ImpedanceShmClientError(
                f"Shared memory segment '{shm_name}' has bad magic "
                f"0x{magic:08x} (expected 0x{SHM_MAGIC:08x})."
            )
        if layout_version != LAYOUT_VERSION:
            del layout
            self._close_shm()
            raise ImpedanceShmClientError(
                f"Shared memory segment '{shm_name}' has layout_version "
                f"{layout_version}, but this client expects {LAYOUT_VERSION}. "
                f"Rebuild/restart the Rust daemon and this Python package together."
            )
        self._layout = layout

    def _close_shm(self) -> None:
        # `ShmLayout.from_buffer(...)` holds an exported buffer reference on the mapping; that
        # reference must be dropped (by clearing `self._layout`, its only owner) before
        # `mmap.close()` will succeed, or it raises `BufferError: cannot close exported pointers`.
        self._layout = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None

    def close(self) -> None:
        """Detaches from the segment. Never unlinks it -- the daemon owns its lifetime."""
        self._close_shm()

    @staticmethod
    def _monotonic_ns() -> int:
        # Same clock domain (CLOCK_MONOTONIC) as the Rust daemon's `monotonic_ns()`, so
        # timestamps written by either side are directly comparable.
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    def write_input(
        self,
        target_pos: list[float],
        k_gain: list[float],
        d_gain: list[float],
        target_vel: list[float] | None = None,
        context: list[float] | None = None,
    ) -> None:
        """Seqlock write of the input region. `target_pos`/`k_gain`/`d_gain`/`target_vel` must
        have length `NUM_MOTORS`, ordered per the robot config's `impedance_joints` (the 5 arm
        joints followed by the gripper).

        `context` is the pontine channel (length `NUM_CONTEXT`); see its comment above for the two
        things that decide whether it does anything. Omitting it writes zeros, which is the
        no-context case."""
        if target_vel is None:
            target_vel = [0.0] * NUM_MOTORS
        if context is None:
            context = [0.0] * NUM_CONTEXT

        region = self._layout.input
        seq = region.seq
        region.seq = seq + 1  # odd: writer in progress
        region.data.timestamp_mono_ns = self._monotonic_ns()
        region.data.target_pos[:] = target_pos
        region.data.target_vel[:] = target_vel
        region.data.k_gain[:] = k_gain
        region.data.d_gain[:] = d_gain
        region.data.context[:] = context
        region.seq = seq + 2  # back to even: stable

    def read_output(self, max_retries: int = 8) -> dict:
        """Seqlock read of the output region, returning a plain dict snapshot.

        On retry exhaustion (writer never observed stable within `max_retries`, under heavy
        contention) this returns the last known-good snapshot rather than ever handing back a
        possibly-torn read -- an earlier version returned the last (unvalidated) in-progress
        attempt here, which the equivalent Rust-side stress test
        (`rust/so101_impedance_ctrl/tests/shm_layout_tests.rs`) caught actually leaking torn
        data under contention. If no stable read has ever succeeded yet, there is no known-good
        value to fall back to, so this raises instead of returning fabricated/torn data.
        """
        region = self._layout.output
        for _ in range(max_retries):
            s1 = region.seq
            if s1 % 2 != 0:
                continue  # writer in progress, retry
            snapshot = {
                "timestamp_mono_ns": region.data.timestamp_mono_ns,
                "present_pos": list(region.data.present_pos),
                "present_vel": list(region.data.present_vel),
                "present_current_avg": list(region.data.present_current_avg),
                "pwm_cmd": list(region.data.pwm_cmd_debug),
                "ff_pwm": list(region.data.ff_pwm_debug),
                "cerebellum_flags": region.data.cerebellum_flags,
                "fault_flags": region.data.fault_flags,
                "leader_gripper_pos": region.data.leader_gripper_pos,
                "leader_gripper_vel": region.data.leader_gripper_vel,
                "leader_gripper_pwm": region.data.leader_gripper_pwm,
                "supply_decivolts": region.data.supply_decivolts,
                "case_temp_c": region.data.case_temp_c,
                "health_motor_id": region.data.health_motor_id,
            }
            s2 = region.seq
            if s1 == s2:
                self._last_output = snapshot
                return snapshot
        if self._last_output is not None:
            return self._last_output
        raise ImpedanceShmClientError(
            f"Could not obtain a stable read of the output region within {max_retries} retries, "
            "and no prior known-good snapshot exists yet. Is the so101_impedance_ctrl daemon "
            "actually running and writing telemetry?"
        )

    def is_output_fresh(self, max_staleness_s: float = 0.5) -> bool:
        snapshot = self.read_output()
        age_s = (self._monotonic_ns() - snapshot["timestamp_mono_ns"]) / 1e9
        return age_s <= max_staleness_s

    def send_command(self, kind: int, motor_id: int, payload: list[float]) -> int:
        """Submits a command over the low-rate command channel without waiting for the ack.
        Returns the `cmd_seq` used, for passing to `wait_for_ack`."""
        cmd = self._layout.command
        new_seq = (cmd.cmd_seq + 1) & 0xFFFFFFFF
        cmd.cmd_kind = kind
        cmd.motor_id = motor_id
        padded_payload = (list(payload) + [0.0, 0.0, 0.0, 0.0])[:4]
        cmd.payload[:] = padded_payload
        cmd.cmd_seq = new_seq
        return new_seq

    def wait_for_ack(self, cmd_seq: int, timeout_s: float = 2.0) -> int:
        """Blocks until the daemon acks `cmd_seq`, returning its status code (0 = ok)."""
        cmd = self._layout.command
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if cmd.ack_seq == cmd_seq:
                return cmd.status
            time.sleep(0.005)
        raise ImpedanceShmClientError(
            f"Timed out after {timeout_s}s waiting for the impedance daemon to ack command "
            f"seq={cmd_seq} (last ack_seq={cmd.ack_seq})."
        )

    def send_command_and_wait(
        self, kind: int, motor_id: int, payload: list[float], timeout_s: float = 2.0
    ) -> int:
        """Submits a command and blocks until it is acked, raising if the daemon reported failure.

        Raising on a nonzero status matters: a silently-dropped `SetOperatingMode` leaves the servo
        in POSITION mode, and the daemon's PWM commands then land in `Goal_Time` (travel time)
        instead of the duty cycle -- the arm drives to a stale `Goal_Position` and holds rigidly,
        which looks like a control-law problem rather than a failed register write.
        """
        seq = self.send_command(kind, motor_id, payload)
        status = self.wait_for_ack(seq, timeout_s)
        if status != 0:
            raise ImpedanceShmClientError(
                f"Impedance daemon reported failure (status={status}) for command kind={kind} "
                f"motor_id={motor_id} payload={payload}. Check the daemon's log for the underlying "
                f"serial/register error."
            )
        return status
