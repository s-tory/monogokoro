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
Tests for the Python side of the shared-memory protocol
(`src/lerobot/robots/so101_impedance_follower/shm_client.py`).

These exercise `ImpedanceShmClient` against a real POSIX shared-memory segment created in-process
(simulating what the `rust/so101_impedance_ctrl` daemon would create), so no Rust binary, serial
port, or RT scheduler is needed. They do NOT verify that the Rust `#[repr(C)]` struct and this
Python `ctypes.Structure` mirror agree byte-for-byte across an actual compiled Rust binary --
that requires building the Rust crate (see rust/so101_impedance_ctrl/README.md) and is out of
scope for this pure-Python test suite.
"""

import ctypes
import gc
import time
from multiprocessing import shared_memory

import pytest

from lerobot.robots.so101_impedance_follower.shm_client import (
    FAULT_COMMS_ERROR,
    FAULT_LEADER_COMMS_ERROR,
    FAULT_WATCHDOG_TIMEOUT,
    LAYOUT_VERSION,
    NUM_CONTEXT,
    SHM_MAGIC,
    CommandKind,
    ImpedanceShmClient,
    ImpedanceShmClientError,
    ShmLayout,
)


@pytest.fixture
def shm_segment():
    """A shared-memory segment initialized the way the Rust daemon's `init_in_place()` would."""
    shm = shared_memory.SharedMemory(create=True, size=ctypes.sizeof(ShmLayout))
    layout = ShmLayout.from_buffer(shm.buf)
    layout.magic = SHM_MAGIC
    layout.layout_version = LAYOUT_VERSION
    layout.input.seq = 0
    layout.output.seq = 0
    layout.command.cmd_seq = 0
    layout.command.ack_seq = 0
    layout.command.status = 0
    del layout  # drop the exported buffer reference before the mmap is closed below
    yield shm
    shm.close()
    shm.unlink()


def _acking_send(client, shm, status: int):
    """Wraps `client.send_command` so the daemon's ack lands immediately with `status`.

    Each call builds and drops its own `ShmLayout` view -- keeping one alive across the test would
    block the fixture's `SharedMemory.close()` with `BufferError: cannot close exported pointers`.
    """
    original_send = client.send_command

    def send_then_ack(*args, **kwargs):
        seq = original_send(*args, **kwargs)
        layout = ShmLayout.from_buffer(shm.buf)
        layout.command.status = status
        layout.command.ack_seq = layout.command.cmd_seq
        del layout
        return seq

    return send_then_ack


def test_shm_layout_has_stable_nonzero_size():
    assert ctypes.sizeof(ShmLayout) > 0


def test_attach_succeeds_against_a_valid_segment(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    client.close()


def test_attach_fails_fast_on_missing_segment():
    with pytest.raises(ImpedanceShmClientError, match="Could not attach"):
        ImpedanceShmClient("definitely_does_not_exist_shm_so101", attach_timeout_s=0.2)


def test_attach_rejects_bad_magic(shm_segment):
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.magic = 0xDEADBEEF
    with pytest.raises(ImpedanceShmClientError, match="bad magic"):
        ImpedanceShmClient(shm_segment.name)


def test_attach_rejects_mismatched_layout_version(shm_segment):
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.layout_version = LAYOUT_VERSION + 1
    with pytest.raises(ImpedanceShmClientError, match="layout_version"):
        ImpedanceShmClient(shm_segment.name)


def test_write_input_is_visible_through_a_second_raw_view(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    # All 6 motors (5 arm joints + gripper) share one impedance-controlled input now.
    target_pos = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    k_gain = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    d_gain = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    client.write_input(target_pos=target_pos, k_gain=k_gain, d_gain=d_gain)

    # Simulates the Rust daemon reading the same segment through its own view.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    assert layout.input.seq % 2 == 0  # left in a stable (even) state
    assert list(layout.input.data.target_pos) == pytest.approx(target_pos)
    assert list(layout.input.data.k_gain) == pytest.approx(k_gain)
    assert list(layout.input.data.d_gain) == pytest.approx(d_gain)
    client.close()


def test_write_input_relays_the_pontine_context(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    args = {
        "target_pos": [0.0] * 6,
        "k_gain": [10.0] * 6,
        "d_gain": [1.0] * 6,
    }

    # Omitted, the context is zeros -- the no-context case, and what a caller written before this
    # field existed leaves behind. That default is what lets an older policy keep driving the arm
    # against the proprioception-only cerebellum instead of against an undefined one.
    client.write_input(**args)
    layout = ShmLayout.from_buffer(shm_segment.buf)
    assert list(layout.input.data.context) == [0.0] * NUM_CONTEXT

    # Supplied, it arrives verbatim. Not squashed and not rescaled: the daemon clamps to [-1, 1]
    # and the granule layer reads what is left, so a value that changed shape in transit would
    # move which cells fire without anything in the loop saying so.
    client.write_input(**args, context=[-1.0, 1.0])
    layout = ShmLayout.from_buffer(shm_segment.buf)
    assert layout.input.seq % 2 == 0
    assert list(layout.input.data.context) == pytest.approx([-1.0, 1.0])
    client.close()


def test_shm_layout_size_matches_the_rust_struct():
    # The mirror in shm_client.py is maintained by hand against shm.rs, and LAYOUT_VERSION only
    # catches someone who remembered to bump it. The Rust side asserts this same number in
    # tests/shm_layout_tests.rs; asserting it here too is what makes a one-sided edit fail on the
    # side that made it rather than on the next person's arm.
    assert ctypes.sizeof(ShmLayout) == 336


def test_read_output_round_trips_values_from_a_simulated_daemon(shm_segment):
    # Simulates the Rust daemon writing telemetry.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.present_pos[:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    layout.output.data.present_current_avg[:] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    layout.output.data.timestamp_mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    layout.output.seq = 2  # stable/even

    client = ImpedanceShmClient(shm_segment.name)
    snapshot = client.read_output()

    assert snapshot["present_pos"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert snapshot["present_current_avg"] == pytest.approx([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    assert client.is_output_fresh(max_staleness_s=5.0)
    client.close()


def test_read_output_raises_if_never_stable_and_no_prior_snapshot(shm_segment):
    # seq stuck odd == the daemon is (simulated to be) perpetually mid-write; with no prior
    # successful read to fall back to, read_output must not fabricate/return torn data.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.seq = 1

    client = ImpedanceShmClient(shm_segment.name)
    with pytest.raises(ImpedanceShmClientError, match="Could not obtain a stable read"):
        client.read_output(max_retries=4)
    client.close()


def test_read_output_falls_back_to_last_known_good_snapshot_on_exhaustion(shm_segment):
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.present_pos[:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    layout.output.seq = 2  # stable/even

    client = ImpedanceShmClient(shm_segment.name)
    first = client.read_output()
    assert first["present_pos"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    # Now simulate the daemon getting stuck mid-write (perpetually odd seq); read_output must
    # return the cached last-known-good snapshot rather than block forever or return torn data.
    layout.output.seq = 1
    layout.output.data.present_pos[:] = [99.0] * 6  # would be torn/in-progress if ever read

    second = client.read_output(max_retries=4)
    assert second["present_pos"] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    client.close()


def test_is_output_fresh_detects_stale_telemetry(shm_segment):
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.timestamp_mono_ns = 0  # arbitrarily far in the past
    layout.output.seq = 2

    client = ImpedanceShmClient(shm_segment.name)
    assert not client.is_output_fresh(max_staleness_s=0.5)
    client.close()


def test_send_command_and_ack_round_trip(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    seq = client.send_command(CommandKind.SET_TORQUE_ENABLE, motor_id=1, payload=[1.0])

    layout = ShmLayout.from_buffer(shm_segment.buf)
    assert layout.command.cmd_kind == CommandKind.SET_TORQUE_ENABLE
    assert layout.command.motor_id == 1
    assert layout.command.cmd_seq == seq

    # Simulates the Rust daemon executing and acking the command.
    layout.command.status = 0
    layout.command.ack_seq = seq

    assert client.wait_for_ack(seq, timeout_s=1.0) == 0
    client.close()


def test_send_command_and_wait_raises_on_nonzero_status(shm_segment):
    """A failed register write must not look like success.

    Regression guard for the bug where a rejected EPROM `Operating_Mode` write was acked with a
    nonzero status that nobody checked: the servos silently stayed in POSITION mode and the arm
    drove to a stale `Goal_Position` and held rigidly.
    """
    client = ImpedanceShmClient(shm_segment.name)
    client.send_command = _acking_send(client, shm_segment, status=1)

    with pytest.raises(ImpedanceShmClientError, match="status=1"):
        client.send_command_and_wait(CommandKind.SET_OPERATING_MODE, 1, [2.0], timeout_s=1.0)
    client.close()


def test_send_command_and_wait_returns_on_zero_status(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    client.send_command = _acking_send(client, shm_segment, status=0)

    assert client.send_command_and_wait(CommandKind.SET_TORQUE_ENABLE, 1, [1.0], timeout_s=1.0) == 0
    client.close()


def test_wait_for_ack_times_out_if_daemon_never_acks(shm_segment):
    client = ImpedanceShmClient(shm_segment.name)
    seq = client.send_command(CommandKind.SET_TORQUE_ENABLE, motor_id=1, payload=[1.0])

    with pytest.raises(ImpedanceShmClientError, match="Timed out"):
        client.wait_for_ack(seq, timeout_s=0.2)
    client.close()


def test_read_output_exposes_leader_gripper_telemetry(shm_segment):
    # The leader's trigger is the operator-facing half of bilateral teleoperation, so it has to
    # survive the shared-memory round trip alongside the follower's arrays -- and it is a scalar,
    # not a seventh element of them: it belongs to a different arm, on a different bus.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.leader_gripper_pos = 2048.0
    layout.output.data.leader_gripper_vel = -37.5
    layout.output.data.leader_gripper_pwm = 120.0
    layout.output.data.timestamp_mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    layout.output.seq = 2  # stable/even

    client = ImpedanceShmClient(shm_segment.name)
    snapshot = client.read_output()

    assert snapshot["leader_gripper_pos"] == pytest.approx(2048.0)
    assert snapshot["leader_gripper_vel"] == pytest.approx(-37.5)
    assert snapshot["leader_gripper_pwm"] == pytest.approx(120.0)
    client.close()


def test_read_output_exposes_the_shared_rail_reading(shm_segment):
    # The rail voltage rides in the same snapshot as the duty it explains. A servo whose power
    # stage shorts pulls the shared supply down for hundreds of ms whenever it is written to,
    # which starves every other joint and inflates the duty they need to hold a pose -- so a
    # holding-duty measurement taken without a concurrent rail reading cannot be distinguished
    # from a stiff mechanism, and cannot be rescued after the fact. A startup-only reading is no
    # substitute: the arm is idle then, and the sag only appears under load.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.supply_decivolts = 46
    layout.output.data.case_temp_c = 41
    layout.output.data.health_motor_id = 3
    layout.output.data.timestamp_mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    layout.output.seq = 2  # stable/even

    client = ImpedanceShmClient(shm_segment.name)
    snapshot = client.read_output()

    assert snapshot["supply_decivolts"] == 46
    assert snapshot["case_temp_c"] == 41
    # Which servo answered has to travel with the reading: the daemon rotates through them one a
    # second, so consecutive samples are different joints on a shared rail.
    assert snapshot["health_motor_id"] == 3
    client.close()


def test_a_daemon_that_never_sampled_the_rail_reports_zero_not_a_plausible_voltage(shm_segment):
    # Zero is the untouched-field value, and it has to stay distinguishable from a real reading so
    # `read_supply` can answer None rather than reporting a 0.0 V rail as though it were measured.
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.data.timestamp_mono_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    layout.output.seq = 2

    client = ImpedanceShmClient(shm_segment.name)
    snapshot = client.read_output()

    assert snapshot["health_motor_id"] == 0
    assert snapshot["supply_decivolts"] == 0
    client.close()


def test_leader_fault_is_distinct_from_the_followers_comms_fault(shm_segment):
    # Losing the leader's bus only drops force feedback; losing the follower's stops the robot.
    # Sharing one flag would make an operator-visible annoyance indistinguishable from a fault
    # that means the arm is no longer tracking.
    assert FAULT_LEADER_COMMS_ERROR != FAULT_COMMS_ERROR
    assert FAULT_LEADER_COMMS_ERROR & (FAULT_COMMS_ERROR | FAULT_WATCHDOG_TIMEOUT) == 0


def test_close_does_not_raise_when_a_traceback_still_holds_a_view(shm_segment):
    """Attach, read, close -- with a live traceback from that read still around.

    Seen 2026-09-01: `examples/check_so101_impedance.py --seconds 60` ran the full 60 s and then
    died in `SO101ImpedanceChecker.__exit__` -> `close()` with `BufferError: cannot close exported
    pointers exist`. No telemetry was lost, but the nonzero exit code fails any measurement script
    wrapped in a shell pipeline whose status is checked.

    The holder is not the client's own `self._layout` -- that one is already cleared. It is a
    frame: the read/write methods bind a `region` sub-view of the mapping while they run, and a
    traceback keeps the frame it was raised from alive. `with ... as checker:` hands exactly such
    a traceback to `__exit__`, which then calls `close()`.
    """
    layout = ShmLayout.from_buffer(shm_segment.buf)
    layout.output.seq = 1  # perpetually mid-write, so read_output() gives up and raises
    del layout

    client = ImpedanceShmClient(shm_segment.name)
    traceback = None
    try:
        client.read_output(max_retries=2)
    except ImpedanceShmClientError as e:
        traceback = e.__traceback__  # what a `with` block passes on to `__exit__`
    assert traceback is not None

    # No `gc.collect()` rescue: the mapping has to be closable by refcount alone, the way it is
    # in the run above.
    gc.disable()
    try:
        client.close()
    finally:
        gc.enable()

    assert client._mmap is None
    client.close()  # idempotent, and still quiet on a second pass


def test_close_tolerates_a_view_it_cannot_drop(shm_segment):
    """A holder that `close()` has no way to reach must not turn shutdown into an exception.

    The traceback above is one such holder and is now released before the raise, but it is not
    the only shape: a KeyboardInterrupt landing inside `read_output`, or a frame kept by a
    debugger or a profiler, leaves the same exported pointer and cannot be reached from here.
    Detaching is best-effort by nature; the segment is the daemon's and is never unlinked either
    way, so the mapping simply outlives `close()` until its last view is dropped.
    """
    client = ImpedanceShmClient(shm_segment.name)
    stray_view = client._layout.output  # stands in for a view held somewhere unreachable

    gc.disable()
    try:
        client.close()
    finally:
        gc.enable()

    assert client._mmap is None
    assert stray_view.seq == 0  # the mapping is still readable through the surviving view
