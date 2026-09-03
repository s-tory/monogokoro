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

"""Exercises the reply validation in `examples/probe_so101_bus.py`.

The checksum check exists because a Feetech status packet does not say which register it answers
for, so a reply that arrived too late for the previous read can be returned as this one's value.
On the arm that check reports zero rejections, which is what a working bus looks like -- and also
what a check that never runs looks like. These tests are the difference between the two.

The last test pins the *blind spot* rather than the guarantee, because the limit is the part worth
not forgetting: two registers of the same width produce replies the checksum cannot tell apart.
"""

import importlib.util
from pathlib import Path

import pytest

PROBE = Path(__file__).resolve().parents[2] / "examples" / "probe_so101_bus.py"

pytest.importorskip("serial", reason="probe_so101_bus.py needs pyserial")


def _load_probe():
    spec = importlib.util.spec_from_file_location("probe_so101_bus", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


class FakeSerial:
    """Returns a canned reply to every request, so a read can be given a specific packet."""

    def __init__(self, reply: bytes):
        self.reply = reply
        self.timeout = 0.005
        self.written = []

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.written.append(data)

    def flush(self):
        pass

    def read(self, n):
        return self.reply[:n]

    def close(self):
        pass


def _bus(reply: bytes):
    bus = probe.Bus.__new__(probe.Bus)
    bus.port, bus.baud, bus.timeout = "fake", 1_000_000, 0.005
    bus.ser = FakeSerial(reply)
    bus.bad_checksums = 0
    return bus


def _status_packet(motor_id: int, params: list[int]) -> bytes:
    body = [motor_id, len(params) + 2, 0, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def test_a_well_formed_reply_is_accepted():
    bus = _bus(_status_packet(6, [0x26, 0x09]))  # position 2342
    assert bus.read_register(6, probe.REG_PRESENT_POSITION) == 2342
    assert bus.bad_checksums == 0


def test_a_corrupted_checksum_is_rejected_and_counted():
    packet = bytearray(_status_packet(6, [0x26, 0x09]))
    packet[-1] ^= 0xFF
    bus = _bus(bytes(packet))
    assert bus.read_register(6, probe.REG_PRESENT_POSITION, retries=1) is None
    assert bus.bad_checksums == 1


def test_the_misattribution_that_actually_happened_is_rejected():
    """A two-byte position reply, read as if it answered a one-byte temperature request.

    Measured 2026-09-03: `Present_Temperature` returned 38 C on a servo a direct read put at 29 C,
    because position 2342's low byte is 38. Header, id and length all pass -- reading 7 bytes of an
    8-byte packet leaves a 7-byte string whose last byte is the position's *high* byte, not a
    checksum. Only the checksum separates the two.
    """
    bus = _bus(_status_packet(6, [0x26, 0x09]))
    assert bus.read_register(6, probe.REG_PRESENT_TEMPERATURE, retries=1) is None
    assert bus.bad_checksums == 1


def test_same_width_registers_are_the_blind_spot():
    """What the checksum cannot do, kept as a test so it is not quietly assumed away.

    `Present_Voltage` and `Present_Temperature` are both one byte, so a late voltage reply is a
    fully valid packet that simply answers a different question. It is accepted, and it must be:
    nothing in the protocol distinguishes them. This is why the stall summary reports a median
    temperature rather than a maximum -- the defence is statistical, because it cannot be a check.
    """
    bus = _bus(_status_packet(6, [46]))  # a 4.6 V reading, arriving late
    assert bus.read_register(6, probe.REG_PRESENT_TEMPERATURE) == 46
    assert bus.bad_checksums == 0


@pytest.mark.parametrize("value", [0, 1, -1, 150, -150, 999, -999])
def test_pwm_sign_magnitude_round_trips(value):
    encoded = probe.encode_sign_magnitude(value, probe.PWM_SIGN_BIT)
    assert probe.decode_sign_magnitude(encoded, probe.PWM_SIGN_BIT) == value


def test_current_sign_decodes_at_bit_15():
    """The magnitudes measured on the bench: 22 one way, 0x8016 the other."""
    assert probe.decode_sign_magnitude(22, probe.CURRENT_SIGN_BIT) == 22
    assert probe.decode_sign_magnitude(0x8016, probe.CURRENT_SIGN_BIT) == -22
