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
Finds out *which servo* is making a Feetech bus unreliable, by talking to it directly.

Run this when the daemon reports comms errors -- `read Present_Position failed`, `bad Feetech
status packet`, or a `loop timing` line whose comms-error count is not zero. It does not need the
daemon, and the daemon must not be running: two programs cannot own the same serial port.

# Why this exists

A failing servo does not announce itself. It answers every read correctly, reports no error bit,
and shows the same model and firmware as its neighbours -- while making the *whole bus* unusable
for everyone else. The symptom you get is "the arm is flaky", which invites a day of swapping power
supplies and USB cables, and none of that is the problem.

The procedure below found exactly that in a few minutes, after those swaps had found nothing:

    probe_so101_bus.py read     --port /dev/ttyACM0    # 59694 replies, zero failures
    probe_so101_bus.py bisect   --port /dev/ttyACM0    # writing to motor 1 -> 39% success
    probe_so101_bus.py voltage  --port /dev/ttyACM0 --motor 1

The last one printed the answer: writing to that servo dropped the shared supply from 4.6 V to
2.4 V for 820 ms, every time, which is a short in its power stage. Its logic was fine, which is why
it looked healthy from every other angle.

After the servo is replaced, `accept` runs all three in order and records what a *healthy* bus
looks like, which is the number that was missing the first time: 4.6 V only became meaningful once
it could be compared against something, and there was nothing to compare it against.

# The modes, in the order to run them

**`read`** -- reads `Present_Position` from every motor as fast as it can, and counts replies. This
is the control, and it matters that it comes first: if reads alone are clean at a few thousand
transactions a second, the bus, the cable, the adapter and the USB bridge are all fine, and the
fault is on the *write* side. That single fact removes most of the search space.

**`bisect`** -- writes to one motor at a time and measures how well reads work afterwards. The
write is a read-modify-write of `Goal_PWM`: whatever value the register already holds is written
straight back, so it commands no change in any operating mode, and any disturbance that follows is
the act of writing rather than anything asked for. The bus is left idle between trials because the
damage outlasts the write.

**`voltage`** -- writes to one motor, then asks a *different* motor what the supply looks like,
as fast as it can. A servo whose power stage is shorted pulls the shared rail down far enough that
the others brown out, and this is what that looks like from the inside.

**`protection`** -- reads the overload-protection EPROM and nothing else: no write, no motion, no
torque. Separate from the four above, because it answers a different question -- not "which servo
is breaking the bus" but "what is this servo told to do when it stalls". Worth running before any
test that deliberately stalls a joint, and after any calibration, since these registers are
non-volatile and outlive whichever program last wrote them. On this arm (2026-09-03) it found
motors 1-5 at the factory values and only the gripper carrying the tighter limits `SOFollower`
writes -- because `so_follower.py` applies them under `if motor == "gripper"`.

**`accept`** -- all three of the bus modes, over every motor, ending in a verdict and a table worth
keeping:

    probe_so101_bus.py accept --port /dev/ttyACM0 --save bus_baseline.json

Run it after swapping a servo, and again whenever the bus next misbehaves with `--compare` pointed
at the saved file. The supply figures are the point. A single reading of 4.6 V says nothing on its
own -- during the diagnosis above it was suspected of being the fault for hours, and it turned out
to be this arm's normal idle voltage. Recorded while the arm is known good, the same number
answers that question in one line.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

try:
    import serial
except ImportError:  # pragma: no cover - the message is the point
    sys.exit("pyserial is required: pip install pyserial")

MOTOR_IDS = (1, 2, 3, 4, 5, 6)
BROADCAST_ID = 0xFE
INST_READ, INST_WRITE = 2, 3
REG_PRESENT_POSITION = (56, 2)
REG_PRESENT_VOLTAGE = (62, 1)
REG_GOAL_PWM = (44, 2)
REG_PRESENT_TEMPERATURE = (63, 1)
REG_PRESENT_CURRENT = (69, 2)
REG_TORQUE_ENABLE = (40, 1)
REG_OPERATING_MODE = (33, 1)
REG_LOCK = (55, 1)
OPERATING_MODE_POSITION, OPERATING_MODE_PWM = 0, 2
# `Goal_PWM` is a 10-bit duty field with bit 10 as the sign, and `Present_Current` signs at bit 15.
# Both were measured on the bench rather than read off a datasheet; see the Rust daemon's
# `feetech.rs`, which carries the evidence for each.
PWM_SIGN_BIT, CURRENT_SIGN_BIT = 10, 15
DUTY_FULL_SCALE = 1000
# An `Operating_Mode` write is a flash commit: 20.4 ms median, 43.5 ms worst, measured 2026-09-02.
EPROM_ACK_TIMEOUT_S = 0.150
EPROM_COMMIT_DELAY_S = 0.020
# Long enough that the previous trial's disturbance is over. Measured: a shorted power stage held
# the rail down for ~820 ms, and reads stayed degraded for seconds after that.
HEAL_S = 4.0
# A dip this far below the run's own maximum counts as a sag. The fault that motivated this script
# dropped 4.6 V to 2.4 V, so the exact figure hardly matters; it only has to clear sensor noise,
# which is one count (0.1 V) of the servo's voltage register.
SAG_V = 0.5


def _framed(body: list[int]) -> bytes:
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def read_packet(motor_id: int, addr: int, size: int) -> bytes:
    return _framed([motor_id, 4, INST_READ, addr, size])


def write_packet(motor_id: int, addr: int, values: list[int]) -> bytes:
    return _framed([motor_id, len(values) + 3, INST_WRITE, addr, *values])


class Bus:
    def __init__(self, port: str, baud: int, timeout: float):
        self.port, self.baud, self.timeout = port, baud, timeout
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def read_register(self, motor_id: int, reg: tuple[int, int], retries: int = 1):
        """The register's value, or None if the servo did not answer.

        The input buffer is discarded first so that a reply which arrived too late for the previous
        request is not mistaken for this one's -- on a bus that is already timing out, that is how
        one failure turns into a run of them.
        """
        addr, size = reg
        for _ in range(retries):
            self.ser.reset_input_buffer()
            self.ser.write(read_packet(motor_id, addr, size))
            reply = self.ser.read(6 + size)
            if len(reply) == 6 + size and reply[:2] == b"\xff\xff" and reply[2] == motor_id:
                return reply[5] if size == 1 else reply[5] | (reply[6] << 8)
        return None

    def write_register(self, motor_id: int, reg: tuple[int, int], value: int) -> bool:
        """Writes one register and waits for the acknowledgement. True if the servo answered."""
        addr, size = reg
        payload = [value & 0xFF] if size == 1 else [value & 0xFF, (value >> 8) & 0xFF]
        self.ser.reset_input_buffer()
        self.ser.write(write_packet(motor_id, addr, payload))
        self.ser.flush()
        return len(self.ser.read(6)) == 6

    def write_operating_mode(self, motor_id: int, mode: int) -> bool:
        """Torque off, unlock EPROM, commit the mode, settle.

        `Operating_Mode` lives in EPROM, so a write that actually changes it is a flash
        erase/program cycle: measured on this arm at 20.4 ms median and 43.5 ms worst against the
        0.23 ms a RAM write takes. The port's normal timeout is far below that, so the ack arrives
        after we have stopped listening and is then read as the *next* transaction's reply. Widen
        the timeout for this one write, and give the flash a beat before anyone reads it back.
        Mirrors `write_operating_mode` in the Rust daemon, which measured those numbers.
        """
        self.write_register(motor_id, REG_TORQUE_ENABLE, 0)
        self.write_register(motor_id, REG_LOCK, 0)
        previous = self.ser.timeout
        self.ser.timeout = EPROM_ACK_TIMEOUT_S
        try:
            ok = self.write_register(motor_id, REG_OPERATING_MODE, mode)
        finally:
            self.ser.timeout = previous
        time.sleep(EPROM_COMMIT_DELAY_S)
        return ok

    def poke(self, motor_id: int) -> bool:
        """Writes `Goal_PWM` back to itself: a write that asks for no change at all.

        Reading first is what makes this safe to point at an arm in any state. Address 44 is the
        duty command in PWM mode and `Goal_Time` in position mode, so writing a constant would mean
        different things depending on how the servo was left; writing back what is already there
        means the same nothing either way.
        """
        current = self.read_register(motor_id, REG_GOAL_PWM, retries=4)
        if current is None:
            return False
        self.ser.reset_input_buffer()
        self.ser.write(write_packet(motor_id, REG_GOAL_PWM[0], [current & 0xFF, (current >> 8) & 0xFF]))
        self.ser.flush()
        self.ser.read(6)  # the servo's acknowledgement, if it sends one
        return True

    def read_success_rate(self, rounds: int, ids=MOTOR_IDS) -> float:
        ok = 0
        for _ in range(rounds):
            for motor_id in ids:
                ok += self.read_register(motor_id, REG_PRESENT_POSITION) is not None
        return ok / (rounds * len(ids)) * 100

    def idle(self, seconds: float = HEAL_S) -> None:
        """Leaves the bus completely alone, which is the only thing that clears the disturbance."""
        self.ser.close()
        time.sleep(seconds)
        self.ser.open()


def read_census(bus: Bus, seconds: float) -> tuple[dict[int, int], dict[int, int]]:
    """Replies and failures per motor, reading `Present_Position` as fast as the bus allows."""
    ok = dict.fromkeys(MOTOR_IDS, 0)
    bad = dict.fromkeys(MOTOR_IDS, 0)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for motor_id in MOTOR_IDS:
            if bus.read_register(motor_id, REG_PRESENT_POSITION) is None:
                bad[motor_id] += 1
            else:
                ok[motor_id] += 1
    return ok, bad


def cmd_read(bus: Bus, args) -> None:
    ok, bad = read_census(bus, args.seconds)
    total_ok, total_bad = sum(ok.values()), sum(bad.values())
    rate = total_bad / max(1, total_ok + total_bad) * 100
    print(f"{args.port}: {total_ok} replies, {total_bad} failures ({rate:.1f}%) in {args.seconds:.0f}s")
    print(f"\n{'motor':<8}{'replies':>10}{'failures':>10}{'fail%':>8}")
    for motor_id in MOTOR_IDS:
        n = ok[motor_id] + bad[motor_id]
        print(f"{motor_id:<8}{ok[motor_id]:>10}{bad[motor_id]:>10}{bad[motor_id] / max(1, n) * 100:>7.1f}%")
    print()
    if total_bad == 0:
        print("Reads alone are clean, so the bus, cabling, adapter and USB bridge are all fine.")
        print("If the daemon still reports comms errors, the fault is on the write side: run")
        print("`bisect` next.")
    else:
        print("Reads are failing on their own, so this is not about writes. Suspect the wiring or")
        print("the bridge before any individual servo -- and check no daemon is holding the port.")


def write_trial(bus: Bus, motor_id: int, rounds: int) -> tuple[float, float | None]:
    """Read success rate before and after writing to one motor; `None` after if it is unreachable.

    The bus is left idle first, because the disturbance from the previous trial outlasts the write
    that caused it.
    """
    bus.idle()
    before = bus.read_success_rate(rounds)
    if not bus.poke(motor_id):
        return before, None
    time.sleep(0.05)
    return before, bus.read_success_rate(rounds)


def cmd_bisect(bus: Bus, args) -> None:
    print("Writing to one motor at a time, then measuring how well every motor reads afterwards.")
    print("The write asks for no change; only the act of writing is under test.\n")
    print(f"{'written to':<14}{'reads before':>14}{'reads after':>13}")
    print("-" * 41)
    suspects = []
    for motor_id in MOTOR_IDS:
        before, after = write_trial(bus, motor_id, args.rounds)
        if after is None:
            print(f"motor {motor_id:<8}{before:>13.1f}%{'unreachable':>13}")
            continue
        flag = "   <-- breaks the bus" if after < args.threshold else ""
        print(f"motor {motor_id:<8}{before:>13.1f}%{after:>12.1f}%{flag}")
        if after < args.threshold:
            suspects.append(motor_id)

    print()
    if not suspects:
        print("No single motor breaks the bus by being written to. If the daemon still sees comms")
        print("errors, they are not coming from this.")
    else:
        for motor_id in suspects:
            print(f"Motor {motor_id} disturbs the bus when written to, while answering reads normally.")
        print(f"\nRun `voltage --motor {suspects[0]}` to see what it does to the supply.")


def supply_after_write(bus: Bus, target: int, witness: int, seconds: float) -> dict | None:
    """Writes to `target`, then watches the supply through `witness` for `seconds`.

    `None` means the target could not be written to at all. A returned `replies` of zero means the
    witness went silent, which is itself the symptom: it browned out rather than answering.
    """
    bus.idle()
    if not bus.poke(target):
        return None
    samples, misses = [], 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        raw = bus.read_register(witness, REG_PRESENT_VOLTAGE)
        if raw is None:
            misses += 1
        else:
            samples.append((time.monotonic() - t0, raw / 10.0))
    if not samples:
        return {"replies": 0, "misses": misses, "min": None, "max": None, "median": None, "sag_ms": None}
    volts = [v for _, v in samples]
    ceiling = max(volts)
    low = [t for t, v in samples if v < ceiling - SAG_V]
    return {
        "replies": len(samples),
        "misses": misses,
        "min": min(volts),
        "max": ceiling,
        "median": statistics.median(volts),
        "sag_ms": max(low) * 1000 if low else 0.0,
    }


def cmd_voltage(bus: Bus, args) -> None:
    witness = next(i for i in MOTOR_IDS if i != args.motor)
    print(f"Writing to motor {args.motor}, then asking motor {witness} what the supply is doing.\n")
    for target, label in ((witness, f"motor {witness} (control)"), (args.motor, f"motor {args.motor}")):
        rail = supply_after_write(bus, target, witness, args.seconds)
        if rail is None:
            print(f"{label}: unreachable")
            continue
        if rail["replies"] == 0:
            print(f"after writing to {label}: no reply at all in {args.seconds:.0f}s")
            continue
        print(f"after writing to {label}: {rail['replies']} replies, {rail['misses']} failures")
        print(
            f"    supply  min {rail['min']:.1f} V   max {rail['max']:.1f} V   median {rail['median']:.1f} V"
        )
        if rail["sag_ms"]:
            print(f"    sagged for the first {rail['sag_ms']:.0f} ms")
    print("\nA supply that collapses only when one particular servo is written to is a short in")
    print("that servo's power stage. Replace it; no amount of cabling will help.")


def cmd_accept(bus: Bus, args) -> None:
    """Every check, over every motor, ending in a verdict and a record of a healthy bus."""
    print(f"Acceptance run on {args.port}. Nothing is commanded to move: every write puts a")
    print("register back to the value it already held. Expect roughly two minutes.\n")

    print(f"[1/2] reads only, {args.seconds:.0f}s ...")
    ok, bad = read_census(bus, args.seconds)
    total_ok, total_bad = sum(ok.values()), sum(bad.values())
    print(f"      {total_ok} replies, {total_bad} failures\n")

    print("[2/2] per motor: write, re-read, then watch the supply through a witness motor ...\n")
    print(f"{'motor':<8}{'read fail':>11}{'reads after write':>20}{'supply median':>16}{'sag':>10}")
    print("-" * 65)
    motors: dict[int, dict] = {}
    failures: list[str] = []
    for motor_id in MOTOR_IDS:
        n = ok[motor_id] + bad[motor_id]
        read_fail = bad[motor_id] / max(1, n) * 100
        _, after = write_trial(bus, motor_id, args.rounds)
        witness = next(i for i in MOTOR_IDS if i != motor_id)
        rail = supply_after_write(bus, motor_id, witness, args.voltage_seconds) if after is not None else None

        row = {
            "read_fail_pct": read_fail,
            "reads_after_write_pct": after,
            "witness": witness,
            "supply_min": rail["min"] if rail else None,
            "supply_max": rail["max"] if rail else None,
            "supply_median": rail["median"] if rail else None,
            "sag_ms": rail["sag_ms"] if rail else None,
        }
        motors[motor_id] = row

        if bad[motor_id]:
            failures.append(f"motor {motor_id} failed {bad[motor_id]} reads while nothing was being written")
        if after is None:
            failures.append(f"motor {motor_id} could not be written to at all")
        elif after < args.threshold:
            failures.append(f"motor {motor_id} drops reads to {after:.1f}% when written to")
        if rail is not None and rail["replies"] == 0:
            failures.append(f"motor {motor_id} silenced motor {witness} completely")
        elif row["sag_ms"] and row["sag_ms"] > args.max_sag_ms:
            failures.append(f"motor {motor_id} sagged the supply for {row['sag_ms']:.0f} ms")

        after_s = "unreachable" if after is None else f"{after:.1f}%"
        median_s = "--" if row["supply_median"] is None else f"{row['supply_median']:.1f} V"
        sag_s = (
            "--" if row["sag_ms"] is None else ("none" if not row["sag_ms"] else f"{row['sag_ms']:.0f} ms")
        )
        print(f"{motor_id:<8}{read_fail:>10.1f}%{after_s:>20}{median_s:>16}{sag_s:>10}")

    verdict = "FAIL" if failures else "PASS"
    print(f"\nverdict: {verdict}")
    if failures:
        for line in failures:
            print(f"  - {line}")
        print("\nRun `bisect` and then `voltage --motor N` on the motor named above for the detail.")
    else:
        medians = [m["supply_median"] for m in motors.values() if m["supply_median"] is not None]
        lo, hi = min(medians), max(medians)
        held = f"{lo:.1f} V" if lo == hi else f"{lo:.1f}-{hi:.1f} V"
        print("  Reads are clean, no motor disturbs the bus when written to, and the supply")
        print(f"  held at {held} throughout. That is this arm's healthy idle voltage, and")
        print("  knowing it is what stops the next investigation suspecting the power supply.")

    record = {
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "port": args.port,
        "baud": args.baud,
        "read_seconds": args.seconds,
        "rounds": args.rounds,
        "voltage_seconds": args.voltage_seconds,
        "verdict": verdict,
        "read": {"replies": total_ok, "failures": total_bad},
        "motors": {str(k): v for k, v in motors.items()},
    }
    if args.compare:
        _print_comparison(record, args.compare)
    if args.save:
        Path(args.save).write_text(json.dumps(record, indent=2) + "\n")
        print(f"\nbaseline written to {args.save}")
        if verdict == "FAIL":
            print("Saved anyway, but do not use a failing run as the reference.")


def _pair(was: dict, now: dict, key: str, unit: str) -> str:
    """`then -> now` for one figure, with `--` where either run has nothing to show."""
    a, b = was.get(key), now.get(key)
    a_s = "--" if a is None else f"{a:.1f}{unit}"
    b_s = "--" if b is None else f"{b:.1f}{unit}"
    return f"{a_s} -> {b_s}"


def _print_comparison(record: dict, path: str) -> None:
    """Puts this run next to a saved one. Absolute volts mean little; a change in them means a lot."""
    try:
        old = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        print(f"\ncould not read baseline {path}: {exc}")
        return

    print(f"\nagainst {path} (recorded {old.get('when', 'unknown')}, verdict {old.get('verdict', '?')}):\n")
    print(f"{'motor':<8}{'supply median':>26}{'reads after write':>26}")
    print("-" * 60)
    for motor_id in MOTOR_IDS:
        was = old.get("motors", {}).get(str(motor_id), {})
        now = record["motors"][str(motor_id)]
        supply = _pair(was, now, "supply_median", " V")
        reads = _pair(was, now, "reads_after_write_pct", "%")
        print(f"{motor_id:<8}{supply:>26}{reads:>26}")


# The overload protection an sts3215 applies to itself, and the values `SOFollower` writes into it
# (`so_follower.py`, `Max_Torque_Limit` 500 / `Protection_Current` 250 / `Overload_Torque` 25).
# These live in EPROM, so they persist across power cycles and outlive whichever program wrote
# them: an arm that has ever been driven by the position-mode follower still carries them when the
# PWM daemon takes over. Reading them is how you find out whether the daemon is running against a
# servo that will cut its own torque under a stall, or against one that will hold a stalled duty
# until something outside the loop stops it.
PROTECTION_REGISTERS = (
    ("Max_Torque_Limit", (16, 2), 500),
    ("Protection_Current", (28, 2), 250),
    ("Protective_Torque", (34, 1), None),
    ("Protection_Time", (35, 1), None),
    ("Overload_Torque", (36, 1), 25),
    ("Over_Current_Protection_Time", (38, 1), None),
    ("Max_Temperature_Limit", (13, 1), None),
    ("Operating_Mode", (33, 1), None),
    ("Torque_Limit", (48, 2), None),
    ("Status", (65, 1), None),
    ("Present_Temperature", (63, 1), None),
)
# Which of the three `SOFollower` writes have to match before we can say it has run on this servo.
FOLLOWER_WRITTEN = tuple(name for name, _, expected in PROTECTION_REGISTERS if expected is not None)


def cmd_protection(bus: Bus, args) -> None:
    """Reads the overload-protection EPROM, and nothing else. No write, no motion, no torque."""
    values: dict[int, dict[str, int | None]] = {}
    for motor_id in MOTOR_IDS:
        values[motor_id] = {
            name: bus.read_register(motor_id, reg, retries=4) for name, reg, _ in PROTECTION_REGISTERS
        }

    width = max(len(name) for name, _, _ in PROTECTION_REGISTERS) + 2
    print(f"{args.port}: overload protection, read-only\n")
    print(f"{'register':<{width}}" + "".join(f"{f'motor {i}':>10}" for i in MOTOR_IDS))
    for name, _, expected in PROTECTION_REGISTERS:
        row = "".join(f"{'--' if values[i][name] is None else values[i][name]:>10}" for i in MOTOR_IDS)
        tail = f"   <- SOFollower writes {expected}" if expected is not None else ""
        print(f"{name:<{width}}{row}{tail}")

    print()
    for motor_id in MOTOR_IDS:
        got = values[motor_id]
        if any(got[name] is None for name in FOLLOWER_WRITTEN):
            print(f"motor {motor_id}: did not answer; nothing can be concluded about it")
            continue
        matched = [
            name
            for name, _, expected in PROTECTION_REGISTERS
            if expected is not None and got[name] == expected
        ]
        if len(matched) == len(FOLLOWER_WRITTEN):
            print(f"motor {motor_id}: carries the SOFollower limits -- overload protection is configured")
        elif matched:
            print(
                f"motor {motor_id}: carries {len(matched)}/{len(FOLLOWER_WRITTEN)} of them ({', '.join(matched)})"
            )
        else:
            print(f"motor {motor_id}: none of the SOFollower limits are present")

    print()
    print("What this does and does not settle: these registers say what the servo has been told to")
    print("do about an overload. Whether its firmware acts on them while `Operating_Mode` is PWM (2)")
    print("is a separate question that this read cannot answer -- it takes a stall on the bench with")
    print("the current logged.")


def decode_sign_magnitude(value: int, sign_bit: int) -> int:
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def encode_sign_magnitude(value: int, sign_bit: int) -> int:
    out = abs(value) & ((1 << sign_bit) - 1)
    return out | (1 << sign_bit) if value < 0 else out


def cmd_stall(bus: Bus, args) -> None:
    """Holds one joint against a stall at a rising duty, and watches whether the servo backs off.

    **This one drives the arm.** Everything else in this file is read-only or writes a value back
    to itself; this commands torque on purpose, and holds it into something that will not move.

    The question it answers cannot be answered by reading. `Protection_Current`, `Overload_Torque`
    and `Over_Current_Protection_Time` say what the servo has been *told* to do about a stall, and
    `protection` prints them -- but the impedance daemon runs the joints in PWM mode, which is
    open-loop duty, and whether the firmware still applies an overload cutback in that mode is
    undocumented. If it does, a saturated duty ends by itself. If it does not, nothing inside the
    control loop can end it, and the release path is a human watching the arm.

    So: raise the duty in steps, hold each one, and log current and temperature throughout. A
    cutback shows up as current falling while the commanded duty stays put.
    """
    motor = args.motor
    steps = [int(d) for d in args.duties.split(",")]
    if any(not 0 < d <= DUTY_FULL_SCALE for d in steps):
        sys.exit(f"duties must each be in 1..{DUTY_FULL_SCALE}")

    entry_mode = bus.read_register(motor, REG_OPERATING_MODE, retries=4)
    temp = bus.read_register(motor, REG_PRESENT_TEMPERATURE, retries=4)
    pos = bus.read_register(motor, REG_PRESENT_POSITION, retries=4)
    volts = bus.read_register(motor, REG_PRESENT_VOLTAGE, retries=4)
    if None in (entry_mode, temp, pos, volts):
        sys.exit(f"motor {motor} did not answer the preflight read; not driving anything")
    if temp >= args.abort_temp:
        sys.exit(f"motor {motor} is already at {temp} C, at or over the {args.abort_temp} C abort")

    print(f"motor {motor}: mode {entry_mode}, {temp} C, position {pos}, rail {volts / 10:.1f} V")
    print(f"about to drive it at duty {steps} x {args.hold:.1f} s, direction {args.direction:+d}")
    print(f"aborting on {args.abort_temp} C, or Ctrl-C, whichever comes first\n")
    print("Put the arm where a stall is safe -- for the gripper, close it onto something solid and")
    print("blunt. Watch it while this runs; nothing here is a substitute for that.")
    input("Press ENTER when the joint is where you want it, or Ctrl-C to stop....")

    rows = []
    verdict = "completed"
    try:
        if not bus.write_operating_mode(motor, OPERATING_MODE_PWM):
            sys.exit("the servo did not acknowledge the mode change; nothing was driven")
        bus.write_register(motor, REG_TORQUE_ENABLE, 1)
        t0 = time.monotonic()
        for duty in steps:
            bus.write_register(
                motor, REG_GOAL_PWM, encode_sign_magnitude(duty * args.direction, PWM_SIGN_BIT)
            )
            step_end = time.monotonic() + args.hold
            while time.monotonic() < step_end:
                raw_i = bus.read_register(motor, REG_PRESENT_CURRENT)
                t = bus.read_register(motor, REG_PRESENT_TEMPERATURE)
                pos = bus.read_register(motor, REG_PRESENT_POSITION)
                v = bus.read_register(motor, REG_PRESENT_VOLTAGE)
                if None in (raw_i, t, pos, v):
                    continue
                rows.append(
                    (time.monotonic() - t0, duty, decode_sign_magnitude(raw_i, CURRENT_SIGN_BIT), t, pos, v)
                )
                if t >= args.abort_temp:
                    verdict = f"aborted: {t} C at duty {duty}"
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        if verdict == "completed":
            verdict = "aborted by hand"
    finally:
        # Order matters and none of it is optional: duty first so the joint stops pushing even if a
        # later write fails, then torque, then the mode. Leaving a servo in PWM is the trap the Rust
        # daemon documents -- the next program to command a position lands in `Goal_Time` instead.
        bus.write_register(motor, REG_GOAL_PWM, 0)
        bus.write_register(motor, REG_TORQUE_ENABLE, 0)
        if entry_mode is not None and entry_mode != OPERATING_MODE_PWM:
            bus.write_operating_mode(motor, entry_mode)
        back = bus.read_register(motor, REG_OPERATING_MODE, retries=6)
        print(f"\nstopped ({verdict}); Operating_Mode restored to {back} (was {entry_mode})")
        if back != entry_mode:
            print("*** the mode did NOT come back. Fix that before running the daemon. ***")

    if not rows:
        print("no samples; nothing to report")
        return

    if args.csv:
        with open(args.csv, "w") as fh:
            fh.write("t_s,duty,current,temp_c,position,rail_dV\n")
            for r in rows:
                fh.write(",".join(str(x) for x in r) + "\n")
        print(f"wrote {len(rows)} samples to {args.csv}")

    print(
        f"\n{'duty':>6}{'n':>7}{'|I| first 1s':>14}{'|I| last 1s':>13}{'peak |I|':>10}{'move':>8}{'max C':>7}"
    )
    for duty in steps:
        step = [r for r in rows if r[1] == duty]
        if not step:
            continue
        t_end = step[-1][0]
        early = [abs(r[2]) for r in step if r[0] <= step[0][0] + 1.0]
        late = [abs(r[2]) for r in step if r[0] >= t_end - 1.0]
        moved = step[-1][4] - step[0][4]
        print(
            f"{duty:>6}{len(step):>7}{statistics.median(early) if early else 0:>14.0f}"
            f"{statistics.median(late) if late else 0:>13.0f}{max(abs(r[2]) for r in step):>10}"
            f"{moved:>8}{max(r[3] for r in step):>7}"
        )
    print()
    print("Read it like this: a step whose current falls from 'first 1s' to 'last 1s' while the")
    print("duty is unchanged, and whose joint did not move, is the servo cutting its own torque --")
    print("the overload protection firing in PWM mode. Current that holds flat at a stall is the")
    print("other answer, and the more expensive one: the release path is not in the servo.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="e.g. /dev/ttyACM0. The daemon must not be running.")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--timeout", type=float, default=0.005, help="Serial round-trip timeout (s).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="Read-only health check. Run this first.")
    r.add_argument("--seconds", type=float, default=20.0)
    r.set_defaults(func=cmd_read)

    pr = sub.add_parser(
        "protection", help="Read the overload-protection EPROM. Read-only; nothing is written."
    )
    pr.set_defaults(func=cmd_protection)

    b = sub.add_parser("bisect", help="Find which motor breaks the bus when written to.")
    b.add_argument("--rounds", type=int, default=40, help="Read rounds per measurement.")
    b.add_argument("--threshold", type=float, default=90.0, help="Below this %% counts as broken.")
    b.set_defaults(func=cmd_bisect)

    v = sub.add_parser("voltage", help="Watch the supply after writing to one motor.")
    v.add_argument("--motor", type=int, required=True)
    v.add_argument("--seconds", type=float, default=2.5)
    v.set_defaults(func=cmd_voltage)

    st = sub.add_parser("stall", help="DRIVES THE ARM. Hold a joint at a rising duty and log current.")
    st.add_argument("--motor", type=int, default=6, help="Default 6, the gripper: lowest threshold.")
    st.add_argument("--direction", type=int, choices=(1, -1), required=True, help="Sign of the duty.")
    st.add_argument("--duties", default="150,300,450", help="Duty magnitudes to hold, out of 1000.")
    st.add_argument(
        "--hold",
        type=float,
        default=4.0,
        help="Seconds to hold each step. Must outlast Over_Current_Protection_Time (200 on this "
        "arm) or a cutback has no chance to appear; 4 s gives it room either way.",
    )
    st.add_argument("--abort-temp", type=int, default=50, help="Stop at this temperature (C).")
    st.add_argument("--csv", help="Write the samples here.")
    st.set_defaults(func=cmd_stall)

    a = sub.add_parser("accept", help="All of the above, over every motor. Run after swapping a servo.")
    a.add_argument("--seconds", type=float, default=20.0, help="Read-only phase (s).")
    a.add_argument("--rounds", type=int, default=40, help="Read rounds per write trial.")
    a.add_argument("--voltage-seconds", type=float, default=2.5, help="Supply watch per motor (s).")
    a.add_argument("--threshold", type=float, default=90.0, help="Below this %% of reads counts as broken.")
    a.add_argument("--max-sag-ms", type=float, default=50.0, help="Supply dip tolerated after a write.")
    a.add_argument("--save", help="Write the run to this JSON file, to compare against later.")
    a.add_argument("--compare", help="Print this run next to a previously saved one.")
    a.set_defaults(func=cmd_accept)

    args = p.parse_args()
    bus = Bus(args.port, args.baud, args.timeout)
    try:
        args.func(bus, args)
    finally:
        bus.ser.close()


if __name__ == "__main__":
    main()
