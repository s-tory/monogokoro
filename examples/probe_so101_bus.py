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

**`accept`** -- all three, over every motor, ending in a verdict and a table worth keeping:

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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="e.g. /dev/ttyACM0. The daemon must not be running.")
    p.add_argument("--baud", type=int, default=1_000_000)
    p.add_argument("--timeout", type=float, default=0.005, help="Serial round-trip timeout (s).")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("read", help="Read-only health check. Run this first.")
    r.add_argument("--seconds", type=float, default=20.0)
    r.set_defaults(func=cmd_read)

    b = sub.add_parser("bisect", help="Find which motor breaks the bus when written to.")
    b.add_argument("--rounds", type=int, default=40, help="Read rounds per measurement.")
    b.add_argument("--threshold", type=float, default=90.0, help="Below this %% counts as broken.")
    b.set_defaults(func=cmd_bisect)

    v = sub.add_parser("voltage", help="Watch the supply after writing to one motor.")
    v.add_argument("--motor", type=int, required=True)
    v.add_argument("--seconds", type=float, default=2.5)
    v.set_defaults(func=cmd_voltage)

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
