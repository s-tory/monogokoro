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
Counts supply collapses in a daemon log, and writes down the window and the rule it counted under.

The daemon logs every change to a servo's error byte as one line:

    servo_error mono_ns=15848250423014 motor=5 0x00 -> 0x01

That stream is the fastest voltmeter on this rig -- `supply_decivolts` is sampled round-robin once
a second and cannot render an 833 ms dip, while the servos report bit0 at the 400 Hz loop rate.
Counting collapses off it is therefore the measurement, not a cross-check.

# Why this exists rather than a grep

On 2026-09-14 a supply A/B was counted by hand out of a 56-minute log, and the answer written into
the docs was "8 episodes over 100 ms became 1, of 390 ms". Re-counting on 2026-09-15 gave **6**
episodes, of 821/825/827/1890/1896/4158 ms -- and neither number could be reproduced against the
other, because **the 45-second window the first count used was never recorded**. The disagreement
may be the window, may be the threshold, may be the number of motors required; with the window
gone there is no way to find out. The numbers themselves move from rig to rig and barely matter.
Not being able to check your own result does.

So this prints the window and the rule with every count, and `--json` writes both next to the
result. A count that comes out of here can be re-run by whoever reads it.

# The rule, and why it is an argument rather than a constant

`README_DETAILS_EN.md` defines a supply collapse as bit0 raised by **every motor in the same
tick**: one servo protecting itself is heat or load, the whole rail sagging is the supply. That
definition is what `--min-motors 6` means, and it is the default here.

It is also the parameter that decided the disagreement above. The 390 ms episode that the docs
still record as surviving the better supply was, on re-count, **two motors** -- so it never met
the definition it was being reported under. Sweeping `--min-motors` is therefore the first thing
this prints: the same log, counted at 1..6 motors, in one table. If a result only exists at a
threshold, that is a property of the threshold.

    # What happened in this log, at every strictness:
    python examples/analyze_servo_error_log.py daemon_run3_old_pwm700.log

    # The documented rule, in a named window, recorded to a file:
    python examples/analyze_servo_error_log.py daemon_run3_old_pwm700.log \
        --window 120:165 --min-motors 6 --min-ms 100 --json run3.collapses.json

    # Two conditions side by side (each still carries its own window in the JSON):
    python examples/analyze_servo_error_log.py old.log new.log --min-motors 2

Times are reported both as seconds from the first `servo_error` line in the file and as the raw
`mono_ns`, because the first is what a human picks a window with and the second is what survives
being pasted into a note. `--window` accepts either: `120:165` is seconds, `mono_ns=...:...` is raw.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# `servo_error mono_ns={now_ns} motor={id} {from:#04x} -> {to:#04x}`, as emitted by main.rs. The
# log prefix in front of it (timestamp, level, module) is whatever env_logger was configured with,
# so it is skipped rather than parsed.
LINE = re.compile(
    r"servo_error\s+mono_ns=(?P<ns>\d+)\s+motor=(?P<motor>\d+)\s+"
    r"(?P<frm>0x[0-9a-fA-F]+)\s*->\s*(?P<to>0x[0-9a-fA-F]+)"
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000

# 400 Hz loop. Two edges closer together than one tick came from the same round of reads, so
# "in the same tick" is a 2.5 ms grace and not an exact timestamp match.
TICK_NS = 2_500_000


@dataclass
class Interval:
    """One stretch of time with the bit raised, on one motor."""

    motor: int
    start_ns: int
    end_ns: int | None  # None when the log ends with the bit still up.

    def duration_ns(self, log_end_ns: int) -> int:
        return (self.end_ns if self.end_ns is not None else log_end_ns) - self.start_ns


@dataclass
class Episode:
    """A stretch where at least `min_motors` motors held the bit at once."""

    start_ns: int
    end_ns: int
    motors: set[int] = field(default_factory=set)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / NS_PER_MS


def parse_log(path: Path) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Returns (transitions, first_ns, last_ns). A transition is (ns, motor, from_byte, to_byte)."""
    transitions: list[tuple[int, int, int, int]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if m is None:
            continue
        transitions.append((int(m["ns"]), int(m["motor"]), int(m["frm"], 16), int(m["to"], 16)))
    if not transitions:
        return [], 0, 0
    transitions.sort(key=lambda t: t[0])
    return transitions, transitions[0][0], transitions[-1][0]


def intervals_for_bit(
    transitions: list[tuple[int, int, int, int]], bit: int, log_end_ns: int
) -> list[Interval]:
    """Reconstructs, per motor, every stretch during which `bit` was set.

    The log records edges of the whole byte, so a motor going 0x01 -> 0x05 has not changed bit0 and
    must not open a second interval. Only the masked value is looked at.
    """
    open_at: dict[int, int] = {}
    out: list[Interval] = []
    for ns, motor, frm, to in transitions:
        was, now = bool(frm & bit), bool(to & bit)
        if now and not was:
            # A repeated rising edge without a fall between them means the daemon restarted or the
            # log was concatenated; keep the earlier start, which is the conservative reading.
            open_at.setdefault(motor, ns)
        elif was and not now and motor in open_at:
            out.append(Interval(motor, open_at.pop(motor), ns))
    for motor, start in open_at.items():
        out.append(Interval(motor, start, None))
    out.sort(key=lambda iv: iv.start_ns)
    return out


def merge_per_motor(intervals: list[Interval], log_end_ns: int, grace_ns: int = TICK_NS) -> list[Interval]:
    """Joins, on each motor separately, intervals that are within one tick of each other.

    The daemon reads all six motors in one round, so a bit that stays up across two rounds can be
    logged as two intervals a tick apart rather than one. Without this the same motor appears
    twice at once in the sweep below, and its own second interval closes the first one early --
    which chops a single long collapse into many short ones and *raises* the episode count as the
    threshold gets stricter. That is what this looked like when it was wrong.
    """
    by_motor: dict[int, list[Interval]] = {}
    for iv in intervals:
        end = iv.end_ns if iv.end_ns is not None else log_end_ns
        by_motor.setdefault(iv.motor, []).append(Interval(iv.motor, iv.start_ns, end))

    out: list[Interval] = []
    for motor, ivs in by_motor.items():
        ivs.sort(key=lambda i: i.start_ns)
        cur = ivs[0]
        for nxt in ivs[1:]:
            if nxt.start_ns - (cur.end_ns or 0) <= grace_ns:
                cur = Interval(motor, cur.start_ns, max(cur.end_ns or 0, nxt.end_ns or 0))
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    out.sort(key=lambda i: i.start_ns)
    return out


def episodes(intervals: list[Interval], min_motors: int, log_end_ns: int) -> list[Episode]:
    """Sweeps merged intervals and returns every stretch with >= `min_motors` of them overlapping.

    The overlap is the raw one. The tick grace belongs to `merge_per_motor`, where it answers "is
    this the same stretch on this motor"; applying it here too would pad every reported duration by
    a tick, which is a measurement error dressed as a tolerance.
    """
    edges: list[tuple[int, int, int]] = []  # (ns, delta, motor)
    for iv in intervals:
        end = iv.end_ns if iv.end_ns is not None else log_end_ns
        edges.append((iv.start_ns, +1, iv.motor))
        edges.append((end, -1, iv.motor))
    # Falls before rises at the same instant, so a motor that drops out and back does not read as
    # having stayed up.
    edges.sort(key=lambda e: (e[0], e[1]))

    out: list[Episode] = []
    live: set[int] = set()
    current: Episode | None = None
    peak = 0
    for ns, delta, motor in edges:
        if delta > 0:
            live.add(motor)
        else:
            live.discard(motor)
        if len(live) >= min_motors:
            if current is None:
                current, peak = Episode(start_ns=ns, end_ns=ns, motors=set(live)), len(live)
            elif len(live) > peak:
                # Report who was up together at the episode's peak, not everyone who passed
                # through it -- those are different claims and only the first one is "at once".
                current.motors, peak = set(live), len(live)
        elif current is not None:
            current.end_ns = ns
            out.append(current)
            current, peak = None, 0
    if current is not None:
        current.end_ns = log_end_ns
        out.append(current)
    return out


def parse_window(spec: str | None, first_ns: int, last_ns: int) -> tuple[int, int]:
    """`120:165` is seconds from the first logged transition; `mono_ns=A:B` is raw. Either end may
    be empty, meaning the start or end of the log."""
    if spec is None:
        return first_ns, last_ns
    raw = spec.startswith("mono_ns=")
    body = spec[len("mono_ns=") :] if raw else spec
    if ":" not in body:
        raise SystemExit(f"--window needs a colon: got {spec!r}")
    lo_s, hi_s = body.split(":", 1)

    def to_ns(text: str, default: int) -> int:
        if text.strip() == "":
            return default
        return int(text) if raw else first_ns + int(round(float(text) * NS_PER_S))

    lo, hi = to_ns(lo_s, first_ns), to_ns(hi_s, last_ns)
    if hi <= lo:
        raise SystemExit(f"--window is empty or reversed: {spec!r}")
    return lo, hi


def clip(intervals: list[Interval], lo: int, hi: int, log_end_ns: int) -> list[Interval]:
    """Keeps the part of each interval inside the window. An interval straddling the edge is cut,
    not dropped: a collapse that was already under way when the window opened was still happening
    inside it, and dropping it would undercount exactly the long events this is looking for."""
    out: list[Interval] = []
    for iv in intervals:
        end = iv.end_ns if iv.end_ns is not None else log_end_ns
        start, stop = max(iv.start_ns, lo), min(end, hi)
        if stop > start:
            out.append(Interval(iv.motor, start, stop))
    return out


def analyse(path: Path, args) -> dict:
    transitions, first_ns, last_ns = parse_log(path)
    if not transitions:
        return {"file": str(path), "transitions": 0}

    lo, hi = parse_window(args.window, first_ns, last_ns)
    raw = clip(intervals_for_bit(transitions, args.bit, last_ns), lo, hi, last_ns)
    ivs = merge_per_motor(raw, hi)

    sweep = {}
    for n in range(1, args.max_motors + 1):
        eps = [e for e in episodes(ivs, n, hi) if e.duration_ms >= args.min_ms]
        sweep[n] = eps

    chosen = sweep.get(args.min_motors, [])
    per_motor = {}
    for iv in ivs:
        d = per_motor.setdefault(iv.motor, {"count": 0, "total_ms": 0.0, "longest_ms": 0.0})
        ms = iv.duration_ns(hi) / NS_PER_MS
        d["count"] += 1
        d["total_ms"] += ms
        d["longest_ms"] = max(d["longest_ms"], ms)

    return {
        "file": str(path),
        # Everything needed to run this again. The window is the part that went missing on
        # 2026-09-14, so it is recorded three ways rather than one.
        "rule": {
            "bit": f"0x{args.bit:02x}",
            "min_motors": args.min_motors,
            "min_ms": args.min_ms,
            "same_tick_grace_ms": TICK_NS / NS_PER_MS,
            "source": "README_DETAILS_EN.md: bit0 on every motor in the same tick is the supply",
        },
        "window": {
            "spec": args.window or "(whole log)",
            "start_mono_ns": lo,
            "end_mono_ns": hi,
            "start_s_into_log": (lo - first_ns) / NS_PER_S,
            "end_s_into_log": (hi - first_ns) / NS_PER_S,
            "length_s": (hi - lo) / NS_PER_S,
        },
        "log": {
            "transitions": len(transitions),
            "first_mono_ns": first_ns,
            "last_mono_ns": last_ns,
            "length_s": (last_ns - first_ns) / NS_PER_S,
        },
        "episodes": [
            {
                "start_mono_ns": e.start_ns,
                "s_into_log": (e.start_ns - first_ns) / NS_PER_S,
                "duration_ms": round(e.duration_ms, 1),
                "motors": sorted(e.motors),
            }
            for e in chosen
        ],
        # `episodes` is not monotonic in `min_motors` and is not supposed to be: a loose threshold
        # can fuse neighbours that a strict one reports separately. `total_ms` *is* monotonic --
        # requiring more motors can only shrink the time that qualifies -- so a run where it rises
        # is a bug in this file, not a finding about the rig. It is printed for that reason.
        "sweep_min_motors": {
            str(n): {
                "episodes": len(eps),
                "total_ms": round(sum(e.duration_ms for e in eps), 1),
                "durations_ms": [round(e.duration_ms, 1) for e in eps],
            }
            for n, eps in sweep.items()
        },
        "per_motor": {
            str(k): {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sorted(per_motor.items())
        },
    }


def report(res: dict, args) -> None:
    print(f"\n{res['file']}")
    if res.get("transitions") == 0:
        print("  no servo_error lines -- nothing was logged, or the log is from an older daemon")
        return

    w, lg = res["window"], res["log"]
    print(
        f"  window   {w['spec']}  =  {w['start_s_into_log']:.1f}..{w['end_s_into_log']:.1f} s "
        f"into the log ({w['length_s']:.1f} s of {lg['length_s']:.1f} s)"
    )
    print(f"           mono_ns={w['start_mono_ns']}:{w['end_mono_ns']}")
    r = res["rule"]
    print(
        f"  rule     bit {r['bit']}, >= {r['min_motors']} motors at once, >= {r['min_ms']:g} ms, "
        f"same-tick grace {r['same_tick_grace_ms']:g} ms"
    )

    print(f"\n  how many episodes survive, by how many motors are required (>= {args.min_ms:g} ms):")
    print(f"    {'motors':>8}{'episodes':>10}{'total ms':>11}   durations (ms)")
    prev_total = None
    for n, s in res["sweep_min_motors"].items():
        ds = ", ".join(f"{d:g}" for d in s["durations_ms"][:8])
        if len(s["durations_ms"]) > 8:
            ds += f", ... (+{len(s['durations_ms']) - 8})"
        mark = "  <- the documented rule" if int(n) == r["min_motors"] else ""
        if prev_total is not None and s["total_ms"] > prev_total:
            mark += "  !! total rose with a stricter rule -- bug in this script, not the rig"
        prev_total = s["total_ms"]
        print(f"    {n:>8}{s['episodes']:>10}{s['total_ms']:>11.1f}   {ds or '--'}{mark}")

    print(f"\n  per motor, every stretch with bit {r['bit']} up inside the window:")
    print(f"    {'motor':>7}{'count':>8}{'total ms':>11}{'longest ms':>13}")
    for motor, d in res["per_motor"].items():
        print(f"    {motor:>7}{int(d['count']):>8}{d['total_ms']:>11.1f}{d['longest_ms']:>13.1f}")

    eps = res["episodes"]
    print(f"\n  episodes under the documented rule: {len(eps)}")
    for e in eps:
        print(
            f"    at {e['s_into_log']:8.1f} s  {e['duration_ms']:8.1f} ms  "
            f"motors {','.join(str(m) for m in e['motors'])}"
        )
    if not eps and any(int(s["episodes"]) for s in res["sweep_min_motors"].values()):
        print(
            "    none. The bit was raised in this window, but never by enough motors at once to\n"
            "    meet the definition -- read the sweep above before calling it a supply collapse."
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logs", nargs="+", type=Path, help="Daemon logs to count.")
    p.add_argument(
        "--window",
        help="Seconds from the first transition (`120:165`), or raw (`mono_ns=A:B`). Either end "
        "may be empty. Defaults to the whole log -- and whatever is used is printed and saved.",
    )
    p.add_argument(
        "--bit",
        type=lambda s: int(s, 0),
        default=0x01,
        help="Error bit to count. Default 0x01, which secondary sources call under-voltage; that "
        "mapping has never been checked against a protocol document (see README_DETAILS).",
    )
    p.add_argument(
        "--min-motors",
        type=int,
        default=6,
        help="How many motors must hold the bit at once. Default 6 = the documented definition of "
        "a supply collapse. Lower it to see what is there, but then say so next to the number.",
    )
    p.add_argument("--max-motors", type=int, default=6, help="Highest row of the sweep table.")
    p.add_argument(
        "--min-ms", type=float, default=100.0, help="Ignore episodes shorter than this. Default 100."
    )
    p.add_argument("--json", type=Path, help="Write the full result, window and rule included.")
    args = p.parse_args()

    results = [analyse(path, args) for path in args.logs]
    for res in results:
        report(res, args)
    print()

    if args.json:
        payload = results[0] if len(results) == 1 else {"runs": results}
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
