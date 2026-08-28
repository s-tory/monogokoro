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
Measures whether the cerebellum's feedforward actually cancels a standing load, against a control.

The claim is specific and so is the test: at an *unchanged* K, does the droop shrink? A feedback
law droops by `holding_duty / K` under a standing load, and the only way a reflex alone can shrink
that is to raise K -- which trades away the compliance the daemon exists to provide. So "the error
got smaller" means nothing on its own; it has to get smaller at the same gains, in the same pose,
against a run where the cerebellum was off.

The same pose is the hard part, and it is why this exists rather than `--hold` on
`check_so101_impedance.py`. That captures whatever pose the arm happens to be in when it starts,
which is never twice the same after the arm has been released and has sagged. Here the pose is
captured once to a file and replayed, so the two runs share a target exactly.

    # 0. Once, with the daemon running and the arm held by hand where it carries its own weight:
    python examples/measure_so101_cerebellum.py capture --pose-file pose.json

    # 1. Baseline: the daemon started with --cerebellum-backend off
    python examples/measure_so101_cerebellum.py hold --pose-file pose.json --label baseline \
        --seconds 180 --out baseline.csv

    # 2. Learning: the daemon restarted with --cerebellum-backend gpu and a FRESH weights path,
    #    because weights persist and a warm start is not a comparison.
    python examples/measure_so101_cerebellum.py hold --pose-file pose.json --label cerebellum \
        --seconds 180 --out learn.csv

    # 3.
    python examples/measure_so101_cerebellum.py compare baseline.summary.json learn.summary.json

Each run eases the target from wherever the arm is resting up to the stored pose, holds, then eases
it back down before releasing, so the arm is never dropped, never snapped at, and both runs see an
identical approach. Gains default to the shipped per-joint values: a single scalar K is either too
soft for the shoulder or stiff enough to make `wrist_roll` buzz at full duty, which was measured
rather than guessed.

Read the `pwm` and `ff` columns together. As the feedforward learns a joint's standing load, `ff`
climbs toward the duty `pwm` was carrying alone, `pwm` falls toward zero, and `err` shrinks with K
untouched. If `err` only improves when K goes up, the feedforward is not doing anything.
"""

import argparse
import contextlib
import json
import math
import signal
import statistics
import sys
import time

from lerobot.robots.so101_impedance_follower.checker import (
    MOTOR_NAMES,
    SO101ImpedanceChecker,
    format_state_table,
)
from lerobot.robots.so101_impedance_follower.config_so101_impedance_follower import (
    SO101ImpedanceFollowerConfig,
)

# The shipped per-joint gains, not a scalar. A single K/D is either too soft for the shoulder or
# needlessly stiff everywhere else -- and an over-large D on a joint that holds nothing (wrist_roll)
# turns the velocity noise floor straight into a full-duty limit cycle. Measured here at K=20/D=1
# flat: wrist_roll swung +/-1000 PWM at 197 counts/s. Both runs of an A/B must use these.
_CFG = SO101ImpedanceFollowerConfig(shm_name="so101_impedance")
DEFAULT_K = dict(zip(MOTOR_NAMES, _CFG.default_k, strict=True))
DEFAULT_D = dict(zip(MOTOR_NAMES, _CFG.default_d, strict=True))


def _gain(value: str):
    parts = value.split(",")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != len(MOTOR_NAMES):
        raise argparse.ArgumentTypeError(f"expected 1 or {len(MOTOR_NAMES)} gains")
    return {m: float(p) for m, p in zip(MOTOR_NAMES, parts, strict=True)}


def _as_dict(g):
    return g if isinstance(g, dict) else dict.fromkeys(MOTOR_NAMES, g)


def cmd_capture(args) -> None:
    with SO101ImpedanceChecker(shm_name=args.shm_name) as checker:
        # Read-only: the watchdog keeps PWM at zero, so the arm stays limp while it is positioned.
        samples = []
        for _ in range(20):
            samples.append({m: s["present_pos"] for m, s in checker.read_state().items()})
            time.sleep(0.02)
        pose = {m: statistics.median(s[m] for s in samples) for m in MOTOR_NAMES}
        spread = {m: max(s[m] for s in samples) - min(s[m] for s in samples) for m in MOTOR_NAMES}
    with open(args.pose_file, "w") as f:
        json.dump(pose, f, indent=2)
    print(f"captured -> {args.pose_file}")
    for m in MOTOR_NAMES:
        note = "  <- moving, hold it still" if spread[m] > 8 else ""
        print(f"  {m:<14}{pose[m]:>9.1f} ticks  (spread {spread[m]:.0f}){note}")


def cmd_hold(args) -> None:
    with open(args.pose_file) as f:
        pose = {m: float(v) for m, v in json.load(f).items()}

    k, d = _as_dict(args.k), _as_dict(args.d)
    rows, stop = [], False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    with SO101ImpedanceChecker(shm_name=args.shm_name) as checker:
        if args.calibration:
            with open(args.calibration) as f:
                checker.write_calibration(json.load(f))
        checker.set_pwm_mode()
        checker.enable_torque()

        start = {m: s["present_pos"] for m, s in checker.read_state().items()}
        drift = max(abs(pose[m] - start[m]) for m in MOTOR_NAMES)
        print(f"resting pose is {drift:.0f} ticks from the stored target; ramping over {args.ramp:.0f}s")
        print(checker.describe_cerebellum())

        t0 = time.monotonic()
        total = args.ramp + args.seconds
        try:
            while not stop:
                t = time.monotonic() - t0
                if t >= total:
                    break
                # Ramp in, then hold. A step onto a compliant arm saturates PWM; a ramp does not.
                frac = min(1.0, t / args.ramp) if args.ramp > 0 else 1.0
                target = {m: start[m] + (pose[m] - start[m]) * frac for m in MOTOR_NAMES}
                checker.move_to(target, k=k, d=d)

                state = checker.read_state()
                row = {"t": t, "phase": "ramp" if frac < 1.0 else "hold"}
                for m in MOTOR_NAMES:
                    s = state[m]
                    row[f"{m}.target"] = target[m]
                    row[f"{m}.pos"] = s["present_pos"]
                    row[f"{m}.err"] = target[m] - s["present_pos"]
                    row[f"{m}.vel"] = s["present_vel"]
                    row[f"{m}.pwm"] = s["pwm_cmd"]
                    row[f"{m}.ff"] = s["ff_pwm"]
                    row[f"{m}.cur"] = s["present_current_avg"]
                rows.append(row)

                if args.print_every and len(rows) % max(1, int(args.print_every / args.interval)) == 0:
                    print(f"\n[{args.label}] t={t:6.1f}s  {checker.describe_cerebellum()}")
                    print(format_state_table(state, checker.describe_faults(), targets=target))
                time.sleep(args.interval)
        finally:
            # Never just stop writing: the watchdog would zero PWM and the arm would fall from the
            # pose it was holding. Walk the target back down to where it started, then release.
            print("\nlowering back to the resting pose...")
            held = {m: rows[-1][f"{m}.target"] for m in MOTOR_NAMES} if rows else start
            tr = time.monotonic()
            while (e := time.monotonic() - tr) < args.ramp:
                frac = e / args.ramp
                checker.move_to({m: held[m] + (start[m] - held[m]) * frac for m in MOTOR_NAMES}, k=k, d=d)
                time.sleep(args.interval)
            checker.move_to({}, k=0.0, d=0.0)

    if not rows:
        sys.exit("no samples recorded")

    with open(args.out, "w") as f:
        cols = list(rows[0])
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")

    summary = _summarise(rows, args)
    with open(args.out.replace(".csv", "") + ".summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _print_summary(summary)
    print(f"\nsamples -> {args.out}")


def _summarise(rows: list[dict], args) -> dict:
    hold = [r for r in rows if r["phase"] == "hold"]
    if not hold:
        sys.exit("the run never left the ramp -- raise --seconds")
    t_end = hold[-1]["t"]
    window = [r for r in hold if r["t"] >= t_end - args.window]
    k = _as_dict(args.k)
    per_motor = {}
    for m in MOTOR_NAMES:
        err = [r[f"{m}.err"] for r in window]
        per_motor[m] = {
            "k": k[m],
            "err_mean": statistics.fmean(err),
            "abs_err_mean": statistics.fmean(abs(e) for e in err),
            "err_sd": statistics.stdev(err) if len(err) > 1 else 0.0,
            "pwm_mean": statistics.fmean(r[f"{m}.pwm"] for r in window),
            "ff_mean": statistics.fmean(r[f"{m}.ff"] for r in window),
            "cur_mean": statistics.fmean(r[f"{m}.cur"] for r in window),
        }
    return {
        "label": args.label,
        "pose_file": args.pose_file,
        "seconds": args.seconds,
        "window_s": args.window,
        "samples_in_window": len(window),
        "motors": per_motor,
    }


def _print_summary(s: dict) -> None:
    print(
        f"\n=== {s['label']}: steady state over the last {s['window_s']:.0f}s "
        f"({s['samples_in_window']} samples) ==="
    )
    print(f"{'motor':<14}{'K':>6}{'err':>10}{'|err|':>9}{'sd':>7}{'pwm':>9}{'ff':>9}{'cur':>8}")
    print("-" * 72)
    for m, v in s["motors"].items():
        print(
            f"{m:<14}{v['k']:>6.0f}{v['err_mean']:>10.2f}{v['abs_err_mean']:>9.2f}"
            f"{v['err_sd']:>7.2f}{v['pwm_mean']:>9.1f}{v['ff_mean']:>9.1f}{v['cur_mean']:>8.1f}"
        )


def cmd_compare(args) -> None:
    with open(args.baseline) as f:
        a = json.load(f)
    with open(args.learned) as f:
        b = json.load(f)
    _print_summary(a)
    _print_summary(b)

    print(f"\n=== {b['label']} vs {a['label']} ===")
    print(
        f"{'motor':<14}{'K same?':>9}{'|err| A':>9}{'|err| B':>9}{'change':>9}{'ff B':>8}{'pwm A':>8}{'ff/pwmA':>9}"
    )
    print("-" * 76)
    verdicts = []
    for m in MOTOR_NAMES:
        va, vb = a["motors"][m], b["motors"][m]
        same_k = "yes" if math.isclose(va["k"], vb["k"]) else "NO"
        ea, eb = va["abs_err_mean"], vb["abs_err_mean"]
        change = (eb - ea) / ea * 100 if ea else float("nan")
        share = vb["ff_mean"] / va["pwm_mean"] if abs(va["pwm_mean"]) > 1e-6 else float("nan")
        print(
            f"{m:<14}{same_k:>9}{ea:>9.2f}{eb:>9.2f}{change:>8.0f}%{vb['ff_mean']:>8.1f}"
            f"{va['pwm_mean']:>8.1f}{share:>9.2f}"
        )
        # Only joints that actually carried a standing load can show anything.
        if abs(va["pwm_mean"]) >= args.min_duty and same_k == "yes":
            verdicts.append((m, change))

    print()
    if not verdicts:
        print(
            f"INCONCLUSIVE: no joint held more than {args.min_duty:.0f} duty in the baseline, so "
            "there was no droop to cancel. Pick a more gravity-loaded pose."
        )
        return
    for m, change in verdicts:
        if change <= -args.threshold:
            print(f"  {m:<14} droop {abs(change):.0f}% smaller at the same K -- the feedforward worked.")
        elif change >= args.threshold:
            print(f"  {m:<14} droop {change:.0f}% LARGER -- the feedforward is fighting the reflex.")
        else:
            print(f"  {m:<14} droop within +/-{args.threshold:.0f}% -- no measurable effect.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="Record the target pose while the arm is limp.")
    c.add_argument("--shm-name", default="so101_impedance")
    c.add_argument("--pose-file", default="pose.json")
    c.set_defaults(func=cmd_capture)

    h = sub.add_parser("hold", help="Hold the stored pose and log the droop.")
    h.add_argument("--shm-name", default="so101_impedance")
    h.add_argument("--pose-file", default="pose.json")
    h.add_argument("--label", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--calibration", default=None)
    h.add_argument(
        "--k",
        type=_gain,
        default=DEFAULT_K,
        help=f"Per-joint spring gains; defaults to the shipped {tuple(_CFG.default_k)}.",
    )
    h.add_argument(
        "--d",
        type=_gain,
        default=DEFAULT_D,
        help=f"Per-joint damper gains; defaults to the shipped {tuple(_CFG.default_d)}. Raising "
        "this past ~K/40 buzzes the joint rather than damping it.",
    )
    h.add_argument("--ramp", type=float, default=5.0, help="Seconds to ease onto (and off) the pose.")
    h.add_argument("--seconds", type=float, default=180.0, help="Hold time after the ramp.")
    h.add_argument("--window", type=float, default=30.0, help="Trailing window the summary averages.")
    h.add_argument("--interval", type=float, default=0.02)
    h.add_argument(
        "--print-every", type=float, default=10.0, help="Seconds between printed tables; 0 to silence."
    )
    h.set_defaults(func=cmd_hold)

    d = sub.add_parser("compare", help="Baseline vs cerebellum, from two summary JSONs.")
    d.add_argument("baseline")
    d.add_argument("learned")
    d.add_argument("--threshold", type=float, default=15.0, help="Percent change called a result.")
    d.add_argument("--min-duty", type=float, default=20.0, help="Baseline duty a joint needs to count.")
    d.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
