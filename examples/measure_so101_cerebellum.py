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

    # `capture` switches the servos into the control loop's own PWM mode before sampling and
    # records that in the file, because `Present_Position` means a different number under
    # POSITION mode and a pose replayed in the wrong frame is a saturated drive at the stops.
    # A pose file written before this was recorded is refused rather than guessed at; re-capture.

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
import random
import select
import signal
import statistics
import sys
import time
from datetime import UTC, datetime

from lerobot.robots.so101_impedance_follower.checker import (
    MOTOR_NAMES,
    SO101ImpedanceChecker,
    format_state_table,
)
from lerobot.robots.so101_impedance_follower.config_so101_impedance_follower import (
    SO101ImpedanceFollowerConfig,
)
from lerobot.robots.so101_impedance_follower.shm_client import (
    describe_fault_flags,
    describe_servo_error,
)
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# The shipped per-joint gains, not a scalar. A single K/D is either too soft for the shoulder or
# needlessly stiff everywhere else -- and an over-large D on a joint that holds nothing (wrist_roll)
# turns the velocity noise floor straight into a full-duty limit cycle. Measured here at K=20/D=1
# flat: wrist_roll swung +/-1000 PWM at 197 counts/s. Both runs of an A/B must use these.
_CFG = SO101ImpedanceFollowerConfig(shm_name="so101_impedance")
DEFAULT_K = dict(zip(MOTOR_NAMES, _CFG.default_k, strict=True))
DEFAULT_D = dict(zip(MOTOR_NAMES, _CFG.default_d, strict=True))
# The joint the blind trial names its gains after: the one carrying the most gravity, so it is the
# one a hand judges first. Every other joint is scaled from it by the shipped ratio.
LIFT_JOINT = "shoulder_lift"


def _gain(value: str):
    parts = value.split(",")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != len(MOTOR_NAMES):
        raise argparse.ArgumentTypeError(f"expected 1 or {len(MOTOR_NAMES)} gains")
    return {m: float(p) for m, p in zip(MOTOR_NAMES, parts, strict=True)}


def _as_dict(g):
    return g if isinstance(g, dict) else dict.fromkeys(MOTOR_NAMES, g)


# The frame a pose file is written in, recorded inside the file itself.
#
# `Present_Position` means two different numbers depending on `Operating_Mode`: the raw encoder
# count under PWM (2), and that count minus `Homing_Offset` under POSITION (0). A pose captured in
# one and replayed in the other is wrong by the homing offset -- up to 1937 counts, 170 deg, on this
# arm -- and `cmd_hold` turns that straight into saturated PWM against the stops.
#
# This bit us on 2026-09-08 and the reason it had never bitten before is the whole lesson:
# `lerobot-calibrate` leaves all six servos in POSITION mode, while every `hold` leaves them in PWM.
# So capture-then-hold agreed on the frame in every session that had already run a hold, and
# disagreed exactly once -- on the first run after a calibration. A latent bug that is *masked by
# having run the tool before* cannot be found by running the tool again.
#
# So the frame is neither inferred nor assumed. `capture` puts the servos into the mode the control
# loop runs in and writes down which one that was; `hold` refuses a file that does not say.
POSE_FRAME = "pwm_raw"


# Where a pose's counts sit on the joint, as opposed to on the encoder.
#
# Under PWM the daemon works in raw encoder counts, and a joint's travel can straddle the 4095/0
# wrap in that frame: shoulder_lift here runs 3519 -> 4095 -> 0 -> 2219. A ramp that interpolates
# raw counts in a straight line therefore walks the target the long way round, through the part of
# the circle the joint cannot reach. On 2026-09-17 that drove shoulder_lift and elbow_flex into
# their folded stops at full duty for as long as the target stayed within half a turn (the servos
# raised 0x20), and then -- because the daemon takes the *shorter* way to any target -- the error
# flipped sign and the arm was thrown the other way. That was the "oscillation" of that morning, and
# very likely of the 3105-count approach on 2026-09-16 that was put down to speed.
#
# `Homing_Offset` exists precisely to move the travel off the wrap: `raw - offset` (mod 4096) is
# the position-mode count, in which the travel is one unbroken interval. So ramps interpolate
# there and convert back. The offsets are read from the calibration file rather than assumed; a
# run without one is refused, because the failure it prevents is silent until the arm hits a stop.
Travel = dict[str, tuple[float, float, float]]
_ENCODER = 4096.0


def load_travel(robot_id: str) -> Travel:
    path = HF_LEROBOT_CALIBRATION / "robots" / "so101_impedance_follower" / f"{robot_id}.json"
    if not path.is_file():
        raise SystemExit(
            f"no calibration at {path}: ramps need each joint's Homing_Offset to know which way "
            "round the encoder its travel runs. Pass --robot-id for the arm being driven."
        )
    with open(path) as f:
        calib = json.load(f)
    travel = {
        m: (float(calib[m]["homing_offset"]), float(calib[m]["range_min"]), float(calib[m]["range_max"]))
        for m in MOTOR_NAMES
    }
    print(f"travel from {path}: " + ", ".join(f"{m} offset {int(v[0])}" for m, v in travel.items()))
    return travel


def _along(motor: str, raw: float, travel: Travel) -> float:
    return (raw - travel[motor][0]) % _ENCODER


def interpolate(a: dict, b: dict, frac: float, travel: Travel) -> dict[str, float]:
    """The target `frac` of the way from `a` to `b`, moving along each joint's travel."""
    out = {}
    for m in MOTOR_NAMES:
        ca, cb = _along(m, a[m], travel), _along(m, b[m], travel)
        out[m] = (ca + (cb - ca) * frac + travel[m][0]) % _ENCODER
    return out


def along_error(motor: str, target: float, present: float, travel: Travel) -> float:
    """`target - present` measured along the travel, so a pose across the wrap does not read ~4096."""
    return _along(motor, target, travel) - _along(motor, present, travel)


def travel_distance(a: dict, b: dict, travel: Travel) -> float:
    """The largest distance any joint has to cover between `a` and `b`, along its travel."""
    return max(abs(_along(m, b[m], travel) - _along(m, a[m], travel)) for m in MOTOR_NAMES)


ROBOT_ID_HELP = (
    "Calibration to read Homing_Offset from, under the so101_impedance_follower calibration "
    "directory. Ramps move along each joint's travel, which only the offset locates."
)


def _load_pose(path: str) -> dict[str, float]:
    with open(path) as f:
        blob = json.load(f)
    if not isinstance(blob, dict) or "pose" not in blob:
        raise SystemExit(
            f"{path} has no frame recorded, so the ticks in it cannot be interpreted -- it was "
            "written before this file was versioned. Re-run `capture`; a pose is 20 samples and "
            "costs nothing, and a pose replayed in the wrong frame drives the arm into its stops."
        )
    if blob.get("frame") != POSE_FRAME:
        raise SystemExit(
            f"{path} was captured in frame {blob.get('frame')!r}, but the control loop runs in "
            f"{POSE_FRAME!r}. Re-run `capture`."
        )
    return {m: float(blob["pose"][m]) for m in MOTOR_NAMES}


def cmd_capture(args) -> None:
    with SO101ImpedanceChecker(shm_name=args.shm_name) as checker:
        # Into the loop's own mode *before* sampling, so there is only ever one frame in play.
        # This does not energise anything: the watchdog holds PWM at zero with no client writing,
        # so the arm stays limp and backdrivable exactly as it was.
        checker.set_pwm_mode()
        samples = []
        for _ in range(20):
            samples.append({m: s["present_pos"] for m, s in checker.read_state().items()})
            time.sleep(0.02)
        pose = {m: statistics.median(s[m] for s in samples) for m in MOTOR_NAMES}
        spread = {m: max(s[m] for s in samples) - min(s[m] for s in samples) for m in MOTOR_NAMES}
    with open(args.pose_file, "w") as f:
        json.dump(
            {
                "frame": POSE_FRAME,
                "captured": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                "pose": pose,
            },
            f,
            indent=2,
        )
    print(f"captured -> {args.pose_file}  (frame {POSE_FRAME})")
    for m in MOTOR_NAMES:
        note = "  <- moving, hold it still" if spread[m] > 8 else ""
        print(f"  {m:<14}{pose[m]:>9.1f} ticks  (spread {spread[m]:.0f}){note}")


def cmd_hold(args) -> None:
    pose = _load_pose(args.pose_file)
    # Approaching the same target from two sides is the only way to separate a joint that is
    # balanced from one that is stuck: static friction lets the arm stop anywhere inside a band, and
    # a single approach cannot say where in that band it landed. Ramping straight back down at the
    # end of a run (see the `finally` below) means two consecutive runs both start from the resting
    # pose, so the second approach has to happen inside one process.
    approach = _load_pose(args.approach_from) if args.approach_from else None
    travel = load_travel(args.robot_id)

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
        drift = travel_distance(start, pose, travel)
        print(f"resting pose is {drift:.0f} ticks from the stored target; ramping over {args.ramp:.0f}s")
        print(checker.describe_cerebellum())

        t0 = time.monotonic()
        # start -> approach -> settle -> pose -> hold. Without `--approach-from` the first two
        # stages have zero length and the timeline is the original one.
        t_settle_start = args.ramp if approach else 0.0
        t_settle_end = t_settle_start + (args.settle if approach else 0.0)
        t_hold_start = t_settle_end + (args.ramp if approach else 0.0)
        total = t_hold_start + args.seconds
        try:
            while not stop:
                t = time.monotonic() - t0
                if t >= total:
                    break

                # Ramp in, then hold. A step onto a compliant arm saturates PWM; a ramp does not.
                def _lerp(a, b, frac):
                    return interpolate(a, b, frac, travel)

                if approach is None:
                    frac = min(1.0, t / args.ramp) if args.ramp > 0 else 1.0
                    target = _lerp(start, pose, frac)
                    phase = "ramp" if frac < 1.0 else "hold"
                elif t < t_settle_start:
                    target = _lerp(start, approach, t / args.ramp if args.ramp > 0 else 1.0)
                    phase = "approach"
                elif t < t_settle_end:
                    target = dict(approach)
                    phase = "settle"
                elif t < t_hold_start:
                    frac = (t - t_settle_end) / args.ramp if args.ramp > 0 else 1.0
                    target = _lerp(approach, pose, min(1.0, frac))
                    phase = "ramp"
                else:
                    target = dict(pose)
                    phase = "hold"
                checker.move_to(target, k=k, d=d)

                state = checker.read_state()
                # `t` is monotonic and run-relative, so two CSVs cannot be ordered against each
                # other -- or against anything else that happened that day. The wall clock is what
                # lets a run be placed either side of a hardware fault discovered later.
                row = {
                    "wall": datetime.now(UTC).astimezone().isoformat(timespec="milliseconds"),
                    "t": t,
                    "phase": phase,
                }
                # Logged every sample, next to the duty it explains. An under-volted arm needs more
                # duty to hold the same pose, so a droop number without a concurrent rail reading
                # cannot be told apart from a stiff mechanism.
                supply = checker.read_supply()
                row["supply_v"] = supply["volts"] if supply else float("nan")
                row["case_temp_c"] = supply["temp_c"] if supply else float("nan")
                row["health_motor_id"] = supply["motor_id"] if supply else float("nan")
                # Logged raw so a finished run can be re-read for faults nobody was looking for at
                # the time. The watchdog bit is the one this rig can trip without anything else
                # showing it: PWM held at zero is indistinguishable from a gain that is simply too
                # soft if all you have is the duty columns.
                # Both from one read: the flag says a servo complained, the byte says which
                # protection. Sampled apart they can disagree by a tick.
                row["fault_flags"], row["servo_error"] = checker.fault_snapshot
                for m in MOTOR_NAMES:
                    s = state[m]
                    row[f"{m}.target"] = target[m]
                    row[f"{m}.pos"] = s["present_pos"]
                    row[f"{m}.err"] = along_error(m, target[m], s["present_pos"], travel)
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
                checker.move_to(interpolate(held, start, frac, travel), k=k, d=d)
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


def scaled_gains(k_value: float, joint: str | None = None) -> tuple[dict, dict]:
    """Gains for the whole arm, named by one number.

    With no `joint`, the number is what the lift joint gets and every other joint moves by the same
    factor: that sweeps the arm's overall compliance and keeps the ratios fixed. A flat K is either
    too soft at the shoulder or stiff enough at `wrist_roll` to turn its velocity noise into a
    full-duty limit cycle.

    With a `joint`, only that joint's K is set and the rest stay at the shipped values. That sweeps
    the *ratio*, which the scaled mode cannot: on 2026-09-16 the judge of the scaled sweep said the
    joints nearer the tip felt stiffer than the shoulder, and a single factor has no way to answer
    that. D follows K on the swept joint only, keeping that joint's own D/K.
    """
    if joint is None:
        f = k_value / DEFAULT_K[LIFT_JOINT]
        return ({m: DEFAULT_K[m] * f for m in MOTOR_NAMES}, {m: DEFAULT_D[m] * f for m in MOTOR_NAMES})
    k, d = dict(DEFAULT_K), dict(DEFAULT_D)
    k[joint] = k_value
    d[joint] = DEFAULT_D[joint] * k_value / DEFAULT_K[joint]
    return k, d


def blind_plan(candidates: list[float], repeats: int, seed: int, order: str = "shuffled") -> list[float]:
    """The trial order: every candidate once, then `repeats` of them again.

    Generated before a single verdict is heard and written out with the results, so the mapping can
    be checked afterwards rather than trusted.

    `shuffled` is the blind: knowing that the last one was the stiff one makes whatever follows
    feel softer, whichever gain it actually carries. `descending` gives that effect up on purpose,
    and buys something back -- walking one direction makes the *boundary* legible ("from here it is
    too soft"), needs fewer trials, and starts stiff, which matters when the arm has to be lifted
    out of limp before the first trial. The bias it admits has a known sign, so it can be carried
    in the record instead of being removed. **Whichever is used belongs next to the verdicts**, and
    is written to the result file for that reason.

    Back-to-back duplicates are rejected either way. The repeats exist to catch an unstable
    judgement, and a repeat the judge can *see* coming -- "that felt identical, it must be the same
    one" -- measures their memory instead. If no arrangement avoids it (two candidates, say), the
    best available order is returned rather than looping forever.
    """
    if order == "descending":
        # Repeats go at the end, taken from the middle of the range, where the boundary usually
        # sits and where a wobbling judgement costs the most.
        head = sorted(candidates, reverse=True)
        mid = head[len(head) // 2 :] + head[: len(head) // 2]
        tail = [k for k in mid if k != head[-1]][:repeats] if repeats else []
        return head + tail
    plan = list(candidates) + list(candidates[:repeats])
    rng = random.Random(seed)
    for _ in range(200):
        rng.shuffle(plan)
        if all(a != b for a, b in zip(plan, plan[1:], strict=False)):
            break
    return plan


def cmd_blind(args) -> None:
    """Holds one pose while K is switched underneath it, without ever releasing the arm.

    Why this exists rather than a loop of `hold` runs: `hold` walks the target back to wherever the
    arm was when it started and then releases, which is correct when that place is safe and wrong
    when it is the pose being measured. Running the trials as separate processes therefore left the
    arm unheld for the seconds between them -- process teardown, startup, and the next ramp -- and
    on 2026-09-16 it fell out of a horizontal reach during exactly that gap. Nothing electrical
    showed it: the servos logged no error byte, because a released arm is not a fault.

    So the sequence lives inside one process, the gains cross-fade instead of stepping, and the
    release at the end goes to `--rest-pose` (a pose captured with the arm somewhere it can be left
    limp) rather than to wherever this run happened to begin.

    The trial order is shuffled from `--seed` and written out with the verdicts, so the mapping
    exists before any verdict is heard and can be checked afterwards. Nothing about which gain is
    loaded is printed while the run is going: the point of the blind is that "20 is stiff" must not
    colour the next answer.
    """
    pose = _load_pose(args.pose_file)
    rest = _load_pose(args.rest_pose) if args.rest_pose else None
    travel = load_travel(args.robot_id)

    def _scaled(k_value: float) -> tuple[dict, dict]:
        return scaled_gains(k_value, args.joint)

    swept = args.joint or f"{LIFT_JOINT} (others scaled with it)"
    print(f"sweeping K on: {swept}")
    candidates = [float(x) for x in args.k_candidates.split(",")]
    plan = blind_plan(candidates, args.repeats, args.seed, args.order)

    trials: list[dict] = []
    stop = False

    def _sigint(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigint)

    with SO101ImpedanceChecker(shm_name=args.shm_name) as checker:
        checker.set_pwm_mode()
        checker.enable_torque()
        start = {m: s["present_pos"] for m, s in checker.read_state().items()}
        drift = travel_distance(start, pose, travel)
        # A fixed ramp is a fixed *time*, so the further the arm has to travel the faster it is
        # driven. On 2026-09-16 a run starting 474 ticks away rode on at 79 ticks/s and was fine; the
        # next one started 3105 ticks away, rode the same 6 s ramp at 517 ticks/s, and oscillated
        # until the arm came down, and that was put down to speed. It was probably not: 3105 is past
        # half a turn, and on 2026-09-17 the same oscillation came back at 100 ticks/s and went away
        # only when the ramp stopped interpolating across the encoder seam (see `interpolate`). The
        # 2026-09-16 start pose was not kept, so this is unverified. The cap stays: it costs nothing.
        ramp = args.ramp
        if args.max_approach_rate > 0:
            ramp = max(ramp, drift / args.max_approach_rate)
        print(
            f"resting pose is {drift:.0f} ticks from the stored target; "
            f"ramping over {ramp:.1f}s ({drift / ramp:.0f} ticks/s)"
        )
        if ramp > args.ramp * 1.01:
            print(
                f"  (stretched from {args.ramp:.0f}s to stay under --max-approach-rate "
                f"{args.max_approach_rate:.0f} ticks/s -- the arm is far from the pose; watch it move)"
            )
        if rest is None:
            print("no --rest-pose given: the arm will be released where this run started")

        # The lift onto the pose runs at the shipped gains. Raising a limp arm at the first
        # trial's K would fail outright whenever that trial is a soft one, and the arm would be
        # dragged along the table instead of lifted.
        k, d = dict(DEFAULT_K), dict(DEFAULT_D)

        def _hold_for(seconds: float, target_of, k_from, k_to, blend_s: float) -> None:
            """Commands the arm for `seconds`, cross-fading the gains over the first `blend_s`."""
            t0 = time.monotonic()
            while (t := time.monotonic() - t0) < seconds and not stop:
                f = min(1.0, t / blend_s) if blend_s > 0 else 1.0
                kk = {m: k_from[0][m] + (k_to[0][m] - k_from[0][m]) * f for m in MOTOR_NAMES}
                dd = {m: k_from[1][m] + (k_to[1][m] - k_from[1][m]) * f for m in MOTOR_NAMES}
                checker.move_to(target_of(t), k=kk, d=dd)
                time.sleep(args.interval)

        try:
            # Onto the pose at the first trial's gains.
            _hold_for(
                ramp,
                lambda t: interpolate(start, pose, min(1.0, t / ramp if ramp else 1.0), travel),
                (k, d),
                (k, d),
                0.0,
            )

            for i, k_value in enumerate(plan):
                if stop:
                    break
                nxt = _scaled(k_value)
                # Cross-fade into this trial's gains while still commanding the pose. A step change
                # in K on a loaded joint is a step change in duty, which reads as a twitch and
                # contaminates the very feel being judged.
                _hold_for(args.blend, lambda _t: dict(pose), (k, d), nxt, args.blend)
                k, d = nxt
                print(f"\n--- trial {i + 1} of {len(plan)} is loaded. Push the arm, then send a verdict.")
                sys.stdout.flush()

                verdict, t0 = None, time.monotonic()
                while verdict is None and not stop:
                    checker.move_to(dict(pose), k=k, d=d)
                    if select.select([sys.stdin], [], [], 0)[0]:
                        line = sys.stdin.readline()
                        verdict = line.strip() if line.strip() else "(blank)"
                    elif time.monotonic() - t0 > args.timeout:
                        verdict = "(timed out)"
                    time.sleep(args.interval)

                state = checker.read_state()
                trials.append(
                    {
                        "trial": i + 1,
                        "k": k_value,
                        "verdict": verdict,
                        "held_s": round(time.monotonic() - t0, 1),
                        "wall": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                        "pwm": {m: state[m]["pwm_cmd"] for m in MOTOR_NAMES},
                        "err": {
                            m: along_error(m, pose[m], state[m]["present_pos"], travel) for m in MOTOR_NAMES
                        },
                    }
                )
                print(f"    recorded: {verdict}")
        finally:
            # The release the separate-process version got wrong. Walk to a pose the arm can be
            # left in, not to wherever it started, and only then drop the gains.
            landing = rest if rest is not None else start
            print("\nlowering to the rest pose...")
            held = dict(pose)
            tr = time.monotonic()
            gap = travel_distance(held, landing, travel)
            down = max(args.ramp, gap / args.max_approach_rate) if args.max_approach_rate > 0 else args.ramp
            while (e := time.monotonic() - tr) < down:
                f = e / down
                checker.move_to(interpolate(held, landing, f, travel), k=k, d=d)
                time.sleep(args.interval)
            checker.move_to({}, k=0.0, d=0.0)

    out = {
        "seed": args.seed,
        "joint": swept,
        "k_all": "joint only, others at the shipped values" if args.joint else "scaled together",
        "shipped_k": DEFAULT_K,
        "candidates": candidates,
        "repeats": args.repeats,
        "order_mode": args.order,
        "pose_file": args.pose_file,
        "rest_pose": args.rest_pose,
        "order": plan,
        "trials": trials,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n=== unblinded ===")
    for t in trials:
        print(f"  trial {t['trial']}: {args.joint or LIFT_JOINT} K = {t['k']:<5} -> {t['verdict']}")
    # A gain that drew two different verdicts is the run telling you the judgement has not settled,
    # which is worth more than the gain itself: it says how much of the next result is noise.
    by_k: dict[float, list[str]] = {}
    for t in trials:
        by_k.setdefault(t["k"], []).append(t["verdict"])
    for k_value, vs in sorted(by_k.items()):
        if len(vs) > 1 and len(set(vs)) > 1:
            print(f"  !! K={k_value} drew different verdicts: {vs} -- the judgement is not stable yet")
    print(f"\n-> {args.out}")


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
    # The minimum matters more than the mean: the rail failure this catches is a sag of a few
    # hundred ms at a time, which barely moves an average but is exactly when the duty was being
    # measured. A `supply_min` well under `supply_mean` means this run's absolute numbers are not
    # comparable with anything else.
    volts = [r["supply_v"] for r in window if r["supply_v"] == r["supply_v"]]
    # OR-ed over the *whole* run, ramp included, and counted separately from the window the duties
    # come from. A single tripped sample is worth knowing about even when the window average looks
    # clean, because every fault here changes what the duty columns mean.
    faulted = [r["fault_flags"] for r in rows if r.get("fault_flags")]
    faults_seen = 0
    for f in faulted:
        faults_seen |= f
    # Kept separately from the fault mask because this one says *what the servo itself* objected
    # to. A run where every motor raises the voltage bit together is a supply that cannot carry the
    # arm -- which looks identical, in every other column, to a bus that drops replies.
    servo_errs = [r["servo_error"] for r in rows if r.get("servo_error")]
    servo_error_seen = 0
    for e in servo_errs:
        servo_error_seen |= e
    episodes = _servo_error_episodes(rows)
    return {
        "label": args.label,
        "pose_file": args.pose_file,
        "seconds": args.seconds,
        "window_s": args.window,
        "samples_in_window": len(window),
        "started_at": rows[0]["wall"],
        "ended_at": rows[-1]["wall"],
        "supply_mean_v": statistics.fmean(volts) if volts else None,
        "supply_min_v": min(volts) if volts else None,
        "faults_seen": faults_seen,
        "fault_samples": len(faulted),
        "faults": describe_fault_flags(faults_seen),
        "servo_error_seen": servo_error_seen,
        "servo_error_samples": len(servo_errs),
        "servo_errors": describe_servo_error(servo_error_seen),
        "servo_error_episodes": len(episodes),
        "servo_error_episode_ms": [round(ms) for ms in episodes],
        "servo_error_per_min": (
            round(len(episodes) / (rows[-1]["t"] - rows[0]["t"]) * 60, 2)
            if len(rows) > 1 and rows[-1]["t"] > rows[0]["t"]
            else None
        ),
        "motors": per_motor,
    }


def _servo_error_episodes(rows: list[dict]) -> list[float]:
    """Lengths in ms of each run of consecutive samples where a servo raised its error byte.

    Counted rather than left in the CSV because the *count* is the number this rig is actually
    steered by, and re-deriving it by hand after every run is how four near-identical analysis
    scripts got written on 2026-09-10. A supply that cannot carry the arm produces a handful of
    long episodes; the noise floor produces many one-sample ones, so the lengths are reported
    alongside the count and not averaged into it.

    **This counter is the coarse instrument, and the numbers below are not its own.** It walks the
    CSV, which `hold` samples at `--interval` (50 Hz by default) against the daemon's 400 Hz loop:
    it sees one tick in eight. That is ample for a rail collapse of several hundred ms and blind to
    the single-tick population, which it can neither reliably catch nor time -- a one-tick blip
    reaches it, if at all, as one sample. For those, read the daemon's own `servo_error` transition
    lines, which are logged per motor at the loop rate.

    Measured for scale, all from the 400 Hz log rather than from here. Bundled 5V4A adapter,
    2026-09-10: one 833 ms episode in 329 s at the defaults, and 8 over 100 ms in 77 s once
    `--pwm-max` was lowered to 700. Regulated 5 V 6 A+ adapter, 2026-09-14, same pose and gains:
    nothing at all over 5 ms in 233 s at the defaults, and one 390 ms episode still present at
    `--pwm-max 700`. The single-tick population survived the swap in both conditions and appears
    with the arm limp as well, so it is not the rail -- see `describe_servo_error`.

    Neither adapter was measured against the other in an alternating order; each is one run.
    """
    episodes: list[float] = []
    start: float | None = None
    prev: float | None = None
    for r in rows:
        t = r["t"]
        if r.get("servo_error"):
            if start is None:
                start = t
        elif start is not None:
            episodes.append((prev - start) * 1000 if prev is not None else 0.0)
            start = None
        prev = t
    if start is not None and prev is not None:
        episodes.append((prev - start) * 1000)
    return episodes


def _print_summary(s: dict) -> None:
    print(
        f"\n=== {s['label']}: steady state over the last {s['window_s']:.0f}s "
        f"({s['samples_in_window']} samples) ==="
    )
    if s.get("supply_mean_v") is None:
        print(
            "supply: not sampled -- this daemon predates the rail telemetry, absolute duties are unverifiable"
        )
    else:
        sag = s["supply_mean_v"] - s["supply_min_v"]
        note = "  <-- the rail sagged; absolute duties from this run are not comparable" if sag >= 0.3 else ""
        print(f"supply: mean {s['supply_mean_v']:.2f} V, min {s['supply_min_v']:.2f} V{note}")
    # `.get` because a summary written before this field existed is still readable by `compare`.
    if s.get("faults_seen"):
        print(f"faults: {s['fault_samples']} sample(s) with a flag set over the whole run")
        for f in s["faults"]:
            print(f"  - {f}")
    if s.get("servo_error_seen"):
        # Printed on its own line rather than folded into the fault list because it is the one
        # reading here that comes from the servos instead of from this software, and because the
        # supply reading two lines up cannot contradict it: `supply_*` is sampled once a second and
        # a dip lasts under a second. A run can print a healthy rail and this line together, and
        # when it does, this line is the one that saw what happened.
        print(
            f"servo protection: {s['servo_error_samples']} sample(s), error byte {s['servo_error_seen']:#04x}"
        )
        for e in s.get("servo_errors", []):
            print(f"  - {e}")
        n = s.get("servo_error_episodes")
        if n:
            lens = s.get("servo_error_episode_ms", [])
            long = [ms for ms in lens if ms >= 100]
            rate = s.get("servo_error_per_min")
            print(
                f"  {n} episode(s)"
                + (f", {rate}/min" if rate is not None else "")
                + (f" -- {len(long)} over 100 ms: {long} ms" if long else " -- all under 100 ms")
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
    h.add_argument(
        "--approach-from",
        default=None,
        help="Pose file to settle on first, so the target is reached from a chosen direction. "
        "Use it with a pose above and a pose below the target: the gap between where the arm stops "
        "in the two runs is the static-friction band, which a single approach cannot measure.",
    )
    h.add_argument(
        "--settle", type=float, default=10.0, help="Seconds held at --approach-from before moving on."
    )
    h.add_argument("--label", required=True)
    h.add_argument("--out", required=True)
    h.add_argument("--calibration", default=None)
    h.add_argument("--robot-id", default="my_awesome_follower_arm", help=ROBOT_ID_HELP)
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

    b = sub.add_parser(
        "blind",
        help="Switch K underneath a held pose and collect verdicts, without releasing the arm.",
    )
    b.add_argument("--shm-name", default="so101_impedance")
    b.add_argument("--robot-id", default="my_awesome_follower_arm", help=ROBOT_ID_HELP)
    b.add_argument("--pose-file", default="pose.json", help="The pose judged in every trial.")
    b.add_argument(
        "--rest-pose",
        default=None,
        help="Pose to walk to before releasing. Capture it with the arm somewhere it can be left "
        "limp. Without it the arm is released where the run began, which is the pose under test -- "
        "that is how an arm was dropped out of a horizontal reach on 2026-09-16.",
    )
    b.add_argument(
        "--k-candidates",
        default="20,12,8,5,3",
        help=f"Gains, comma separated. Without --joint they are for {LIFT_JOINT} and every other "
        "joint is scaled from the shipped set by the same factor; with --joint they are for that "
        "joint alone.",
    )
    b.add_argument(
        "--joint",
        choices=MOTOR_NAMES,
        default=None,
        help="Sweep this joint's K alone, leaving the others at the shipped values. Push that "
        "joint's segment, and say which joint is being judged before the run -- the blind is on "
        "the value, not on the joint.",
    )
    b.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many candidates to show a second time. A gain that draws two different verdicts "
        "says the judgement has not settled, which bounds how much of the result is noise.",
    )
    b.add_argument("--seed", type=int, default=0, help="Shuffles the order; recorded with the result.")
    b.add_argument(
        "--order",
        choices=("shuffled", "descending"),
        default="shuffled",
        help="`shuffled` is the blind. `descending` walks stiff to soft, which makes the boundary "
        "legible and lifts a limp arm safely, at the cost of a known bias: each trial follows a "
        "stiffer one and so feels softer than it is. Recorded with the verdicts either way.",
    )
    b.add_argument(
        "--ramp", type=float, default=5.0, help="Seconds to ease on at the start and off at the end."
    )
    b.add_argument(
        "--max-approach-rate",
        type=float,
        default=100.0,
        help="Ticks per second the approach may move at; the ramp is stretched past --ramp to stay "
        "under it: a fixed ramp drives a far-away pose proportionally faster. 0 disables the cap.",
    )
    b.add_argument(
        "--blend",
        type=float,
        default=2.0,
        help="Seconds to cross-fade between one trial's gains and the next. A step change in K on "
        "a loaded joint is a step change in duty, which is felt as a twitch.",
    )
    b.add_argument(
        "--timeout", type=float, default=300.0, help="Seconds to wait for a verdict before moving on."
    )
    b.add_argument("--interval", type=float, default=0.02)
    b.add_argument("--out", required=True, help="JSON: the order, the verdicts, and the seed.")
    b.set_defaults(func=cmd_blind)

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
