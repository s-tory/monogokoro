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
A quick bring-up/smoke-test tool for the SO101 + `rust/so101_impedance_ctrl` daemon, built on
`lerobot.robots.so101_impedance_follower.checker.SO101ImpedanceChecker`. No calibration file,
camera, or dataset is needed -- positions are raw encoder ticks (0-4095).

Two modes:

**Monitor mode (default)** -- read-only. This script never writes to the input region, so the
daemon's watchdog zeroes PWM within its `--watchdog-timeout-ms` (default 75ms): every motor stays
torque-limp. Move the joints by hand and confirm position/current readouts track your movement and
the fault flags are clean once telemetry starts flowing.

**Hold mode** (`--hold`) -- captures the current position once at startup and repeatedly commands
every motor back to that fixed target under impedance control (`--k`/`--d`). Try gently pushing a
joint: it should spring back toward the captured position -- this is the actual impedance-control
behavior check, not just telemetry plumbing.

Examples:

    # Attach to an already-running daemon and just watch telemetry (read-only, safe to move by hand):
    python examples/check_so101_impedance.py --shm-name so101_impedance

    # Also launch the daemon, then test that a gentle push on any joint springs back:
    python examples/check_so101_impedance.py \\
        --shm-name so101_impedance_check --port /dev/ttyACM0 --start-daemon \\
        --hold --k 15 --d 1.0
"""

import argparse
import json
import logging
import time

from lerobot.robots.so101_impedance_follower.checker import (
    MOTOR_NAMES,
    SO101ImpedanceChecker,
    format_state_table,
)


def _gain_arg(value: str) -> float | dict[str, float]:
    """Parses `--k`/`--d` as either one gain for every joint or one gain per joint.

    Per-joint gains are the normal case rather than a refinement: `shoulder_lift` and `elbow_flex`
    hold the arm's weight, so they need roughly an order of magnitude more K than `wrist_roll`,
    which holds nothing. A single scalar therefore has to be either too soft for the shoulder or
    needlessly stiff everywhere else.
    """
    parts = value.split(",")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != len(MOTOR_NAMES):
        raise argparse.ArgumentTypeError(
            f"expected 1 or {len(MOTOR_NAMES)} comma-separated gains "
            f"({','.join(MOTOR_NAMES)}), got {len(parts)}"
        )
    return {motor: float(p) for motor, p in zip(MOTOR_NAMES, parts, strict=True)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--shm-name", default="so101_impedance_check")
    parser.add_argument("--port", default=None, help="Serial port, required with --start-daemon.")
    parser.add_argument(
        "--start-daemon", action="store_true", help="Also launch the Rust binary; terminate it on exit."
    )
    parser.add_argument("--binary-path", default="so101_impedance_ctrl")
    parser.add_argument("--cpu-core", type=int, default=None, help="Forwarded to the daemon if starting it.")
    parser.add_argument("--priority", type=int, default=None, help="Forwarded to the daemon if starting it.")
    parser.add_argument("--loop-hz", type=float, default=None, help="Forwarded to the daemon if starting it.")
    parser.add_argument(
        "--leader-port",
        default=None,
        help="Leader arm's serial port, to include its gripper in the loop. Forwarded to the "
        "daemon if starting it. On its own this only measures the cost of the second bus -- no "
        "force is rendered until --force-feedback-gain is nonzero.",
    )
    parser.add_argument(
        "--force-feedback-gain",
        type=float,
        default=None,
        help="Leader duty per count of follower gripper error. Signed: if the trigger ASSISTS "
        "your squeeze rather than resisting it, negate this and stop -- that polarity is positive "
        "feedback through your own hand.",
    )

    parser.add_argument(
        "--calibration",
        default=None,
        metavar="PATH",
        help="LeRobot calibration JSON to push into the servos before testing. Strongly "
        "recommended: without its homing offsets a joint's travel can straddle the 4095/0 encoder "
        "wrap, and any closed-loop test then slams that joint.",
    )
    parser.add_argument(
        "--probe-direction",
        metavar="MOTOR",
        default=None,
        help="Measure which way positive PWM moves one joint (e.g. --probe-direction wrist_roll), "
        "then exit. Safe: one motor, bounded duty, sub-second, auto-abort.",
    )
    parser.add_argument("--hold", action="store_true", help="Actively hold the startup position (see above).")
    parser.add_argument(
        "--k",
        type=_gain_arg,
        default=20.0,
        help="Spring gain for --hold. One value for every joint, or 6 comma-separated values in "
        f"the order {','.join(MOTOR_NAMES)} -- the gravity-loaded joints need far more K than the "
        "wrist, and the gripper wants the least of all.",
    )
    parser.add_argument(
        "--d", type=_gain_arg, default=1.0, help="Damper gain for --hold; same formats as --k."
    )

    parser.add_argument(
        "--interval", type=float, default=0.05, help="Telemetry refresh / command period (s)."
    )
    parser.add_argument(
        "--seconds", type=float, default=0.0, help="Run for this long, or forever if 0 (Ctrl+C to stop)."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    daemon_args = {}
    if args.cpu_core is not None:
        daemon_args["cpu-core"] = args.cpu_core
    if args.priority is not None:
        daemon_args["priority"] = args.priority
    if args.loop_hz is not None:
        daemon_args["loop-hz"] = args.loop_hz
    if args.leader_port is not None:
        daemon_args["leader-port"] = args.leader_port
    if args.force_feedback_gain is not None:
        daemon_args["force-feedback-gain"] = args.force_feedback_gain

    with SO101ImpedanceChecker(
        shm_name=args.shm_name,
        port=args.port,
        start_daemon=args.start_daemon,
        binary_path=args.binary_path,
        daemon_args=daemon_args,
    ) as checker:
        if args.calibration:
            with open(args.calibration) as f:
                checker.write_calibration(json.load(f))
        elif args.hold or args.probe_direction:
            print(
                "WARNING: no --calibration given. Without homing offsets a joint whose travel\n"
                "crosses the 4095/0 encoder wrap reports a full-scale position jump mid-motion,\n"
                "which the impedance law turns into a saturated command. Pass --calibration.\n"
            )

        if args.probe_direction:
            checker.set_pwm_mode()
            checker.enable_torque()
            print(f"Probing {args.probe_direction} -- keep clear of the arm.\n")
            r = checker.probe_direction(args.probe_direction)
            print(f"  commanded {r['commanded_pwm']:.0f} PWM toward +position")
            print(f"  {r['start']:.0f} -> {r['end']:.0f} ticks (delta {r['delta']:+.0f})\n")
            if not r["moved"]:
                print("  INCONCLUSIVE: the joint barely moved. It may be against a stop, or the")
                print("  duty was too low to overcome friction/gravity -- retry with the joint free,")
                print("  or raise --k on a hold test instead.")
            elif r["inverted"]:
                print("  INVERTED: positive PWM drives position DOWN. The daemon must run with the")
                print("  opposite --invert-pwm setting to what it is using now, otherwise the")
                print("  impedance law is positive feedback and the joint runs to a stop.")
            else:
                print("  CORRECT: positive PWM drives position UP. The daemon's current")
                print("  --invert-pwm setting is right; a runaway is a gain problem, not a sign one.")
            return

        hold_targets = None
        if args.hold:
            checker.set_pwm_mode()
            checker.enable_torque()
            initial_state = checker.read_state()
            hold_targets = {motor: s["present_pos"] for motor, s in initial_state.items()}
            print(f"Holding at: { {m: round(p, 1) for m, p in hold_targets.items()} }\n")
        else:
            print("Monitor mode: no commands are sent, motors are torque-limp. Move joints by hand.\n")
            if checker.read_leader() is not None:
                print(
                    "A leader gripper is attached. Force feedback is gated on fresh input, so in\n"
                    "monitor mode the trigger stays slack no matter what --force-feedback-gain\n"
                    "says -- use this to confirm the LEADER row tracks your trigger, then switch\n"
                    "to --hold to feel actual force.\n"
                )

        start = time.perf_counter()
        try:
            while args.seconds <= 0 or (time.perf_counter() - start) < args.seconds:
                tick_start = time.perf_counter()

                if hold_targets is not None:
                    checker.move_to(hold_targets, k=args.k, d=args.d)

                state = checker.read_state()
                print("\x1b[2J\x1b[H", end="")  # clear terminal for a live-updating view
                print(
                    format_state_table(
                        state,
                        checker.describe_faults(),
                        targets=hold_targets,
                        leader=checker.read_leader(),
                    )
                )
                print(checker.describe_cerebellum())
                if hold_targets is not None:
                    print(
                        "\nerr = target - pos (ticks, 11.4/deg)."
                        "\n  err settled, vel ~0    -> holding. pwm IS the duty gravity+friction"
                        "\n                            demand at this pose, so size the gain:"
                        "\n                            K_new = K * err / err_wanted."
                        "\n  err growing, pwm ~0    -> too soft. Raise --k for that joint."
                        "\n  err growing, pwm %max  -> drive direction inverted; the daemon needs"
                        "\n                            the opposite --invert-pwm (K cannot fix it)."
                        "\n  err ~0 but buzzing     -> D too high for the velocity noise floor;"
                        "\n                            lower --d or raise --vel-filter-window."
                        "\n  ff climbing, pwm falling, err shrinking at unchanged K"
                        "\n                         -> the cerebellum is taking the load over."
                        "\n                            That is the whole claim; if err only"
                        "\n                            improves when you raise K, it is not."
                    )

                elapsed = time.perf_counter() - tick_start
                if elapsed < args.interval:
                    time.sleep(args.interval - elapsed)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
