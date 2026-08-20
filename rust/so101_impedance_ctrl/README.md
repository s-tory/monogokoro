# so101_impedance_ctrl

A PREEMPT_RT impedance controller for all 6 of the SO101's Feetech STS3215 servos -- the 5 arm
joints **and** the gripper. The gripper is impedance-controlled too, not left in position mode: a
rigid gripper keeps commanding full force toward its target regardless of contact, crushing fragile
objects before it can "feel" them. A compliant K/D law (with a softer default K, see
`SO101ImpedanceFollowerConfig.default_k`) is what makes gentle grasping possible.

Standalone Cargo project, deliberately **not** wired into the repo's `pyproject.toml`/`uv` build --
the Python package is pure-`setuptools` and RT setup is a deployment concern. Build and run it
directly on the machine wired to the arm.

- Python-side robot class: `src/lerobot/robots/so101_impedance_follower/`
- RT host setup (kernel params, IRQ isolation, troubleshooting): [PREEMPT_RT.md](PREEMPT_RT.md)

## Why open-loop PWM

The STS3215 has no host-streamable torque/current command register (unlike this repo's
Robstride/Damiao CAN actuators with MIT-mode `kp`/`kd`/torque packets). This daemon switches all 6
motors into `Operating_Mode = PWM` and computes the duty cycle host-side from `K`, `D`,
target/present position and finite-differenced velocity. Accepted trade-off: noisier than true
torque control, but usable without different servo hardware.

## Build

```bash
cargo build --release
```

> **Every build wipes the `setcap` capability**, because `cargo` writes a fresh binary and file
> capabilities live on the inode. Re-run the `setcap` line from [Run](#run) after _every_ build --
> otherwise the daemon starts fine but silently falls back to non-RT scheduling.

## Run

**Grant the capability once, then run without `sudo`:**

```bash
sudo setcap cap_sys_nice+ep ./target/release/so101_impedance_ctrl

./target/release/so101_impedance_ctrl \
  --port /dev/ttyACM1 \
  --shm-name so101_impedance \
  --cpu-core 3 \
  --priority 99 \
  --loop-hz 1000
```

Only `SCHED_FIFO` actually needs privileges. Opening the serial port just needs membership of the
`dialout` group (`sudo usermod -aG dialout $USER`, then re-login), and pinning your own process to
a core needs nothing at all. Both privileged bits degrade to a logged warning if unavailable, so
this still runs as a normal user on a dev laptop.

> **`setcap` is lost on every rebuild** -- `cargo` writes a fresh binary and file capabilities live
> on the inode. Re-run the `setcap` line after _every_ `cargo build`.

The failure is silent by design (the daemon keeps running, just not at RT priority), so check
rather than assume. Either inspect the binary:

```bash
getcap ./target/release/so101_impedance_ctrl   # want: cap_sys_nice=ep
```

or watch the daemon's own startup log -- it says which of the two privileged steps it got:

```
INFO  pinned to CPU core 3
INFO  acquired SCHED_FIFO priority 99          <- good
WARN  failed to set SCHED_FIFO priority 99: Operation not permitted (...)   <- setcap missing
```

Running under `sudo` works too -- the daemon detects `SUDO_UID`/`SUDO_GID` and chowns
`/dev/shm/<shm-name>` back to the invoking user, since the segment would otherwise be root-owned
mode 0600 and the unprivileged Python robot could not attach to it.

Start this **before** the Python robot -- Python only attaches to an already-created segment and
fails fast if the name or `layout_version` doesn't match
(`SO101ImpedanceFollowerConfig.shm_name` must equal `--shm-name`).

### Loop rate is bounded by the servo link, not CPU

The isolated core spends nearly the whole tick blocked on I/O, so the servo link sets the ceiling --
and the dominant cost is the **USB round trip**, not baud rate. Each transaction measures ~256 us
against the ~160 us its bytes take at 1 Mbaud, and CDC-ACM exposes no latency knob (`latency_timer`
is an FTDI feature these CH343 bridges do not have).

So count transactions, not bytes:

| configuration                                    | transactions/tick | measured mean | sustains |
| ------------------------------------------------ | ----------------- | ------------- | -------- |
| per-motor READ, current batched every 10th tick  | 7, but 13 on the batched tick | 2.30 ms | 300 Hz with 10% overruns |
| per-motor READ, current round-robin              | 8                 | 2.39 ms       | 300 Hz cleanly |
| **SYNC_READ, current round-robin (default)**     | **3**             | ~0.8 ms       | **400 Hz with headroom** |

The middle row is the useful lesson: batching all six current reads onto one tick made that tick
twice as expensive as the rest, and those fat ticks were *every single overrun*. Sampling one
motor per tick round-robin costs the same one extra transaction every time, which removes the
spike instead of making it rarer -- and refreshes each motor more often than the batched version
did. Current only feeds a moving average that ACT reads at camera rate, so staggering the six
samples in time costs nothing.

Do not chase rate beyond this. What limits how the arm *feels* is open-loop PWM and gearbox
friction, not the loop period; 400 Hz is already far past the arm's mechanical bandwidth and more
than 10x ACT's ~30 Hz. Size `--loop-hz` from the summary the daemon logs every second -- pick a
period above the observed `max` -- rather than from arithmetic.

### Validating SYNC_READ on unfamiliar hardware

`--sync-read` is protocol-0 only, which covers the SO101's `sts3215` (the protocol-1 SCS series has
no SYNC_READ -- the same restriction `FeetechMotorsBus` enforces in Python). It is on by default
here because it has been validated on this arm, but re-check it on hardware you have not tried,
because the failure mode is quiet rather than loud:

A misparsed reply does not raise -- it returns a plausible-looking but wrong position. The
impedance law then chases an error that never converges and drives the joint continuously, which
looks exactly like a runaway. Crucially, **flipping `--invert-pwm` does not fix it**; that only
reverses which way the joint runs. So "it runs away with the flag both on and off" points at bad
telemetry, not a wrong drive direction. The checker's `pwm` column separates them: a real sign
error pegs PWM at `%max`, whereas a joint that is merely too soft sits near zero.

Validate with **monitor mode**, which cannot run away: Python writes nothing, so the watchdog holds
PWM at zero and the arm stays limp. Move each joint by hand and confirm the positions track it with
no comms errors in the daemon's per-second summary. Fall back with `--sync-read false --loop-hz 300`.

## Force feedback on the leader gripper

With `--leader-port`, the daemon also owns the leader arm's gripper servo and drives it as a haptic
display, so the operator feels what the follower is holding. Only the gripper: the other five leader
servos stay torque-off and backdrivable, and Python keeps reading their positions as before.

Grip force is the one thing a demonstrator currently cannot express -- a torque-off leader is a
position sensor and nothing else, so every recorded demonstration squeezes with whatever K the
config happened to hold. It is also the axis where being wrong breaks the object, and the axis where
an unstable loop only buzzes a trigger rather than an arm with the operator attached to it.

### How it works

The feedback is driven by the **follower's own tracking error**, not by a measured force. Free
gripper: it reaches its target, error ~0, trigger slack. Blocked gripper: the commanded position
runs ahead of the achieved one, and that gap grows with how hard the operator is asking it to
squeeze. Render the gap as leader duty and it becomes resistance in their hand. This is classic
position-position bilateral -- no load cell, no current sensing.

Deliberately *not* driven by the follower's `Present_Current`: that is averaged over ~0.5 s so it is
usable as an ACT observation, which is an eternity for haptics.

### Bring-up, in this order

`--leader-port` alone renders nothing (`--force-feedback-gain` defaults to 0) but pays the full bus
cost, so step 1 measures whether bilateral fits before any force reaches a hand:

```bash
# 1. Measure. Trigger is read and held at zero duty; check the per-second timing summary.
./target/release/so101_impedance_ctrl --port /dev/ttyACM0 --leader-port /dev/ttyACM1 ...

# 2. Confirm the LEADER row tracks the trigger, in monitor mode (the watchdog holds it slack).
python examples/check_so101_impedance.py --shm-name so101_impedance

# 3. Only then, a small gain, and hold something soft.
./target/release/so101_impedance_ctrl ... --leader-port /dev/ttyACM1 --force-feedback-gain 0.5
```

**The gain is signed and the sign must be measured.** Which encoder direction means "closed" is a
property of each gripper's calibration and the two arms need not agree. If the trigger *assists*
your squeeze instead of resisting it, stop and negate the gain: that polarity is positive feedback
through your own hand. `--leader-pwm-max` (default 250, far below `--pwm-max`) is what bounds a
wrong sign to something you can overpower.

### Cost, and why the rate matters more here

The two arms are on separate ports, so the half-duplex constraint does not couple them; the leader
adds 2 transactions to the tick's 3. Sequentially that is ~1.3 ms against a 2.5 ms period at 400 Hz,
which fits without doing anything clever. If it ever stops fitting, the lever is that ~256 us per
transaction is USB *waiting*, not work: issuing both ports' requests before blocking on either reply
overlaps the two round trips. Measure before reaching for it -- the summary reports the leader's
share separately for exactly this decision.

Do not trade rate away here the way you can for unilateral control. A haptic display's maximum
stable stiffness goes as `b / T`, so halving the rate halves the stiffness renderable before the
trigger buzzes in the operator's hand.

### One loop, two arms

The leader shares the follower's control loop rather than running its own thread. Two independent
loops would let their phase free-run, injecting up to a full period of *variable* delay into the
coupling -- and variable delay is what destabilises a bilateral loop. A single tick keeps both
arms' samples in lockstep by construction, which is simpler and also more correct.

A missing or unconfigurable leader degrades to follower-only with a logged error: force feedback
enhances teleoperation, it is never a prerequisite for it. `FAULT_LEADER_COMMS_ERROR` is likewise
kept distinct from `FAULT_COMMS_ERROR`, because losing the leader's bus drops force feedback while
losing the follower's stops the robot.

## Sizing K and D

Gains are per joint because the load is. `shoulder_lift` and `elbow_flex` hold the arm's weight;
`wrist_roll` holds nothing.

Measure rather than guess: **hold the arm at its most gravity-loaded pose with `--k 1`**. Then
`pwm == err`, so the `pwm` column reads out directly as the duty each joint needs to hold itself.
On this arm, outstretched: 17 / 87 / 61 / ~0 / ~0 / 0 counts for pan / lift / elbow / wrist_flex /
wrist_roll / gripper.

A PD law droops under a constant load by `err = holding_duty / K`, so K follows from the droop you
accept -- and more generally `K_new = K * err / err_wanted` from any hold test. Targeting ~5 counts
(0.4 deg) gives the shipped `SO101ImpedanceFollowerConfig.default_k`. Note the flip side: K also
sets where PWM saturates, at `pwm_max / K` counts -- K=20 is full duty at 4.4 deg, which is the
compliance range you actually feel. Wanting both a small droop *and* a wide compliance range means
adding gravity feedforward, not raising K.

D is bounded from above by velocity noise, not by stability. Position is quantised to whole counts,
so the filtered finite difference has a noise floor near `1 / (vel_filter_window * dt)` -- ~50
counts/s at the defaults -- and D turns that straight into PWM chatter. `K/40` keeps it under ~2%
duty. Raising `--loop-hz` makes this *worse*, which is why `--vel-filter-window` exists: averaging
N per-tick differences telescopes exactly to the N-tick difference, dividing the noise by N for
N/2 ticks of lag, with no attenuation of a real velocity.

## Protocol notes

Feetech register addresses in `src/feetech.rs` mirror
`src/lerobot/motors/feetech/tables.py::STS_SMS_SERIES_CONTROL_TABLE` -- keep the two in sync.
Notably, address 44 (`Goal_Time` in tables.py, used for POSITION mode) doubles as the PWM
duty-cycle command register when `Operating_Mode = PWM`, with **bit 10** as the sign bit -- not
bit 15 like `Goal_Position`, and not the bit 11 that `feetech.py`'s `OperatingMode` docstring
claims. See the Drive direction section for the measurement that settled it.

## Testing

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

All tests are pure unit/integration tests over packet framing, the shared-memory seqlock and
control-law math -- none need a serial port, shared-memory segment or RT privileges. `rt.rs`'s
actual `sched_setaffinity`/`SCHED_FIFO` calls only run in `main.rs`, since their success depends on
host privileges.

## Drive direction

Two coupled settings, both measured rather than assumed, both shipped as defaults:

| setting            | value  | why                                                              |
| ------------------ | ------ | ---------------------------------------------------------------- |
| `--pwm-sign-bit`   | **10** | bit 11 (per `feetech.py`'s docstring) does not reverse the joint  |
| `--invert-pwm`     | `true` | with the sign bit right, positive duty still lowers the encoder   |

The probe output that settled it:

```
# bit 11 (documented), flag clear -> down     ; flag set -> further down, not reversed
  2391 -> 2164 ticks (delta -227)
  2906 -> 2495 ticks (delta -411)
# bit 10, flag set
  2415 -> 2545 ticks (delta +130)      <- CORRECT
```

The middle line is the trap: setting a *wrong* sign bit still changes the motion, because the bit
gets consumed as extra magnitude and clamps to full duty. So "the joint behaved differently" is not
evidence the bit is right -- only the delta changing **sign** is. Bit 10 also fits the hardware: a
0-1000 duty scale needs 10 bits, leaving bit 10 as the flag.

This ships as the default rather than as a flag to discover because getting it wrong is not a
subtle degradation: the impedance law becomes positive feedback, and a nudged joint accelerates
into a stop instead of springing back. Re-run the probe if you suspect different wiring -- it
drives one joint, at a fraction of full duty, for under a second, with an auto-abort, so it cannot
run away the way a hold test can.
