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
