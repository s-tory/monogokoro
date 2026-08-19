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

The isolated core spends nearly the whole tick blocked on I/O, so the servo link sets the ceiling.
1 Mbaud is already the STS3215 maximum; at 8N1 that is 10 us/byte, giving this **wire-time floor**:

| read strategy                                                | bytes/tick   | wire time |
| ------------------------------------------------------------ | ------------ | --------- |
| one READ per motor (**default**)                             | 218          | 2.18 ms   |
| `--sync-read`, current every tick (`--current-read-divisor 1`) | 150          | 1.50 ms   |
| `--sync-read`, current decimated (divisor 10)                | 88 (typical) | 0.88 ms   |

12 per-motor READs (6 motors x 2 registers, ~160 us each) blow a 1 ms budget on their own.
`SYNC_READ` collapses each register into one request plus back-to-back replies (620 us per
register). Only `Present_Position` is needed at full rate; `Present_Current` is moving-averaged and
consumed by ACT at camera rate (~30 Hz), so `--current-read-divisor` skips it on most ticks (they
republish the existing average). The averaging window then spans
`current_avg_window * divisor / loop_hz` seconds -- 0.32 s at the defaults.

**Wire time is a floor, not a prediction.** It ignores the USB round trip, which dominates in
practice. Measured on the reference setup (CH343 bridge presenting as CDC-ACM `/dev/ttyACM*`,
SYNC_READ with divisor 10):

```
loop timing over 1000 ticks: min 1.127471ms / mean 1.309507ms / max 2.290276ms (period 1ms); 1000 overruns
```

That is ~250 us above the 0.88 ms wire-time figure, and the min never drops below ~1.1 ms because
each tick needs two USB bulk round trips (one SYNC_READ, one SYNC_WRITE) that no amount of baud
rate removes. CDC-ACM exposes no latency knob either -- `latency_timer` is an FTDI feature and does
not exist for these bridges. **So size `--loop-hz` from a measurement, not from the table above:**
run for a few seconds, read the summary the daemon logs every second, and pick a period above the
observed `max`. Here that means ~400 Hz (2.5 ms), not the 1000 Hz the wire math suggests.

The daemon logs that summary once a second rather than warning per overrun -- at 1 kHz a saturated
bus would emit a thousand lines a second, and writing them costs more time than they report. A
nonzero overrun count every second means the link cannot sustain the requested rate: lower
`--loop-hz` or raise `--current-read-divisor`.

### SYNC_READ is opt-in, and why

`--sync-read` is protocol-0 only, which covers the SO101's `sts3215` (the protocol-1 SCS series has
no SYNC_READ -- the same restriction `FeetechMotorsBus` enforces in Python). Its framing is
unit-tested but **not validated against physical servos**, and it is off by default because its
failure mode is nasty rather than obvious:

A misparsed reply does not raise -- it returns a plausible-looking but wrong position. The
impedance law then computes an error that never converges and drives the joint continuously, which
looks exactly like a runaway. Crucially, **flipping `--invert-pwm` does not fix it**; it only
reverses which way the joint runs, so "it runs away with the flag both on and off" is a symptom of
bad telemetry, not of a wrong drive direction. Watching `pwm` in the checker's table separates
them: a real sign error pegs PWM at `%max`, whereas a joint that is simply too soft sits near zero.

Turn `--sync-read` on only after per-motor reads are working and you have confirmed the positions
it reports track reality.

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
