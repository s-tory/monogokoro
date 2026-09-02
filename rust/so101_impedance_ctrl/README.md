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

Building needs `glslc` on `PATH` (`sudo apt install glslc`) to compile the cerebellum's compute
shaders. Running needs only a Vulkan ICD (`mesa-vulkan-drivers`), and only if you actually enable
`--cerebellum-backend gpu`.

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

| configuration                                   | transactions/tick             | measured mean | sustains                 |
| ----------------------------------------------- | ----------------------------- | ------------- | ------------------------ |
| per-motor READ, current batched every 10th tick | 7, but 13 on the batched tick | 2.30 ms       | 300 Hz with 10% overruns |
| per-motor READ, current round-robin             | 8                             | 2.39 ms       | 300 Hz cleanly           |
| **SYNC_READ, current round-robin (default)**    | **3**                         | ~0.8 ms       | **400 Hz with headroom** |

The middle row is the useful lesson: batching all six current reads onto one tick made that tick
twice as expensive as the rest, and those fat ticks were _every single overrun_. Sampling one
motor per tick round-robin costs the same one extra transaction every time, which removes the
spike instead of making it rarer -- and refreshes each motor more often than the batched version
did. Current only feeds a moving average that ACT reads at camera rate, so staggering the six
samples in time costs nothing.

Do not chase rate beyond this. What limits how the arm _feels_ is open-loop PWM and gearbox
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

One shape of that fault is now caught rather than argued about. `--travel-gate` (on by default)
reads each joint's calibrated travel out of the servos' `Min/Max_Position_Limit` -- the registers
`lerobot-calibrate` writes next to the homing offset -- and refuses a reported position more than
`--travel-margin` (200 counts) outside it. Measured on this arm on 2026-09-02, `shoulder_pan`
reported whole-turn jumps of +-4083 counts three times in one 25-second hand sweep, while its
mechanical stops sit at 867 and 3280, repeatable to 3-6 counts, with 815 counts of clearance to the
encoder wrap on either side. The joint had not moved; the reading was simply false.

This deliberately does **not** go through the slew gate, which holds a doubted batch and accepts it
if the next read agrees. That is right for a joint that moved while the bus was quiet and wrong
here: a whole-turn misreport holds perfectly still, so it corroborates itself on the next tick and
the impedance law then answers ~4090 counts of error with saturated duty in one direction. That is
what drove a joint into its own stop earlier the same day -- motor 1 went from 29 C to 42 C in 36
seconds and pulled the shared rail from 4.6 V to 4.0 V before a human stopped it. An uncalibrated
joint reports the whole circle and is not checked, which is also the honest answer for
`wrist_roll`. **The misreport itself is still unexplained**, and this bounds its consequences
rather than removing it.

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

Deliberately _not_ driven by the follower's `Present_Current`: that is averaged over ~0.5 s so it is
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
property of each gripper's calibration and the two arms need not agree. If the trigger _assists_
your squeeze instead of resisting it, stop and negate the gain: that polarity is positive feedback
through your own hand. `--leader-pwm-max` (default 250, far below `--pwm-max`) is what bounds a
wrong sign to something you can overpower.

### Cost, and why the rate matters more here

The two arms are on separate ports, so the half-duplex constraint does not couple them; the leader
adds 2 transactions to the tick's 3. Sequentially that is ~1.3 ms against a 2.5 ms period at 400 Hz,
which fits without doing anything clever. If it ever stops fitting, the lever is that ~256 us per
transaction is USB _waiting_, not work: issuing both ports' requests before blocking on either reply
overlaps the two round trips. Measure before reaching for it -- the summary reports the leader's
share separately for exactly this decision.

Do not trade rate away here the way you can for unilateral control. A haptic display's maximum
stable stiffness goes as `b / T`, so halving the rate halves the stiffness renderable before the
trigger buzzes in the operator's hand.

### One loop, two arms

The leader shares the follower's control loop rather than running its own thread. Two independent
loops would let their phase free-run, injecting up to a full period of _variable_ delay into the
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
compliance range you actually feel. Wanting both a small droop _and_ a wide compliance range means
adding gravity feedforward, not raising K.

D is bounded from above by velocity noise, not by stability. Position is quantised to whole counts,
so the filtered finite difference has a noise floor near `1 / (vel_filter_window * dt)` -- ~50
counts/s at the defaults -- and D turns that straight into PWM chatter. `K/40` keeps it under ~2%
duty. Raising `--loop-hz` makes this _worse_, which is why `--vel-filter-window` exists: averaging
N per-tick differences telescopes exactly to the N-tick difference, dividing the noise by N for
N/2 ticks of lag, with no attenuation of a real velocity.

## Cerebellum: an adaptive feedforward on the iGPU

The impedance law is a spinal reflex -- it only ever reacts to an error that has already happened.
Under a standing load that is a permanent offset: a joint carrying `g` PWM of gravity settles
exactly `g / K` counts below its target, and the only way a pure feedback law can shrink that droop
is to raise `K`, i.e. to trade away the compliance this daemon exists to provide. Predicting the
load decouples the two:

```
pwm = K*(x_t - x) + D*(v_t - v)  +  ff(sensory state)
      \___________ ____________/    \_______ _______/
      reflex, 400 Hz, isolated core   cerebellum, Vulkan compute
```

Off by default (`--cerebellum-backend off`). Turning it on is safe with the arm already holding a
position: the learned weights start at zero, so an untrained network contributes precisely nothing.

### Architecture

Marr-Albus-Ito, mapped onto the hardware more or less directly:

| cerebellum        | here                                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- |
| mossy fibres      | 32 signals: per joint, encoder phase as `(sin, cos)`, velocity, tracking error, current -- plus 2 pontine context channels |
| granule cells     | 16384 units, each reading 4 mossy fibres through a **fixed random** projection                                             |
| Golgi inhibition  | one global subtractive threshold, driven by feedback on the measured active fraction                                       |
| parallel fibres   | the granule code, L2-normalised, with a ~150 ms eligibility trace                                                          |
| Purkinje cells    | 6 outputs, a linear readout -- **the only learned layer**                                                                  |
| climbing fibres   | the reflex's own standing duty, low-passed and gated                                                                       |
| PF->PC plasticity | `dW = rate * (cf * e - leak * W * e)` -- three-factor Hebbian, heterosynaptic decay, both terms gated on a non-zero `cf`   |

**There is no backpropagation, and none is needed.** The expansion layer is not learned, so there
is exactly one layer of adjustable synapses, and its error is already expressed in the units the
output is in (PWM). The credit-assignment problem backprop exists to solve does not arise. What it
costs instead is parameters: 16k granule cells to get the separation a trained hidden layer would
get with a few hundred -- which is precisely the trade an idle iGPU is there to absorb.

Position enters as a phase pair rather than a count for the same reason `wrapped_delta` exists:
`Present_Position` rolls over mid-travel, so a raw count would make the network see a full-scale
jump where the joint moved one tick -- and learn a feedforward with a cliff in it. (This used to
add "and `wrist_roll` is calibrated over the full turn" as the example. That was true only because
the calibration script assigned `0-4095` without measuring; on this arm the joint reaches 340 deg
and stops. The seam argument never needed that example -- a homing offset moves the seam, it does
not remove it.)

### The pontine relay

Proprioception cannot tell two payloads apart. A gripper holding 20 g and one holding 200 g pass
through the same joint angles on the way to the same place, and the difference only becomes visible
_after_ the load has already pulled the arm out of position -- which is the one thing a feedforward
exists to prevent. So `src/pontine.rs` carries two channels down from the policy layer and lands
them at the tail of the mossy-fibre vector.

It is a **sibling of the cerebellum, not a part of it**. The pontine nuclei are brainstem, and
their entire anatomical job is to sit between cortex and the cerebellar mossy fibres -- which is
exactly where this module sits in the data flow. Filing it under `cerebellum/` would have put a
structure inside the one it projects to.

**It relays an identity, not a mass.** The policy is not asked how heavy the object is. Biology
hands the cerebellum the object and keeps the weight-to-force map in the cerebellum: grip force is
scaled correctly _before_ lift-off, from a memory indexed by which object this is, and cerebellar
damage is what takes that anticipation away. Asking a policy for grams would move the cerebellum's
job up a layer and require it to learn a calibration nothing in this loop can teach it. Anything
separable will do.

**And it does not compute.** There is no learned layer here, deliberately. The expansion into a
separable code is already paid for by 16384 granule cells on a fixed random projection, so a layer
here would duplicate the one part of the design that is pointedly not learned. Worse, a _trained_
layer upstream of the granule code would need its error routed back through the expansion and the
readout to reach it -- which is precisely the credit assignment this design does not have, and the
reason it needs no backward pass.

What is here instead is a first-order lag. The policy publishes at inference rate and the
cerebellum reads at 200 Hz, so an unfiltered channel steps; the readout is linear in the granule
code, so a step in the code is a step in the PWM. The feedforward is slew-limited downstream, but a
slew limiter turns a step into a ramp at a fixed rate regardless of distance, whereas a lag makes
the code itself move continuously -- so what the readout sees was always a state the network could
have been in.

#### Two numbers that came out of the CPU reference, not an arm

Both are pinned in `tests/cerebellum_net_tests.rs`, and both were measured before any of this ran
on hardware -- which was the point of measuring them there.

**Swing every channel to `+/-1`. Do not use a `0/1` flag.** Contexts are separated by the granule
cells that happen to draw a context fibre, so what buys separation is _how many fibres differ_, not
what the numbers mean. Interleaved training, 16384 cells, two loads 80 counts apart:

| context encoding            | fibres differing | separation recovered |
| --------------------------- | ---------------- | -------------------- |
| flag, `0,0` vs `1,0`        | 1 of 32          | 64 of 80             |
| one-hot, `1,0` vs `0,1`     | 2 of 32          | 78 of 80             |
| antipodal, `-1,-1` vs `1,1` | 2 of 32          | 80 of 80             |

They cost exactly the same to compute.

**Interleave the contexts while learning. Do not train one to convergence and then the other.**
Most granule cells draw no context fibre at all, so their weights are shared between contexts, and
a run that converges on one drags those shared weights with it. Blocked training, best encoding:
after learning the loaded case, the empty case reads 98 where it should read 40. The
context-sensitive minority cannot pull the shared majority back. A policy that picks things up and
puts them down interleaves by itself; a person at a bench has to do it deliberately. (Biology has
the same property, and calls it the contextual interference effect.)

#### Running it

Nothing fills the channel yet: demonstrations carry no label for anything below the policy, so
`InputData::context` is written as zeros by everything that exists today -- and zero is the neutral
value, so the network degrades to the proprioception-only cerebellum rather than to something
undefined. `--cerebellum-context -1,-1` pins it by hand, which is enough to run the whole
experiment with an arm and a weight:

```bash
# alternating, not blocked -- see above
--cerebellum-context -1,-1   # empty
--cerebellum-context 1,1     # loaded
```

The falsifiable claim is narrow: at one pose, with `K` unchanged, the two contexts should settle to
two different `ff` values in the checker's table, and switching between them should not require
relearning. If the feedforward is the same in both, the context is not reaching the granule code.

Widening the mossy-fibre vector reshuffles every granule draw, so **every weight learned before
this change is invalid**. The weights file's header records `MF_DIM` and refuses such a file rather
than loading it.

The granule code is normalised to unit length so that `--cerebellum-rate` means one thing
regardless of layer size: _the fraction of the remaining error corrected per step_, i.e. a time
constant of `1 / rate` steps. Without it the effective step size scales with `sum_j g_j^2`, and
changing `--cerebellum-gc-dim` would silently retune how fast the arm learns.

### Why it does not run in the control loop

Measured on the reference machine (Arc 140V / Mesa ANV 26.0.3), one full step -- submit plus fence
wait, four dispatches:

| `gc_dim` | mean   | max     |
| -------- | ------ | ------- |
| 1 024    | 174 us | 1036 us |
| 4 096    | 190 us | 1078 us |
| 16 384   | 307 us | 933 us  |
| 65 536   | 748 us | 1705 us |

Two things fall out of that. Below ~4k cells the cost is _entirely_ submission latency and the
compute is free, so a large granule layer is nearly as cheap as a small one. And the **max** is
about a millisecond whatever the size -- on an otherwise idle desktop. A 400 Hz tick is 2.5 ms and
already spends ~0.8 ms on the bus.

That max is a floor, not a bound. The same 16384-cell configuration measured from inside the
running daemon (the per-second `cerebellum [...]` log line, with the control loop and the rest of
the machine competing for the GPU) reports mean 450-650 us and **max 2969 us** -- a single step
longer than the entire control period.

That the control loop does not pay for any of it was checked rather than assumed, by alternating
the two configurations 3 x 20 s each against a dead bus at 200 Hz, 10800 ticks per condition:

| control loop                                       | mean tick | overruns   | worst tick |
| -------------------------------------------------- | --------- | ---------- | ---------- |
| `--cerebellum-backend off`                         | 3219 us   | 10 / 10800 | 7344 us    |
| `--cerebellum-backend gpu --cerebellum-cpu-core 1` | 3228 us   | 9 / 10800  | 10971 us   |

Mean cost and overrun rate are indistinguishable -- 0.3% and one fewer overrun, i.e. nothing. The
worst-case tick swings by milliseconds in _both_ columns, and one repetition had the cerebellum-off
run produce the worse outlier, so that tail belongs to the laptop rather than to the GPU thread.
Reproduce it by watching the daemon's own `loop timing` line with and without the flag; if enabling
the cerebellum moves the mean or the overrun count, the core assignment is wrong.

That jitter is not something core isolation can remove:

- The iGPU exposes **one queue family with one queue**, shared with graphics. Compute submissions
  queue behind whatever the compositor is doing; there is no async compute queue to escape to.
- The GPU's kernel-side service path -- driver workqueues, the DRM scheduler, completion interrupts
  -- runs on housekeeping cores _by construction_, because steering interrupts away from the RT
  core is exactly what `irqaffinity=` does.

So the cerebellum runs on its own thread, exchanging data with the control loop through two
seqlocks (the same non-blocking pattern `shm.rs` uses across the process boundary, and for the same
reason -- a mutex shared between a `SCHED_FIFO` loop and a normal-priority thread is a textbook
priority inversion). The control loop publishes a snapshot and reads whatever feedforward is
currently available. It never waits, and nothing is lost by that: the load being predicted is
quasi-static, so a feedforward a few milliseconds old is still correct. Biology puts the cerebellum
outside the stretch reflex's arc too.

`--cerebellum-cpu-core` pins the thread; a **housekeeping** core, and the daemon refuses the RT
core outright. A second _isolated_ core is not worth taking, for the two reasons above -- there is
nothing about a fence wait that isolation can make deterministic, and the isolation that actually
matters (this thread can never preempt the reflex) already follows from the reflex's core being
isolated from everything else.

### Safety envelope

A learned term that can push a compliant arm around is the one genuinely new hazard here, so it is
bounded four ways, none of which depend on the network having learned anything sensible:

1. **Zero at rest.** Weights start at zero; enabling it cannot change the arm's behaviour until it
   has learned.
2. **Clamped** to `--cerebellum-ff-max` (300), far below `--pwm-max` (1000).
3. **Slew-limited** to `--cerebellum-ff-slew` (500/s), in both directions -- including on the way
   back to zero, since dropping a held feedforward instantly is a step input into a compliant
   joint.
4. **Fail-safe.** Zeroed on watchdog timeout, on blind ticks, if the thread stops publishing, and
   if the backend faults. A learned term must not be the one part of the controller that survives
   its own fail-safe.

Two gates decide when it may learn at all:

- `--cerebellum-vel-gate` (80 counts/s): a moving joint's duty is inertia and damping, neither of
  which is a function of pose.
- `--cerebellum-error-gate` (200 counts): **this is what separates droop from contact.** Gravity
  droop settles small, at `duty / K`; an arm resting on the table holds a large standing error that
  never closes. Both look identical to the velocity gate.

And the gripper is **not** in `--cerebellum-joints` by default, deliberately. A gripper holding an
object shows exactly the signature this layer cancels -- a large, motionless, standing duty -- but
that duty _is the grasp_. Learning it makes the gripper squeeze harder at the same commanded
position, and keep squeezing after the object is gone. On the arm joints contact is the exception
and the error gate handles it; on the gripper contact is the normal case, so no gate can, and it is
left out entirely.

### Running it

```bash
./target/release/so101_impedance_ctrl \
  --port /dev/ttyACM0 --shm-name so101_impedance --cpu-core 3 --priority 99 \
  --cerebellum-backend gpu \
  --cerebellum-cpu-core 1 \
  --cerebellum-weights ~/.local/share/so101/cerebellum.bin
```

`--cerebellum-weights` is loaded at startup and written on a clean exit (Ctrl-C, `SIGTERM`, or
Python's Shutdown command), so learning accumulates across sessions. The file is refused if
`--cerebellum-gc-dim` or `--cerebellum-seed` changed: the weights are only meaningful against the
random projection that produced them, and reusing them under a different one would not be degraded,
it would be arbitrary -- on a real arm.

The daemon logs a summary alongside the loop timing:

```
cerebellum [gpu: Intel(R) Graphics (LNL)]: 200 steps (188 with plasticity), 307 us mean /
  933 us max per step, 0 errors; granule activity 2.01% (theta 0.412); ff [12, 96, 61, 8, 3, 0]
```

Watch `granule activity` track `--cerebellum-sparsity`: if it is pinned at 0% or 100% the Golgi
integrator has saturated and nothing downstream can learn.

### Bring-up

The checker's table gained an `ff` column, and watching it against `pwm` _is_ the procedure:

```bash
python examples/check_so101_impedance.py --shm-name so101_impedance
```

Hold a pose. As the feedforward learns a joint's standing load, `ff` should climb toward the duty
`pwm` was carrying alone, `pwm` should fall toward zero, and `err` -- the droop -- should shrink
**without anyone raising K**. That last part is the whole point; if `err` only improves when you
raise K, the feedforward is not doing anything.

`describe_cerebellum()` reports why the feedforward is what it is, which is otherwise unanswerable
from the outside: a zero `ff` could mean not enabled, nothing learned yet, gated off this tick, or
the backend died, and those call for very different responses.

### Backends and build requirements

`--cerebellum-backend cpu` runs the reference implementation in `src/cerebellum/net.rs` instead.
That module is the _definition_ of the math; the shaders are a second implementation of it, and
`tests/cerebellum_gpu_tests.rs` runs both step-by-step through learning and compares. A compute
shader is not debuggable by reading it and its failure mode is wrong numbers rather than a crash,
so the readable version is the authority and the fast one is held against it.

Asking for `gpu` and not getting it **disables** the cerebellum rather than silently falling back
to `cpu` -- "the feedforward is running" and "the feedforward is running somewhere else at a
different speed" are things an operator has to be able to tell apart.

Building needs `glslc` on `PATH` (`sudo apt install glslc`); running needs only a Vulkan ICD
(`mesa-vulkan-drivers`). The `.spv` blobs are compiled by `build.rs` rather than committed, so a
shader edit cannot ship without its binary being rebuilt.

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

Almost all of it is pure unit/integration tests over packet framing, the shared-memory seqlock,
the control law and the cerebellum's math -- none need a serial port, shared-memory segment or RT
privileges. `rt.rs`'s actual `sched_setaffinity`/`SCHED_FIFO` calls only run in `main.rs`, since
their success depends on host privileges.

The exception is `tests/cerebellum_gpu_tests.rs`, which needs a Vulkan device and **skips** rather
than fails without one, so the suite still runs headless. On the robot host it has to actually run:
a skipped cross-check is not a passing one, and it is the only thing standing between a mistake in
a compute shader and a confidently wrong feedforward on a real arm. Check for the skip line:

```bash
cargo test --release --test cerebellum_gpu_tests -- --nocapture
```

That also prints the measured per-step latency table reproduced above, so the numbers in this file
can be re-derived on any host rather than taken on faith.

## Drive direction

Two coupled settings, both measured rather than assumed, both shipped as defaults:

| setting          | value  | why                                                              |
| ---------------- | ------ | ---------------------------------------------------------------- |
| `--pwm-sign-bit` | **10** | bit 11 (per `feetech.py`'s docstring) does not reverse the joint |
| `--invert-pwm`   | `true` | with the sign bit right, positive duty still lowers the encoder  |

The probe output that settled it:

```
# bit 11 (documented), flag clear -> down     ; flag set -> further down, not reversed
  2391 -> 2164 ticks (delta -227)
  2906 -> 2495 ticks (delta -411)
# bit 10, flag set
  2415 -> 2545 ticks (delta +130)      <- CORRECT
```

The middle line is the trap: setting a _wrong_ sign bit still changes the motion, because the bit
gets consumed as extra magnitude and clamps to full duty. So "the joint behaved differently" is not
evidence the bit is right -- only the delta changing **sign** is. Bit 10 also fits the hardware: a
0-1000 duty scale needs 10 bits, leaving bit 10 as the flag.

This ships as the default rather than as a flag to discover because getting it wrong is not a
subtle degradation: the impedance law becomes positive feedback, and a nudged joint accelerates
into a stop instead of springing back. Re-run the probe if you suspect different wiring -- it
drives one joint, at a fraction of full duty, for under a second, with an auto-abort, so it cannot
run away the way a hold test can.
