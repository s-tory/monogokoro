# ものごころ/MONOGOKORO, Thinks of Things, 物心 — details

← back to [README_EN.md](README_EN.md)

The long half of the README: the per-layer write-ups from "What this fork adds", and the A/B latency work from "Measured, not assumed".

---

### 1. Impedance control on a PREEMPT_RT isolated core

[`rust/so101_impedance_ctrl/`](rust/so101_impedance_ctrl/) is a standalone Rust daemon that
exclusively owns the follower's serial bus and runs

```
pwm = clamp(K · (target_pos − present_pos) + D · (target_vel − present_vel))
```

for **all six motors, the gripper included**, at **400 Hz** on a `SCHED_FIFO` thread pinned to an
`isolcpus`-isolated core. Python never touches the bus; it exchanges targets and telemetry through
a `#[repr(C)]` shared-memory segment guarded by a seqlock.

The gripper is deliberately not left in position mode. A rigid gripper is exactly the thing that
snaps the chip, so it runs the same K/D law as the arm with a much softer default K.

Open-loop PWM, because the STS3215 exposes no host-streamable torque register. Noisier than true
torque control, and an accepted trade-off rather than a hidden one.

Loop rate is bounded by the servo link, not the CPU: three bus transactions per tick at ~256 µs of
USB round trip each is ~0.8 ms against a 2.5 ms period. **The measured tick is longer**: 1.1-1.5 ms
mean over intervals with a healthy bus (2026-09-09, defaults; 1.1 ms with
`--current-read-divisor 4`), so about 1.5x the figure above, with the margin still intact.
Beyond 400 Hz there is nothing to gain --
what limits how the arm feels is open-loop PWM and gearbox friction, and 400 Hz is already far past
the arm's mechanical bandwidth.

### 2. Force feedback on the leader gripper

With `--leader-port`, the same loop also drives the **leader** arm's gripper as a haptic display, so
the operator feels what the follower is holding. The other five leader servos stay torque-off and
backdrivable.

The force is derived from the follower's own tracking error, not from a force sensor: a free gripper
reaches its target and the trigger stays slack; a blocked one lets the commanded position run ahead
of the achieved one, and that gap grows with how hard the operator is asking it to squeeze.

Both arms share **one loop on one core**. Their ports are separate, so the half-duplex constraint
does not couple them, and a single tick keeps the two arms' samples in lockstep -- two independent
loops would let their phase free-run, injecting a full period of variable delay into the coupling,
which is precisely what destabilises a bilateral loop.

### 3. ACT reasons about force and compliance

No changes to `modeling_act.py`. Both projections already derive their width from the feature
shapes, so widening the robot's declared features is sufficient.

|                           | stock SO-101     | this fork                                                                            |
| ------------------------- | ---------------- | ------------------------------------------------------------------------------------ |
| `observation.state`       | `pos` ×6 → **6** | (`pos` + `current_avg` + `pwm_cmd` + `ff_pwm`) ×6 + rail + cerebellum flags → **26** |
| `action` (per chunk step) | `pos` ×6 → **6** | `pos` ×6 + `K` ×6 + `D` ×6 → **18**                                                  |

**Input.** Each motor's `Present_Current` is sampled one servo per tick round-robin and averaged in
Rust over a fixed window (~0.5 s at the defaults). ACT reads the pre-averaged value at camera rate,
so it sees contact force without the per-tick noise. The supply rail travels alongside it, because a
current read without a concurrent rail reading cannot be told apart from a stiff mechanism.

The two duty columns are the loop's own account of itself: `ff_pwm` is the cerebellum's share and
`pwm_cmd` the total, so their difference is the feedback share -- the quantity the climbing fibre is
derived from. Recording both is what lets a frame's salience (which frames the cerebellum could not
predict) be judged after the fact rather than having to be decided before the first episode. The
case temperature is deliberately absent; it belongs to whichever servo the health poll last read, so
recording it truthfully would mean recording that round-robin id as a column too.

**Output.** ACT's action chunking is unchanged -- it still predicts `chunk_size` steps ahead -- but
each step now carries a per-joint stiffness and damping alongside the position. The policy chooses
its own compliance over the horizon; K/D are clamped in Python and again in Rust before reaching a
servo.

### 4. A cerebellum, learned online on the integrated GPU

[`rust/so101_impedance_ctrl/src/cerebellum/`](rust/so101_impedance_ctrl/src/cerebellum/) predicts
the load the reflex would otherwise carry as a standing error:

```
pwm = K·(x_t − x) + D·(v_t − v) + ff(sensory state)
```

The structure is Marr-Albus-Ito, mapped onto the hardware directly:

| cerebellum       | here                                                                                          |
| ---------------- | --------------------------------------------------------------------------------------------- |
| mossy fibres     | 30 signals -- per joint: encoder phase as `sin`/`cos`, velocity, tracking error, current      |
| granule cells    | 16384 units, each reading 4 mossy fibres through a **fixed random**, never-learned projection |
| Golgi inhibition | one global threshold, on a feedback loop against measured sparsity (~2% left active)          |
| parallel fibres  | that sparse code, normalised, carrying a ~150 ms eligibility trace                            |
| Purkinje cells   | a linear readout, 6 outputs -- **the only learned layer**                                     |
| climbing fibres  | the reflex's own standing duty                                                                |

```
ΔW = rate · (cf · e  −  leak · W · e)
```

Three factors -- parallel-fibre eligibility, climbing fibre, and nothing else. The `leak` term is
what makes it a _modified_ Hebbian rule rather than a runaway: a bare Hebbian product only grows
once the error has a consistent sign, which is exactly what gravity produces. Both terms are gated
on a live climbing fibre: a `cf` of zero is the absence of an error signal, not an error of zero,
and a decay left running without one dismantles the very feedforward that made the error go away.

**No backpropagation, and none is needed.** Only one layer has adjustable synapses, and its error is
already expressed in the units its output is in (PWM), so the credit-assignment problem backprop
exists to solve never arises. What it costs instead is parameters rather than depth -- 16k granule
cells to get the separation a trained hidden layer would get with a few hundred, which is precisely
the trade an otherwise idle iGPU absorbs for nothing.

The teaching signal is a quantity the daemon already computes every tick: whatever duty the spring
is _still_ having to hold is, by definition, what the prediction failed to cancel. So learning is
online and unconditional -- no dataset, no training phase, no episode boundary. The arm learns while
it is teleoperated, while ACT drives it, and while it sits still; point `--cerebellum-weights` at a
file and what it learns accumulates across runs.

It never runs inside the control loop; see [Measured, not assumed](README_EN.md#measured-not-assumed) for the
two numbers that make that non-negotiable. The handoff is a seqlock in both directions, so a slow or
dead cerebellum cannot stall the reflex, and nothing is lost by the delay while the iGPU is not
carrying heavy work of its own -- the load being predicted is quasi-static
([Known limitations](README_EN.md#known-limitations)).

**Off by default**, and safe to switch on mid-hold: the weights start at zero, so an untrained
network contributes exactly nothing. Its output is clamped, slew-limited in both directions, and
zeroed by every fail-safe the reflex has. The gripper is excluded from it entirely -- a gripper that
learns its own grasp keeps squeezing after the object is gone. Full safety envelope and tuning:
[`rust/so101_impedance_ctrl/README.md`](rust/so101_impedance_ctrl/README.md#cerebellum-an-adaptive-feedforward-on-the-igpu).

### 5. A pontine relay, so the cerebellum can be told what it is holding

The mossy fibres above carry proprioception only, which means the cerebellum cannot tell two
payloads apart: 20 g and 200 g pass through the same joint angles on the way to the same place, and
the difference only shows up _after_ the load has pulled the arm down -- the one thing a
feedforward exists to prevent. [`rust/so101_impedance_ctrl/src/pontine.rs`](rust/so101_impedance_ctrl/src/pontine.rs)
adds two channels from the policy layer to the tail of that vector, taking it from 30 signals to 32.

It relays an **identity, not a mass**. The policy is never asked how heavy the object is. Biology
hands the cerebellum the object and keeps the weight-to-force map in the cerebellum -- grip force
is scaled correctly before lift-off, from a memory indexed by which object this is -- so asking a
policy for grams would move the cerebellum's job up a layer and demand a calibration nothing in
this loop can teach it.

And it **does not compute**. It is a first-order lag and nothing else, because the expansion into a
separable code is already paid for by 16384 granule cells on a fixed random projection. A trained
layer here would duplicate that, and would need its error routed back through the granule layer and
the readout to learn -- which is exactly the credit assignment this design does not have and does
not want. It is a sibling of the cerebellum in the source tree for the same reason it is one in the
brainstem.

**The channel has a source now, and it is still a constant.** `--robot.pontine_context` is the
operator's declaration of which of two indistinguishable situations is being demonstrated:
`send_action` clamps it to `[-1, 1]` and writes it to shared memory, and
`PontineContextProcessorStep` records it into the dataset as `context.<i>` action columns.
Recording it as an _action_ is the point: ACT has no context of its own -- its CVAE latent is
zeroed at inference by construction -- but a context inside the action it is trained to reproduce
is one it can learn to emit from the images, reproducing the recorded constant first exactly as the
`.k`/`.d` columns do. `--cerebellum-context` still pins the channel by hand, bypassing shared
memory and the lag both, which is what the bench wants. Two constants that came out of the CPU
reference rather than an arm, and are regression tests now: swing every channel to `+/-1` rather
than raising a `0/1` flag (one differing fibre recovers 64 of an 80-count separation, two recover
all 80, for the same cost), and interleave the contexts while learning rather than training one to
convergence and then the other (most granule cells draw no context fibre, so their weights are
shared, and blocked training leaves the first context reading 98 where it should read 40) --
`--robot.pontine_context_cycle` rotates them per kept episode so one recording run interleaves
on its own.

---

The cerebellum step and the ACT forward pass are why the cerebellum has its own thread rather than a slot in the tick. That it
stays out of the way was then checked rather than assumed -- 3 × 20 s each way, alternating,
10800 control ticks per condition. (Against a stub serial port, so the absolute tick cost is
timeout-dominated and means nothing on its own; the _comparison_ is what is being made.)

| control loop           | mean tick | overruns   | worst tick |
| ---------------------- | --------- | ---------- | ---------- |
| cerebellum off         | 3219 µs   | 10 / 10800 | 7344 µs    |
| cerebellum on the iGPU | 3228 µs   | 9 / 10800  | 10971 µs   |

Mean cost and overrun rate are indistinguishable. The worst-case tick swings by milliseconds in
_both_ columns -- one repetition had the quiet run produce the worse outlier -- so that tail belongs
to the laptop, not to the cerebellum.

**Policy inference on the same iGPU was checked the same way** (2026-09-04: a dead bus on a pty, no
plasticity, no `SCHED_FIFO`, 25 s per condition, load from `examples/load_igpu_with_act.py`).

| condition                                                            | cerebellar step, mean |
| -------------------------------------------------------------------- | --------------------- |
| cerebellum alone                                                     | 588-646 us            |
| + ACT inference (once every 3.3 s -- the default `n_action_steps`)   | 504-605 us            |
| + ACT inference (back to back -- the temporal-ensembling worst case) | **311-427 us**        |

No condition dropped a single 200 Hz step (600-604 steps per 3 s), with zero errors and zero
rejections. **The cerebellum is faster under the heavier load.** A dispatch this size costs
submission latency rather than compute, and an idle iGPU has clocked down; a continuous load keeps
it awake. The contention this was measured to find is not there, and what is there points the other
way.

The tooling for re-deriving all of it ships too: `--probe-direction` measures drive direction with a
bounded, auto-aborting nudge; the checker's live table separates "too soft" from "driven the wrong
way", which look identical from across the room; and `cargo test --test cerebellum_gpu_tests --
--nocapture` reprints the latency table on whatever host you are on.
