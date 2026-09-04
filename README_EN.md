# ものごころ ≒ MONOGOKORO, Thinks of Things, 物心

[日本語](README.md) | **English** | [简体中文](README_CN.md)

A fork of [LeRobot](https://github.com/huggingface/lerobot) that gives the SO-101 two of the motor
layers that sit _underneath_ a policy: a **spinal reflex** whose joints and gripper yield to contact
instead of driving through it, and a **cerebellum** that learns, online, to cancel a load before the
reflex has to feel it. ACT sees the resulting forces and commands how hard to resist them. A third
layer sits below both and is not software at all -- a compliant fingertip, which answers contact
sooner than any loop here could be scheduled to.

Synced with upstream through [`d36d404b`](https://github.com/huggingface/lerobot/commit/d36d404b)
(2026-08-31, on the 0.6.2 line). Everything upstream still works unchanged -- this adds a robot, a real-time
controller, an online-learning feedforward layer, and two extra dimensions of what ACT reasons
about. No policy architecture is modified.

## Why

Almost everything called _Physical AI_ is a story about the cortex. The layer that actually touches
physics -- the one that answers contact in real time -- has been left empty.

The goal is to build the layers below the policy out of hardware anyone can buy and software anyone
can read. Not a lab rig: 3D-printed arms, hobby servos, a laptop, a mainline kernel, and the
integrated GPU that was already in it.

Stock SO-101 control writes `Goal_Position` and lets the servo's internal PID drive there. That
controller has no notion of contact: blocked by an object, it keeps increasing effort toward a
position it will never reach. A potato chip snaps before the arm can be said to have felt it. And
the policy has no vocabulary for the difference between _press firmly_ and _hold gently_ --
`Goal_Position` is the only thing it can say.

Biology does not solve this in the brain. The stretch reflex is a spring-damper closed in the
spinal cord, and descending commands do not specify force -- they set an equilibrium position and,
via gamma motor neurons, the _gain_ of that reflex.

A reflex alone is not enough, and its shortfall is exactly measurable. It can only answer an error
that has already happened, so a joint carrying a standing load sits `holding_duty / K` below its
target forever, and the only way a feedback law can shrink that droop is to raise `K` -- to hand
back the compliance it was there to provide. Cancelling the load _before_ the error appears is a
different job, and biology gives it to a different structure.

So there are four layers here, and each is where it is for a reason:

|                                         | biology                    | here                                             | rate   |
| --------------------------------------- | -------------------------- | ------------------------------------------------ | ------ |
| answers contact with no loop at all     | preflex: muscle and tissue | three silicone finger caps, stacked, on each jaw | --     |
| fast local loop, brain not involved     | stretch reflex             | Rust daemon, `SCHED_FIFO`, isolated core         | 400 Hz |
| prediction, learned from its own errors | cerebellum                 | Vulkan compute on the iGPU, own thread           | 200 Hz |
| slow loop through perception            | visual feedback            | ACT                                              | ~30 Hz |

The ~13x separation between the reflex and ACT is roughly the one biology runs at, and it is the
reason the control law does not live in Python. The cerebellum sits between them and, like its
namesake, _outside_ the reflex arc -- it corrects the loop without ever being inside it.

The pontine relay has no row in that table, because it is not a rung on this ladder -- it is a
path. It carries context down from the top of the stack to the cerebellum's mossy fibres, closes no
loop of its own, and therefore has no rate. What it carries, and what it deliberately does not, is
[section 5](#5-a-pontine-relay-so-the-cerebellum-can-be-told-what-it-is-holding).

The top row is the cheapest thing in this repository and possibly the most load-bearing. Contact
transients are faster than any loop on the list: a fingertip meeting an object produces its force
spike well inside the reflex's 2.5 ms tick, so whatever answers it _first_ cannot be a controller at
all. Biology's answer is the **preflex** -- the intrinsic mechanical response of muscle and tissue,
at zero latency, before any reflex arc has been traversed. Here it is three silicone finger caps
stacked on each jaw -- the thick anti-chapping kind, ~2 mm wall, not the thin office cots sold for
counting paper -- and it is why a soft-fingered animal can be careless with a fragile object in a way
this arm cannot.

<p align="center">
  <img src="media/readme/gripper_fingertips.jpg" width="360"
       alt="The SO-101 gripper held in a hand, three silicone finger caps stacked on each jaw, holding an intact potato chip over a bowl of them" />
</p>

It also gives the gripper a finer sense of touch, which is less obvious. Grip force was always
readable -- after contact the commanded position keeps advancing while the achieved one stops, and
`pwm = K * err` follows the squeeze. What a compliant fingertip changes is the _scale_: the same
range of force now spreads across far more encoder counts, which is exactly why a load cell has a
flexure, to turn force into a displacement large enough to measure. The signal was already there;
the padding gives it a finer ruler. By how much, on this arm, is not measured yet -- and the
candidate that could eat the whole effect is the static friction described under Known limitations.

None of these four layers is a new idea. Impedance control is Hogan, 1985. A granule expansion read
out linearly and taught by a climbing fibre is Marr, Albus and Ito -- and Albus built a controller
out of it in 1975. Compliance in series with a sensor, so that force becomes a displacement big
enough to measure, is what a series elastic actuator has been since 1995. Bilateral teleoperation
driven by position error is older than any of them.

What is new is where they run. Each arrived attached to hardware a person could not simply buy: a
torque-controlled arm, a dSPACE box or a DSP card to close the fast loop, something substantial to
do the learning on. All four now fit on a laptop -- the adaptive layer on the integrated GPU that
came with it, the real-time loop on a mainline kernel, since PREEMPT_RT was merged upstream in 2024
and has only been ordinary for about two years.

So the contribution here is not a mechanism. It is the port, and the numbers that come with it: what
a Marr-Albus layer actually costs on an Arc 140V, why it cannot go inside the control tick, what a
hobby servo bus will do at 400 Hz. None of those could be looked up. They are why the section below
on what was measured is as long as it is.

## What this fork adds

```
   operator's hand                                                        camera
        │  ▲                                                                 │
   ┌────┴──┴────┐                                                            │
   │ SO-101     │  gripper: force feedback ──┐                               │
   │ leader     │  5 joints: backdriven      │                               ▼
   └────────────┘                            │                    ┌──────────────────┐
        │ pos                                │                    │ ACT              │
        ▼                                    │                    │                  │
   ╔═══════════════════════════════════════╗ │                    │ in:  images      │
   ║ Rust RT daemon  ·  400 Hz             ║◄┘   shared memory    │      pos    ×6   │
   ║ SCHED_FIFO, isolated core             ║◄──── seqlock ───────►│      current×6   │
   ║  pwm = K·Δx + D·Δv + ff   (6 motors)  ║                      │                  │
   ║  owns both serial buses               ║                      │ out: pos ×6 ┐    │
   ╚═══════════════════════════════════════╝                      │      K   ×6 ├ ×N │
     │ PWM   ▲ pos, current    │ state  ▲ ff                      │      D   ×6 ┘    │
     ▼       │                 ▼        │                         └──────────────────┘
   ┌───────────┐          ╔═════════════════════════╗
   │ SO-101    │          ║ cerebellum · 200 Hz     ║
   │ follower  │          ║ Vulkan on the Intel iGPU║
   │ 6×STS3215 │          ║ 16384 granule → 6 PC    ║
   └───────────┘          ║ three-factor Hebbian    ║
                          ╚═════════════════════════╝
```

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
USB round trip each is ~0.8 ms against a 2.5 ms period. Beyond 400 Hz there is nothing to gain --
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

It never runs inside the control loop; see [Measured, not assumed](#measured-not-assumed) for the
two numbers that make that non-negotiable. The handoff is a seqlock in both directions, so a slow or
dead cerebellum cannot stall the reflex, and nothing is lost by the delay while the iGPU is not
carrying heavy work of its own -- the load being predicted is quasi-static
([Known limitations](#known-limitations)).

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

## Quick start

```bash
# 1. Build and grant the one privileged capability. setcap is lost on every rebuild.
cd rust/so101_impedance_ctrl && cargo build --release
sudo setcap cap_sys_nice+ep ./target/release/so101_impedance_ctrl

# 2. Start the daemon. It must be running before any Python attaches.
#    Without RUST_LOG=info it prints nothing at all. The cerebellum backend it
#    actually got, and the first supply reading, are only ever reported there.
RUST_LOG=info ./target/release/so101_impedance_ctrl \
  --port /dev/ttyACM0 --shm-name so101_impedance --cpu-core 3 --priority 99

# 3. Confirm telemetry, with the arm torque-limp and safe to move by hand.
python examples/check_so101_impedance.py --shm-name so101_impedance
```

Then teleoperate or record with `--robot.type=so101_follower_impedance`; both fill in per-joint K/D
from the robot's config automatically.

The cerebellum is opt-in. Add to step 2 (needs `glslc` to build, a Vulkan ICD to run, and a
housekeeping core that is **not** `--cpu-core`):

```bash
  --cerebellum-backend gpu --cerebellum-cpu-core 1 \
  --cerebellum-weights ~/.local/share/so101/cerebellum.bin
```

- Setting up the isolated core: [`rust/so101_impedance_ctrl/PREEMPT_RT.md`](rust/so101_impedance_ctrl/PREEMPT_RT.md)
- Tuning gains, cerebellum bring-up, protocol notes: [`rust/so101_impedance_ctrl/README.md`](rust/so101_impedance_ctrl/README.md)
- General LeRobot usage (recording, training, eval): [`AGENT_GUIDE.md`](AGENT_GUIDE.md)

## Measured, not assumed

Development is on a ThinkPad X1 Carbon Gen 13 -- Core Ultra 7 258V (Lunar Lake), Arc 140V iGPU.
Several numbers here were settled on that hardware after their documented or intuitive values turned
out to be wrong. They are specific to it and worth re-measuring on another machine:

| what                | value                                        | how it was settled                                                                                                                                                                               |
| ------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PWM sign bit        | **10**                                       | bit 11 (per upstream's docstring) does not reverse the joint -- it is consumed as extra magnitude                                                                                                |
| `--invert-pwm`      | **true**                                     | with the right sign bit, positive duty still lowers the encoder                                                                                                                                  |
| per-joint K         | 10/20/15/10/8/5                              | holding at K=1 makes the reported PWM read out as each joint's gravity+friction duty                                                                                                             |
| per-joint D         | ≈ K/40                                       | bounded from above by the velocity quantisation noise floor, not by stability                                                                                                                    |
| iGPU compute queues | **1**                                        | one queue family, one queue, shared with graphics -- a compute submission cannot be scheduled around the compositor, and no CPU isolation changes that                                           |
| cerebellum step     | 307 µs mean idle, **2969 µs max under load** | one step can outlast an entire 2.5 ms control period; the max barely moves with layer size, so it is submission jitter rather than compute                                                       |
| ACT forward pass    | **30 ms** (bf16), 44 ms (fp32)               | on the iGPU, two 480x640 cameras; 16 ms with one, so vision dominates. `n_action_steps` defaults to 100, so a 30 Hz robot infers once every 3.3 s -- 0.9% duty. `examples/load_igpu_with_act.py` |

Those last two are why the cerebellum has its own thread rather than a slot in the tick. That it
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

## Known limitations

- **The pontine context is not verified on hardware, and nothing yet _infers_ it.** The channel is
  wired end to end -- config to shared memory to mossy fibres, with the demonstration's context
  recorded as an action column -- but its two constants were measured against the CPU reference
  rather than an arm, and a policy reproduces whatever the operator labelled until it learns to
  predict it from the images.
- **The cerebellum's droop numbers are withdrawn. The whole measurement has to be retaken.** The
  2026-08-28 session reported `shoulder_pan` 3.00 -> 0.00, `elbow_flex` 9.00 -> 0.00 and
  `shoulder_lift` 12.57 -> 3.00 counts of droop at unchanged K, against baselines that were
  `err = holding_duty / K` to the decimal. Do not rely on any of it. `shoulder_pan`'s power stage
  had shorted, and writing to it pulls the shared supply from 4.6 V to 2.4 V for ~820 ms -- with
  the daemon writing to all six every 2.5 ms, the rail was down for most of the session. An
  under-volted servo makes less torque per PWM count, so the holding duties and the
  `--cerebellum-ff-max` clamp being reached on two joints are inflated by an unknown amount.
  This page previously claimed the _comparison_ survived because both sides ran under the same
  fault. That was wrong: the fault began partway through the session, so which side of it each run
  fell on decides whether the effect is understated or overstated -- and the CSVs are gone, with
  only a run-relative monotonic clock in them, so it can no longer be established either way. The
  daemon now publishes the rail voltage into the same telemetry snapshot as the duty, and the
  measurement CSV records it per sample alongside a wall clock, so a future run carries the
  evidence needed to judge it.
- **The feedforward decayed instead of settling. Fixed; the fix is unmeasured.** In both learning
  runs it bled away with a time constant of minutes while the joint sat perfectly still, then
  snapped back to the clamp once the arm finally slipped. The cause was the rule, not the arm: the
  decay was gated on the eligibility trace but not on the climbing fibre, so the fixed point was
  `w = cf / leak` -- weights that need a standing error to hold them up. The residual that leaves is
  under a PWM count, narrower than the stick band, and inside that band the joint cannot move to
  report the error at all. Both halves are now gated on a live climbing fibre, and
  `--cerebellum-cf-deadband` (default 5.0) sets where the reflex counts as silent. That default is
  derived rather than measured -- above the leak's residual, below one encoder count times the joint
  stiffness -- so it wants confirming on a healthy arm; `0` restores the old behaviour for an A/B
  inside one session. Whether the stick band is a property of the gearboxes or an artefact of the
  supply collapsing is still what the re-measurement has to separate.
- **Touch stops at how hard, not where or whether it is slipping.** A compliant fingertip turns grip
  force into encoder counts, and that is the whole of the tactile sense here: one scalar per jaw,
  available to the reflex at 400 Hz. Where on the finger contact happened, and the micro-vibration
  that says an object has _begun_ to slip, both need a purpose-built sensor rather than a commodity
  part -- and what this repository is trying to show is how much of the stack can be built without
  one. The wrist camera can see that something has slipped, at ~30 Hz; it cannot see it starting.
- **What it can learn is bounded by its mossy fibres.** They carry pose, velocity, tracking error
  and current, so it can learn gravity, joint friction and a fixed payload -- but nothing tells it
  which of two payloads is in the gripper, so it cannot tell them apart. Camera features are the
  obvious missing bundle.
- **Demonstrations do not label the layers below the policy.** A torque-off leader is a position
  sensor and nothing else, so episodes are labelled with the config's default K/D and ACT trained on
  them learns to reproduce those gains, not to vary them. The cerebellum's weights likewise persist
  to a file and not into any dataset. The leader gripper's force feedback is the first step toward
  fixing the first half; deriving stiffness from cross-demonstration variance is the likely next.
- **Open-loop PWM** is noisier than true torque control -- the STS3215 has no host-streamable torque
  register, so this is a constraint of the hardware rather than a choice.
- **The daemon is not part of the Python build.** It is a separate Cargo project, deployed by hand.
- **Interactive calibration and `setup-motors`** are not implemented for the impedance robot. Run
  both with the stock `so101_follower` against the same servos, then copy the calibration across --
  the two robot types write to different directories.

## Upstream

Everything not listed above is upstream LeRobot, unmodified, including all other robots, policies,
datasets and scripts. Upstream documentation applies directly:

- [Documentation](https://huggingface.co/docs/lerobot) · [Hub](https://huggingface.co/lerobot) · [Discord](https://discord.gg/q8Dzzpym3f)

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascale, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

Apache 2.0, as upstream. See [LICENSE](LICENSE).
