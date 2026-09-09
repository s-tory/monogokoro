# ものごころ/MONOGOKORO, Thinks of Things, 物心

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
[section 5](README_DETAIL_EN.md#5-a-pontine-relay-so-the-cerebellum-can-be-told-what-it-is-holding).

The top row is the cheapest thing in this repository and possibly the most load-bearing. Contact
transients are faster than any loop on the list: a fingertip meeting an object produces its force
spike well inside the reflex's 2.5 ms tick, so whatever answers it _first_ cannot be a controller at
all. Biology's answer is the **preflex** -- the intrinsic mechanical response of muscle and tissue,
at zero latency, before any reflex arc has been traversed. Here it is three silicone finger caps
stacked on each jaw -- the thick anti-chapping kind, ~2 mm wall (ours are Rimikuru finger
supporters), not the thin office cots sold for counting paper -- and it is why a soft-fingered
animal can be careless with a fragile object in a way this arm cannot.

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

**Each layer is written up in [README_DETAIL_EN.md](README_DETAIL_EN.md)** — how it is built, why, and what did not work.

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

| what                  | value                                        | how it was settled                                                                                                                                                                                                           |
| --------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PWM sign bit          | **10**                                       | bit 11 (per upstream's docstring) does not reverse the joint -- it is consumed as extra magnitude                                                                                                                            |
| `--invert-pwm`        | **true**                                     | with the right sign bit, positive duty still lowers the encoder                                                                                                                                                              |
| per-joint K           | 10/20/15/10/8/5                              | holding at K=1 makes the reported PWM read out as each joint's gravity+friction duty                                                                                                                                         |
| per-joint D           | ≈ K/40                                       | bounded from above by the velocity quantisation noise floor, not by stability                                                                                                                                                |
| wrist stereo baseline | **55 mm**                                    | estimated 57.8 mm from disparity, measured 55 mm with a ruler. The 5% gap is a constant rotation offset riding on the disparity (+39 px), matching the +42 px measured independently in the vertical direction               |
| recording resolution  | **640x360**                                  | this camera's 4:3 modes do not add height, they crop 25% off the width. At 640x480 the stereo overlap falls from 58% to 43%, and it costs 25% more pixels than 640x360. ACT does not resize, so pixel count is training cost |
| iGPU compute queues   | **1**                                        | one queue family, one queue, shared with graphics -- a compute submission cannot be scheduled around the compositor, and no CPU isolation changes that                                                                       |
| cerebellum step       | 307 µs mean idle, **2969 µs max under load** | one step can outlast an entire 2.5 ms control period; the max barely moves with layer size, so it is submission jitter rather than compute                                                                                   |
| ACT forward pass      | **30 ms** (bf16), 44 ms (fp32)               | on the iGPU, two 480x640 cameras; 16 ms with one, so vision dominates. `n_action_steps` defaults to 100, so a 30 Hz robot infers once every 3.3 s -- 0.9% duty. `examples/load_igpu_with_act.py`                             |

The full A/B measurements are in [README_DETAIL_EN.md](README_DETAIL_EN.md).

<p align="center">
  <img src="media/readme/wrist_stereo.jpg" width="420"
       alt="Two U20CAM-1080P camera boards on a printed bracket at the SO-101 wrist, each held by two standoffs at diagonal corners, with the silicone fingertips behind them" />
</p>

The two wrist-stereo rows above are legible in this photograph. That the lenses are wide is visible.
That each camera board is held by only two standoffs at diagonal corners is also visible -- and the
second is the most likely source of the constant offset riding on the disparity. Neither the
registers nor the feature matching said that much.

## Known limitations

- **The pontine context is not verified on hardware, and nothing yet _infers_ it.** The channel is
  wired end to end -- config to shared memory to mossy fibres, with the demonstration's context
  recorded as an action column -- but its two constants were measured against the CPU reference
  rather than an arm, and a policy reproduces whatever the operator labelled until it learns to
  predict it from the images.
- **The cerebellum's droop numbers were retaken on 2026-09-08; the 2026-08-28 ones are withdrawn.**
  The old session reported `shoulder_pan` 3.00 -> 0.00, `elbow_flex` 9.00 -> 0.00 and
  `shoulder_lift` 12.57 -> 3.00 counts of droop at unchanged K. Do not rely on any of it.
  `shoulder_pan`'s power stage had shorted, and writing to it pulled the shared rail from 4.6 V to
  2.4 V for ~820 ms. This page previously claimed the _comparison_ survived because both sides ran
  under the same fault. That was wrong: the fault began partway through the session, so which side
  of it each run fell on decides the direction of the effect -- and the CSVs are gone.
  **The retaken numbers** (rail mean 4.52 V, min 4.50 V; one pose file across all runs; K
  unchanged; two stereo cameras carried on the wrist): droop of `shoulder_lift` 5.00 -> 1.02 and
  `elbow_flex` 21.00 -> 1.00 counts. ~~against baselines where `err = pwm / K` held to the decimal
  (100/20, 315/15, sd 0.00). The instrument for a holding duty is the cerebellum-off droop~~ --
  **both sentences were withdrawn on 2026-09-09.** `err = pwm / K` is an **identity** when nothing
  but a PD law is in the path: it agreed to the decimal for arithmetic reasons, not physical ones.
  And sd 0.00 only says the arm is stationary, which **stiction produces as readily as balance**.
  Four runs at identical settings put `shoulder_lift`'s holding duty at 100 / 180 / 200 / 255, every
  one of them at sd 0.00. **A holding duty is not a value but a band.** Measured by approaching the
  same target from above and from below, that band is 18.0 counts wide on `shoulder_lift`
  (duty 360) and 12.9 counts on `elbow_flex` (duty 194) -- **the same order as the holding duty it
  brackets**. 100/315 is not the band's value; it is the edge reached from above (`--approach-from`,
  2026-09-09, rail 4.40-4.51 V, case 26-32 C, 25% RH). The ff readout still becomes path-dependent
  with `--cerebellum-cf-deadband`. And the `--cerebellum-ff-max` clamp being reached on two joints
  **was not inflated**: on a healthy rail `elbow_flex` still asks for 315 and still hits the 300
  clamp. **A clamp that binds in normal operation cannot tell normal from abnormal**, so it needs
  re-siting -- not yet done, because 315 is one edge of one pose's band and the legitimate maximum
  across poses is unmeasured.
- **The feedforward decayed instead of settling. Fixed; the fix is unmeasured.** In both learning
  runs it bled away with a time constant of minutes while the joint sat perfectly still, then
  snapped back to the clamp once the arm finally slipped. The cause was the rule, not the arm: the
  decay was gated on the eligibility trace but not on the climbing fibre, so the fixed point was
  `w = cf / leak` -- weights that need a standing error to hold them up. The residual that leaves is
  under a PWM count, narrower than the stick band, and inside that band the joint cannot move to
  report the error at all. Both halves are now gated on a live climbing fibre, and
  `--cerebellum-cf-deadband` (default 5.0) sets where the reflex counts as silent. **That default was measured on a
  healthy arm on 2026-09-08 and is too low.** It was derived as "above the leak's residual, below one
  encoder count times the joint stiffness" -- but position is quantised and the error never reaches
  zero. `wrist_flex` (K=10) alternates between +/-2 counts, so its feedback duty never drops below
  20 and never enters a band of 5; the ff hunted between 0 and 126 with a **69-second period**.
  Raising it to 25 removes the oscillation entirely (0.2 counts over 173 s, and within 4% of the
  true load) -- but the ff then freezes the moment the error enters the band, so **it stops being a
  measurement of the load and becomes a function of how far the transient got**. Which way to settle
  it is undecided. `0` restores the old behaviour. ~~Whether the stick band is a property of the
  gearboxes or an artefact of the supply collapsing is still what the re-measurement has to
  separate.~~ **Separated on 2026-09-09: the band appears on a healthy rail (4.44-4.51 V mean,
  4.40 V min), so it is not an artefact of the supply.** Its width is now measured -- approaching
  one target from above and from below leaves `shoulder_lift` resting 18.0 counts apart (duty 120
  vs 480) and `elbow_flex` 12.9 counts apart (duty 300 vs 106). **The band is the same order as the
  holding duty it brackets**, and four runs at identical settings, every one at sd 0.00, put that
  duty at 100/180/200/255. **Neither `err = pwm/K` holding nor sd 0.00 is evidence of balance**:
  the first is an identity when only a PD law is in the path, and the second says the arm is
  stationary, which stiction produces as readily as balance.
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
- **The bus produces comms errors while driving: 0% to 43% of samples across runs at identical
  settings. Cause unsettled.** Only visible since `fault_flags` was logged on 2026-09-09. The errors
  arrive in bursts of a fixed **~800 ms** (39-41 samples at 20 ms), 75% of which start within 60 ms
  of a PWM saturation. A burst past `--max-blind-ticks` zeroes the duty and the arm falls, which in
  the duty columns is indistinguishable from a gain that is merely too soft -- and was read as
  exactly that. **With the arm limp the rate is 0.00% on the same hardware**, so the trigger is the
  drive, not the bus at rest. `--current-read-divisor 4` cuts tick time 16% and leaves the rate
  unchanged, so it is not congestion; `--serial-timeout-ms 2` makes it **worse** (2 ms cuts replies
  that were going to arrive, and a late reply is read as the answer to the next question). The
  800 ms matches `Protection_Time`/`Over_Current_Protection_Time`, both 200 on every motor, if that
  register counts in 4 ms units -- **unverified**, and if true it means the servo's own overload
  protection fires in PWM mode. **Every number measured here sits on top of this.**
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
