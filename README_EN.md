# ものごころ=MONOGOKORO, Thinks of Things, 物心

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

|                                         | biology                    | here                                                   | rate   |
| --------------------------------------- | -------------------------- | ------------------------------------------------------ | ------ |
| answers contact with no loop at all     | preflex: muscle and tissue | three silicone finger caps per jaw, under a rubber cot | --     |
| fast local loop, brain not involved     | stretch reflex             | Rust daemon, `SCHED_FIFO`, isolated core               | 400 Hz |
| prediction, learned from its own errors | cerebellum                 | Vulkan compute on the iGPU, own thread                 | 200 Hz |
| slow loop through perception            | visual feedback            | ACT                                                    | ~30 Hz |

The ~13x separation between the reflex and ACT is roughly the one biology runs at, and it is the
reason the control law does not live in Python. The cerebellum sits between them and, like its
namesake, _outside_ the reflex arc -- it corrects the loop without ever being inside it.

The pontine relay has no row in that table, because it is not a rung on this ladder -- it is a
path. It carries context down from the top of the stack to the cerebellum's mossy fibres, closes no
loop of its own, and therefore has no rate. What it carries, and what it deliberately does not, is
[section 5](README_DETAILS_EN.md#5-a-pontine-relay-so-the-cerebellum-can-be-told-what-it-is-holding).

The top row is the cheapest thing in this repository and possibly the most load-bearing. Contact
transients are faster than any loop on the list: a fingertip meeting an object produces its force
spike well inside the reflex's 2.5 ms tick, so whatever answers it _first_ cannot be a controller at
all. Biology's answer is the **preflex** -- the intrinsic mechanical response of muscle and tissue,
at zero latency, before any reflex arc has been traversed. Here it is three silicone finger caps
stacked on each jaw -- the thick anti-chapping kind, ~2 mm wall (ours are Rimikuru finger
supporters), not the thin office cots sold for counting paper -- and it is why a soft-fingered
animal can be careless with a fragile object in a way this arm cannot. **Bare silicone split after
about two weeks** (found 2026-09-16); a **natural-rubber finger cot** with a dotted grip surface
now goes over the top of it.

<p align="center">
  <img src="media/fingertip_rubber_20260916.jpg" width="360"
       alt="The SO-101 gripper held in a hand, orange natural-rubber finger cots with a dotted grip surface pulled over each jaw, covering the silicone caps underneath" />
</p>

It also gives the gripper a finer sense of touch, which is less obvious. Grip force was always
readable -- after contact the commanded position keeps advancing while the achieved one stops, and
`pwm = K * err` follows the squeeze. What a compliant fingertip changes is the _scale_: the same
range of force now spreads across far more encoder counts, which is exactly why a load cell has a
flexure, to turn force into a displacement large enough to measure. The signal was already there;
the padding gives it a finer ruler. By how much, on this arm, is not measured yet -- and the
candidate that could eat the whole effect is the static friction described under [Known limitations](#known-limitations).

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

**Each layer is written up in [README_DETAILS_EN.md](README_DETAILS_EN.md)** — how it is built, why, and what did not work.

## Quick start

```bash
# 1. Build, and install the daemon as a systemd unit, once. The unit grants
#    realtime itself, so a rebuild no longer loses it (setcap did).
#    Edit User= and the paths in so101-impedance.service for your machine first.
cd rust/so101_impedance_ctrl && cargo build --release
sudo install -m 0644 so101-impedance.service /etc/systemd/system/ && sudo systemctl daemon-reload

# 2. Start the daemon. It must be running before any Python attaches.
#    It never starts at boot -- start it when you are there to watch the arm.
sudo systemctl start so101-impedance
journalctl -u so101-impedance -f   # "acquired SCHED_FIFO priority 99" means it got realtime

# 3. Confirm telemetry, with the arm torque-limp and safe to move by hand.
python examples/check_so101_impedance.py --shm-name so101_impedance
```

Step 3 is the first thing that needs the Python environment; building it is
[`SETUP_XPU.md`](SETUP_XPU.md). The daemon in steps 1-2 is Rust and runs without it.

Then teleoperate or record with `--robot.type=so101_follower_impedance`; both fill in per-joint K/D
from the robot's config automatically.

The cerebellum is opt-in. Add to the unit's `ExecStart=` and `sudo systemctl daemon-reload` (a full path -- `~` is not expanded there; needs `glslc` to build, a Vulkan ICD to run, and a
housekeeping core that is **not** `--cpu-core`):

```bash
  --cerebellum-backend gpu --cerebellum-cpu-core 1 \
  --cerebellum-weights /home/<you>/.local/share/so101/cerebellum.bin
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

The full A/B measurements are in [README_DETAILS_EN.md](README_DETAILS_EN.md).

<p align="center">
  <img src="media/readme/wrist_stereo.jpg" width="420"
       alt="Two U20CAM-1080P camera boards on a printed bracket at the SO-101 wrist, each held by two standoffs at diagonal corners, with the silicone fingertips behind them" />
</p>

The two wrist-stereo rows above are legible in this photograph. The cameras are two InnoMaker
U20CAM-1080P boards (1080P USB 2.0 UVC). That the lenses are wide is visible. That each camera
board is held by only two standoffs at diagonal corners is also visible -- **M2.6 hex standoff spacers,
self-tapped straight into the printed part** -- and that is the most likely source of the constant
offset riding on the disparity. Neither the registers nor the feature matching said that much.

## Known limitations

- [The pontine context is unverified on hardware](README_DETAILS_EN.md#the-pontine-context-is-unverified-on-hardware)
- [The cerebellum droop numbers were retaken](README_DETAILS_EN.md#the-cerebellum-droop-numbers-were-retaken) -- the 2026-08-28 numbers are withdrawn.
- [The feedforward decayed instead of settling](README_DETAILS_EN.md#the-feedforward-decayed-instead-of-settling) -- fixed, the fix unmeasured.
- [Touch has no where and no slip](README_DETAILS_EN.md#touch-has-no-where-and-no-slip)
- [The mossy fibres bound what can be learned](README_DETAILS_EN.md#the-mossy-fibres-bound-what-can-be-learned) -- the camera-derived features are missing.
- [Demonstrations do not label the layers below](README_DETAILS_EN.md#demonstrations-do-not-label-the-layers-below)
- **Open-loop PWM** -- the STS3215 has no host-streamable torque register. Noisier than true torque control, and a hardware constraint rather than a choice.
- [The bundled supply collapses at the defaults](README_DETAILS_EN.md#the-bundled-supply-collapses-at-the-defaults) -- one 833 ms collapse every 329 s. **Swapping in an off-the-shelf 5 V 6 A brick clears it at the defaults** (a plug change, measured).
- [The comms errors were the supply](README_DETAILS_EN.md#the-comms-errors-were-the-supply)
- [A better supply did not remove every collapse](README_DETAILS_EN.md#a-better-supply-did-not-remove-every-collapse)
- [The single-tick bit0 survived the 5 V swap and went at 7.4 V](README_DETAILS_EN.md#the-single-tick-bit0-survived-the-5-v-swap-and-went-at-74-v) -- limp, none in 603 s.
- [A 7.4 V supply, built](README_DETAILS_EN.md#a-74-v-supply-built) -- for more margin still. Needs a soldering iron.
- **The daemon is not part of the Python build.** It is a separate Cargo project, deployed by hand.
- [Interactive calibration is not implemented](README_DETAILS_EN.md#interactive-calibration-is-not-implemented)
- [An iGPU training OOM takes the desktop down](README_DETAILS_EN.md#an-igpu-training-oom-takes-the-desktop-down) -- a cgroup does not prevent it.

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
