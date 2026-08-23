# monogokoro

A fork of [LeRobot](https://github.com/huggingface/lerobot) that makes the SO-101 **compliant**:
its joints and its gripper yield to contact instead of driving through it, and the ACT policy both
sees the resulting forces and commands how hard to resist them.

Forked at upstream [`0d383d09`](https://github.com/huggingface/lerobot/commit/0d383d09) (2026-07-24,
on the 0.6.1 line). Everything upstream still works unchanged -- this adds a robot, a real-time
controller, and two extra dimensions of what ACT reasons about. It does not modify any policy
architecture.

## Why

Stock SO-101 control writes `Goal_Position` and lets the servo's internal PID drive there. That
controller has no notion of contact: blocked by an object, it keeps increasing effort toward a
position it will never reach. A paper cup is crushed before the arm can be said to have felt it.
And the policy has no vocabulary for the difference between *press firmly* and *hold gently* --
`Goal_Position` is the only thing it can say.

Compliance fixes both ends. The arm is driven by a spring-damper law instead of a position
servo, so contact produces a bounded force. And because that law is parameterised, the policy can
choose its own stiffness per joint, per timestep.

## What this fork adds

```
   operator's hand                                                        camera
        │  ▲                                                                 │
   ┌────┴──┴────┐                                                            │
   │ SO-101     │  gripper: force feedback ──┐                               │
   │ leader     │  5 joints: backdriven      │                               │
   └────────────┘                            │                               │
        │ pos                                │                               │
        ▼                                    │                               ▼
   ╔═══════════════════════════════════════╗ │                    ┌──────────────────┐
   ║ Rust RT daemon  ·  400 Hz             ║◄┘                    │ ACT              │
   ║ SCHED_FIFO, isolated core             ║                      │                  │
   ║                                       ║   shared memory      │ in:  images      │
   ║  pwm = K·Δx + D·Δv     (all 6 motors) ║◄────── seqlock ─────► │      pos    ×6   │
   ║  owns both serial buses               ║                      │      current×6   │
   ╚═══════════════════════════════════════╝                      │                  │
        │ PWM              ▲ pos, current                         │ out: pos ×6 ┐    │
        ▼                  │                                      │      K   ×6 ├ ×N │
   ┌────────────┐                                                 │      D   ×6 ┘    │
   │ SO-101     │  6× Feetech STS3215                             └──────────────────┘
   │ follower   │  all impedance-controlled
   └────────────┘
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
crushes the cup, so it runs the same K/D law as the arm with a much softer default K.

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

| | stock SO-101 | this fork |
| --- | --- | --- |
| `observation.state` | `pos` ×6 → **6** | `pos` ×6 + `current_avg` ×6 → **12** |
| `action` (per chunk step) | `pos` ×6 → **6** | `pos` ×6 + `K` ×6 + `D` ×6 → **18** |

**Input.** Each motor's `Present_Current` is sampled one servo per tick round-robin and averaged in
Rust over a fixed window (~0.5 s at the defaults). ACT reads the pre-averaged value at camera rate,
so it sees contact force without the per-tick noise.

**Output.** ACT's action chunking is unchanged -- it still predicts `chunk_size` steps ahead -- but
each step now carries a per-joint stiffness and damping alongside the position. The policy chooses
its own compliance over the horizon; K/D are clamped in Python and again in Rust before reaching a
servo.

## Quick start

```bash
# 1. Build and grant the one privileged capability. setcap is lost on every rebuild.
cd rust/so101_impedance_ctrl && cargo build --release
sudo setcap cap_sys_nice+ep ./target/release/so101_impedance_ctrl

# 2. Start the daemon. It must be running before any Python attaches.
./target/release/so101_impedance_ctrl \
  --port /dev/ttyACM0 --shm-name so101_impedance --cpu-core 3 --priority 99

# 3. Confirm telemetry, with the arm torque-limp and safe to move by hand.
python examples/check_so101_impedance.py --shm-name so101_impedance
```

Then teleoperate or record with `--robot.type=so101_follower_impedance`; both fill in per-joint K/D
from the robot's config automatically.

- Setting up the isolated core: [`rust/so101_impedance_ctrl/PREEMPT_RT.md`](rust/so101_impedance_ctrl/PREEMPT_RT.md)
- Tuning gains, force-feedback bring-up, protocol notes: [`rust/so101_impedance_ctrl/README.md`](rust/so101_impedance_ctrl/README.md)
- General LeRobot usage (recording, training, eval): [`AGENT_GUIDE.md`](AGENT_GUIDE.md)

## Measured, not assumed

Several constants here were settled on real hardware after their documented or intuitive values
turned out to be wrong. They are specific to this arm and worth re-measuring on another:

| constant | value | how it was settled |
| --- | --- | --- |
| PWM sign bit | **10** | bit 11 (per upstream's docstring) does not reverse the joint -- it is consumed as extra magnitude |
| `--invert-pwm` | **true** | with the right sign bit, positive duty still lowers the encoder |
| per-joint K | 10/20/15/10/8/5 | holding at K=1 makes the reported PWM read out as each joint's gravity+friction duty |
| per-joint D | ≈ K/40 | bounded from above by the velocity quantisation noise floor, not by stability |

The tooling for re-deriving them ships too: `--probe-direction` measures drive direction with a
bounded, auto-aborting nudge, and the checker's live table separates "too soft" from "driven the
wrong way", which look identical from across the room.

## Known limitations

- **Recorded K/D are constant.** A torque-off leader is a position sensor and nothing else, so
  demonstrations are labelled with the config's default gains. ACT trained on such data learns to
  reproduce those gains, not to vary them. The leader gripper's force feedback is the first step
  toward fixing this; deriving stiffness from cross-demonstration variance is the likely next one.
- **Open-loop PWM** is noisier than torque control, and no gravity feedforward is implemented, so a
  gravity-loaded joint droops by `holding_duty / K`.
- **The daemon is not part of the Python build.** It is a separate Cargo project, deployed by hand.
- **Interactive calibration** for the impedance robot is not implemented; copy a calibration
  produced with the stock `so101_follower` against the same servos.

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
