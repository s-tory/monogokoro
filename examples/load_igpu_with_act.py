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

"""Put the GPU load of a running ACT policy on the device, without a robot or a dataset.

The cerebellum is Vulkan compute on the same iGPU an ACT policy runs on, and that GPU has one
compute queue shared with graphics. Whether a policy's inference starves the 200 Hz cerebellar
thread is therefore a question about *this* load, and it is the only load worth asking about: the
arm never moves while a training run holds the GPU, so a co-scheduled trainer is a state the system
does not operate in.

Two things this deliberately does not do. It does not normalise -- the tensors are random and the
outputs are meaningless; only the shapes, the dtype and the cadence matter for occupancy. And it
does not run the model once per control step: ACT emits `n_action_steps` actions per invocation
(100 by default), so a 30 Hz robot runs one forward pass every 3.3 s and is idle in between. A
continuous 30 Hz stream would be a load the robot never produces -- pass `--period 0` if you want
that upper bound anyway.

    python examples/load_igpu_with_act.py --measure          # one-pass latency, then exit
    python examples/load_igpu_with_act.py --seconds 60       # hold the real cadence for a minute
"""

import argparse
import contextlib
import statistics
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# `SO101ImpedanceFollower`: (pos + current_avg + pwm_cmd + ff_pwm) x 6 + rail volts + cerebellum
# flags, and (pos + k + d) x 6 out. See the observation/action tables in the top-level README.
STATE_DIM = 26
ACTION_DIM = 18


def build_policy(args: argparse.Namespace) -> tuple[ACTPolicy, dict[str, torch.Tensor]]:
    image_keys = [f"{OBS_IMAGES}.cam{i}" for i in range(args.cameras)]
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **{
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.height, args.width))
                for key in image_keys
            },
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    )
    policy = ACTPolicy(config).to(args.device).eval()

    batch = {OBS_STATE: torch.randn(1, STATE_DIM, device=args.device)}
    for key in image_keys:
        batch[key] = torch.rand(1, 3, args.height, args.width, device=args.device)
    return policy, batch


def synchronize(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def one_pass(policy: ACTPolicy, batch: dict[str, torch.Tensor], args: argparse.Namespace) -> float:
    # XPU autocast rejects float32 outright, so fp32 has to run without the context manager rather
    # than through it: asking for a no-op cast is an error, not a no-op.
    if args.torch_dtype == torch.float32:
        cast = contextlib.nullcontext()
    else:
        cast = torch.autocast(device_type=args.device.split(":")[0], dtype=args.torch_dtype)
    start = time.perf_counter()
    with torch.no_grad(), cast:
        policy.predict_action_chunk(batch)
    synchronize(args.device)
    return (time.perf_counter() - start) * 1e3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", action="store_true", help="Time N passes back to back and exit.")
    parser.add_argument("--passes", type=int, default=30, help="Passes to time with --measure.")
    parser.add_argument("--seconds", type=float, default=60.0, help="How long to hold the load.")
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Seconds between passes. Default: n_action_steps / --fps, the cadence a robot "
        "actually produces. Zero means back to back.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.torch_dtype = getattr(torch, args.dtype)

    policy, batch = build_policy(args)
    period = args.period
    if period is None:
        period = policy.config.n_action_steps / args.fps

    for _ in range(args.warmup):
        one_pass(policy, batch, args)

    label = f"{args.device} {args.dtype}, {args.cameras}x{args.height}x{args.width}"
    if args.measure:
        times = [one_pass(policy, batch, args) for _ in range(args.passes)]
        times.sort()
        print(
            f"ACT forward [{label}]: {len(times)} passes, "
            f"{statistics.mean(times):.1f} ms mean / {times[len(times) // 2]:.1f} ms median / "
            f"{times[-1]:.1f} ms max"
        )
        return

    print(f"ACT forward [{label}]: one pass every {period:.2f} s for {args.seconds:.0f} s", flush=True)
    deadline = time.perf_counter() + args.seconds
    times = []
    while time.perf_counter() < deadline:
        times.append(one_pass(policy, batch, args))
        print(f"  pass {len(times):3d}: {times[-1]:7.1f} ms", flush=True)
        if period > 0:
            time.sleep(max(0.0, period - times[-1] / 1e3))
    print(f"held {len(times)} passes, {statistics.mean(times):.1f} ms mean / {max(times):.1f} ms max")


if __name__ == "__main__":
    main()
