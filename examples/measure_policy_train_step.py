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

"""Time one training step of a policy on this host, with no dataset and no robot.

The number this exists to produce is a *ratio* between two policies on one machine. The per-policy
table in `AGENT_GUIDE.md` was measured elsewhere, with SGD, at batch 1, so using it to decide "is
this policy N times more expensive than the one I have working" compares three things at once.
Here both policies run through the same optimiser (AdamW, LeRobot's default), the same batch, the
same camera count and the same dtype, on the machine that has to pay for it.

Random, unnormalised tensors: only shapes, dtype and the optimiser state matter for step cost.

    python examples/measure_policy_train_step.py --policy act --batch 16
    python examples/measure_policy_train_step.py --policy smolvla --batch 2   # needs transformers
"""

import argparse
import contextlib
import statistics
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import (
    ACTION,
    OBS_IMAGES,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# `SO101ImpedanceFollower`: (pos + current_avg + pwm_cmd + ff_pwm) x 6 + rail volts + cerebellum
# flags in, (pos + k + d) x 6 out. See the observation/action tables in the top-level README.
STATE_DIM = 26
ACTION_DIM = 18


def build(args):
    features = {
        "input_features": {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            **{
                f"{OBS_IMAGES}.cam{i}": PolicyFeature(
                    type=FeatureType.VISUAL, shape=(3, args.height, args.width)
                )
                for i in range(args.cameras)
            },
        },
        "output_features": {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
    }

    if args.policy == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        config = ACTConfig(**features)
        policy = ACTPolicy(config)
    elif args.policy == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        # Weights are random either way; `load_vlm_weights` only decides whether the pretrained VLM
        # is fetched, and a randomly initialised backbone of the same shape costs the same per step.
        config = SmolVLAConfig(**features, load_vlm_weights=args.load_vlm_weights)
        policy = SmolVLAPolicy(config)
    else:
        raise ValueError(args.policy)

    policy = policy.to(args.device).train()
    chunk = policy.config.chunk_size
    batch = {
        OBS_STATE: torch.randn(args.batch, STATE_DIM, device=args.device),
        ACTION: torch.randn(args.batch, chunk, ACTION_DIM, device=args.device),
        "action_is_pad": torch.zeros(args.batch, chunk, dtype=torch.bool, device=args.device),
        "task": ["pick up the chip"] * args.batch,
    }
    for i in range(args.cameras):
        batch[f"{OBS_IMAGES}.cam{i}"] = torch.rand(args.batch, 3, args.height, args.width, device=args.device)

    # A language-conditioned policy is handed tokens, not a string: tokenising is a processor step
    # upstream of the model. The ids are arbitrary here, but the sequence length is not -- it sets
    # how many prefix tokens the attention has to carry.
    tokens = getattr(policy.config, "tokenizer_max_length", None)
    if tokens is not None:
        batch[OBS_LANGUAGE_TOKENS] = torch.randint(0, 1000, (args.batch, tokens), device=args.device)
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.ones(
            args.batch, tokens, dtype=torch.bool, device=args.device
        )
    return policy, batch


def synchronize(device: str) -> None:
    if device.startswith("xpu"):
        torch.xpu.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy", default="act", choices=["act", "smolvla"])
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--load-vlm-weights", action="store_true", help="smolvla: fetch the real VLM.")
    args = parser.parse_args()
    torch_dtype = getattr(torch, args.dtype)

    policy, batch = build(args)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    cast = (
        contextlib.nullcontext()
        if torch_dtype == torch.float32
        else torch.autocast(device_type=args.device.split(":")[0], dtype=torch_dtype)
    )

    def step() -> float:
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with cast:
            loss, _ = policy.forward(batch)
        loss.backward()
        optimizer.step()
        synchronize(args.device)
        return (time.perf_counter() - start) * 1e3

    for _ in range(args.warmup):
        step()
    if args.device.startswith("xpu"):
        torch.xpu.reset_peak_memory_stats()

    times = [step() for _ in range(args.steps)]
    peak = torch.xpu.max_memory_allocated() / 1e9 if args.device.startswith("xpu") else float("nan")
    mean = statistics.mean(times)
    params = sum(p.numel() for p in policy.parameters()) / 1e6
    print(
        f"{args.policy} train step [{args.device} {args.dtype}, batch {args.batch}, "
        f"{args.cameras}x{args.height}x{args.width}, AdamW]: {params:.0f} M params, "
        f"{mean:.1f} ms mean / {min(times):.1f} min / {max(times):.1f} max, "
        f"{args.batch / mean * 1e3:.1f} samples/s, peak {peak:.2f} GB"
    )


if __name__ == "__main__":
    main()
