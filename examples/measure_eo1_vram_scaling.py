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

"""Fit peak training memory against batch size for EO-1, and say what fits on a given card.

The companion of `measure_act_vram_scaling.py`, for a policy three orders of magnitude larger.
EO-1 carries a full Qwen2.5-VL backbone -- 3.755 B parameters, of which 0.669 B is the vision
tower -- and exposes no freezing knob of its own, so the only way to train it on a small card is
LeRobot's policy-agnostic PEFT path. `--lora` measures that; the default measures full training.

Full training does not fit on anything small, and arithmetic says so before a run does: at
bf16 weights + bf16 grads + fp32 AdamW moments the floor is 12 bytes per parameter, so 3.755 B
parameters need 42 GiB before a single byte of activations. That is over this machine's entire
shared pool, which is why the full-training line cannot be measured here at all -- the numbers
below for `--lora` are the useful ones.

**Simulating a card you do not have.** `--target-gib` caps the allocator with
`torch.xpu.set_per_process_memory_fraction`, so a batch that would not fit on that card raises
`torch.OutOfMemoryError` here. Use it rather than a cgroup: on an integrated GPU, device
allocations are *not* charged to the process cgroup (measured 2026-09-11 -- 2 GiB on the XPU moved
`memory.current` by 0.00 GiB), so `systemd-run -p MemoryMax=` cannot bound them and would report a
false pass. The allocator cap also fails as a Python exception instead of reaching the system OOM
killer, which on a shared-memory iGPU has taken the desktop down before.

    python examples/measure_eo1_vram_scaling.py --lora --target-gib 11.5
    python examples/measure_eo1_vram_scaling.py --lora --batches 1,2,4,8 --target-gib 11.5

Random pixels and random states: only shapes, dtype and optimiser state move peak memory.
"""

import argparse
import gc
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.eo1.configuration_eo1 import EO1Config
from lerobot.policies.eo1.modeling_eo1 import EO1Policy
from lerobot.policies.eo1.processor_eo1 import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_STATE_TOKEN,
    EO1_SPECIAL_TOKENS,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
)


def build_policy(args) -> EO1Policy:
    image_names = [f"observation.images.cam_{i}" for i in range(args.cameras)]
    cfg = EO1Config(
        input_features={
            **{
                name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.height, args.width))
                for name in image_names
            },
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))},
        device=args.device,
        dtype=args.dtype,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    return EO1Policy(cfg)


def apply_lora(policy: EO1Policy, args):
    """Wrap the VLM backbone in LoRA adapters and freeze everything else in it."""
    from peft import LoraConfig, inject_adapter_in_model

    backbone = policy.model.vlm_backbone
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha or args.lora_r,
        target_modules=args.lora_targets,
        lora_dropout=0.0,
        bias="none",
    )
    # `get_peft_model` would replace the backbone with a PeftModel wrapper, and EO-1 reaches into
    # `vlm_backbone.model` expecting the inner Qwen2_5_VLModel -- the wrapper's `.model` is the
    # outer ForConditionalGeneration instead, and the forward dies on a missing
    # `last_hidden_state`. `inject_adapter_in_model` mutates in place and leaves every attribute
    # path intact, which is what the policy's own code needs.
    for param in backbone.parameters():
        param.requires_grad = False
    inject_adapter_in_model(lora_cfg, backbone)
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(
        f"LoRA r={lora_cfg.r} targets={args.lora_targets}: trainable {trainable / 1e6:.1f} M / {total / 1e9:.3f} B"
    )
    return policy


def build_batch(policy: EO1Policy, args, batch: int) -> dict:
    """Build one supervised batch through the real Qwen processor, as the EO-1 pipeline does."""
    from transformers import Qwen2_5_VLProcessor

    cfg = policy.config
    proc = Qwen2_5_VLProcessor.from_pretrained(cfg.vlm_base, use_fast=cfg.use_fast_processor)
    proc.tokenizer.add_tokens(EO1_SPECIAL_TOKENS, special_tokens=True)

    images = torch.randint(0, 256, (batch, args.cameras, 3, args.height, args.width), dtype=torch.uint8)
    text = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}{args.task}{TASK_VLA_TOKEN}"
    answer = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN * cfg.chunk_size}{ACTION_END_TOKEN}"
    messages = [
        [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": images[i, c]} for c in range(args.cameras)],
                    {"type": "text", "text": text},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        for i in range(batch)
    ]
    inputs = proc.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        padding_side="right",
        min_pixels=cfg.image_min_pixels,
        max_pixels=cfg.image_max_pixels,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    dev = args.device
    out = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    out["state_token_id"] = proc.tokenizer.convert_tokens_to_ids(DEFAULT_STATE_TOKEN)
    out["action_token_id"] = proc.tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_TOKEN)
    out["observation.state"] = torch.rand(batch, args.state_dim, device=dev)
    out["action"] = torch.rand(batch, cfg.chunk_size, args.action_dim, device=dev)
    return out


def measure(policy: EO1Policy, args, batch: int) -> tuple[float, float, int]:
    """Return (peak GiB, seconds per step, sequence length), or raise torch.OutOfMemoryError."""
    dev_mod = getattr(torch, args.device)
    sample = build_batch(policy, args, batch)
    seq_len = int(sample["input_ids"].shape[1])
    optimiser = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-5)

    dev_mod.empty_cache()
    dev_mod.reset_peak_memory_stats()
    # Time the last tenth of the run, not one step. A laptop drops to its sustained power limit
    # inside a minute, so an early step reports a throughput the machine cannot hold -- and one
    # step alone is noisy. Averaging the tail measures what a long run will actually get.
    timed_from = max(1, args.iters - max(1, args.iters // 10))
    start = None
    for i in range(args.iters):
        if i == timed_from:
            dev_mod.synchronize()
            start = time.perf_counter()
        loss, _ = policy.forward(sample)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    dev_mod.synchronize()
    elapsed = (time.perf_counter() - start) / (args.iters - timed_from)
    peak = dev_mod.max_memory_allocated() / 2**30

    del sample, optimiser, loss
    gc.collect()
    dev_mod.empty_cache()
    return peak, elapsed, seq_len


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="xpu")
    p.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float32"])
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--state-dim", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--task", default="pick up the chip")
    p.add_argument("--batches", default="1,2,4,8")
    p.add_argument("--iters", type=int, default=3, help="steps per batch; the last one is timed")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--lora", action="store_true", help="freeze the backbone and train LoRA adapters")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None)
    p.add_argument("--lora-targets", default="all-linear")
    p.add_argument(
        "--target-gib",
        type=float,
        required=True,
        help=(
            "cap the allocator to simulate a card of this size (e.g. 11.5 for a 12GB B580). "
            "Required, and deliberately so: without a cap this script reached the system OOM "
            "killer on 2026-09-11 and it took an editor and part of a chat client with it. "
            "Pass the device's own total to run uncapped, and mean it."
        ),
    )
    args = p.parse_args()

    dev_mod = getattr(torch, args.device)
    total = dev_mod.get_device_properties(0).total_memory / 2**30
    dev_mod.set_per_process_memory_fraction(min(args.target_gib / total, 1.0))
    print(
        f"device total {total:.2f} GiB -> capped at {args.target_gib} GiB (simulating that card)", flush=True
    )

    policy = build_policy(args)
    if args.lora:
        policy = apply_lora(policy, args)
    policy.to(args.device)

    params = sum(t.numel() for t in policy.parameters())
    print(
        f"EO-1 {params / 1e9:.3f} B params  ·  {args.cameras} camera(s) {args.width}x{args.height}  ·  "
        f"chunk_size={policy.config.chunk_size}  ·  dtype={args.dtype}  ·  device={args.device}"
        f"{'  ·  gradient checkpointing' if args.gradient_checkpointing else ''}",
        flush=True,
    )
    print(f"\n{'batch':>6} {'seq':>6} {'peak GiB':>10} {'GiB/sample':>11} {'s/step':>8}", flush=True)

    points: list[tuple[int, float]] = []
    for b in (int(x) for x in args.batches.split(",")):
        try:
            peak, secs, seq = measure(policy, args, b)
        except torch.OutOfMemoryError:
            cap = args.target_gib
            print(f"{b:>6} {'':>6} {'OOM':>10}   <- does not fit in {cap:.1f} GiB", flush=True)
            dev_mod.empty_cache()
            break
        points.append((b, peak))
        print(f"{b:>6} {seq:>6} {peak:>10.2f} {peak / b:>11.3f} {secs:>8.2f}", flush=True)

    if len(points) >= 2:
        (b0, p0), (b1, p1) = points[0], points[-1]
        a = (p1 - p0) / (b1 - b0)
        c = p0 - a * b0
        print(f"\nline: peak = {a:.3f} * batch + {c:.2f} GiB")
        if args.target_gib:
            fits = int((args.target_gib - c) / a) if a > 0 else 0
            print(
                f"  -> {args.target_gib} GiB affords batch {fits} by this line (verify it; the fit is two points)"
            )


if __name__ == "__main__":
    main()
