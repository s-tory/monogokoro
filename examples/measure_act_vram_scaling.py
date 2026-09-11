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

"""Fit peak training memory against batch size for ACT, to size a GPU before buying one.

`measure_policy_train_step.py` answers "how long is one step"; this answers "what fits". The
output is a straight line, `peak = a*batch + b` GiB, which is what turns a card's advertised
VRAM into a batch you can actually run. Card memory is never all usable -- the driver and, on a
card driving a display, the framebuffer take a slice -- so `--usable-fraction` applies a haircut
before reporting the batch each capacity affords.

Camera count and resolution are arguments because they dominate the slope: ACT runs one ResNet18
per camera and does not resize, so activation memory scales with total pixels, not with the
policy's parameter count. Measure the shape you are going to record in, not a default.

Random tensors: only shapes, dtype and optimiser state matter for peak memory.

    python examples/measure_act_vram_scaling.py --device xpu --cameras 2 --height 360 --width 640
    python examples/measure_act_vram_scaling.py --device cuda --batches 8,16,32 --capacities 12,24

On an integrated GPU the pool is shared with system RAM, so a large batch can succeed here and
still be impossible on a discrete card with less dedicated memory -- and an out-of-memory kill
there takes the desktop with it.

A cgroup does not stop that, though this docstring recommended one until 2026-09-11. Device
allocations are not charged to the process cgroup: 2 GiB on the XPU moved a scope's
`memory.current` by 0.00 GiB, so `MemoryMax` bounds the host side only. Cap the allocator instead,
which fails as a `torch.OutOfMemoryError` in-process rather than reaching the kernel:

    import torch
    total = torch.xpu.get_device_properties(0).total_memory / 2**30
    torch.xpu.set_per_process_memory_fraction(11.5 / total)   # behave like an 11.5 GiB card

`measure_eo1_vram_scaling.py` takes that ceiling as a required `--target-gib`, and is the better
model to copy.
"""

import argparse
import gc
import time

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


def _backend(device: str):
    """Return the torch.<backend> module owning memory stats for this device, or None."""
    mod = getattr(torch, device, None)
    return mod if mod is not None and hasattr(mod, "max_memory_allocated") else None


def build_policy(args) -> ACTPolicy:
    image_names = [f"observation.images.cam_{i}" for i in range(args.cameras)]
    cfg = ACTConfig(
        input_features={
            **{
                name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.height, args.width))
                for name in image_names
            },
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.action_dim,))},
        device=args.device,
    )
    return ACTPolicy(cfg).to(args.device)


def measure(policy: ACTPolicy, args, batch: int, backend) -> tuple[float, float]:
    """Return (peak GiB, seconds per step) for one batch size, or raise on OOM."""
    cfg = policy.config
    dev = args.device
    sample = {
        **{
            f"observation.images.cam_{i}": torch.rand(batch, 3, args.height, args.width, device=dev)
            for i in range(args.cameras)
        },
        "observation.state": torch.rand(batch, args.state_dim, device=dev),
        "action": torch.rand(batch, cfg.chunk_size, args.action_dim, device=dev),
        "action_is_pad": torch.zeros(batch, cfg.chunk_size, dtype=torch.bool, device=dev),
    }
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-5)
    if backend:
        backend.empty_cache()
        backend.reset_peak_memory_stats()

    # One warm-up step allocates the optimiser state, which would otherwise land in the timed step.
    for i in range(args.iters):
        if i == args.iters - 1 and backend:
            backend.synchronize()
            start = time.perf_counter()
        loss, _ = policy.forward(sample)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)
    if backend:
        backend.synchronize()
    elapsed = time.perf_counter() - start
    peak = backend.max_memory_allocated() / 2**30 if backend else float("nan")

    del sample, optimiser, loss
    gc.collect()
    if backend:
        backend.empty_cache()
    return peak, elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="xpu", help="xpu, cuda, ... (needs memory stats to report peaks)")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--state-dim", type=int, default=6)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--batches", default="2,4,8,16,24,32,48")
    p.add_argument("--capacities", default="10,12,16,24", help="card VRAM in GB to report batches for")
    p.add_argument("--usable-fraction", type=float, default=0.93)
    p.add_argument("--iters", type=int, default=3, help="steps per batch; the last one is timed")
    args = p.parse_args()

    backend = _backend(args.device)
    if backend is None:
        print(f"note: torch.{args.device} exposes no memory stats; peaks will be nan")

    policy = build_policy(args)
    params = sum(t.numel() for t in policy.parameters())
    print(
        f"ACT {params / 1e6:.1f} M params  ·  {args.cameras} camera(s) {args.width}x{args.height}  ·  "
        f"chunk_size={policy.config.chunk_size}  ·  device={args.device}"
    )
    print(f"\n{'batch':>6} {'peak GiB':>10} {'GiB/sample':>11} {'s/step':>8}")

    points: list[tuple[int, float]] = []
    for batch in (int(b) for b in args.batches.split(",")):
        try:
            peak, elapsed = measure(policy, args, batch, backend)
        except (torch.OutOfMemoryError, RuntimeError) as exc:  # RuntimeError: some backends still raise it
            print(f"{batch:6d}   stopped: {type(exc).__name__}: {str(exc)[:60]}")
            break
        points.append((batch, peak))
        print(f"{batch:6d} {peak:10.2f} {peak / batch:11.3f} {elapsed:8.2f}")

    if len(points) < 2 or any(p != p for _, p in points):  # NaN check without importing math
        return

    # Least squares on peak = a*batch + b. The intercept is what the model and optimiser cost
    # before any activations, so it is the part a bigger batch never amortises away.
    n = len(points)
    sx = sum(b for b, _ in points)
    sy = sum(v for _, v in points)
    sxx = sum(b * b for b, _ in points)
    sxy = sum(b * v for b, v in points)
    a = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    b = (sy - a * sx) / n
    print(f"\nfit: peak ≈ {a:.3f}·batch + {b:.2f} GiB")

    print(f"\nbatch that fits, at {args.usable_fraction:.0%} of advertised VRAM:")
    for gb in (float(c) for c in args.capacities.split(",")):
        usable = gb * args.usable_fraction
        fits = int((usable - b) / a)
        print(f"  {gb:5.0f} GB card   usable {usable:5.1f} GiB   ->  batch {max(fits, 0):3d}")


if __name__ == "__main__":
    main()
