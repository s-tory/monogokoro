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

"""Write a small LeRobotDataset of synthetic frames, so training can be exercised with no robot.

This exists to answer "does the training path work on this machine, at the shape I am going to
record in" before spending a day recording. Give it the camera count, resolution and joint count
you actually plan to use: ACT runs one ResNet18 per camera and does not resize, so a run at the
wrong shape measures the wrong machine.

    python examples/make_dummy_dataset.py --root /tmp/dummy --cameras 2 --height 360 --width 640

Then train from scratch, and finetune from what that produced:

    lerobot-train --dataset.repo_id=local/dummy --dataset.root=/tmp/dummy \\
        --policy.type=act --policy.device=xpu --batch_size=8 --steps=60 --save_freq=60 \\
        --output_dir=/tmp/scratch --wandb.enable=false --policy.push_to_hub=false

    lerobot-train --policy.path=/tmp/scratch/checkpoints/000060/pretrained_model \\
        --dataset.repo_id=local/dummy --dataset.root=/tmp/dummy --policy.device=xpu \\
        --batch_size=8 --steps=80 --output_dir=/tmp/ft --wandb.enable=false --policy.push_to_hub=false

The finetune has loaded the checkpoint if its loss *continues* the first run's rather than
restarting an order of magnitude higher; that is the only check that distinguishes a finetune
from a fresh run with extra steps.

Two things this file exists to remember:

- The frames carry a moving square on a gradient, not uniform noise. Noise does not compress, so
  a dataset of it reports video sizes and decode costs nothing like a real recording's.
- `finalize()` is not optional. Without it the episode metadata stays buffered, `meta/episodes`
  is never written, and loading the dataset falls through to the Hub -- where it fails with a
  404/401 on a local repo id, which reads as a network problem and is not one.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def frame_pair(height: int, width: int, phase: float, cameras: int) -> list[np.ndarray]:
    """One gradient background plus a moving white square, shifted per camera to fake parallax."""
    base = np.zeros((height, width, 3), np.uint8)
    base[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    base[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = int((np.sin(phase) * 0.4 + 0.5) * width)
    y = int((np.cos(phase) * 0.4 + 0.5) * height)
    half = max(height // 12, 4)
    base[max(0, y - half) : y + half, max(0, x - half) : x + half] = 255
    return [np.roll(base, 40 * i, axis=1) for i in range(cameras)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--repo-id", default="local/dummy")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--joints", type=int, default=6)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--frames", type=int, default=150, help="per episode; keep above the policy's chunk")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--task", default="pick the dummy block")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.root} exists; pass --overwrite to replace it")
        shutil.rmtree(args.root)

    names = (
        JOINTS[: args.joints] if args.joints <= len(JOINTS) else [f"joint_{i}" for i in range(args.joints)]
    )
    image_keys = [f"observation.images.cam_{i}" for i in range(args.cameras)]
    features = {
        **{
            key: {
                "dtype": "video",
                "shape": (args.height, args.width, 3),
                "names": ["height", "width", "channels"],
            }
            for key in image_keys
        },
        "observation.state": {"dtype": "float32", "shape": (args.joints,), "names": names},
        "action": {"dtype": "float32", "shape": (args.joints,), "names": names},
    }

    dataset = LeRobotDataset.create(
        args.repo_id, fps=args.fps, features=features, root=args.root, use_videos=True
    )
    for episode in range(args.episodes):
        phases = np.linspace(0, 4 * np.pi, args.frames)
        for phase in phases:
            images = frame_pair(args.height, args.width, float(phase), args.cameras)
            state = np.sin(phase + np.arange(args.joints) * 0.5).astype(np.float32)
            dataset.add_frame(
                {
                    **dict(zip(image_keys, images, strict=True)),
                    "observation.state": state,
                    "action": (state + 0.01).astype(np.float32),
                    "task": args.task,
                }
            )
        dataset.save_episode()
        print(f"  episode {episode + 1}/{args.episodes}")

    dataset.finalize()  # writes meta/episodes; without it the dataset looks remote and 404s
    print(f"\n{dataset.num_frames} frames, {dataset.num_episodes} episodes at {args.root}")


if __name__ == "__main__":
    main()
