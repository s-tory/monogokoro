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

"""Pins the frame check on the pose file in `examples/measure_so101_cerebellum.py`.

`Present_Position` is two different numbers depending on `Operating_Mode`: the raw encoder count
under PWM, and that count minus `Homing_Offset` under POSITION. `capture` used not to record which
one it had written, and `hold` used to read the file as bare ticks. On 2026-09-08 that drove the
arm at saturated PWM toward a target 1737 counts (152 deg) from where it actually was, on
`shoulder_pan`, and the run was stopped by a human hearing it -- no telemetry threshold fired,
because saturated PWM against a distant target is what a correct controller does.

The reason it had never fired before is the part worth pinning. `lerobot-calibrate` leaves the
servos in POSITION mode; every `hold` leaves them in PWM. So capture-then-hold agreed in every
session that had already run a hold that day, and disagreed exactly once: on the first run after a
calibration. **A bug masked by having run the tool before is not found by running the tool again**,
which is why this is a test on the file format rather than another careful run on hardware.

These tests do not check that the frame is *correct* -- nothing off-hardware can. They check that an
unlabelled or mislabelled pose is refused instead of interpreted, which is the part that turns a
silent wrong number into a stop.
"""

import importlib.util
import json
from pathlib import Path

import pytest

MEASURE = Path(__file__).resolve().parents[2] / "examples" / "measure_so101_cerebellum.py"

pytest.importorskip(
    "lerobot.robots.so101_impedance_follower.checker",
    reason="measure_so101_cerebellum.py needs the impedance checker",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("measure_so101_cerebellum", MEASURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = _load_module()

_TICKS = {
    "shoulder_pan": 2031.0,
    "shoulder_lift": 3164.0,
    "elbow_flex": 967.0,
    "wrist_flex": 1813.0,
    "wrist_roll": 2039.0,
    "gripper": 2097.0,
}


def _write(tmp_path: Path, blob) -> str:
    path = tmp_path / "pose.json"
    path.write_text(json.dumps(blob))
    return str(path)


def test_pose_in_the_loops_frame_loads(tmp_path):
    path = _write(tmp_path, {"frame": measure.POSE_FRAME, "pose": _TICKS})
    assert measure._load_pose(path) == pytest.approx(_TICKS)


def test_legacy_flat_pose_is_refused(tmp_path):
    """The exact file that caused the 2026-09-08 run: bare ticks, frame unrecorded.

    Refusing is the only correct answer -- the ticks are real, and there is no way to tell from the
    file which of the two frames produced them.
    """
    path = _write(tmp_path, _TICKS)
    with pytest.raises(SystemExit, match="no frame recorded"):
        measure._load_pose(path)


def test_pose_from_the_other_frame_is_refused(tmp_path):
    path = _write(tmp_path, {"frame": "position_offset", "pose": _TICKS})
    with pytest.raises(SystemExit, match="captured in frame"):
        measure._load_pose(path)


def test_refusal_names_the_file_and_the_remedy(tmp_path):
    """The message has to say what to do: re-capturing is 20 samples and costs nothing, so an
    operator who is told that will do it rather than hand-edit a frame tag onto a stale file."""
    path = _write(tmp_path, _TICKS)
    with pytest.raises(SystemExit) as excinfo:
        measure._load_pose(path)
    assert path in str(excinfo.value)
    assert "capture" in str(excinfo.value)
