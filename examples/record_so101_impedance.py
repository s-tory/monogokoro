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

"""
Records a teleoperation dataset for the impedance-controlled SO101
(`--robot.type=so101_follower_impedance`), labeling every recorded frame's `.k`/`.d` action
dimensions -- for all 6 motors, including the gripper -- with constant default gains via
`ImpedanceGainDefaultsProcessorStep`.

This is a *dedicated* script rather than a change to the generic `lerobot-record` CLI
(`src/lerobot/scripts/lerobot_record.py`) on purpose: that script is robot-agnostic by
convention, and `record()`'s `teleop_action_processor` override is a programmatic-only
parameter (not exposed over the CLI), so injecting this robot-specific processor step has to
happen from a small wrapper script like this one.

IMPORTANT limitation: this records *constant* default K/D (from `--robot.default_k`/
`--robot.default_d`, or `ImpedanceGainDefaultsProcessorStep`'s own defaults if unset), not
variable teleop-taught gains -- the plain position-mode leader arm has no way to teleoperate K/D
today. ACT trained on data collected this way will initially learn to reproduce the configured
default gains rather than gains that vary with the demonstration; teaching variable gains is a
follow-up, out of scope here.

Before running this script:
1. Start the `rust/so101_impedance_ctrl` RT daemon (see its README.md) -- this robot only
   attaches to its shared-memory segment, it never opens the serial port itself.
2. Have a valid calibration file in place (see `SO101ImpedanceFollower.calibrate()`'s docstring
   -- calibrate once with the plain `so101_follower` robot type against the same servos first).

Example:

```shell
python examples/record_so101_impedance.py \\
    --robot.type=so101_follower_impedance \\
    --robot.shm_name=so101_impedance \\
    --robot.id=my_impedance_arm \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --teleop.type=so101_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --display_data=true
```
"""

from lerobot.processor import (
    ImpedanceGainDefaultsProcessorStep,
    RobotProcessorPipeline,
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots import so101_impedance_follower  # noqa: F401 -- registers the RobotConfig subclass
from lerobot.scripts.lerobot_record import RecordConfig, record  # noqa: F401 -- RecordConfig used by draccus
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.import_utils import register_third_party_plugins


def make_impedance_teleop_action_processor() -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    return RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[ImpedanceGainDefaultsProcessorStep()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )


def main():
    register_third_party_plugins()
    record(
        teleop_action_processor=make_impedance_teleop_action_processor(),
        robot_action_processor=make_default_robot_action_processor(),
        robot_observation_processor=make_default_robot_observation_processor(),
    )


if __name__ == "__main__":
    main()
