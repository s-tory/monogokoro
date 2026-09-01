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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction

from .pipeline import ProcessorStepRegistry, RobotActionProcessorStep


@ProcessorStepRegistry.register("pontine_context_processor")
@dataclass
class PontineContextProcessorStep(RobotActionProcessorStep):
    """
    Labels an impedance robot's action with the pontine context the demonstration was performed
    under, so it lands in the dataset as ground truth rather than only reaching the arm.

    The context channel is what the policy layer declares to the cerebellum -- object *identity*,
    not mass; see `rust/so101_impedance_ctrl/src/shm.rs`'s `NUM_CONTEXT`. A teleoperator has no
    way to supply it (a leader arm only ever produces `.pos`), so during recording it is a
    constant the operator sets to say which of two otherwise indistinguishable situations this
    episode is.

    This step must run in the *teleop* action pipeline, upstream of `robot.send_action()`, for the
    same reason `ImpedanceGainDefaultsProcessorStep` does: `record_loop()` writes the teleop
    pipeline's *output* to the dataset, so a context filled in by `send_action` would steer the
    arm correctly but be absent from the recorded action columns.

    Recording it as an *action* column is the point, not a side effect. It is what gives an
    imitation policy a port onto the pontine channel at all: ACT has no context variable of its
    own (its CVAE latent is zeroed at inference by construction, and it has no task conditioning),
    but a context that appears in the action it is trained to reproduce is one it learns to emit
    from the observations -- which is precisely "infer which object this is, and tell the
    cerebellum". Until then it reproduces whatever constant was recorded, exactly the known v1
    limitation already documented for K/D.

    Attributes:
        context: The context currently being demonstrated, nominally in `[-1, 1]` per channel.
            Its length is the number of channels and must match the daemon's `NUM_CONTEXT`.
            All-zero is the neutral "no context" value, which degrades the cerebellum to
            proprioception-only rather than to garbage.
        cycle: Contexts to rotate through, one per kept episode. Empty means a fixed `context`.
            This exists because the cerebellum has to be taught the contexts *interleaved* --
            most granule cells draw no context fibre, so their weights are shared, and training
            one context to convergence before the other leaves the first reading 98 where it
            should read 40. Rotating per episode makes a single `lerobot-record` run interleave
            on its own, which is otherwise the operator's job to remember.
    """

    # Zero-length would silently record nothing, so the default is the daemon's two channels at
    # the neutral value: wiring the step up without configuring it is a no-op, not a surprise.
    context: tuple[float, ...] = (0.0, 0.0)
    cycle: tuple[tuple[float, ...], ...] = ()
    cycle_index: int = field(default=0, init=False)

    def __post_init__(self):
        if self.cycle:
            for entry in self.cycle:
                self._validate(tuple(entry))
            if len({len(entry) for entry in self.cycle}) != 1:
                raise ValueError(f"every cycle entry must have the same length, got {self.cycle}")
            # The cycle is the source of truth once configured, so the first episode records its
            # first entry rather than a `context` the caller may not have bothered to match.
            self.cycle = tuple(tuple(float(c) for c in entry) for entry in self.cycle)
            self.context = self.cycle[0]
        self._validate(self.context)

    @staticmethod
    def _validate(context: tuple[float, ...]) -> None:
        if len(context) == 0:
            raise ValueError("context must have at least one channel")
        if any(not -1.0 <= float(c) <= 1.0 for c in context):
            raise ValueError(f"context values must lie in [-1, 1], got {context}")

    def cycle_context(self) -> tuple[float, ...]:
        """Advance to the next context in `cycle` and return it (a no-op without a cycle).

        Called by `lerobot-record` after an episode is *kept*, never after a re-record: the
        operator has already staged the object for the context that was just aborted, so
        advancing there would label the retake with the wrong one.
        """
        if not self.cycle:
            return self.context
        self.cycle_index = (self.cycle_index + 1) % len(self.cycle)
        self.context = self.cycle[self.cycle_index]
        return self.context

    def action(self, action: RobotAction) -> RobotAction:
        new_action = dict(action)
        for i, value in enumerate(self.context):
            new_action.setdefault(f"context.{i}", float(value))
        return new_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for i in range(len(self.context)):
            key = f"context.{i}"
            if key not in features[PipelineFeatureType.ACTION]:
                features[PipelineFeatureType.ACTION][key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features
