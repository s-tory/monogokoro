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

"""Covers the hook `lerobot-record` uses to rotate a declared context between episodes."""

import pytest

pytest.importorskip("datasets", reason="lerobot.scripts.lerobot_record requires the `dataset` extra")

from lerobot.processor import PontineContextProcessorStep  # noqa: E402
from lerobot.scripts.lerobot_record import _context_cycling_steps  # noqa: E402


class _Pipeline:
    def __init__(self, steps):
        self.steps = steps


class _PlainStep:
    """Stands in for every step that declares nothing below the policy -- i.e. almost all of them."""


def test_a_pipeline_without_cycling_steps_contributes_none():
    pipeline = _Pipeline([_PlainStep(), _PlainStep()])

    assert _context_cycling_steps(pipeline) == []


def test_a_step_with_an_empty_cycle_is_not_collected():
    # A fixed context must not make `record` announce a rotation it will never perform.
    step = PontineContextProcessorStep(context=(1.0, 0.0))

    assert _context_cycling_steps(_Pipeline([step])) == []


def test_a_step_with_a_cycle_is_collected():
    step = PontineContextProcessorStep(cycle=((1.0, 0.0), (-1.0, 0.0)))

    assert _context_cycling_steps(_Pipeline([step])) == [step]


def test_an_object_without_a_pipeline_of_steps_is_tolerated():
    # The hook is duck-typed; an identity processor need not expose `.steps` at all.
    assert _context_cycling_steps(object()) == []
