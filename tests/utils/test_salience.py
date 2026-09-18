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

"""Unit tests for the per-episode salience sidecar.

The property under test is the positional contract -- line N is episode N -- because
everything else in the design rests on it: it is what lets the file carry no indices
and no timestamps.
"""

import logging

from lerobot.utils import salience


def test_tags_are_written_in_episode_order(tmp_path):
    for tag in (salience.ORDINARY, salience.GOOD, salience.ORDINARY, salience.NEAR_MISS):
        salience.append_salience(tmp_path, tag)
    assert salience.read_salience(tmp_path) == [
        salience.ORDINARY,
        salience.GOOD,
        salience.ORDINARY,
        salience.NEAR_MISS,
    ]


def test_ordinary_survives_the_round_trip(tmp_path):
    """``ORDINARY`` is the one non-ASCII tag, so it is the one an encoding slip would eat."""
    salience.append_salience(tmp_path, salience.ORDINARY)
    assert salience.read_salience(tmp_path) == ["→"]


def test_reading_a_dataset_with_no_sidecar_yields_nothing(tmp_path):
    assert salience.read_salience(tmp_path) == []


def test_align_pads_a_short_sidecar(tmp_path):
    """Episodes recorded before the sidecar existed get ``?``, not ``ordinary``.

    Padding with ``ORDINARY`` would claim a judgement nobody made; ``?`` states that
    none was given, which is what actually happened.
    """
    salience.append_salience(tmp_path, salience.GOOD)
    salience.align_salience(tmp_path, num_episodes=4)
    assert salience.read_salience(tmp_path) == [
        salience.GOOD,
        salience.UNLABELLED,
        salience.UNLABELLED,
        salience.UNLABELLED,
    ]


def test_align_leaves_a_matching_sidecar_alone(tmp_path):
    salience.append_salience(tmp_path, salience.GOOD)
    salience.append_salience(tmp_path, salience.NEAR_MISS)
    salience.align_salience(tmp_path, num_episodes=2)
    assert salience.read_salience(tmp_path) == [salience.GOOD, salience.NEAR_MISS]


def test_align_on_a_fresh_dataset_writes_no_file(tmp_path):
    salience.align_salience(tmp_path, num_episodes=0)
    assert not salience.salience_path(tmp_path).exists()


def test_align_does_not_truncate_a_longer_sidecar(tmp_path, caplog):
    """More tags than episodes means the two disagree; guessing which is right destroys one."""
    for tag in (salience.GOOD, salience.NEAR_MISS, salience.ORDINARY):
        salience.append_salience(tmp_path, tag)
    with caplog.at_level(logging.ERROR):
        salience.align_salience(tmp_path, num_episodes=1)
    assert len(salience.read_salience(tmp_path)) == 3
    assert "Line N no longer means" in caplog.text
