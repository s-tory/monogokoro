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

import pytest

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


# --- Filtering by tag -------------------------------------------------------
def _write(tmp_path, tags):
    for tag in tags:
        salience.append_salience(tmp_path, tag)


def test_episodes_with_tags_returns_matching_indices(tmp_path):
    _write(tmp_path, [salience.GOOD, salience.GAVE_UP, salience.ORDINARY, salience.GAVE_UP])
    assert salience.episodes_with_tags(tmp_path, [salience.GAVE_UP], 4) == [1, 3]


def test_episodes_with_tags_accepts_several_tags(tmp_path):
    _write(tmp_path, [salience.GOOD, salience.GAVE_UP, salience.UNLABELLED])
    got = salience.episodes_with_tags(tmp_path, [salience.GAVE_UP, salience.UNLABELLED], 3)
    assert got == [1, 2]


def test_episodes_with_tags_matching_nothing_is_empty_not_an_error(tmp_path):
    _write(tmp_path, [salience.GOOD, salience.ORDINARY])
    assert salience.episodes_with_tags(tmp_path, [salience.GAVE_UP], 2) == []


def test_a_short_sidecar_raises_rather_than_dropping_the_wrong_episodes(tmp_path):
    """Line N is episode N only while the file is as long as the dataset."""
    _write(tmp_path, [salience.GAVE_UP])
    with pytest.raises(ValueError, match="Line N means episode N"):
        salience.episodes_with_tags(tmp_path, [salience.GAVE_UP], 5)


def test_a_missing_sidecar_raises(tmp_path):
    with pytest.raises(ValueError, match="holds 0 tags"):
        salience.episodes_with_tags(tmp_path, [salience.GAVE_UP], 2)


def test_an_unknown_tag_raises_instead_of_matching_nothing(tmp_path):
    """A typo that quietly keeps everything looks exactly like a filter that worked."""
    _write(tmp_path, [salience.GAVE_UP, salience.GOOD])
    with pytest.raises(ValueError, match="Unknown salience tag"):
        salience.episodes_with_tags(tmp_path, ["X"], 2)


def test_the_legend_names_every_key_that_ends_an_episode():
    """The legend is the only thing the operator reads, so a key missing from it is a key
    nobody presses. This is the check that the dropped arrows would have failed."""
    for key in ("g", "b", "x", "n", "Right", "r", "Left", "q", "Esc"):
        assert key in salience.KEY_LEGEND, key


def test_every_tag_that_can_be_written_can_also_be_asked_for(tmp_path):
    """Whatever the recorder can put in the file, the filter must accept as a name.

    Two lists would drift, and the drift is asymmetric: a tag the operator pressed but the
    filter rejects costs a run, while a typo costs a retyped flag.
    """
    for tag in salience.ALL_TAGS:
        salience.append_salience(tmp_path, tag)
    for tag in salience.ALL_TAGS:
        assert salience.episodes_with_tags(tmp_path, [tag], len(salience.ALL_TAGS))
