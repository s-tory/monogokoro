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

"""Exercises the collapse counter in `examples/analyze_servo_error_log.py`.

Two of these pin bugs the first version actually had, and both were invisible in the output until
a number was compared against a hand count:

- the same motor's neighbouring intervals closed each other early, which *raised* the episode
  count as the rule got stricter -- a shape that reads like a finding about the rig;
- the same-tick grace was added to both ends of every interval, so every duration came out one
  tick long. Against a known hand count that showed up as a uniform +2.5 ms, which is small enough
  to be mistaken for rounding.

The last test is the one that would have caught both without knowing the right answer: requiring
more motors can only shrink the qualifying time, so `total_ms` must never rise down that sweep.
"""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "analyze_servo_error_log.py"


def _load():
    spec = importlib.util.spec_from_file_location("analyze_servo_error_log", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because the module defines dataclasses, and `@dataclass`
    # resolves annotations through `sys.modules[cls.__module__]`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ana = _load()

MS = ana.NS_PER_MS


def _log(lines: list[str], tmp_path: Path) -> Path:
    path = tmp_path / "daemon.log"
    path.write_text("\n".join(lines) + "\n")
    return path


def _edge(ns: int, motor: int, frm: int, to: int) -> str:
    return (
        f"[2026-09-16T00:00:00Z WARN  so101_impedance_ctrl] "
        f"servo_error mono_ns={ns} motor={motor} {frm:#04x} -> {to:#04x}"
    )


def test_a_transition_line_is_parsed_out_of_the_log_prefix(tmp_path):
    path = _log([_edge(1000, 5, 0x00, 0x01), "unrelated line", _edge(2000, 5, 0x01, 0x00)], tmp_path)
    transitions, first, last = ana.parse_log(path)
    assert transitions == [(1000, 5, 0x00, 0x01), (2000, 5, 0x01, 0x00)]
    assert (first, last) == (1000, 2000)


def test_a_change_to_another_bit_does_not_open_a_second_interval():
    # 0x01 -> 0x05 sets the overload bit and leaves bit0 alone. Counting whole-byte edges instead
    # of masked ones would end the collapse here and start a new one.
    transitions = [
        (0, 3, 0x00, 0x01),
        (10 * MS, 3, 0x01, 0x05),
        (20 * MS, 3, 0x05, 0x04),
    ]
    ivs = ana.intervals_for_bit(transitions, 0x01, 20 * MS)
    assert len(ivs) == 1
    assert ivs[0].start_ns == 0
    assert ivs[0].end_ns == 20 * MS


def test_intervals_still_open_at_the_end_of_the_log_are_closed_there():
    ivs = ana.intervals_for_bit([(0, 2, 0x00, 0x01)], 0x01, 500 * MS)
    assert ivs[0].end_ns is None
    assert ivs[0].duration_ns(500 * MS) == 500 * MS


def test_one_motors_neighbouring_intervals_merge_within_a_tick():
    # The regression: a bit held across two read rounds is logged twice, and without merging the
    # motor is in the sweep twice at once -- where its own second interval closes the first.
    raw = [
        ana.Interval(6, 0, 2 * MS),
        ana.Interval(6, 4 * MS, 6 * MS),  # 2 ms later: within the 2.5 ms tick
        ana.Interval(6, 100 * MS, 102 * MS),  # a separate event
    ]
    merged = ana.merge_per_motor(raw, 200 * MS)
    assert [(iv.start_ns, iv.end_ns) for iv in merged] == [(0, 6 * MS), (100 * MS, 102 * MS)]


def test_the_tick_grace_does_not_lengthen_a_reported_episode():
    # Two motors up for exactly 100 ms is a 100 ms episode, not 102.5. Padding here is a
    # measurement error wearing a tolerance's clothes.
    ivs = [ana.Interval(1, 0, 100 * MS), ana.Interval(2, 0, 100 * MS)]
    (episode,) = ana.episodes(ivs, 2, 100 * MS)
    assert episode.duration_ms == 100.0


def test_an_episode_reports_who_was_up_at_its_peak():
    # Two motors hold the whole episode and a third joins in the middle. The episode is credited
    # with three, because that is how many were up *at once* -- which is the claim being made.
    ivs = [
        ana.Interval(1, 0, 100 * MS),
        ana.Interval(2, 0, 100 * MS),
        ana.Interval(3, 30 * MS, 60 * MS),
    ]
    (episode,) = ana.episodes(ivs, 2, 100 * MS)
    assert episode.motors == {1, 2, 3}
    assert episode.duration_ms == 100.0


def test_a_motor_that_joins_only_after_another_has_gone_does_not_join_the_same_episode():
    # motor 3 arrives after motor 1 left, so the pair is never three and the stretch in between
    # has only one motor up: two episodes, not one episode of three.
    ivs = [
        ana.Interval(1, 0, 40 * MS),
        ana.Interval(2, 0, 100 * MS),
        ana.Interval(3, 60 * MS, 100 * MS),
    ]
    first, second = ana.episodes(ivs, 2, 100 * MS)
    assert (first.motors, round(first.duration_ms)) == ({1, 2}, 40)
    assert (second.motors, round(second.duration_ms)) == ({2, 3}, 40)


def test_a_motor_that_drops_out_and_back_breaks_the_episode():
    ivs = [
        ana.Interval(1, 0, 100 * MS),
        ana.Interval(2, 0, 40 * MS),
        ana.Interval(2, 60 * MS, 100 * MS),
    ]
    eps = ana.episodes(ana.merge_per_motor(ivs, 100 * MS), 2, 100 * MS)
    assert [round(e.duration_ms) for e in eps] == [40, 40]


def test_an_interval_straddling_the_window_edge_is_cut_not_dropped():
    # Dropping it would undercount exactly the long collapses the window was opened to look at.
    ivs = [ana.Interval(4, 0, 300 * MS)]
    (clipped,) = ana.clip(ivs, 100 * MS, 200 * MS, 300 * MS)
    assert (clipped.start_ns, clipped.end_ns) == (100 * MS, 200 * MS)


def test_a_window_outside_every_interval_keeps_nothing():
    assert ana.clip([ana.Interval(4, 0, 50 * MS)], 100 * MS, 200 * MS, 300 * MS) == []


def test_window_accepts_seconds_and_raw_mono_ns():
    first, last = 1_000_000_000, 11_000_000_000
    assert ana.parse_window("2:5", first, last) == (3_000_000_000, 6_000_000_000)
    assert ana.parse_window("mono_ns=2000:5000", first, last) == (2000, 5000)
    assert ana.parse_window(":3", first, last) == (first, 4_000_000_000)
    assert ana.parse_window(None, first, last) == (first, last)


def test_a_reversed_window_is_refused_rather_than_silently_empty():
    import pytest

    with pytest.raises(SystemExit):
        ana.parse_window("5:2", 0, 10_000_000_000)


def test_requiring_more_motors_can_only_shrink_the_qualifying_time():
    # The invariant that catches the merge bug without knowing the right answer: episode *counts*
    # may rise as the rule tightens (a loose rule fuses neighbours), but the total time cannot.
    ivs = []
    for motor in range(1, 7):
        for k in range(5):
            start = k * 200 * MS + motor * 3 * MS
            ivs.append(ana.Interval(motor, start, start + 80 * MS))
    merged = ana.merge_per_motor(ivs, 2000 * MS)
    totals = [sum(e.duration_ms for e in ana.episodes(merged, n, 2000 * MS)) for n in range(1, 7)]
    assert totals == sorted(totals, reverse=True)
    assert totals[0] > 0
