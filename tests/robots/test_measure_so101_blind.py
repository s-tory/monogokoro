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

"""Exercises the gain scaling and trial order of `measure_so101_cerebellum.py blind`.

Neither of these touches hardware, which is the point: the parts of a blind trial that can be
wrong without anyone noticing are the mapping from one named gain to six joints, and the order the
trials are shown in. A scaling error shows up as "that one felt odd" and nothing else; a plan that
puts a repeat next to its twin measures the judge's memory rather than their hand.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "examples" / "measure_so101_cerebellum.py"


def _load():
    spec = importlib.util.spec_from_file_location("measure_so101_cerebellum", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


measure = _load()

CANDIDATES = [20.0, 12.0, 8.0, 5.0, 3.0]


def test_the_named_gain_lands_on_the_lift_joint():
    k, _ = measure.scaled_gains(8.0)
    assert k[measure.LIFT_JOINT] == 8.0


def test_the_shipped_value_reproduces_the_shipped_gains_exactly():
    # The identity case. If this drifts, every trial is being run against a different arm than the
    # one the shipped defaults were chosen for.
    k, d = measure.scaled_gains(measure.DEFAULT_K[measure.LIFT_JOINT])
    assert k == measure.DEFAULT_K
    assert d == measure.DEFAULT_D


def test_every_joint_keeps_its_shipped_ratio():
    # A flat K is what this scaling exists to avoid: too soft at the shoulder, and stiff enough at
    # wrist_roll to turn its velocity noise into a full-duty limit cycle.
    k, _ = measure.scaled_gains(5.0)
    f = 5.0 / measure.DEFAULT_K[measure.LIFT_JOINT]
    for motor, shipped in measure.DEFAULT_K.items():
        assert k[motor] == pytest.approx(shipped * f)


def test_damping_scales_with_the_spring():
    # D past ~K/40 buzzes the joint instead of damping it, so the ratio has to travel with K.
    k, d = measure.scaled_gains(3.0)
    for motor in measure.MOTOR_NAMES:
        assert d[motor] / k[motor] == pytest.approx(measure.DEFAULT_D[motor] / measure.DEFAULT_K[motor])


def test_a_single_joint_moves_alone():
    # The ratio sweep. If a neighbour moves with it, the verdict is about two joints and the ratio
    # read off it is wrong in a way no one pushing the arm would notice.
    k, d = measure.scaled_gains(0.84, joint="wrist_roll")
    assert k["wrist_roll"] == 0.84
    for motor in measure.MOTOR_NAMES:
        if motor != "wrist_roll":
            assert k[motor] == measure.DEFAULT_K[motor]
            assert d[motor] == measure.DEFAULT_D[motor]


def test_a_single_joint_keeps_its_own_damping_ratio():
    k, d = measure.scaled_gains(2.0, joint="wrist_flex")
    j = "wrist_flex"
    assert d[j] / k[j] == pytest.approx(measure.DEFAULT_D[j] / measure.DEFAULT_K[j])


def test_a_single_joint_at_its_shipped_value_is_the_shipped_arm():
    for motor in measure.MOTOR_NAMES:
        k, d = measure.scaled_gains(measure.DEFAULT_K[motor], joint=motor)
        assert k == measure.DEFAULT_K
        assert d == pytest.approx(measure.DEFAULT_D)


def test_the_plan_holds_every_candidate_plus_the_repeats():
    plan = measure.blind_plan(CANDIDATES, 2, seed=0)
    assert len(plan) == len(CANDIDATES) + 2
    assert set(plan) == set(CANDIDATES)


def test_a_repeat_never_follows_its_own_twin():
    # A repeat the judge can see coming measures their memory. Checked across many seeds because
    # one passing seed says nothing about the shuffle.
    for seed in range(50):
        plan = measure.blind_plan(CANDIDATES, 2, seed)
        assert all(a != b for a, b in zip(plan, plan[1:], strict=False)), (seed, plan)


def test_the_same_seed_gives_the_same_order():
    # The order is written into the result file as the record of what was shown. It has to be
    # reproducible from the seed or that record cannot be checked.
    assert measure.blind_plan(CANDIDATES, 2, 7) == measure.blind_plan(CANDIDATES, 2, 7)


def test_different_seeds_give_different_orders():
    orders = {tuple(measure.blind_plan(CANDIDATES, 2, s)) for s in range(20)}
    assert len(orders) > 1


def test_an_impossible_arrangement_returns_an_order_rather_than_hanging():
    # Two candidates and two repeats cannot avoid adjacency. The loop is bounded, so this returns.
    plan = measure.blind_plan([10.0, 4.0], 2, seed=0)
    assert sorted(plan) == [4.0, 4.0, 10.0, 10.0]


def test_zero_repeats_shows_each_candidate_once():
    plan = measure.blind_plan(CANDIDATES, 0, seed=3)
    assert sorted(plan) == sorted(CANDIDATES)


def test_descending_walks_stiff_to_soft():
    plan = measure.blind_plan(CANDIDATES, 0, seed=0, order="descending")
    assert plan == sorted(CANDIDATES, reverse=True)


def test_descending_still_never_repeats_back_to_back():
    # The bias descending admits is the order effect, not a visible repeat.
    plan = measure.blind_plan(CANDIDATES, 2, seed=0, order="descending")
    assert all(a != b for a, b in zip(plan, plan[1:], strict=False)), plan
    assert len(plan) == len(CANDIDATES) + 2


def test_descending_ignores_the_seed():
    # It is a fixed walk; a seed in the record would imply a randomness that is not there.
    a = measure.blind_plan(CANDIDATES, 2, 0, "descending")
    b = measure.blind_plan(CANDIDATES, 2, 999, "descending")
    assert a == b
