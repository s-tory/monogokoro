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
Verifies that ACT needs zero architecture changes to consume the widened
`observation.state`/`action` features produced by `SO101ImpedanceFollower` (+6 state dims for
per-motor current-avg, +12 action dims for per-motor K/D across all 6 motors -- the gripper is
impedance-controlled too, not left in plain position mode): `action_head` and
`vae_encoder_action_input_proj` (modeling_act.py) both derive their width purely from
`config.action_feature.shape[0]`, and `encoder_robot_state_input_proj`/
`vae_encoder_robot_state_input_proj` from `config.robot_state_feature.shape[0]`.
"""

import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

# 6 motors x (.pos + .current_avg).
_STATE_DIM = 12
# 6 motors x (.pos + .k + .d).
_ACTION_DIM = 18


def _make_small_config(**overrides) -> ACTConfig:
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(_STATE_DIM,)),
        # ACTConfig.validate_features() requires an image or environment_state input; a tiny
        # env-state feature avoids pulling in a vision backbone for this architecture-only test.
        OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(1,)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(_ACTION_DIM,)),
    }
    defaults = {
        "chunk_size": 8,
        "n_action_steps": 8,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_vae_encoder_layers": 1,
        "dim_model": 32,
        "n_heads": 2,
        "dim_feedforward": 64,
    }
    defaults.update(overrides)
    return ACTConfig(input_features=input_features, output_features=output_features, **defaults)


def test_action_head_and_state_projection_widths_match_features():
    config = _make_small_config()
    policy = ACTPolicy(config)

    assert policy.model.action_head.out_features == _ACTION_DIM
    assert policy.model.vae_encoder_action_input_proj.in_features == _ACTION_DIM
    assert policy.model.encoder_robot_state_input_proj.in_features == _STATE_DIM
    assert policy.model.vae_encoder_robot_state_input_proj.in_features == _STATE_DIM


def test_predict_action_chunk_produces_widened_action_dim():
    config = _make_small_config()
    policy = ACTPolicy(config)

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, _STATE_DIM),
        OBS_ENV_STATE: torch.randn(batch_size, 1),
    }

    actions = policy.predict_action_chunk(batch)
    assert actions.shape == (batch_size, config.chunk_size, _ACTION_DIM)


def test_forward_computes_finite_loss_over_widened_action():
    config = _make_small_config()
    policy = ACTPolicy(config)

    batch_size = 2
    batch = {
        OBS_STATE: torch.randn(batch_size, _STATE_DIM),
        OBS_ENV_STATE: torch.randn(batch_size, 1),
        ACTION: torch.randn(batch_size, config.chunk_size, _ACTION_DIM),
        "action_is_pad": torch.zeros(batch_size, config.chunk_size, dtype=torch.bool),
    }

    loss, loss_dict = policy.forward(batch)
    assert torch.isfinite(loss)
    assert "l1_loss" in loss_dict
