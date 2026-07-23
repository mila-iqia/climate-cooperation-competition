"""
tests/test_rice_base.py

Smoke tests for the base Rice environment (no MRIO / external csv_asset).

Run from rice_jax/ with:
    conda activate rice-jax
    pytest tests/test_rice_base.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
import pytest
from jaxnasium.algorithms import PPO

from rice_jax import Rice
from rice_jax.utils import i_to_agent_str, load_region_yamls

NUM_REGIONS = 3
SEED = 0

_BASE_KWARGS = dict(
    num_regions=NUM_REGIONS,
    num_discrete_action_levels=10,
    diff_reward_mode=True,
    negotiation_on=False,
)

_PPO_KWARGS = dict(
    total_timesteps=8,
    num_steps=4,
    num_envs=2,
    num_minibatches=2,
    num_epochs=1,
    learning_rate_start=3e-4,
    ent_coef_start=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    clip_coef=0.2,
    clip_coef_vf=0.5,
    vf_coef=0.5,
    normalize_observations=True,
    normalize_rewards=False,
)


@pytest.fixture(scope="module")
def region_params():
    return load_region_yamls(NUM_REGIONS)


@pytest.fixture(scope="module")
def rice_env(region_params):
    return Rice(region_params=region_params, **_BASE_KWARGS)


@pytest.fixture(scope="module")
def wrapped_env(rice_env):
    return jym.LogWrapper(rice_env)


def _assert_finite_temperature(env_state: dict) -> None:
    temp = np.array(env_state["global_temperature"])
    assert np.isfinite(temp).all(), f"non-finite temperature after step: {temp}"


class TestBaseRiceSmoke:
    def test_reset(self, wrapped_env):
        key = jax.random.PRNGKey(SEED)
        obs, state = wrapped_env.reset(key)

        assert isinstance(obs, dict)
        assert len(obs) == NUM_REGIONS
        for i in range(NUM_REGIONS):
            agent_key = i_to_agent_str(i)
            assert agent_key in obs
            assert hasattr(obs[agent_key], "observation")
            assert hasattr(obs[agent_key], "action_mask")

        env_state = state.env_state
        assert "global_temperature" in env_state
        assert env_state["global_temperature"].shape == (2,)

    def test_step_with_sampled_action(self, rice_env, wrapped_env):
        key = jax.random.PRNGKey(SEED)
        obs, state = wrapped_env.reset(key)

        step_key, act_key = jax.random.split(key)
        actions = rice_env.sample_action(act_key)

        (next_obs, reward, terminated, truncated, info), next_state = wrapped_env.step(
            step_key, state, actions
        )

        assert isinstance(next_obs, dict)
        assert len(next_obs) == NUM_REGIONS
        assert isinstance(reward, dict)
        assert len(reward) == NUM_REGIONS
        assert isinstance(terminated, (bool, dict))
        assert isinstance(truncated, (bool, dict))
        _assert_finite_temperature(next_state.env_state)
        assert next_state.env_state["activity_timestep"] > state.env_state["activity_timestep"]

    def test_ppo_init_train_and_step(self, wrapped_env):
        key = jax.random.PRNGKey(SEED)
        obs, state = wrapped_env.reset(key)

        ppo = PPO(**_PPO_KWARGS)
        agent = ppo.train(key, wrapped_env)

        assert hasattr(agent, "get_action")

        act_key, step_key = jax.random.split(key)
        actions = agent.get_action(act_key, obs)

        assert isinstance(actions, dict)
        assert len(actions) == NUM_REGIONS

        (next_obs, reward, terminated, truncated, info), next_state = wrapped_env.step(
            step_key, state, actions
        )

        assert len(next_obs) == NUM_REGIONS
        assert len(reward) == NUM_REGIONS
        _assert_finite_temperature(next_state.env_state)
        assert next_state.env_state["activity_timestep"] > state.env_state["activity_timestep"]
