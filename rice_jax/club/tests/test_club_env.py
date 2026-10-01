"""Fixed-action sanity checks for the club mediator env.

Run from rice_jax/ with:
    conda activate rice-jax
    pytest club/tests/test_club_env.py -v
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
import optax
import pytest
from jaxnasium.algorithms import PPO

from club.env import MEDIATOR, RiceClubMediator
from rice_jax import Rice
from rice_jax.utils import i_to_agent_str, load_region_yamls

NUM_REGIONS = 3
SEED = 0
KEY = jax.random.PRNGKey(SEED)


@pytest.fixture(scope="module")
def region_params():
    return load_region_yamls(NUM_REGIONS)


def _make_club_env(region_params, **kwargs) -> RiceClubMediator:
    defaults = dict(
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=10,
        diff_reward_mode=True,
    )
    defaults.update(kwargs)
    return RiceClubMediator(region_params=region_params, **defaults)


def _zero_actions(env):
    return optax.tree.zeros_like(env.sample_action(KEY))


def _set_action(actions, agent, key, level):
    actions[agent][key] = jnp.full_like(actions[agent][key], level)


def _step(env, state, actions):
    (_, reward, _, _, _), state = env.step_env(KEY, state, actions)
    return state, reward


class TestCanonicalNull:
    def test_null_club_matches_base_rice(self, region_params):
        """fixed_club_params=(0,0) + all-reject == base Rice(negotiation_on=False)."""
        club = _make_club_env(region_params, fixed_club_params=(0.0, 0.0))
        base = Rice(
            region_params=region_params,
            num_regions=NUM_REGIONS,
            num_discrete_action_levels=10,
            diff_reward_mode=True,
            negotiation_on=False,
        )
        assert club.episode_length == 3 * base.episode_length

        _, club_state = club.reset_env(KEY)
        _, base_state = base.reset_env(KEY)

        club_actions = _zero_actions(club)
        base_actions = _zero_actions(base)
        for i in range(NUM_REGIONS):
            agent = i_to_agent_str(i)
            for actions in (club_actions, base_actions):
                _set_action(actions, agent, "savings_rate", 2)
                _set_action(actions, agent, "mitigation_rate", 3)

        check_keys = [
            "global_temperature",
            "global_carbon_mass",
            # NOTE: "global_emissions" excluded: base Rice never updates it
            # under carbon_model="base"; the club env fixes that on purpose
            "capital_all_regions",
            "gross_output_all_regions",
            "utility_all_regions",
            "aggregate_consumption",
            "utility_times_welfloss_all_regions",
        ]
        for cycle in range(4):
            for _ in range(3):  # propose, evaluate, climate
                club_state, _ = _step(club, club_state, club_actions)
            base_state, _ = _step(base, base_state, base_actions)
            for k in check_keys:
                np.testing.assert_allclose(
                    np.asarray(club_state[k]),
                    np.asarray(base_state[k]),
                    rtol=1e-5,
                    atol=1e-6,
                    err_msg=f"state[{k}] diverged at cycle {cycle}",
                )


class TestClubMechanics:
    def _run_propose_evaluate(self, env, joins):
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        for i, join in enumerate(joins):
            _set_action(actions, i_to_agent_str(i), "club_join", int(join))
        state, _ = _step(env, state, actions)  # t=1: propose
        state, reward = _step(env, state, actions)  # t=2: evaluate
        return state, actions, reward

    def test_membership_and_mitigation_mask_floor(self, region_params):
        env = _make_club_env(region_params, fixed_club_params=(0.7, 0.5))
        state, _, _ = self._run_propose_evaluate(env, joins=[1, 1, 0])

        np.testing.assert_allclose(np.asarray(state["club_membership"]), [1, 1, 0])
        np.testing.assert_allclose(
            np.asarray(state["minimum_mitigation_rate_all_regions"]), [0.7, 0.7, 0.0]
        )

        mask = env.generate_action_masks(state)
        member_expected = np.arange(10) >= 7
        for member in (0, 1):
            np.testing.assert_array_equal(
                np.asarray(mask[i_to_agent_str(member)]["mitigation_rate"]),
                member_expected,
            )
        np.testing.assert_array_equal(
            np.asarray(mask[i_to_agent_str(2)]["mitigation_rate"]),
            np.ones(10, dtype=bool),
        )
        # mediator's economy actions stay pinned to level 0
        assert np.asarray(mask[MEDIATOR]["mitigation_rate"])[1:].sum() == 0

    def test_club_tariff_floor_on_non_members(self, region_params):
        env = _make_club_env(region_params, fixed_club_params=(0.0, 0.5))
        state, actions, _ = self._run_propose_evaluate(env, joins=[1, 1, 0])
        state, _ = _step(env, state, actions)  # t=3: climate

        tariffs = np.asarray(state["import_tariffs_all_regions"])
        # member -> non-member floored at club rate
        assert tariffs[0, 2] == pytest.approx(0.5)
        assert tariffs[1, 2] == pytest.approx(0.5)
        # member -> member and non-member rows keep the (zero) chosen tariff
        assert tariffs[0, 1] == 0.0
        assert tariffs[1, 0] == 0.0
        assert np.all(tariffs[2] == 0.0)
        assert np.all(np.diag(tariffs) == 0.0)
        # welfloss channel sees the effective tariff matrix
        np.testing.assert_allclose(np.asarray(state["import_tariffs"]), tariffs)

    def test_mediator_action_defines_club(self, region_params):
        """Without fixed_club_params, club terms come from the mediator's action."""
        env = _make_club_env(region_params)
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        _set_action(actions, MEDIATOR, "club_min_rate", 6)
        _set_action(actions, MEDIATOR, "club_tariff", 4)
        state, _ = _step(env, state, actions)  # propose

        assert float(state["club_min_mitigation"]) == pytest.approx(0.6)
        assert float(state["club_tariff_rate"]) == pytest.approx(0.4)


class TestMediatorReward:
    def test_members_mode(self, region_params):
        env = _make_club_env(
            region_params,
            fixed_club_params=(0.0, 0.0),
            mediator_reward_mode="members",
        )
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        for i, join in enumerate([1, 1, 0]):
            _set_action(actions, i_to_agent_str(i), "club_join", join)
        state, _ = _step(env, state, actions)  # propose
        state, reward = _step(env, state, actions)  # evaluate
        assert float(reward[MEDIATOR]) == pytest.approx(2.0 / 3.0)

    def test_emissions_mode(self, region_params):
        env = _make_club_env(
            region_params,
            fixed_club_params=(0.0, 0.0),
            mediator_reward_mode="emissions",
        )
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        for _ in range(3):  # propose, evaluate, climate
            state, reward = _step(env, state, actions)
        expected = -float(state["global_emissions"]) * env.mediator_reward_scale
        assert float(reward[MEDIATOR]) == pytest.approx(expected)


class TestInterface:
    def test_obs_shapes_match_across_agents(self, region_params):
        env = _make_club_env(region_params)
        obs, _ = env.reset_env(KEY)
        assert set(obs) == {i_to_agent_str(i) for i in range(NUM_REGIONS)} | {MEDIATOR}
        region_shape = obs[i_to_agent_str(0)].observation.shape
        for agent, agent_obs in obs.items():
            assert agent_obs.observation.shape == region_shape, agent

    def test_ppo_smoke(self, region_params):
        env = jym.LogWrapper(_make_club_env(region_params))
        ppo = PPO(
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
        agent, _ = ppo.train(KEY, env)
        assert agent is not None
