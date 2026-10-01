"""Canonical null + mechanics tests for the PLS club envs and shield.

Run from rice_jax/ with:
    conda activate rice-jax
    pytest pls_club/tests/test_pls_club.py -v
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
import numpy as np
import optax
import pytest
from jaxnasium import Discrete
from jaxnasium.algorithms.core._input_output import CategoricalLayer

from club.env import RiceClubMediator
from pls_club.env import PLSClubMediator, PLSClubTariffAmbition, PLSNaiveClubRice
from pls_club.shield import ShieldedCategoricalLayer, mitigation_safety_weights
from rice_jax import BasicClubTariffAmbition, Rice
from rice_jax.utils import i_to_agent_str, load_region_yamls

NUM_REGIONS = 3
L = 10
SEED = 0
KEY = jax.random.PRNGKey(SEED)

CHECK_KEYS = [
    "global_temperature",
    "global_carbon_mass",
    # NOTE: "global_emissions" excluded: base Rice never updates it under
    # carbon_model="base"; the club envs fix that on purpose
    "capital_all_regions",
    "gross_output_all_regions",
    "utility_all_regions",
    "aggregate_consumption",
    "utility_times_welfloss_all_regions",
]


@pytest.fixture(scope="module")
def region_params():
    return load_region_yamls(NUM_REGIONS)


def _zero_actions(env):
    return optax.tree.zeros_like(env.sample_action(KEY))


def _set_action(actions, agent, key, level):
    actions[agent][key] = jnp.full_like(actions[agent][key], level)


def _step(env, state, actions):
    (_, reward, _, _, _), state = env.step_env(KEY, state, actions)
    return state, reward


def _assert_states_close(state_a, state_b, msg=""):
    for k in CHECK_KEYS:
        np.testing.assert_allclose(
            np.asarray(state_a[k]),
            np.asarray(state_b[k]),
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"state[{k}] diverged {msg}",
        )


# --------------------------------------------------------------------- shield


class TestShieldedLayer:
    def _layers(self, in_features=8):
        k = jax.random.PRNGKey(42)
        stock = CategoricalLayer(in_features, Discrete(L), key=k)
        shielded = ShieldedCategoricalLayer(in_features, Discrete(L), key=k)
        x = jax.random.normal(jax.random.PRNGKey(7), (in_features,))
        return stock, shielded, x

    def test_ones_mask_matches_stock(self):
        stock, shielded, x = self._layers()
        ones = jnp.ones(L)
        np.testing.assert_allclose(
            shielded(x, action_mask=ones).probs, stock(x, action_mask=ones).probs,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            shielded(x).probs, stock(x).probs, rtol=1e-6
        )

    def test_hard_mask_equivalent(self):
        stock, shielded, x = self._layers()
        mask = jnp.array([0.0] * 5 + [1.0] * 5)
        np.testing.assert_allclose(
            shielded(x, action_mask=mask).probs,
            stock(x, action_mask=mask).probs,
            atol=1e-7,
        )

    def test_fractional_weights_are_exact_pls(self):
        """Shielded probs == P(safe|a) * pi(a) / P_pi(safe)  (Def. 3.1)."""
        _, shielded, x = self._layers()
        w = jnp.array([0.2] * 5 + [1.0] * 5)
        base_probs = shielded(x).probs
        expected = base_probs * w / jnp.sum(base_probs * w)
        np.testing.assert_allclose(
            shielded(x, action_mask=w).probs, expected, rtol=1e-5
        )
        # Defection still possible: shielded probability below floor is nonzero
        assert shielded(x, action_mask=w).probs[:5].sum() > 0.0


class TestShieldWeights:
    def test_null_shield_strength_zero(self):
        w = mitigation_safety_weights(
            jnp.float32(0.5), jnp.ones(NUM_REGIONS), L, shield_strength=0.0
        )
        np.testing.assert_allclose(w, np.ones((NUM_REGIONS, L)))

    def test_nonmembers_unshielded(self):
        w = mitigation_safety_weights(
            jnp.float32(0.5), jnp.zeros(NUM_REGIONS), L, shield_strength=0.8
        )
        np.testing.assert_allclose(w, np.ones((NUM_REGIONS, L)))

    def test_member_downweighted_below_floor(self):
        membership = jnp.array([1.0, 0.0, 1.0])
        w = mitigation_safety_weights(
            jnp.float32(0.5), membership, L, shield_strength=0.8
        )
        expected_member = np.array([0.2] * 5 + [1.0] * 5, dtype=np.float32)
        np.testing.assert_allclose(w[0], expected_member, rtol=1e-6)
        np.testing.assert_allclose(w[1], np.ones(L))
        np.testing.assert_allclose(w[2], expected_member, rtol=1e-6)

    def test_graded_mode(self):
        w = mitigation_safety_weights(
            jnp.float32(0.5), jnp.ones(1), L, mode="graded", kappa=10.0
        )
        np.testing.assert_allclose(w[0, 5:], np.ones(5))  # compliant -> 1
        assert np.all(np.diff(np.asarray(w[0, :5])) > 0)  # monotone in shortfall
        np.testing.assert_allclose(w[0, 0], np.exp(-5.0), rtol=1e-5)


# ------------------------------------------------------------------ naive env


def _make_naive(region_params, **kwargs) -> PLSNaiveClubRice:
    defaults = dict(
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=L,
        diff_reward_mode=True,
        negotiation_on=False,
    )
    defaults.update(kwargs)
    return PLSNaiveClubRice(region_params=region_params, **defaults)


class TestNaiveEnv:
    def test_null_vs_base_rice(self, region_params):
        """No joiners + zero tariffs == base Rice(negotiation_on=False)."""
        pls = _make_naive(region_params)
        base = Rice(
            region_params=region_params,
            num_regions=NUM_REGIONS,
            num_discrete_action_levels=L,
            diff_reward_mode=True,
            negotiation_on=False,
        )
        _, pls_state = pls.reset_env(KEY)
        _, base_state = base.reset_env(KEY)

        pls_actions = _zero_actions(pls)
        base_actions = _zero_actions(base)
        for i in range(NUM_REGIONS):
            agent = i_to_agent_str(i)
            for actions in (pls_actions, base_actions):
                _set_action(actions, agent, "savings_rate", 2)
                _set_action(actions, agent, "mitigation_rate", 3)

        for t in range(5):
            pls_state, _ = _step(pls, pls_state, pls_actions)
            base_state, _ = _step(base, base_state, base_actions)
            _assert_states_close(pls_state, base_state, msg=f"at step {t}")

    def test_shield_weights_in_action_mask(self, region_params):
        env = _make_naive(region_params, shield_strength=0.8, club_min_rate=0.5)
        _, state = env.reset_env(KEY)
        state["club_membership"] = jnp.array([1.0, 0.0, 0.0])
        mask = env.generate_action_masks(state)
        member = np.asarray(mask[i_to_agent_str(0)]["mitigation_rate"])
        outsider = np.asarray(mask[i_to_agent_str(1)]["mitigation_rate"])
        np.testing.assert_allclose(member, [0.2] * 5 + [1.0] * 5, rtol=1e-6)
        np.testing.assert_allclose(outsider, np.ones(L))
        # No hard mask anywhere: every mitigation level stays reachable
        assert member.min() > 0.0

    def test_defector_sanctioned(self, region_params):
        env = _make_naive(region_params, club_min_rate=0.5, club_tariff=0.5)
        _, state = env.reset_env(KEY)

        # Step 1: regions 0 and 1 join (membership binds from next step)
        actions = _zero_actions(env)
        for i in (0, 1):
            _set_action(actions, i_to_agent_str(i), "club_join", 1)
        state, _ = _step(env, state, actions)
        np.testing.assert_allclose(state["club_membership"], [1.0, 1.0, 0.0])
        np.testing.assert_allclose(state["club_defectors"], [0.0, 0.0, 0.0])

        # Step 2: region 0 complies (0.6 >= 0.5), region 1 defects (0.4 < 0.5)
        _set_action(actions, i_to_agent_str(0), "mitigation_rate", 6)
        _set_action(actions, i_to_agent_str(1), "mitigation_rate", 4)
        state, _ = _step(env, state, actions)

        np.testing.assert_allclose(state["club_defectors"], [0.0, 1.0, 0.0])
        tariffs = np.asarray(state["import_tariffs"])
        # Complying member 0 sanctions defector 1 and outsider 2, nobody else
        np.testing.assert_allclose(tariffs[0], [0.0, 0.5, 0.5])
        np.testing.assert_allclose(tariffs[1], np.zeros(NUM_REGIONS))
        np.testing.assert_allclose(tariffs[2], np.zeros(NUM_REGIONS))


# --------------------------------------------------------------- mediator env


def _make_mediator(region_params, cls=PLSClubMediator, **kwargs):
    defaults = dict(
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=L,
        diff_reward_mode=True,
    )
    defaults.update(kwargs)
    return cls(region_params=region_params, **defaults)


class TestMediatorEnv:
    def test_null_vs_club_env(self, region_params):
        """fixed_club_params=(0,0): PLS mediator == mask-based club env."""
        pls = _make_mediator(region_params, fixed_club_params=(0.0, 0.0))
        club = _make_mediator(
            region_params, cls=RiceClubMediator, fixed_club_params=(0.0, 0.0)
        )
        _, pls_state = pls.reset_env(KEY)
        _, club_state = club.reset_env(KEY)

        pls_actions = _zero_actions(pls)
        club_actions = _zero_actions(club)
        for i in range(NUM_REGIONS):
            agent = i_to_agent_str(i)
            for actions in (pls_actions, club_actions):
                _set_action(actions, agent, "savings_rate", 2)
                _set_action(actions, agent, "mitigation_rate", 3)
                _set_action(actions, agent, "club_join", 1)

        for cycle in range(3):
            for _ in range(3):  # propose, evaluate, climate
                pls_state, _ = _step(pls, pls_state, pls_actions)
                club_state, _ = _step(club, club_state, club_actions)
            _assert_states_close(pls_state, club_state, msg=f"at cycle {cycle}")

    def test_shield_replaces_hard_mask(self, region_params):
        env = _make_mediator(
            region_params, fixed_club_params=(0.5, 0.3), shield_strength=0.8
        )
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        for i in range(NUM_REGIONS):
            _set_action(actions, i_to_agent_str(i), "club_join", 1)
        state, _ = _step(env, state, actions)  # t=1: propose
        state, _ = _step(env, state, actions)  # t=2: evaluate

        # Hard-mask channel stays off; shield weights carry compliance instead
        np.testing.assert_allclose(
            state["minimum_mitigation_rate_all_regions"], np.zeros(NUM_REGIONS)
        )
        mask = env.generate_action_masks(state)
        member = np.asarray(mask[i_to_agent_str(0)]["mitigation_rate"])
        np.testing.assert_allclose(member, [0.2] * 5 + [1.0] * 5, rtol=1e-6)
        assert member.min() > 0.0

    def test_defector_tariffed_in_climate_step(self, region_params):
        env = _make_mediator(region_params, fixed_club_params=(0.5, 0.3))
        _, state = env.reset_env(KEY)
        actions = _zero_actions(env)
        for i in range(NUM_REGIONS):
            _set_action(actions, i_to_agent_str(i), "club_join", 1)
        _set_action(actions, i_to_agent_str(0), "mitigation_rate", 6)  # complies
        _set_action(actions, i_to_agent_str(1), "mitigation_rate", 4)  # defects
        _set_action(actions, i_to_agent_str(2), "mitigation_rate", 6)  # complies

        state, _ = _step(env, state, actions)  # propose
        state, _ = _step(env, state, actions)  # evaluate
        state, _ = _step(env, state, actions)  # climate

        np.testing.assert_allclose(state["club_defectors"], [0.0, 1.0, 0.0])
        tariffs = np.asarray(state["import_tariffs"])
        np.testing.assert_allclose(tariffs[0], [0.0, 0.3, 0.0], atol=1e-7)
        np.testing.assert_allclose(tariffs[2], [0.0, 0.3, 0.0], atol=1e-7)
        np.testing.assert_allclose(tariffs[1], np.zeros(NUM_REGIONS), atol=1e-7)


# ----------------------------------------------------- tariff-ambition variant


def _make_ta(region_params, cls=PLSClubTariffAmbition, **kwargs):
    defaults = dict(
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=L,
        diff_reward_mode=True,
        negotiation_on=True,
    )
    defaults.update(kwargs)
    return cls(region_params=region_params, **defaults)


class TestTariffAmbitionEnv:
    def _run_cycles(self, env, actions, cycles=2):
        _, state = env.reset_env(KEY)
        for _ in range(cycles * 3):  # propose, evaluate, climate
            state, _ = _step(env, state, actions)
        return state

    def test_null_vs_basic_when_compliant(self, region_params):
        """Same actions, mitigation at/above floor: shield == hard-mask env."""
        pls = _make_ta(region_params)
        base = _make_ta(region_params, cls=BasicClubTariffAmbition)
        pls_actions = _zero_actions(pls)
        base_actions = _zero_actions(base)
        for i in range(NUM_REGIONS):
            agent = i_to_agent_str(i)
            for actions in (pls_actions, base_actions):
                _set_action(actions, agent, "savings_rate", 2)
                _set_action(actions, agent, "mitigation_rate", 5)
                _set_action(actions, agent, "proposal", 3)
                _set_action(actions, agent, "proposal_decisions", 1)
        pls_state = self._run_cycles(pls, pls_actions)
        base_state = self._run_cycles(base, base_actions)
        _assert_states_close(pls_state, base_state, msg="tariff-ambition null")

    def test_shield_replaces_hard_floor_mask(self, region_params):
        env = _make_ta(region_params, shield_strength=0.8)
        _, state = env.reset_env(KEY)
        state["minimum_mitigation_rate_all_regions"] = jnp.array([0.5, 0.0, 0.0])
        mask = env.generate_action_masks(state)
        floored = np.asarray(mask[i_to_agent_str(0)]["mitigation_rate"])
        free = np.asarray(mask[i_to_agent_str(1)]["mitigation_rate"])
        np.testing.assert_allclose(floored, [0.2] * 5 + [1.0] * 5, rtol=1e-6)
        np.testing.assert_allclose(free, np.ones(L))
        # defection possible: below-floor levels keep nonzero weight
        assert floored.min() > 0.0
        # hard-mask env zeroes those same levels
        base = _make_ta(region_params, cls=BasicClubTariffAmbition)
        _, bstate = base.reset_env(KEY)
        bstate["minimum_mitigation_rate_all_regions"] = jnp.array([0.5, 0.0, 0.0])
        bmask = base.generate_action_masks(bstate)
        assert np.asarray(bmask[i_to_agent_str(0)]["mitigation_rate"])[:5].max() == 0

    def test_tariff_ambition_mask_inherited(self, region_params):
        """Below-floor realized mitigation still triggers the tariff floor mask."""
        env = _make_ta(region_params)
        _, state = env.reset_env(KEY)
        state["minimum_mitigation_rate_all_regions"] = jnp.array([0.5, 0.0, 0.0])
        state["mitigation_rates_all_regions"] = jnp.array([0.5, 0.2, 0.5])
        mask = env.generate_action_masks(state)
        tariff_mask = np.asarray(mask[i_to_agent_str(0)]["import_tariff"])
        # region 1 is 0.3 below region 0's floor -> first 3 tariff levels banned
        np.testing.assert_allclose(tariff_mask[1][:3], np.zeros(3))
        assert tariff_mask[1][3:].min() == 1
