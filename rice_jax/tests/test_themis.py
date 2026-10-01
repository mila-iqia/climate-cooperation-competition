"""
tests/test_themis.py

Canonical null + invariant tests for the Themis mechanism overlay
(Rasmussen 2025 concept note) on both ThemisRice (core) and
ThemisRiceMRIO (MRIO).

Run from rice_jax/ with:
    conda activate rice-jax
    pytest tests/test_themis.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
import pytest

from cbam.config.canonical_config import (
    CANONICAL_MRIO_ROOT,
    CANONICAL_YAML_DIR,
    EU_REGION_IDX,
    NUM_REGIONS as MRIO_NUM_REGIONS,
)
from rice_jax import Rice, RiceMRIO, ThemisRice, ThemisRiceMRIO
from rice_jax.core.scenarios import compute_themis_payments
from rice_jax.utils import i_to_agent_str, load_region_yamls

SEED = 0
KEY = jax.random.PRNGKey(SEED)
CORE_NUM_REGIONS = 3

# Keys that must be unaffected by a null (p=0) Themis overlay.
_NULL_CHECK_KEYS = [
    "global_temperature",
    "global_carbon_mass",
    "production_all_regions",
    "aggregate_consumption",
    "utility_all_regions",
    "utility_times_welfloss_all_regions",
]

_MRIO_KWARGS = dict(
    num_regions=MRIO_NUM_REGIONS,
    mrio_data_root=CANONICAL_MRIO_ROOT,
    mrio_trade=True,
    cbam_tariff_rate=0.0,
    cbam_randomize=False,
    eu_region_idx=EU_REGION_IDX,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=0.0,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    sectoral_welfloss=False,
    fixed_savings_rate=False,
    no_mitigation=False,
    sector_granularity="emissions-simple",
    welfare_loss_per_unit_tariff=5.0,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core_params():
    return load_region_yamls(CORE_NUM_REGIONS)


@pytest.fixture(scope="module")
def mrio_params():
    return load_region_yamls(MRIO_NUM_REGIONS, yaml_dir=CANONICAL_YAML_DIR)


def _zero_actions(env):
    return optax.tree.zeros_like(env.sample_action(KEY))


def _set_action(actions, agent_id, key, level):
    agent = i_to_agent_str(agent_id)
    actions[agent][key] = jnp.full_like(actions[agent][key], level)


def _default_actions(env, join_levels=None):
    """Zero actions with mild savings/mitigation; optional per-region join."""
    actions = _zero_actions(env)
    for i in range(env.num_regions):
        _set_action(actions, i, "savings_rate", 2)
        _set_action(actions, i, "mitigation_rate", i + 1)  # heterogeneous μ
        if join_levels is not None and "themis_join" in actions[i_to_agent_str(i)]:
            _set_action(actions, i, "themis_join", join_levels[i])
    return actions


def _step(env, state, actions):
    (_, reward, _, _, _), state = env.step_env(KEY, state, actions)
    return state, reward


# ---------------------------------------------------------------------------
# Core (ThemisRice)
# ---------------------------------------------------------------------------


class TestThemisCore:
    def test_null_price_zero_matches_base(self, core_params):
        """p=0 + universal membership ≡ base Rice (canonical null)."""
        base = Rice(region_params=core_params, num_regions=CORE_NUM_REGIONS)
        themis = ThemisRice(
            region_params=core_params,
            num_regions=CORE_NUM_REGIONS,
            themis_price_schedule=(0.0,),
            themis_membership_mode="all",
        )
        _, s_base = base.reset_env(KEY)
        _, s_them = themis.reset_env(KEY)

        a_base = _default_actions(base)
        a_them = _default_actions(themis)

        for _ in range(5):
            s_base, r_base = _step(base, s_base, a_base)
            s_them, r_them = _step(themis, s_them, a_them)
            for k in _NULL_CHECK_KEYS:
                assert jnp.allclose(s_base[k], s_them[k], atol=1e-5), k
            for i in range(CORE_NUM_REGIONS):
                agent = i_to_agent_str(i)
                assert jnp.allclose(r_base[agent], r_them[agent], atol=1e-5)
            assert jnp.allclose(s_them["themis_payments_all_regions"], 0.0)

    def test_cost_neutrality_and_signs(self, core_params):
        """Σ payments = 0 over members; above-average per-capita emitter pays."""
        themis = ThemisRice(
            region_params=core_params,
            num_regions=CORE_NUM_REGIONS,
            themis_price_schedule=(50.0,),
            themis_membership_mode="all",
        )
        _, state = themis.reset_env(KEY)
        actions = _default_actions(themis)
        for _ in range(3):
            pre_labor = state["labor_all_regions"]  # settlement uses pre-step labor
            state, _ = _step(themis, state, actions)

        payments = state["themis_payments_all_regions"]
        emissions = state["themis_emissions_all_regions"]

        assert jnp.allclose(payments.sum(), 0.0, atol=1e-6)
        assert jnp.any(payments < 0) and jnp.any(payments > 0)
        # Highest per-capita emitter must be a net contributor and vice versa.
        per_capita = emissions / pre_labor
        assert payments[jnp.argmax(per_capita)] < 0
        assert payments[jnp.argmin(per_capita)] > 0

    def test_analytical_payment_one_step(self, core_params):
        """Payments match the hand-computed Rasmussen (2025) §1 formula."""
        price = 100.0
        themis = ThemisRice(
            region_params=core_params,
            num_regions=CORE_NUM_REGIONS,
            themis_price_schedule=(price,),
            themis_membership_mode="all",
        )
        _, state = themis.reset_env(KEY)
        pre_labor = state["labor_all_regions"]
        actions = _default_actions(themis)
        state, _ = _step(themis, state, actions)

        emissions = state["themis_emissions_all_regions"]
        expected = compute_themis_payments(
            jnp.float32(price), emissions, pre_labor, jnp.ones(CORE_NUM_REGIONS)
        )
        assert jnp.allclose(state["themis_payments_all_regions"], expected, atol=1e-6)
        # Consumption shifted by exactly the payment (no clamping active here).
        base = Rice(region_params=core_params, num_regions=CORE_NUM_REGIONS)
        _, s_base = base.reset_env(KEY)
        s_base, _ = _step(base, s_base, _default_actions(base))
        assert jnp.allclose(
            state["aggregate_consumption"],
            s_base["aggregate_consumption"] + expected,
            atol=1e-5,
        )

    def test_fixed_membership_nonmember_pays_nothing(self, core_params):
        themis = ThemisRice(
            region_params=core_params,
            num_regions=CORE_NUM_REGIONS,
            themis_price_schedule=(50.0,),
            themis_membership_mode="fixed",
            themis_fixed_membership=(1.0, 1.0, 0.0),
        )
        _, state = themis.reset_env(KEY)
        actions = _default_actions(themis)
        for _ in range(3):
            state, _ = _step(themis, state, actions)

        payments = state["themis_payments_all_regions"]
        assert jnp.allclose(payments[2], 0.0)
        assert jnp.allclose(payments.sum(), 0.0, atol=1e-6)
        assert jnp.allclose(
            state["themis_membership_all_regions"], jnp.array([1.0, 1.0, 0.0])
        )

    def test_join_action_sets_membership(self, core_params):
        themis = ThemisRice(
            region_params=core_params,
            num_regions=CORE_NUM_REGIONS,
            themis_price_schedule=(50.0,),
            themis_membership_mode="action",
        )
        _, state = themis.reset_env(KEY)
        actions = _default_actions(themis, join_levels=(1, 1, 0))
        state, _ = _step(themis, state, actions)

        assert jnp.allclose(
            state["themis_membership_all_regions"], jnp.array([1.0, 1.0, 0.0])
        )
        assert jnp.allclose(state["themis_payments_all_regions"][2], 0.0)


# ---------------------------------------------------------------------------
# MRIO (ThemisRiceMRIO)
# ---------------------------------------------------------------------------


class TestThemisMRIO:
    def test_null_price_zero_matches_base(self, mrio_params):
        """p=0 + universal membership ≡ base RiceMRIO (canonical null)."""
        base = RiceMRIO(region_params=mrio_params, **_MRIO_KWARGS)
        themis = ThemisRiceMRIO(
            region_params=mrio_params,
            themis_price_schedule=(0.0,),
            themis_membership_mode="all",
            **_MRIO_KWARGS,
        )
        _, s_base = base.reset_env(KEY)
        _, s_them = themis.reset_env(KEY)

        a_base = _default_actions(base)
        a_them = _default_actions(themis)

        for _ in range(3):
            s_base, r_base = _step(base, s_base, a_base)
            s_them, r_them = _step(themis, s_them, a_them)
            for k in _NULL_CHECK_KEYS:
                assert jnp.allclose(s_base[k], s_them[k], atol=1e-5), k
            for i in range(MRIO_NUM_REGIONS):
                agent = i_to_agent_str(i)
                assert jnp.allclose(r_base[agent], r_them[agent], atol=1e-5)
            assert jnp.allclose(s_them["themis_payments_all_regions"], 0.0)

    def test_cost_neutrality_and_join(self, mrio_params):
        """Σ payments = 0; non-joiner excluded; consumption shifted by payment."""
        themis = ThemisRiceMRIO(
            region_params=mrio_params,
            themis_price_schedule=(50.0,),
            themis_membership_mode="action",
            **_MRIO_KWARGS,
        )
        _, state = themis.reset_env(KEY)
        join = [1] * MRIO_NUM_REGIONS
        join[0] = 0  # RoW stays out
        actions = _default_actions(themis, join_levels=join)
        for _ in range(3):
            state, _ = _step(themis, state, actions)

        payments = state["themis_payments_all_regions"]
        membership = state["themis_membership_all_regions"]
        assert jnp.allclose(membership[0], 0.0)
        assert jnp.allclose(payments[0], 0.0)
        assert jnp.allclose(payments.sum(), 0.0, atol=1e-5)
        assert jnp.any(payments < 0) and jnp.any(payments > 0)
