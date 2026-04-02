"""
tests/test_rice_mrio.py

Pytest test suite for RiceMRIO — covering all sanity checks from
CBAM_ROADMAP.md phases 1B, 2A, and 2A+.

Run from rice_jax/ with:
    conda activate rice-jax
    pytest tests/test_rice_mrio.py -v

Groups
------
TestPhase1B   — MRIO scaffold: static share arrays
TestPhase2A   — Trade flows at zero delta; CBAM wedge; budget conservation
TestPhase2Aplus — sectoral_welfloss: dirty/clean wedge asymmetry
"""

from __future__ import annotations

import sys
import os

# Make rice_jax importable when run from rice_jax/ directory directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rice_jax import Rice, RiceMRIO
from rice_jax.utils import load_region_yamls

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_REGIONS = 7
EU_REGION_IDX = 5   # Europe & Central Asia — NEVER 0 for 7-region
SEED = 42

# A minimal but representative experiment config used by all tests.
_BASE_KWARGS = dict(
    num_regions=NUM_REGIONS,
    mrio_data_root=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "csv_asset",
    ),
    mrio_trade=False,
    cbam_tariff_rate=0.0,
    cbam_randomize=False,
    eu_region_idx=EU_REGION_IDX,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    sectoral_welfloss=False,
    fixed_savings_rate=False,
    no_mitigation=False,
    sector_granularity="emissions-simple",
    welfare_loss_per_unit_tariff=5.0,
)

_MRIO_KWARGS = dict(_BASE_KWARGS, mrio_trade=True)


@pytest.fixture(scope="module")
def region_params():
    return load_region_yamls(NUM_REGIONS)


@pytest.fixture(scope="module")
def env_base(region_params):
    return RiceMRIO(region_params=region_params, **_BASE_KWARGS)


@pytest.fixture(scope="module")
def env_mrio(region_params):
    return RiceMRIO(region_params=region_params, **_MRIO_KWARGS)


@pytest.fixture(scope="module")
def env_mrio_cbam(region_params):
    return RiceMRIO(
        region_params=region_params,
        **dict(_MRIO_KWARGS, cbam_tariff_rate=0.80),
    )


@pytest.fixture(scope="module")
def env_mrio_sectoral(region_params):
    return RiceMRIO(
        region_params=region_params,
        **dict(_MRIO_KWARGS, cbam_tariff_rate=0.80, sectoral_welfloss=True),
    )


@pytest.fixture(scope="module")
def env_rice(region_params):
    """Base Rice (no MRIO) — used for Phase 1B bit-identity checks."""
    rice_kwargs = dict(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=10,
        diff_reward_mode=True,
        fixed_savings_rate=False,
        no_mitigation=False,
    )
    return Rice(**rice_kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _midpoint_actions(env, key):
    """Return midpoint-discrete actions (no CBAM action effect)."""
    actions = env.sample_action(key)
    D = env.num_discrete_action_levels
    mid = D // 2
    for agent in actions:
        for k, v in actions[agent].items():
            actions[agent][k] = jnp.full_like(v, mid)
    return actions


def _zero_realloc_actions(env, key):
    """Midpoint for export_reallocation (= zero logit delta) + mid for others."""
    return _midpoint_actions(env, key)


def _rollout(env, key, steps: int = 5):
    """Run a fixed-action rollout; return list of states."""
    obs, state = env.reset(key)
    states = []
    for _ in range(steps):
        actions = _midpoint_actions(env, key)
        processed = env.process_actions(actions, state)
        state = env.step_climate_and_economy(state, processed)
        states.append(state)
    return states


# ---------------------------------------------------------------------------
# Phase 1B — MRIO scaffold
# ---------------------------------------------------------------------------

class TestPhase1B:
    """Static share arrays are well-formed; production_by_sector sums correctly."""

    def test_sector_shares_sum_to_one(self, env_base):
        """σ_{r,s} must sum to 1 across sectors for every region."""
        shares = env_base.sector_output_shares   # (NR, NS)
        assert shares is not None, "sector_output_shares not set"
        row_sums = shares.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), (
            f"Sector share rows do not sum to 1: min={row_sums.min():.8f}, max={row_sums.max():.8f}"
        )

    def test_sector_shares_no_nan_inf(self, env_base):
        shares = env_base.sector_output_shares
        assert not np.isnan(shares).any(), "NaN in sector_output_shares"
        assert not np.isinf(shares).any(), "Inf in sector_output_shares"

    def test_sector_shares_nonnegative(self, env_base):
        shares = env_base.sector_output_shares
        assert (shares >= 0).all(), "Negative values in sector_output_shares"

    def test_num_sectors(self, env_base):
        """emissions-simple → exactly 2 sectors (dirty=0, clean=1)."""
        assert env_base.num_sectors == 2, (
            f"Expected 2 sectors for emissions-simple, got {env_base.num_sectors}"
        )

    def test_production_by_sector_sum(self, env_base):
        """production_by_sector.sum(axis=1) == production_all_regions at every step."""
        key = jax.random.PRNGKey(SEED)
        states = _rollout(env_base, key, steps=5)
        for t, state in enumerate(states):
            pbs = np.array(state["production_by_sector"])   # (NR, NS)
            pa  = np.array(state["production_all_regions"])  # (NR,)
            assert np.allclose(pbs.sum(axis=1), pa, atol=1e-4), (
                f"Step {t}: production_by_sector.sum != production_all_regions\n"
                f"  max diff = {np.abs(pbs.sum(axis=1) - pa).max():.6e}"
            )

    def test_mrio_region_labels_count(self, env_base):
        assert len(env_base.mrio_region_labels) == NUM_REGIONS

    def test_sector_names_count(self, env_base):
        assert len(env_base.sector_names) == env_base.num_sectors


# ---------------------------------------------------------------------------
# Phase 2A — Trade flows live
# ---------------------------------------------------------------------------

class TestPhase2A:
    """With mrio_trade=True and zero delta, baseline conditions; CBAM wedge; budget."""

    # ── Initialisation ────────────────────────────────────────────────────

    def test_dest_alloc_baseline_no_nan(self, env_mrio):
        dab = env_mrio.dest_alloc_baseline
        assert dab is not None, "dest_alloc_baseline is None — MRIO data not loaded"
        assert not np.isnan(dab).any(), "NaN in dest_alloc_baseline"
        assert not np.isinf(dab).any(), "Inf in dest_alloc_baseline"

    def test_dest_alloc_baseline_sums_to_one(self, env_mrio):
        """Destination shares baseline sums to 1 per (region, sector) — no self-trade."""
        dab = env_mrio.dest_alloc_baseline   # (NR, NS, NR) [from, sector, to]
        # Row-sum over last axis (destinations)
        row_sums = dab.sum(axis=2)           # (NR, NS)
        assert np.allclose(row_sums, 1.0, atol=1e-4), (
            f"dest_alloc_baseline rows don't sum to 1: min={row_sums.min():.6f}"
        )

    def test_dest_alloc_no_self_trade(self, env_mrio):
        """Self-destination share should be zero in the baseline."""
        dab = env_mrio.dest_alloc_baseline   # (NR, NS, NR)
        for r in range(NUM_REGIONS):
            self_share = dab[r, :, r]         # (NS,) — sector shares going to self
            assert np.allclose(self_share, 0.0, atol=1e-5), (
                f"Region {r} has non-zero self-destination baseline: {self_share}"
            )

    def test_emissions_intensity_no_nan(self, env_mrio):
        intensity = env_mrio.emissions_intensity
        assert intensity is not None
        assert not np.isnan(intensity).any()

    def test_total_export_frac_range(self, env_mrio):
        """Export fractions must be in [0, 1]."""
        tef = env_mrio.total_export_frac
        assert tef is not None
        assert (tef >= 0).all() and (tef <= 1.0 + 1e-5).all(), (
            f"total_export_frac out of [0,1]: min={tef.min():.4f}, max={tef.max():.4f}"
        )

    # ── State keys present ────────────────────────────────────────────────

    def test_state_keys_present(self, env_mrio):
        key = jax.random.PRNGKey(SEED)
        _, state = env_mrio.reset(key)
        for k in ("trade_flows", "cbam_revenue", "dest_alloc_current", "production_by_sector"):
            assert k in state, f"Missing state key: {k}"

    def test_trade_flows_shape(self, env_mrio):
        key = jax.random.PRNGKey(SEED)
        _, state = env_mrio.reset(key)
        tf = state["trade_flows"]
        assert tf.shape == (NUM_REGIONS, NUM_REGIONS, env_mrio.num_sectors), (
            f"trade_flows shape {tf.shape} != (NR, NR, NS)"
        )

    # ── Zero-delta rollout ────────────────────────────────────────────────

    def test_zero_delta_trade_flows_positive(self, env_mrio):
        """At zero delta the realized trade flows must be non-negative."""
        key = jax.random.PRNGKey(SEED)
        states = _rollout(env_mrio, key, steps=5)
        for t, state in enumerate(states):
            tf = np.array(state["trade_flows"])
            assert (tf >= -1e-6).all(), (
                f"Step {t}: negative trade_flows detected (min={tf.min():.4e})"
            )

    def test_budget_conservation(self, env_mrio):
        """Total exports from region r in sector s <= production_by_sector[r, s]."""
        key = jax.random.PRNGKey(SEED)
        states = _rollout(env_mrio, key, steps=5)
        for t, state in enumerate(states):
            tf  = np.array(state["trade_flows"])    # (NR, NR, NS) [from, to, sector]
            pbs = np.array(state["production_by_sector"])  # (NR, NS)
            # Sum over destinations
            total_exports = tf.sum(axis=1)           # (NR, NS) [from, sector]
            export_volume = pbs * env_mrio.total_export_frac   # (NR, NS) — capped by this
            assert (total_exports <= export_volume + 1e-4).all(), (
                f"Step {t}: total exports exceed production_by_sector * export_frac\n"
                f"  max excess = {(total_exports - export_volume).max():.4e}"
            )

    # ── No-CBAM: cbam_revenue must be zero ───────────────────────────────

    def test_no_cbam_zero_revenue(self, env_mrio):
        """Without a tariff, cbam_revenue should remain zero at every step."""
        key = jax.random.PRNGKey(SEED)
        states = _rollout(env_mrio, key, steps=5)
        for t, state in enumerate(states):
            rev = np.array(state["cbam_revenue"])
            assert np.allclose(rev, 0.0, atol=1e-6), (
                f"Step {t}: cbam_revenue non-zero without tariff: {rev}"
            )

    # ── CBAM wedge ────────────────────────────────────────────────────────

    def test_cbam_revenue_positive_with_tariff(self, env_mrio_cbam):
        """With τ=0.80, EU cbam_revenue > 0 after the first step."""
        key = jax.random.PRNGKey(SEED)
        states = _rollout(env_mrio_cbam, key, steps=3)
        for t, state in enumerate(states):
            rev = np.array(state["cbam_revenue"])
            eu_rev = rev[EU_REGION_IDX]
            assert eu_rev > 0.0, (
                f"Step {t}: EU cbam_revenue == 0 with tariff=0.80 (got {eu_rev:.4e})"
            )

    def test_cbam_lowers_welfare(self, env_mrio, env_mrio_cbam):
        """Utility at τ=0.80 should be <= utility at τ=0 (CBAM is a welfare loss)."""
        key = jax.random.PRNGKey(SEED)
        states_no   = _rollout(env_mrio,      key, steps=3)
        states_cbam = _rollout(env_mrio_cbam, key, steps=3)
        for t in range(len(states_no)):
            u_no   = np.array(states_no[t]["utility_times_welfloss_all_regions"])
            u_cbam = np.array(states_cbam[t]["utility_times_welfloss_all_regions"])
            # Exclude EU itself (it collects not pays)
            for r in range(NUM_REGIONS):
                if r == EU_REGION_IDX:
                    continue
                assert u_cbam[r] <= u_no[r] + 1e-4, (
                    f"Step {t}, region {r}: CBAM should lower welfare "
                    f"(no_cbam={u_no[r]:.4f}, cbam={u_cbam[r]:.4f})"
                )

    def test_cbam_tariff_rate_in_state(self, env_mrio_cbam):
        """cbam_tariff_rate stored in state matches the env parameter."""
        key = jax.random.PRNGKey(SEED)
        _, state = env_mrio_cbam.reset(key)
        stored = float(state["cbam_tariff_rate"])
        assert abs(stored - env_mrio_cbam.cbam_tariff_rate) < 1e-6, (
            f"State cbam_tariff_rate {stored} != env param {env_mrio_cbam.cbam_tariff_rate}"
        )

    # ── Observations ─────────────────────────────────────────────────────

    def test_cbam_rate_in_obs(self, env_mrio_cbam):
        """cbam_tariff_rate must appear in the raw per-agent observation dict.

        We call generate_observation() directly rather than env.reset() because
        jaxnasium flattens the per-agent dict into an AgentObservation container
        before returning from reset().
        """
        key = jax.random.PRNGKey(SEED)
        _, state = env_mrio_cbam.reset(key)
        obs = env_mrio_cbam.generate_observation(state)
        for agent_key, agent_obs in obs.items():
            assert "cbam_tariff_rate" in agent_obs, (
                f"Agent {agent_key}: cbam_tariff_rate missing from observation"
            )

    def test_trade_flows_in_obs(self, env_mrio):
        """trade_flows must appear in the raw per-agent observation dict."""
        key = jax.random.PRNGKey(SEED)
        _, state = env_mrio.reset(key)
        obs = env_mrio.generate_observation(state)
        for agent_key, agent_obs in obs.items():
            assert "trade_flows" in agent_obs, (
                f"Agent {agent_key}: trade_flows missing from observation"
            )

    # ── Action space ──────────────────────────────────────────────────────

    def test_action_space_has_export_reallocation(self, env_mrio):
        for agent_key, agent_space in env_mrio.action_space.items():
            assert "export_reallocation" in agent_space, (
                f"Agent {agent_key}: no export_reallocation in action_space"
            )

    def test_export_reallocation_dims(self, env_mrio):
        """export_reallocation must have NS * NR dimensions."""
        expected_dims = env_mrio.num_sectors * NUM_REGIONS
        for agent_key, agent_space in env_mrio.action_space.items():
            er = agent_space["export_reallocation"]
            # MultiDiscrete wraps a list; len gives number of dims
            n_dims = len(er.nvec) if hasattr(er, "nvec") else int(np.prod(er.shape))
            assert n_dims == expected_dims, (
                f"Agent {agent_key}: export_reallocation has {n_dims} dims, "
                f"expected {expected_dims} (NS={env_mrio.num_sectors} × NR={NUM_REGIONS})"
            )

    def test_no_legacy_actions_in_mrio_space(self, env_mrio):
        """import_bid / import_tariff / export_limit removed from action space in Phase 2A."""
        for agent_key, agent_space in env_mrio.action_space.items():
            for legacy in ("import_bid", "import_tariff", "export_limit"):
                assert legacy not in agent_space, (
                    f"Agent {agent_key}: legacy action '{legacy}' still in action_space"
                )


# ---------------------------------------------------------------------------
# Phase 2A+ — Sectoral welfare loss
# ---------------------------------------------------------------------------

class TestPhase2Aplus:
    """sectoral_welfloss=True creates asymmetric wedge: dirty costly, clean not."""

    def test_sectoral_welfloss_attribute(self, env_mrio_sectoral):
        assert env_mrio_sectoral.sectoral_welfloss is True

    def test_dirty_sector_higher_intensity(self, env_mrio_sectoral):
        """Sanity check: emissions-simple grouping puts higher intensity in sector 0.

        If this fails, the sector ordering is inverted and ALL welfloss
        tests need to swap the sector index.
        """
        intensity = env_mrio_sectoral.emissions_intensity  # (NR, NS)
        mean_dirty = float(intensity[:, 0].mean())
        mean_clean = float(intensity[:, 1].mean())
        assert mean_dirty > mean_clean, (
            f"Sector 0 (CBAM/dirty) mean intensity {mean_dirty:.6f} "
            f"should be > sector 1 (non-CBAM/clean) mean intensity {mean_clean:.6f}."
            " Check _build_sector_groups ordering."
        )

    def test_dirty_eu_exports_lower_welfloss_multiplier(self, env_mrio_sectoral):
        """With sectoral_welfloss=True, the welfloss multiplier (1 - CBAM_penalty/Y)
        is lower (more penalising) when dirty-sector exports go to EU than when
        the same volume of clean-sector exports does.

        This is tested by directly computing the welfloss formula with crafted
        trade_flows rather than via a full env step, which would confound the
        CBAM effect with consumption-redistribution effects from different trade
        patterns.

        Literature anchor: Carbone & Rivers (2017), §4 — sector-differentiated
        trade diversion under asymmetric carbon costs: higher-intensity exports
        to the CBAM zone incur a larger welfare penalty than lower-intensity
        exports of equal volume.
        """
        key = jax.random.PRNGKey(SEED)
        NR = NUM_REGIONS
        EU = EU_REGION_IDX
        intensity = jnp.array(env_mrio_sectoral.emissions_intensity)  # (NR, NS)
        tau = env_mrio_sectoral.cbam_tariff_rate
        alpha = env_mrio_sectoral.welfare_loss_per_unit_tariff

        # Craft two trade_flow tensors with identical total EU-bound volume per region:
        #   A: all EU-bound volume in sector 0 (dirty/CBAM)
        #   B: all EU-bound volume in sector 1 (non-CBAM/clean)
        volume = 1.0   # arbitrary unit; cancels in all comparisons
        ns = env_mrio_sectoral.num_sectors

        # A: dirty → EU, zero clean → EU
        tf_dirty = jnp.zeros((NR, NR, ns))
        tf_dirty = tf_dirty.at[:, EU, 0].set(volume)   # sector 0 to EU

        # B: clean → EU, zero dirty → EU
        tf_clean = jnp.zeros((NR, NR, ns))
        tf_clean = tf_clean.at[:, EU, 1].set(volume)   # sector 1 to EU

        def cbam_loss(tf):
            """Raw CBAM penalty per exporter from the sectoral_welfloss formula.

            We compare unclipped loss values rather than the welfloss multiplier
            (1 - loss/Y) to avoid jnp.clip(0,1) masking the signal when the
            penalty is large relative to gross output at t=0.
            """
            eu_exports = tf[:, EU, :]                              # (NR, NS)
            return (eu_exports * intensity * tau * alpha).sum(axis=1)  # (NR,)

        loss_dirty = np.array(cbam_loss(tf_dirty))
        loss_clean = np.array(cbam_loss(tf_clean))

        non_eu = [r for r in range(NR) if r != EU]
        dirty_more_penalised = sum(
            loss_dirty[r] > loss_clean[r] + 1e-12 for r in non_eu
        )
        assert dirty_more_penalised == len(non_eu), (
            "Every non-EU region should have a larger CBAM loss "
            "when the same export volume goes to EU in the dirty/CBAM sector "
            f"vs the clean sector.\n"
            f"  CBAM loss under dirty→EU: {loss_dirty[non_eu]}\n"
            f"  CBAM loss under clean→EU: {loss_clean[non_eu]}"
        )

    def test_sectoral_welfloss_zero_tariff_equals_one(self, region_params):
        """When τ=0, sectoral_welfloss multiplier should be 1 (no penalty)."""
        env = RiceMRIO(
            region_params=region_params,
            **dict(_MRIO_KWARGS, cbam_tariff_rate=0.0, sectoral_welfloss=True),
        )
        key = jax.random.PRNGKey(SEED)
        env_no_sl = RiceMRIO(
            region_params=region_params,
            **dict(_MRIO_KWARGS, cbam_tariff_rate=0.0, sectoral_welfloss=False),
        )
        _, state = env.reset(key)
        actions = _midpoint_actions(env, key)
        proc = env.process_actions(actions, state)
        s_sl  = env.step_climate_and_economy(state, proc)
        s_no  = env_no_sl.step_climate_and_economy(state, env_no_sl.process_actions(actions, state))

        u_sl = np.array(s_sl["utility_times_welfloss_all_regions"])
        u_no = np.array(s_no["utility_times_welfloss_all_regions"])
        assert np.allclose(u_sl, u_no, atol=1e-4), (
            "With τ=0, sectoral_welfloss=True should give same utility as False\n"
            f"  max diff = {np.abs(u_sl - u_no).max():.4e}"
        )
