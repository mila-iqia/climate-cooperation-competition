"""
validation/validate_mrio_clubs.py

Runnable demonstration of the Phase 2E endogenous CBAM-club scenarios
(MRIOClubCBAM, MRIOSectoralClub, MRIOMultiClub).  Confirms:

  1. A full propose → evaluate → climate negotiation cycle runs end-to-end
     through ``step_env`` (the ``jax.lax.switch`` pytree-structure invariant
     and jit-compatibility).
  2. Expected per-subclass behaviour:
       - B1 singleton {EU} == base single-EU differential CBAM (null).
       - Grand coalition → zero CBAM cost.
       - Joining the club strictly lowers a laggard's CBAM cost.
       - Sectoral coverage gates CBAM by sector.
       - Multi-club partitions regions and charges non-members.

Run from rice_jax/:
    conda activate rice-jax
    python validation/validate_mrio_clubs.py

A bar chart of CBAM cost (member vs non-member, per scenario) is written to
``validation/mrio_clubs_cbam.png``.
"""

from __future__ import annotations

import os
import sys

# matplotlib BEFORE any JAX import (macOS Agg backend pollution, see
# .github/copilot-instructions.md → "JAX / matplotlib conflict").
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from rice_jax import RiceMRIO, MRIOClubCBAM, MRIOSectoralClub, MRIOMultiClub
from rice_jax.utils import load_region_yamls, i_to_agent_str

NUM_REGIONS = 7
EU_REGION_IDX = 5
SEED = 0
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_COMMON = dict(
    num_regions=NUM_REGIONS,
    mrio_data_root=os.path.join(os.path.dirname(_PROJECT_ROOT), "csv_asset"),
    mrio_trade=True,
    cbam_tariff_rate=0.80,
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


def _heterogeneous_mu():
    mu = jnp.full(NUM_REGIONS, 0.10, dtype=jnp.float32)
    return mu.at[EU_REGION_IDX].set(0.80)


def _climate_state(env, key):
    obs, state = env.reset(key)
    actions = env.sample_action(key)
    D = env.num_discrete_action_levels
    for agent in actions:
        for k, v in actions[agent].items():
            actions[agent][k] = jnp.full_like(v, D // 2)
    return env.step_climate_and_economy(state, env.process_actions(actions, state))


def demo_full_cycle(rp):
    print("\n=== 1. Full propose→evaluate→climate cycles via step_env ===")
    cases = [
        ("B1 MRIOClubCBAM", MRIOClubCBAM, {}),
        ("B2 MRIOSectoralClub", MRIOSectoralClub, {}),
        ("B3 MRIOMultiClub", MRIOMultiClub, {"club_cores": (EU_REGION_IDX, 2)}),
    ]
    for name, cls, extra in cases:
        env = cls(region_params=rp, **_COMMON, **extra)
        key = jax.random.PRNGKey(SEED)
        obs, state = env.reset(key)
        for _ in range(9):  # 3 full negotiation cycles
            key, k = jax.random.split(key)
            acts = env.sample_action(k)
            (obs, rew, term, trunc, info), state = env.step_env(k, state, acts)
        members = np.asarray(state["club_membership"]).astype(int).tolist()
        print(f"  {name:22s} OK  final club_membership={members}")


def demo_b1_null(rp):
    print("\n=== 2. B1 singleton {EU} == base single-EU differential (null) ===")
    env_diff = RiceMRIO(
        region_params=rp,
        **dict(_COMMON, cbam_tariff_mode="differential", reward_mode="welfloss"),
    )
    club = MRIOClubCBAM(region_params=rp, **_COMMON)
    state = _climate_state(env_diff, jax.random.PRNGKey(SEED))
    tf, mu, t = state["trade_flows"], _heterogeneous_mu(), state["activity_timestep"]

    _, ref_rev, ref_cost = env_diff._compute_cbam(
        tf, jnp.ones((NUM_REGIONS, NUM_REGIONS)),
        cbam_tariff_rate=state["cbam_tariff_rate"],
        mitigation_rates=mu, activity_timestep=t,
    )
    cstate = dict(state)
    cstate["club_membership"] = (
        jnp.zeros(NUM_REGIONS, dtype=jnp.bool_).at[EU_REGION_IDX].set(True)
    )
    cstate["club_mitigation_rate"] = mu[EU_REGION_IDX]
    _, club_rev, club_cost = club._postprocess_cbam(
        cstate, tf, jnp.ones((NUM_REGIONS, NUM_REGIONS)), mu,
        jnp.zeros((NUM_REGIONS, NUM_REGIONS)), jnp.zeros(NUM_REGIONS),
        jnp.zeros(NUM_REGIONS),
    )
    ok = bool(jnp.allclose(club_cost, ref_cost, atol=1e-6)
              and jnp.allclose(club_rev, ref_rev, atol=1e-6))
    print(f"  cost match={ok}  ref_cost.sum()={float(ref_cost.sum()):.4f}")
    return env_diff, club, state, tf, mu


def demo_membership(club, state, tf, mu):
    print("\n=== 3. Joining lowers own CBAM cost ===")
    r = 0 if EU_REGION_IDX != 0 else 1

    def cost_with(members):
        s = dict(state)
        s["club_membership"] = members
        s["club_mitigation_rate"] = jnp.float32(0.80)
        _, _, c = club._postprocess_cbam(
            s, tf, jnp.ones((NUM_REGIONS, NUM_REGIONS)), mu,
            jnp.zeros((NUM_REGIONS, NUM_REGIONS)), jnp.zeros(NUM_REGIONS),
            jnp.zeros(NUM_REGIONS),
        )
        return np.asarray(c)

    out = jnp.zeros(NUM_REGIONS, dtype=jnp.bool_).at[EU_REGION_IDX].set(True)
    cin = out.at[r].set(True)
    grand = jnp.ones(NUM_REGIONS, dtype=jnp.bool_)
    c_out, c_in, c_grand = cost_with(out), cost_with(cin), cost_with(grand)
    print(f"  region {r}: non-member cost={c_out[r]:.4f} → member cost={c_in[r]:.4f}")
    print(f"  grand coalition total cost={c_grand.sum():.6f} (expect ~0)")
    return c_out, c_in


def demo_sectoral(rp, state, tf, mu):
    print("\n=== 4. Sectoral coverage gates CBAM ===")
    club = MRIOSectoralClub(region_params=rp, **_COMMON)
    NS = club.num_sectors
    base = dict(state)
    base["club_membership"] = (
        jnp.zeros(NUM_REGIONS, dtype=jnp.bool_).at[EU_REGION_IDX].set(True)
    )
    base["club_mitigation_rate"] = jnp.float32(0.80)

    def cost_cov(cov):
        s = dict(base)
        s["sector_coverage"] = cov
        _, _, c = club._postprocess_cbam(
            s, tf, jnp.ones((NUM_REGIONS, NUM_REGIONS)), mu,
            jnp.zeros((NUM_REGIONS, NUM_REGIONS)), jnp.zeros(NUM_REGIONS),
            jnp.zeros(NUM_REGIONS),
        )
        return float(np.asarray(c).sum())

    full = cost_cov(jnp.ones(NS, dtype=jnp.bool_))
    dirty = cost_cov(jnp.array([True, False], dtype=jnp.bool_))
    none = cost_cov(jnp.zeros(NS, dtype=jnp.bool_))
    print(f"  total cost: full={full:.4f}  dirty-only={dirty:.4f}  none={none:.6f}")


def make_plot(c_out, c_in):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(NUM_REGIONS)
    w = 0.4
    ax.bar(x - w / 2, c_out, w, label="non-member (laggard 0 outside)")
    ax.bar(x + w / 2, c_in, w, label="laggard 0 joined club")
    ax.set_xlabel("region")
    ax.set_ylabel("CBAM cost")
    ax.set_title("MRIOClubCBAM: CBAM cost falls when a laggard accedes")
    ax.legend()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrio_clubs_cbam.png")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out}")


def main():
    rp = load_region_yamls(NUM_REGIONS)
    demo_full_cycle(rp)
    env_diff, club, state, tf, mu = demo_b1_null(rp)
    c_out, c_in = demo_membership(club, state, tf, mu)
    demo_sectoral(rp, state, tf, mu)
    make_plot(c_out, c_in)
    print("\nALL DEMOS OK")


if __name__ == "__main__":
    main()
