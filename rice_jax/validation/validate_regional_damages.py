"""validate_regional_damages.py

Compares three damage-function configurations over a fixed-action rollout:

  A  Uniform damages     — use_regional_damage_coeff=False (baseline, yaml xa_2)
  B  Regional uniform    — use_regional_damage_coeff=True, coeff = xa_2 everywhere
                           → must be numerically identical to A
  C  Regional adjusted   — use_regional_damage_coeff=True, calibrated per-region coeffs
                           → must differ from A in a directionally expected way

Checks:
  CHECK 1  B vs A: max |damages_B - damages_A| < 1e-6  at every timestep
  CHECK 2  C vs A: damages differ by >1e-4 on at least one region/timestep
  CHECK 3  C ordering: regions with coeff < xa_2 survive better; regions with
           coeff > xa_2 survive worse (directional sanity)

Output: plots/regional_damages_validation_<TIMESTAMP>.png

Usage (from rice_jax/ directory):
    conda activate rice-jax
    python validation/validate_regional_damages.py
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import os

import jax
import jax.numpy as jnp
import numpy as np

from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls
from validation.canonical_config import (
    CANONICAL_MRIO_ROOT,
    CANONICAL_YAML_DIR,
    EU_REGION_IDX,
    NUM_REGIONS,
    REGION_NAMES,
)

# ── Configuration ─────────────────────────────────────────────────────────────

SEED = 42
ROLLOUT_STEPS = 20

# Baseline xa_2 value (uniform default from region yamls)
XA2_BASELINE = 0.00236

# Literature-calibrated quadratic temperature-sensitivity coefficients (a_2, °C⁻²).
# Derived via level-equivalent back-out: a_2 = D(T) / (T² · (1 − D(T)))
# where D(T) is the projected % GDP loss at T°C from CGE / empirical panel sources.
# Global RICE-2010 baseline: a_2 = 0.00236 (≈2.08% GDP loss at 3°C).
# Order matches REGION_NAMES:
#   0=RoW, 1=Russia+Eur, 2=MENA, 3=EU, 4=SSA-Mining,
#   5=Americas, 6=SE Asia, 7=China, 8=India
EXAMPLE_REGIONAL_COEFFS = np.array([
    0.00480,   # 0  RoW        — GDP-weighted aggregation of RICE-2010 sub-regions
               #                 and Kompas, Pham & Che (2018) tropical/temperate mix;
               #                 Hansel et al. (2020) non-market upscaling. (2.03× baseline)
    0.00130,   # 1  Russia+Eur — Cold-baseline temperate buffer; Russia RICE-2010 (a_2≈0.00114),
               #                 Turkey warmer-baseline scaling; Kompas et al. (2018) CGE.
               #                 NOTE: does not capture CBAM transition risk. (0.55× baseline)
    0.00720,   # 2  MENA       — Extreme outdoor heat / arid baseline; Kompas et al. (2018)
               #                 labor-productivity CGE shocks; Ricke et al. (2018) high CSCC
               #                 for Saudi Arabia/UAE; Hansel et al. (2020) a_2≈0.0068.
               #                 (3.05× baseline)
    0.00310,   # 3  EU         — Temperate near-optimum; Hansel et al. (2020) + Howard &
               #                 Sterner (2017) meta-analysis; Kompas et al. (2018) North/South
               #                 Europe split. Market-only calibration would be ≈0.00120.
               #                 (1.31× baseline)
    0.01350,   # 4  SSA-Mining — Highest physical vulnerability; Kompas et al. (2018) projects
               #                 ~15% GDP loss at 3°C for SSA; RICE-2010 SSA a_2≈0.00550;
               #                 Ricke et al. (2018) Nigeria high absolute CSCC. (5.72× baseline)
    0.00240,   # 5  Americas   — GDP-weighted US+Canada (low) vs Latin America (high);
               #                 Hansel et al. (2020) global meta-analysis update;
               #                 Kompas et al. (2018) moderate N.American losses. (1.02× baseline)
    0.00980,   # 6  SE Asia    — Tropical coastal exposure; Kompas et al. (2018) ~12% GDP
               #                 loss at 3°C for SE Asia; IPCC AR6 WGII Ch.16 coastal/monsoon.
               #                 (4.15× baseline)
    0.00290,   # 7  China      — Temperate baseline, large coastal exposure; Ricke et al. (2018)
               #                 2nd-largest absolute CSCC; Hansel et al. (2020) a_2≈0.00290;
               #                 Kompas et al. (2018) ~3–4% long-run GDP loss. (1.23× baseline)
    0.01200,   # 8  India      — Highest median CSCC globally (Ricke et al. 2018, ~$86/tCO2);
               #                 Kompas et al. (2018) ~10% GDP loss at 3°C; IPCC AR6 WGII Ch.16
               #                 extreme wet-bulb / monsoon failure. (5.08× baseline)
], dtype=np.float64)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_env(region_params, use_regional: bool, coeff=None) -> RiceMRIO:
    kwargs = dict(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        mrio_data_root=CANONICAL_MRIO_ROOT,
        eu_region_idx=EU_REGION_IDX,
        mrio_trade=False,       # disable MRIO trade — isolates damage function
        cbam_tariff_rate=0.0,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        fixed_savings_rate=False,
        sector_granularity="emissions-simple",
        use_regional_damage_coeff=use_regional,
    )
    if coeff is not None:
        kwargs["regional_damage_coeff"] = coeff
    return RiceMRIO(**kwargs)


def _midpoint_actions(env, key):
    """Midpoint-discrete actions (neutral, no reallocation)."""
    actions = env.sample_action(key)
    mid = env.num_discrete_action_levels // 2
    for agent in actions:
        for k in actions[agent]:
            actions[agent][k] = jnp.full_like(actions[agent][k], mid)
    return actions


def _rollout(env, key):
    """Run ROLLOUT_STEPS with fixed midpoint actions; return damages per step."""
    _, state = env.reset(key)
    actions = _midpoint_actions(env, key)
    proc = env.process_actions(actions, state)

    damage_trace = []
    for _ in range(ROLLOUT_STEPS):
        state = env.step_climate_and_economy(state, proc)
        damage_trace.append(np.array(state["damages_all_regions"]))  # (NR,)
    return np.stack(damage_trace, axis=0)  # (T, NR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    key = jax.random.PRNGKey(SEED)
    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=CANONICAL_YAML_DIR)

    print("Building environments …")
    env_A = _build_env(region_params, use_regional=False)
    env_B = _build_env(region_params, use_regional=True,
                       coeff=np.full(NUM_REGIONS, XA2_BASELINE))
    env_C = _build_env(region_params, use_regional=True,
                       coeff=EXAMPLE_REGIONAL_COEFFS)

    print(f"Rolling out {ROLLOUT_STEPS} steps …\n")
    dmg_A = _rollout(env_A, key)   # (T, NR)
    dmg_B = _rollout(env_B, key)
    dmg_C = _rollout(env_C, key)

    # ── CHECK 1: B == A ──────────────────────────────────────────────────────
    max_diff_BA = float(np.abs(dmg_B - dmg_A).max())
    status_1 = "PASS" if max_diff_BA < 1e-6 else "FAIL"
    print(f"CHECK 1  B (regional uniform) ≡ A (baseline yaml)   max|B-A| = {max_diff_BA:.2e}  [{status_1}]")

    # ── CHECK 2: C != A ──────────────────────────────────────────────────────
    max_diff_CA = float(np.abs(dmg_C - dmg_A).max())
    status_2 = "PASS" if max_diff_CA > 1e-4 else "FAIL"
    print(f"CHECK 2  C (regional adjusted) ≠ A (baseline yaml)  max|C-A| = {max_diff_CA:.2e}  [{status_2}]")

    # ── CHECK 3: directional ordering at final step ───────────────────────────
    final_A = dmg_A[-1]   # (NR,) — survival fractions
    final_C = dmg_C[-1]
    low_coeff_mask  = EXAMPLE_REGIONAL_COEFFS < XA2_BASELINE   # should survive better
    high_coeff_mask = EXAMPLE_REGIONAL_COEFFS > XA2_BASELINE   # should survive worse
    low_ok  = bool(np.all(final_C[low_coeff_mask]  > final_A[low_coeff_mask]))
    high_ok = bool(np.all(final_C[high_coeff_mask] < final_A[high_coeff_mask]))
    status_3 = "PASS" if (low_ok and high_ok) else "FAIL"
    print(f"CHECK 3  Directional ordering (lower coeff → less damage, higher → more)   [{status_3}]")
    if status_3 == "FAIL":
        print(f"  low_coeff regions ok : {low_ok}")
        print(f"  high_coeff regions ok: {high_ok}")

    # ── Per-region damage table (mean over rollout) ───────────────────────────
    mean_A = dmg_A.mean(axis=0)   # (NR,)
    mean_C = dmg_C.mean(axis=0)
    delta  = mean_C - mean_A

    print(f"\n{'Region':<25} {'coeff':>8}  {'mean dmg A':>10}  {'mean dmg C':>10}  {'Δ(C−A)':>10}")
    print("─" * 70)
    for r in range(NUM_REGIONS):
        name = REGION_NAMES.get(r, str(r))
        coeff_r = EXAMPLE_REGIONAL_COEFFS[r]
        marker = "  ←" if abs(coeff_r - XA2_BASELINE) > 1e-6 else ""
        print(f"  {name:<23} {coeff_r:8.5f}  {mean_A[r]:10.6f}  {mean_C[r]:10.6f}  {delta[r]:+10.6f}{marker}")

    print()
    n_pass = sum(s == "PASS" for s in [status_1, status_2, status_3])
    print(f"{'='*70}")
    print(f"  {n_pass}/3 checks PASS")
    if n_pass < 3:
        raise SystemExit(1)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot(dmg_A, dmg_C, status_1, status_2, status_3)


def _plot(dmg_A, dmg_C, status_1, status_2, status_3):
    """Two-panel figure:
      Left  — damage survival fraction trajectories per region (A vs C).
      Right — bar chart of Δ(C−A) at final step, sorted by coefficient.
    """
    T = dmg_A.shape[0]
    steps = np.arange(1, T + 1)

    # Colour each region; EU gets a dashed border
    region_colors = plt.cm.tab10(np.linspace(0, 1, NUM_REGIONS))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Regional damage coefficients — validation rollout\n"
        f"CHECK 1 (B≡A): {status_1}   CHECK 2 (C≠A): {status_2}   "
        f"CHECK 3 (ordering): {status_3}",
        fontsize=11,
    )

    # ── Panel 1: damage trajectories ─────────────────────────────────────────
    ax = axes[0]
    for r in range(NUM_REGIONS):
        name = REGION_NAMES.get(r, str(r))
        c = region_colors[r]
        ls_A = "--" if r == EU_REGION_IDX else "-"
        ax.plot(steps, dmg_A[:, r], color=c, lw=1.2, ls=ls_A, alpha=0.5,
                label=f"{name} (uniform)" if r == 0 else "_")
        ax.plot(steps, dmg_C[:, r], color=c, lw=2.0, ls=ls_A,
                label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Damage survival fraction  (1 = no damage)")
    ax.set_title("Damage trajectories  A (faint) vs C (bold)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Δ(C−A) bar chart at final step ───────────────────────────────
    ax2 = axes[1]
    final_delta = dmg_C[-1] - dmg_A[-1]
    region_names = [REGION_NAMES.get(r, str(r)) for r in range(NUM_REGIONS)]
    bar_colors = ["#e05c2a" if d < 0 else "#5577cc" for d in final_delta]
    ax2.barh(region_names, final_delta, color=bar_colors)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Δ survival fraction  (C − A)  at final step")
    ax2.set_title("Effect of regional calibration\n(red = more damage, blue = less)")
    # Annotate with coefficient values
    for r, (name, delta_r) in enumerate(zip(region_names, final_delta)):
        coeff_r = EXAMPLE_REGIONAL_COEFFS[r]
        offset = 0.0001 if delta_r >= 0 else -0.0001
        ax2.text(delta_r + offset, r, f"  xa_2={coeff_r:.5f}",
                 va="center", ha="left" if delta_r >= 0 else "right", fontsize=7)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"plots/regional_damages_validation_{ts}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {_os.path.abspath(out)}")


if __name__ == "__main__":
    main()
