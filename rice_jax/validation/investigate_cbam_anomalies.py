"""
investigate_cbam_anomalies.py
==============================
Phase 1–3 diagnostic script for three CBAM anomalies:
  A1. SA (South Asia / RoW) responds to CBAM but MENA does not.
  A2. "no-CBAM dirty > CBAM dirty" holds for some regions but not MENA.
  A3. EU exports increase despite imposing CBAM.

Run from rice_jax/:
    python investigate_cbam_anomalies.py

Outputs:
  - Console tables with Phase 1 static audit
  - plots/cbam_anomaly_*.png with Phase 2 trajectory diagnostics
  - Console summary of Phase 3 mechanism isolation tests
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

os.makedirs("plots", exist_ok=True)
MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from rice_jax._rice_mrio import RiceMRIO, _CBAM_SECTORS, _HIGH_EMISSIONS_SECTORS
from rice_jax.utils import load_region_yamls, full_state_info_log_fn
from _experiment_util import FixedActionAgent, run_single_episode
import jaxnasium as jym
from jaxnasium.algorithms import PPO

# ── Configuration ─────────────────────────────────────────────────────────────
# Standard RICE-7 region ordering (from CountryClass_7.csv):
#   idx 0 = Sub-Saharan Africa
#   idx 1 = South Asia
#   idx 2 = North America
#   idx 3 = Middle East & North Africa (MENA)
#   idx 4 = Latin America & Caribbean
#   idx 5 = Europe & Central Asia  ← the actual EU
#   idx 6 = East Asia & Pacific
SETUPS = {
    "7-region": {
        "num_regions": 7,
        "eu_idx": 5,                   # Europe & Central Asia
        "region_labels": [
            "Sub-Saharan Africa",      # 0
            "South Asia",              # 1
            "North America",           # 2
            "MENA",                    # 3
            "Latin America",           # 4
            "Europe & C.Asia",         # 5  ← EU
            "East Asia & Pacific",     # 6
        ],
        "mena_indices": [3],           # MENA
        "sa_index": 1,                 # South Asia
        "yamls_dir": None,             # use standard load_region_yamls
    },
}

# CBAM rates to test
CBAM_RATES = [0.0, 0.15, 0.5, 0.80]

# Training config (reduced for quick diagnostics)
TOTAL_TIMESTEPS = 2_000_000
NUM_ENVS = 4
NUM_STEPS = 100
SEED = 42
N_EVAL = 4
# ──────────────────────────────────────────────────────────────────────────────


def _load_params(setup):
    """Load region params for a given setup."""
    if setup.get("yamls_dir"):
        from main import _load_region_yamls_from_dir
        params, nr = _load_region_yamls_from_dir(setup["yamls_dir"])
        return params
    return load_region_yamls(setup["num_regions"])


def _make_env(setup, cbam_rate, sectoral=False, persistence=1.0,
              welfare_amp=0.4, log_fn=None,
              sector_granularity="full",
              fixed_savings_rate=False, no_mitigation=False,
              baseline_decay=0.0,
              cbam_randomize=False, cbam_tariff_rates=None):
    """Create a RiceMRIO environment."""
    params = _load_params(setup)
    kwargs = dict(
        num_regions=setup["num_regions"],
        region_params=params,
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        cbam_tariff_rate=cbam_rate,
        eu_region_idx=setup["eu_idx"],
        diff_reward_mode=True,
        dest_alloc_persistence=persistence,
        dest_alloc_baseline_decay=baseline_decay,
        sectoral_welfloss=sectoral,
        welfare_loss_per_unit_tariff=welfare_amp,
        sector_granularity=sector_granularity,
        fixed_savings_rate=fixed_savings_rate,
        no_mitigation=no_mitigation,
    )
    if cbam_randomize:
        kwargs["cbam_randomize"] = True
        if cbam_tariff_rates is not None:
            kwargs["cbam_tariff_rates"] = cbam_tariff_rates
    if log_fn is not None:
        kwargs["log_info_fn"] = log_fn
    env = RiceMRIO(**kwargs)
    if env.dest_alloc_baseline is None:
        print(f"[ABORT] MRIO data not found at: {MRIO_ROOT}")
        sys.exit(1)
    return env


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Static Parameter Audit
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_static_audit(setup_name, setup):
    """Extract and tabulate key static arrays, compute theoretical max CBAM cost."""
    print("\n" + "=" * 80)
    print(f"PHASE 1: STATIC PARAMETER AUDIT — {setup_name}")
    print("=" * 80)

    env = _make_env(setup, cbam_rate=0.15)
    NR = env.num_regions
    NS = env.num_sectors
    labels = setup["region_labels"]
    eu = setup["eu_idx"]

    shares = env.sector_output_shares         # (NR, NS)
    tef = env.total_export_frac               # (NR, NS)
    dab = env.dest_alloc_baseline             # (NR, NS, NR)
    intensity = env.emissions_intensity       # (NR, NS)
    sectors = list(env.sector_names)

    # ── 1a. Sector shares & export fractions ──
    print(f"\n{'─'*60}")
    print(f"1a. Sector Output Shares & Export Fractions (top sectors)")
    print(f"    NR={NR}, NS={NS}, sectors: {len(sectors)}")
    print(f"{'─'*60}")

    for r in range(NR):
        top_share = np.argsort(shares[r])[::-1][:5]
        print(f"\n  [{r}] {labels[r]}:")
        print(f"    Top sectors (by output share):")
        for s in top_share:
            print(f"      {sectors[s][:45]:45s}  share={shares[r,s]:.4f}  "
                  f"export_frac={tef[r,s]:.4f}  intensity={intensity[r,s]:.4f}")

    # ── 1b. EU-bound export volume (baseline) ──
    print(f"\n{'─'*60}")
    print(f"1b. Baseline EU-Bound Export Volume (dest_alloc × export_frac × share)")
    print(f"{'─'*60}")

    # eu_bound_vol[r, s] = share[r,s] * tef[r,s] * dab[r, s, eu]
    # (per unit of total production Y_r)
    eu_vol = shares * tef * dab[:, :, eu]  # (NR, NS) — as fraction of Y_r
    total_eu_vol = eu_vol.sum(axis=1)       # (NR,) — total EU-bound as frac of Y_r

    print(f"\n  {'Region':25s} {'Total EU-bound':>15s} {'CBAM sector EU':>15s} {'Dirty EU':>10s}")
    cbam_idx = [i for i, s in enumerate(sectors) if s in _CBAM_SECTORS]
    dirty_set = _CBAM_SECTORS | _HIGH_EMISSIONS_SECTORS
    dirty_idx = [i for i, s in enumerate(sectors) if s in dirty_set]

    for r in range(NR):
        cbam_eu = eu_vol[r, cbam_idx].sum() if cbam_idx else 0.0
        dirty_eu = eu_vol[r, dirty_idx].sum() if dirty_idx else 0.0
        marker = " ◄ EU" if r == eu else ""
        print(f"  [{r}] {labels[r]:20s} {total_eu_vol[r]:15.6f} "
              f"{cbam_eu:15.6f} {dirty_eu:10.6f}{marker}")

    # ── 1c. Theoretical max CBAM cost (as % of Y_r) ──
    print(f"\n{'─'*60}")
    print(f"1c. Theoretical Max CBAM Cost (% of Y_r) at τ=0.15, α=0.4")
    print(f"    Formula: Σ_s share[r,s] * tef[r,s] * dab[r,s,EU] * σ[r,s] * τ * α")
    print(f"{'─'*60}")

    tau = 0.15
    alpha = 0.4
    # welfloss_reduction[r] = Σ_s eu_vol[r,s] * intensity[r,s] * tau * alpha
    wl_reduction = (eu_vol * intensity * tau * alpha).sum(axis=1)  # (NR,)

    print(f"\n  {'Region':25s} {'Welfloss reduction':>18s} {'Welfloss':>10s} {'Effective rate':>15s}")
    for r in range(NR):
        wl = max(1.0 - wl_reduction[r], 0.0)
        # effective_rate = cbam_cost / total_eu_imports
        cbam_cost_r = (eu_vol[r] * intensity[r] * tau).sum()
        eff_rate = cbam_cost_r / (total_eu_vol[r] + 1e-10)
        eff_rate_clipped = min(eff_rate, 1.0)
        clip_flag = " ⚠ CLIPPED" if eff_rate > 1.0 else ""
        print(f"  [{r}] {labels[r]:20s} {wl_reduction[r]:18.8f} {wl:10.6f} "
              f"{eff_rate_clipped:15.6f}{clip_flag}")

    # ── 1d. Gradient analysis ──
    print(f"\n{'─'*60}")
    print(f"1d. Welfloss Gradient ∂welfloss/∂(EU share) per region")
    print(f"    Approach A (sectoral): -share[r,s]*tef[r,s]*σ[r,s]*τ*α")
    print(f"    Approach B (aggregate): -eu_vol_ratio * eff_rate * α")
    print(f"{'─'*60}")

    # Sectoral gradient magnitude per region (sum of absolute per-sector gradients)
    grad_sectoral = np.abs(shares * tef * intensity * tau * alpha)  # (NR, NS)
    grad_sectoral_total = grad_sectoral.sum(axis=1)

    # Aggregate gradient
    grad_aggregate = np.abs(total_eu_vol * wl_reduction)

    print(f"\n  {'Region':25s} {'|∂wl/∂alloc| sectoral':>22s} {'|∂wl/∂alloc| aggregate':>22s} {'Ratio':>8s}")
    for r in range(NR):
        ratio = grad_sectoral_total[r] / (grad_aggregate[r] + 1e-10)
        print(f"  [{r}] {labels[r]:20s} {grad_sectoral_total[r]:22.8f} "
              f"{grad_aggregate[r]:22.8f} {ratio:8.2f}")

    # ── 1e. Effective rate clipping check for high-intensity regions ──
    print(f"\n{'─'*60}")
    print(f"1e. Effective Rate Clipping Check")
    print(f"    Does cbam_cost/eu_gross exceed 1.0 at various τ values?")
    print(f"{'─'*60}")

    for test_tau in [0.05, 0.10, 0.15, 0.30, 0.50]:
        print(f"\n  τ = {test_tau}:")
        for r in range(NR):
            if r == eu:
                continue
            cbam_cost_r = (eu_vol[r] * intensity[r] * test_tau).sum()
            eff = cbam_cost_r / (total_eu_vol[r] + 1e-10)
            clip = " ⚠ WOULD CLIP" if eff > 1.0 else ""
            print(f"    [{r}] {labels[r]:20s}  eff_rate={eff:.6f}{clip}")

    # ── 1f. Destination allocation baseline — EU share per region ──
    print(f"\n{'─'*60}")
    print(f"1f. Baseline Destination Shares to EU (mean across sectors)")
    print(f"{'─'*60}")

    eu_share_mean = dab[:, :, eu].mean(axis=1)
    eu_share_weighted = (shares * tef * dab[:, :, eu]).sum(axis=1) / np.maximum(
        (shares * tef).sum(axis=1), 1e-10
    )
    print(f"\n  {'Region':25s} {'Mean EU share':>13s} {'Weighted EU share':>18s}")
    for r in range(NR):
        marker = " ◄ EU" if r == eu else ""
        print(f"  [{r}] {labels[r]:20s} {eu_share_mean[r]:13.6f} {eu_share_weighted[r]:18.6f}{marker}")

    # ── 1g. Cross-region intensity comparison ──
    print(f"\n{'─'*60}")
    print(f"1g. Emissions Intensity by Sector Group")
    print(f"{'─'*60}")

    print(f"\n  {'Region':25s}", end="")
    if cbam_idx:
        print(f" {'CBAM mean σ':>12s}", end="")
    if dirty_idx:
        print(f" {'Dirty mean σ':>13s}", end="")
    print(f" {'Clean mean σ':>13s} {'Overall mean σ':>15s}")

    clean_idx = [i for i in range(NS) if i not in dirty_idx]
    for r in range(NR):
        print(f"  [{r}] {labels[r]:20s}", end="")
        if cbam_idx:
            print(f" {intensity[r, cbam_idx].mean():12.6f}", end="")
        if dirty_idx:
            print(f" {intensity[r, dirty_idx].mean():13.6f}", end="")
        if clean_idx:
            print(f" {intensity[r, clean_idx].mean():13.6f}", end="")
        print(f" {intensity[r].mean():15.6f}")

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1B: Signal Strength Comparison — Exaggerated vs Realistic
# ═══════════════════════════════════════════════════════════════════════════════

def phase1b_signal_comparison(setup_name, setup):
    """Compare welfloss signal under different mechanism amplification configs.

    Level 1 (exaggerated): Does the mechanism work at all?
        emissions-simple, sectoral_welfloss, fixed_savings, no_mitigation,
        α=5.0, τ=0.80, ρ=0.55, decay=1.0
    Level 2 (realistic):   Can it work with calibrated parameters?
        full 26 sectors, α=0.4, τ=0.15, ρ=0.0
    """
    print("\n" + "=" * 80)
    print(f"PHASE 1B: SIGNAL STRENGTH COMPARISON — {setup_name}")
    print("  Level 1 = exaggerated (mechanism validation)")
    print("  Level 2 = realistic (calibrated conditions)")
    print("=" * 80)

    NR = setup["num_regions"]
    eu = setup["eu_idx"]
    labels = setup["region_labels"]
    mena = setup["mena_indices"]
    sa = setup["sa_index"]

    # Define configurations to compare
    configs = {
        "L1: exaggerated (validate_emissions_simple settings)": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=5.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "L1b: exaggerated, α=15": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=15.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "L1c: exaggerated, full sectors": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=5.0, sector_granularity="full",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "L2: realistic (Nordhaus calibrated)": dict(
            cbam_rate=0.15, persistence=0.0, sectoral=False,
            welfare_amp=0.4, sector_granularity="full",
            fixed_savings_rate=False, no_mitigation=False, baseline_decay=0.0,
        ),
        "L2b: realistic + sectoral": dict(
            cbam_rate=0.15, persistence=0.55, sectoral=True,
            welfare_amp=0.4, sector_granularity="emissions-simple",
            fixed_savings_rate=False, no_mitigation=False, baseline_decay=1.0,
        ),
    }

    print(f"\n  {'Config':55s} ", end="")
    for r in range(NR):
        print(f" {labels[r][:10]:>10s}", end="")
    print()
    print("  " + "─" * (55 + 11 * NR))

    for config_name, cfg in configs.items():
        env = _make_env(setup, **cfg)
        info = _run_episode_extract(env, seed_int=SEED)
        ut_raw = info["utility_all_regions"]
        utility = _dict_to_array(ut_raw) if isinstance(ut_raw, dict) else np.array(ut_raw)
        utw_raw = info["utility_times_welfloss_all_regions"]
        utw = _dict_to_array(utw_raw) if isinstance(utw_raw, dict) else np.array(utw_raw)
        welfloss = utw / (utility + 1e-10)
        mean_wl = welfloss[3:].mean(axis=0)  # skip first few warmup steps

        print(f"  {config_name:55s} ", end="")
        for r in range(NR):
            wl_val = mean_wl[r]
            # Show as 1-welfloss (the penalty) for readability
            penalty = 1.0 - wl_val
            if penalty < 1e-8:
                print(f" {'~0':>10s}", end="")
            else:
                print(f" {penalty:10.6f}", end="")
        print()

    # Also show the static intensity arrays under emissions-simple
    print(f"\n{'─'*60}")
    print(f"Emissions intensity under emissions-simple granularity:")
    print(f"{'─'*60}")
    env_es = _make_env(setup, cbam_rate=0.5, sector_granularity="emissions-simple",
                        sectoral=True)
    intensity_es = env_es.emissions_intensity  # (NR, 2) [dirty, clean]
    sectors_es = list(env_es.sector_names)
    tef_es = env_es.total_export_frac
    shares_es = env_es.sector_output_shares
    dab_es = env_es.dest_alloc_baseline

    print(f"  Sectors: {sectors_es}")
    print(f"\n  {'Region':25s} {'σ_dirty':>10s} {'σ_clean':>10s} {'ratio':>8s} "
          f"{'tef_dirty':>10s} {'tef_clean':>10s} {'EU_share_d':>11s} {'EU_share_c':>11s}")
    for r in range(NR):
        ratio = intensity_es[r, 0] / max(intensity_es[r, 1], 1e-10)
        eu_share_d = dab_es[r, 0, eu] if dab_es is not None else 0
        eu_share_c = dab_es[r, 1, eu] if dab_es is not None else 0
        marker = ""
        if r in mena:
            marker = " ◄ MENA"
        elif r == sa:
            marker = " ◄ SA"
        elif r == eu:
            marker = " ◄ EU"
        print(f"  [{r}] {labels[r]:20s} {intensity_es[r,0]:10.6f} {intensity_es[r,1]:10.6f} "
              f"{ratio:8.1f}× {tef_es[r,0]:10.6f} {tef_es[r,1]:10.6f} "
              f"{eu_share_d:11.6f} {eu_share_c:11.6f}{marker}")

    # Theoretical welfloss penalty under each config
    print(f"\n{'─'*60}")
    print(f"Theoretical welfloss penalty = Σ_s share*tef*dab[EU]*σ*τ*α  (per unit Y)")
    print(f"{'─'*60}")
    for config_name, cfg in configs.items():
        tau = cfg["cbam_rate"]
        alpha = cfg["welfare_amp"]
        env_t = _make_env(setup, **cfg)
        int_t = env_t.emissions_intensity
        tef_t = env_t.total_export_frac
        sh_t = env_t.sector_output_shares
        dab_t = env_t.dest_alloc_baseline
        NS_t = env_t.num_sectors

        eu_vol = sh_t * tef_t * dab_t[:, :, eu]  # (NR, NS)
        penalty_per_r = (eu_vol * int_t * tau * alpha).sum(axis=1)

        print(f"\n  {config_name}:")
        for r in range(NR):
            marker = ""
            if r in mena:
                marker = " ◄ MENA"
            elif r == sa:
                marker = " ◄ SA"
            elif r == eu:
                marker = " ◄ EU"
            print(f"    [{r}] {labels[r]:20s}  penalty = {penalty_per_r[r]:.8f} "
                  f" → welfloss = {max(1-penalty_per_r[r], 0):.8f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Runtime Trajectory Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def _dict_to_array(d):
    """Convert {agent_id: (T,)} dict to (T, NR) array."""
    keys = sorted(d.keys())
    return np.stack([np.array(d[k]) for k in keys], axis=-1)


def _run_episode_extract(env, seed_int=0):
    """Run one episode with FixedActionAgent, return raw info stack."""
    key = jax.random.PRNGKey(seed_int)
    log_env = replace(env, log_info_fn=full_state_info_log_fn)
    agent = FixedActionAgent(log_env)
    info = run_single_episode(key, log_env, agent)
    return info


def phase2_trajectory_diagnostics(setup_name, setup):
    """Run CBAM-on vs CBAM-off episodes and compare trajectories."""
    print("\n" + "=" * 80)
    print(f"PHASE 2: TRAJECTORY DIAGNOSTICS — {setup_name}")
    print("=" * 80)

    NR = setup["num_regions"]
    eu = setup["eu_idx"]
    labels = setup["region_labels"]
    mena = setup["mena_indices"]
    sa = setup["sa_index"]

    results = {}
    for rate in [0.0, 0.15, 0.5]:
        print(f"\n  Running episode with CBAM rate = {rate}...")
        env = _make_env(setup, cbam_rate=rate, persistence=1.0)
        info = _run_episode_extract(env, seed_int=SEED)
        results[rate] = info

    # Extract key time series
    T = None
    traces = {}
    for rate, info in results.items():
        # Handle dict-per-agent vs stacked-array formats
        go_raw = info["gross_output_all_regions"]
        if isinstance(go_raw, dict):
            gross_output = _dict_to_array(go_raw)  # (T, NR)
        else:
            gross_output = np.array(go_raw)

        ut_raw = info["utility_all_regions"]
        if isinstance(ut_raw, dict):
            utility = _dict_to_array(ut_raw)
        else:
            utility = np.array(ut_raw)

        T = gross_output.shape[0]

        tf = np.array(info["trade_flows"])  # (T, NR, NR, NS)
        # EU-bound by region: sum over sectors
        eu_bound = tf[:, :, eu, :].sum(axis=-1)  # (T, NR)
        # EU outbound (EU exports) by destination: sum over sectors
        eu_exports = tf[:, eu, :, :].sum(axis=-1)  # (T, NR)
        total_eu_exports = eu_exports.sum(axis=-1)  # (T,)

        # Welfloss
        utw_raw = info.get("utility_times_welfloss_all_regions")
        if utw_raw is not None:
            if isinstance(utw_raw, dict):
                utw = _dict_to_array(utw_raw)
            else:
                utw = np.array(utw_raw)
            welfloss = utw / (utility + 1e-10)
        else:
            welfloss = np.ones_like(utility)

        # CBAM revenue
        cbam_rev = np.array(info.get("cbam_revenue",
                                      np.zeros((T, NR))))

        traces[rate] = {
            "gross_output": gross_output,
            "utility": utility,
            "welfloss": welfloss,
            "eu_bound": eu_bound,
            "eu_exports": eu_exports,
            "total_eu_exports": total_eu_exports,
            "cbam_revenue": cbam_rev,
        }

    # ── Plot ──
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(4, NR, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle(f"CBAM Trajectory Diagnostics — {setup_name}\n"
                 f"FixedAction baseline, persistence=1.0", fontsize=14)

    timesteps = np.arange(T)
    colors = {0.0: "blue", 0.15: "orange", 0.5: "red"}

    # Row 0: Welfloss per region
    for r in range(NR):
        ax = fig.add_subplot(gs[0, r])
        for rate, tr in traces.items():
            ax.plot(timesteps, tr["welfloss"][:, r],
                    color=colors[rate], label=f"τ={rate}", alpha=0.8)
        ax.set_title(f"Welfloss: {labels[r]}", fontsize=9)
        ax.set_ylim(0.8, 1.02)
        if r == 0:
            ax.set_ylabel("welfloss")
            ax.legend(fontsize=7)

    # Row 1: EU-bound exports per region (as share of output)
    for r in range(NR):
        ax = fig.add_subplot(gs[1, r])
        for rate, tr in traces.items():
            eu_share = tr["eu_bound"][:, r] / (tr["gross_output"][:, r] + 1e-10)
            ax.plot(timesteps, eu_share, color=colors[rate],
                    label=f"τ={rate}", alpha=0.8)
        ax.set_title(f"EU-bound share: {labels[r]}", fontsize=9)
        if r == 0:
            ax.set_ylabel("EU exports / Y")
            ax.legend(fontsize=7)

    # Row 2: EU total exports (the anomaly check)
    ax = fig.add_subplot(gs[2, :3])
    for rate, tr in traces.items():
        ax.plot(timesteps, tr["total_eu_exports"],
                color=colors[rate], label=f"τ={rate}", linewidth=2)
    ax.set_title("EU Total Exports (Anomaly A3: do they increase under CBAM?)", fontsize=11)
    ax.set_ylabel("EU total export volume")
    ax.legend()

    # EU exports per destination
    ax2 = fig.add_subplot(gs[2, 3:])
    for r in range(NR):
        if r == eu:
            continue
        for rate in [0.0, 0.5]:
            tr = traces[rate]
            style = "--" if rate == 0.0 else "-"
            ax2.plot(timesteps, tr["eu_exports"][:, r],
                     linestyle=style, label=f"EU→{labels[r]} τ={rate}", alpha=0.7)
    ax2.set_title("EU Exports by Destination", fontsize=11)
    ax2.legend(fontsize=6, ncol=2)

    # Row 3: CBAM revenue
    ax = fig.add_subplot(gs[3, :3])
    for rate, tr in traces.items():
        if rate == 0.0:
            continue
        ax.plot(timesteps, tr["cbam_revenue"][:, eu],
                color=colors[rate], label=f"EU CBAM rev τ={rate}", linewidth=2)
    ax.set_title("CBAM Revenue (EU)", fontsize=11)
    ax.set_ylabel("Revenue")
    ax.legend()

    # MENA-specific welfloss comparison
    ax = fig.add_subplot(gs[3, 3:])
    for m_idx in mena:
        for rate in [0.0, 0.15, 0.5]:
            tr = traces[rate]
            style = "--" if rate == 0.0 else "-"
            lw = 2 if rate == 0.5 else 1
            ax.plot(timesteps, tr["welfloss"][:, m_idx],
                    color=colors[rate], linestyle=style, linewidth=lw,
                    label=f"{labels[m_idx]} τ={rate}", alpha=0.8)
    # Also SA
    for rate in [0.0, 0.5]:
        tr = traces[rate]
        style = "--" if rate == 0.0 else "-"
        ax.plot(timesteps, tr["welfloss"][:, sa],
                color="green" if rate == 0.0 else "darkgreen",
                linestyle=style, linewidth=2,
                label=f"{labels[sa]} τ={rate}")
    ax.set_title("MENA vs SA Welfloss (Anomaly A1)", fontsize=11)
    ax.legend(fontsize=6)

    outpath = f"plots/cbam_anomaly_trajectories_{setup_name}_{TIMESTAMP}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved trajectory plot: {outpath}")
    plt.close(fig)

    # ── Console summary ──
    print(f"\n{'─'*60}")
    print(f"Phase 2 Summary — Mean welfloss (t>5), CBAM=0.0 vs 0.5:")
    print(f"{'─'*60}")
    print(f"  {'Region':25s} {'τ=0.0':>10s} {'τ=0.5':>10s} {'Δ':>10s} {'% drop':>10s}")
    for r in range(NR):
        wl0 = traces[0.0]["welfloss"][5:, r].mean()
        wl5 = traces[0.5]["welfloss"][5:, r].mean()
        delta = wl5 - wl0
        pct = delta / (wl0 + 1e-10) * 100
        marker = ""
        if r in mena:
            marker = " ◄ MENA"
        elif r == sa:
            marker = " ◄ SA"
        elif r == eu:
            marker = " ◄ EU"
        print(f"  [{r}] {labels[r]:20s} {wl0:10.6f} {wl5:10.6f} {delta:10.6f} {pct:9.3f}%{marker}")

    print(f"\n  EU total exports (mean t>5):")
    for rate in [0.0, 0.15, 0.5]:
        val = traces[rate]["total_eu_exports"][5:].mean()
        print(f"    τ={rate}: {val:.4f}")

    return traces


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Mechanism Isolation Tests
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_mechanism_isolation(setup_name, setup):
    """Test specific hypotheses via controlled variations."""
    print("\n" + "=" * 80)
    print(f"PHASE 3: MECHANISM ISOLATION — {setup_name}")
    print("=" * 80)

    NR = setup["num_regions"]
    eu = setup["eu_idx"]
    labels = setup["region_labels"]
    mena = setup["mena_indices"]
    sa = setup["sa_index"]

    # ── H1a: Amplify MENA signal ──
    print(f"\n{'─'*60}")
    print(f"H1a. Amplified welfare_loss_per_unit_tariff (0.4 → 15.0)")
    print(f"     Using emissions-simple + sectoral_welfloss (validate_emissions_simple config)")
    print(f"{'─'*60}")

    for amp in [0.4, 5.0, 15.0]:
        env = _make_env(setup, cbam_rate=0.80, persistence=0.55, welfare_amp=amp,
                        sectoral=True, sector_granularity="emissions-simple",
                        fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0)
        info = _run_episode_extract(env, seed_int=SEED)
        ut_raw = info["utility_all_regions"]
        utility = _dict_to_array(ut_raw) if isinstance(ut_raw, dict) else np.array(ut_raw)
        utw_raw = info["utility_times_welfloss_all_regions"]
        utw = _dict_to_array(utw_raw) if isinstance(utw_raw, dict) else np.array(utw_raw)
        welfloss = utw / (utility + 1e-10)
        mean_wl_mena = np.mean([welfloss[5:, m].mean() for m in mena])
        mean_wl_sa = welfloss[5:, sa].mean()
        mean_wl_eu = welfloss[5:, eu].mean()
        print(f"  α={amp:4.1f}: MENA welfloss={mean_wl_mena:.6f}, "
              f"SA welfloss={mean_wl_sa:.6f}, EU welfloss={mean_wl_eu:.6f}")

    # ── H1c: Effective rate clipping ──
    print(f"\n{'─'*60}")
    print(f"H1c. Effective Rate in Practice (τ=0.5)")
    print(f"{'─'*60}")

    env = _make_env(setup, cbam_rate=0.5, persistence=1.0)
    info = _run_episode_extract(env, seed_int=SEED)
    if "import_tariffs" in info:
        tariffs = np.array(info["import_tariffs"])  # (T, NR, NR)
        eff_rates = tariffs[:, eu, :]  # (T, NR) — effective rate EU charges each region
        print(f"\n  {'Region':25s} {'mean eff_rate':>13s} {'max eff_rate':>12s} {'clips?':>8s}")
        for r in range(NR):
            if r == eu:
                continue
            m = eff_rates[5:, r].mean()
            mx = eff_rates[5:, r].max()
            clip = "YES" if mx >= 0.999 else "no"
            print(f"  [{r}] {labels[r]:20s} {m:13.6f} {mx:12.6f} {clip:>8s}")

    # ── H2b: CES substitution sensitivity ──
    print(f"\n{'─'*60}")
    print(f"H2b. CES Substitution Rate Sensitivity")
    print(f"     Does EU export advantage disappear with Leontief-like CES?")
    print(f"{'─'*60}")

    # We can't easily change consumption_substitution_rate on RiceMRIO without
    # it being an EnvSettings param. We'll document this as a manual test.
    print(f"  [NOTE] consumption_substitution_rate is an EnvSettings param.")
    print(f"  Manual test: set consumption_substitution_rate=0.01 in EnvSettings")
    print(f"  and re-run Phase 2 trajectory diagnostics.")

    # ── Sectoral vs Aggregate welfloss comparison ──
    print(f"\n{'─'*60}")
    print(f"Sectoral vs Aggregate Welfloss Comparison (τ=0.5)")
    print(f"{'─'*60}")

    for sectoral in [False, True]:
        env = _make_env(setup, cbam_rate=0.5, persistence=1.0, sectoral=sectoral)
        info = _run_episode_extract(env, seed_int=SEED)
        ut_raw = info["utility_all_regions"]
        utility = _dict_to_array(ut_raw) if isinstance(ut_raw, dict) else np.array(ut_raw)
        utw_raw = info["utility_times_welfloss_all_regions"]
        utw = _dict_to_array(utw_raw) if isinstance(utw_raw, dict) else np.array(utw_raw)
        welfloss = utw / (utility + 1e-10)
        mode = "Sectoral" if sectoral else "Aggregate"
        print(f"\n  {mode} welfloss (mean t>5):")
        for r in range(NR):
            marker = ""
            if r in mena:
                marker = " ◄ MENA"
            elif r == sa:
                marker = " ◄ SA"
            elif r == eu:
                marker = " ◄ EU"
            print(f"    [{r}] {labels[r]:20s} {welfloss[5:, r].mean():.6f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2B: Training comparison (CBAM-on vs CBAM-off)
# ═══════════════════════════════════════════════════════════════════════════════

def phase2b_training_comparison(setup_name, setup):
    """Train PPO under CBAM-on vs CBAM-off and compare learned behaviors."""
    print("\n" + "=" * 80)
    print(f"PHASE 2B: TRAINING COMPARISON — {setup_name}")
    print(f"  Training {TOTAL_TIMESTEPS:,} timesteps per condition")
    print("=" * 80)

    NR = setup["num_regions"]
    eu = setup["eu_idx"]
    labels = setup["region_labels"]
    mena = setup["mena_indices"]
    sa = setup["sa_index"]

    trained_traces = {}
    for rate in [0.0, 0.5]:
        print(f"\n  Training with CBAM rate = {rate}...")
        train_env = jym.LogWrapper(
            _make_env(setup, cbam_rate=rate, persistence=1.0)
        )
        eval_env = _make_env(setup, cbam_rate=rate, persistence=1.0,
                             log_fn=full_state_info_log_fn)

        ppo = PPO(
            total_timesteps=TOTAL_TIMESTEPS,
            num_envs=NUM_ENVS,
            num_steps=NUM_STEPS,
            learning_rate=2.5e-4,
            ent_coef=2.0,
            anneal_ent_coef=0.05,
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            clip_coef=0.2,
            clip_coef_vf=0.5,
            vf_coef=0.5,
            num_minibatches=4,
            num_epochs=4,
        )
        trained = ppo.train(jax.random.PRNGKey(SEED), train_env)

        # Eval
        all_info = []
        for ep in range(N_EVAL):
            info = run_single_episode(
                jax.random.PRNGKey(SEED + ep + 1), eval_env, trained
            )
            all_info.append(info)

        # Average over eval episodes
        def _extract_array(info_list, key):
            arrs = []
            for i in info_list:
                raw = i[key]
                if isinstance(raw, dict):
                    arrs.append(_dict_to_array(raw))
                else:
                    arrs.append(np.array(raw))
            return np.stack(arrs).mean(0)

        utility = _extract_array(all_info, "utility_all_regions")
        utw = _extract_array(all_info, "utility_times_welfloss_all_regions")
        welfloss = utw / (utility + 1e-10)

        eu_bound_list = []
        eu_exports_list = []
        go_list = []
        for i in all_info:
            if "trade_flows" in i:
                tf = np.array(i["trade_flows"])
                eu_bound_list.append(tf[:, :, eu, :].sum(axis=-1))
                eu_exports_list.append(tf[:, eu, :, :].sum(axis=-1))
            go_raw = i["gross_output_all_regions"]
            if isinstance(go_raw, dict):
                go_list.append(_dict_to_array(go_raw))
            else:
                go_list.append(np.array(go_raw))

        gross_output = np.stack(go_list).mean(0)
        if eu_bound_list:
            eu_bound = np.stack(eu_bound_list).mean(0)
            eu_exports = np.stack(eu_exports_list).mean(0)
        else:
            eu_bound = np.zeros_like(gross_output)
            eu_exports = np.zeros_like(gross_output)

        trained_traces[rate] = {
            "utility": utility,
            "welfloss": welfloss,
            "eu_bound": eu_bound,
            "eu_exports": eu_exports,
            "gross_output": gross_output,
        }

    # ── Report ──
    print(f"\n{'─'*60}")
    print(f"Phase 2B: Trained Policy Mean Welfloss (t>5)")
    print(f"{'─'*60}")
    print(f"  {'Region':25s} {'τ=0.0':>10s} {'τ=0.5':>10s} {'Δ':>10s}")
    for r in range(NR):
        wl0 = trained_traces[0.0]["welfloss"][5:, r].mean()
        wl5 = trained_traces[0.5]["welfloss"][5:, r].mean()
        marker = ""
        if r in mena:
            marker = " ◄ MENA"
        elif r == sa:
            marker = " ◄ SA"
        elif r == eu:
            marker = " ◄ EU"
        print(f"  [{r}] {labels[r]:20s} {wl0:10.6f} {wl5:10.6f} "
              f"{wl5 - wl0:10.6f}{marker}")

    print(f"\n  Trained EU export share of Y (mean t>5):")
    for rate in [0.0, 0.5]:
        t = trained_traces[rate]
        eu_share = t["eu_exports"][5:].sum(axis=-1).mean() / (t["gross_output"][5:, eu].mean() + 1e-10)
        print(f"    τ={rate}: {eu_share:.6f}")

    # ── Plot ──
    T = trained_traces[0.0]["welfloss"].shape[0]
    fig, axes = plt.subplots(2, NR, figsize=(4 * NR, 8))
    fig.suptitle(f"Trained Policy Trajectories — {setup_name}\n"
                 f"{TOTAL_TIMESTEPS:,} timesteps PPO", fontsize=13)
    timesteps = np.arange(T)
    for r in range(NR):
        for rate, color in [(0.0, "blue"), (0.5, "red")]:
            axes[0, r].plot(timesteps, trained_traces[rate]["welfloss"][:, r],
                            color=color, label=f"τ={rate}")
            eu_share = trained_traces[rate]["eu_bound"][:, r] / (
                trained_traces[rate]["gross_output"][:, r] + 1e-10
            )
            axes[1, r].plot(timesteps, eu_share, color=color, label=f"τ={rate}")
        axes[0, r].set_title(f"{labels[r]}", fontsize=9)
        axes[0, r].set_ylim(0.7, 1.02)
        axes[1, r].set_title(f"EU share: {labels[r]}", fontsize=9)
        if r == 0:
            axes[0, r].set_ylabel("welfloss")
            axes[1, r].set_ylabel("EU exports / Y")
            axes[0, r].legend(fontsize=7)
            axes[1, r].legend(fontsize=7)

    outpath = f"plots/cbam_anomaly_trained_{setup_name}_{TIMESTAMP}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved trained policy plot: {outpath}")
    plt.close(fig)

    return trained_traces


# ═══════════════════════════════════════════════════════════════════════════════
# LEVEL 1: Deep Mechanism Exploration
# ═══════════════════════════════════════════════════════════════════════════════

def level1_deep_exploration(setup_name, setup):
    """Systematic Level 1 exploration: does the CBAM mechanism work AT ALL?

    Tests a grid of configurations with PPO training to find any combination
    that produces detectable EU export diversion for SA/MENA.

    Key metric: Does trained CBAM-on policy produce lower EU export share
    for SA/MENA than CBAM-off policy?
    """
    print("\n" + "█" * 80)
    print(f"LEVEL 1: DEEP MECHANISM EXPLORATION — {setup_name}")
    print(f"  Does the CBAM mechanism produce learned diversion under ANY config?")
    print("█" * 80)

    NR = setup["num_regions"]
    eu = setup["eu_idx"]
    labels = setup["region_labels"]
    mena = setup["mena_indices"]
    sa = setup["sa_index"]

    # ── Step 0: Quick sanity check with corrected eu_idx ──
    print(f"\n{'─'*60}")
    print(f"Step 0: Sanity check — EU = [{eu}] {labels[eu]}")
    print(f"  SA = [{sa}] {labels[sa]}, MENA = {mena} {[labels[m] for m in mena]}")
    print(f"{'─'*60}")

    env_check = _make_env(setup, cbam_rate=0.5, sector_granularity="emissions-simple",
                          sectoral=True, welfare_amp=5.0, fixed_savings_rate=True,
                          no_mitigation=True, persistence=0.55, baseline_decay=1.0)
    print(f"  mrio_region_labels: {env_check.mrio_region_labels}")
    print(f"  eu_region_idx:      {env_check.eu_region_idx}")
    print(f"  num_sectors:        {env_check.num_sectors}")

    dab = env_check.dest_alloc_baseline
    tef = env_check.total_export_frac
    shares = env_check.sector_output_shares
    intensity = env_check.emissions_intensity
    # eu_vol computed per sector: (NR, NS)
    eu_vol_sector = shares * tef * dab[:, :, eu]
    eu_vol = eu_vol_sector.sum(axis=1)  # (NR,)
    print(f"\n  Export volumes to EU [{labels[eu]}] (frac of Y):")
    for r in range(NR):
        marker = ""
        if r in mena: marker = " ◄ MENA"
        elif r == sa: marker = " ◄ SA"
        elif r == eu: marker = " ◄ EU"
        print(f"    [{r}] {labels[r]:20s}: {eu_vol[r]:.6f}{marker}")

    # Theoretical welfloss at various α
    print(f"\n  Theoretical welfloss penalty at τ=0.80:")
    for alpha in [0.4, 5.0, 15.0, 50.0, 100.0]:
        pen = (eu_vol_sector * intensity * 0.80 * alpha).sum(axis=1)  # (NR,)
        pen_sa = pen[sa]
        pen_mena = np.mean([pen[m] for m in mena])
        print(f"    α={alpha:6.1f}: SA penalty={pen_sa:.6f} ({pen_sa*100:.3f}%),"
              f" MENA penalty={pen_mena:.6f} ({pen_mena*100:.3f}%)")

    # ── Step 1: FixedAction baseline comparison (CBAM-off vs CBAM-on) ──
    print(f"\n{'═'*60}")
    print(f"Step 1: FixedAction baseline — Is there ANY welfloss difference?")
    print(f"{'═'*60}")

    baseline_configs = {
        "emissions-simple α=5 τ=0.80": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=5.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "emissions-simple α=50 τ=0.80": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=50.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "emissions-simple α=100 τ=0.80": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=100.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
        "full-sectors α=5 τ=0.80": dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=5.0, sector_granularity="full",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ),
    }

    # Also run CBAM-off variants
    for cfg_name, cfg in list(baseline_configs.items()):
        print(f"\n  --- {cfg_name} ---")
        env_on = _make_env(setup, **cfg)
        info_on = _run_episode_extract(env_on, seed_int=SEED)

        cfg_off = dict(cfg)
        cfg_off["cbam_rate"] = 0.0
        env_off = _make_env(setup, **cfg_off)
        info_off = _run_episode_extract(env_off, seed_int=SEED)

        # Compare welfloss
        ut_on_raw = info_on["utility_all_regions"]
        ut_on = _dict_to_array(ut_on_raw) if isinstance(ut_on_raw, dict) else np.array(ut_on_raw)
        utw_on_raw = info_on["utility_times_welfloss_all_regions"]
        utw_on = _dict_to_array(utw_on_raw) if isinstance(utw_on_raw, dict) else np.array(utw_on_raw)
        wl_on = utw_on / (ut_on + 1e-10)

        ut_off_raw = info_off["utility_all_regions"]
        ut_off = _dict_to_array(ut_off_raw) if isinstance(ut_off_raw, dict) else np.array(ut_off_raw)
        utw_off_raw = info_off["utility_times_welfloss_all_regions"]
        utw_off = _dict_to_array(utw_off_raw) if isinstance(utw_off_raw, dict) else np.array(utw_off_raw)
        wl_off = utw_off / (ut_off + 1e-10)

        # Compare EU-bound trade
        tf_on = np.array(info_on["trade_flows"])   # (T, NR, NR, NS)
        tf_off = np.array(info_off["trade_flows"])
        go_on = _dict_to_array(info_on["gross_output_all_regions"]) if isinstance(info_on["gross_output_all_regions"], dict) else np.array(info_on["gross_output_all_regions"])
        go_off = _dict_to_array(info_off["gross_output_all_regions"]) if isinstance(info_off["gross_output_all_regions"], dict) else np.array(info_off["gross_output_all_regions"])
        eu_share_on = tf_on[:, :, eu, :].sum(axis=-1) / (go_on + 1e-10)  # (T, NR)
        eu_share_off = tf_off[:, :, eu, :].sum(axis=-1) / (go_off + 1e-10)

        # Rewards (utility * welfloss)
        reward_on = (utw_on[3:]).mean(axis=0)
        reward_off = (utw_off[3:]).mean(axis=0)

        print(f"  {'Region':20s} {'welfloss_on':>12s} {'welfloss_off':>13s} {'Δ(penalty)':>11s}"
              f" {'EU_share_on':>12s} {'EU_share_off':>13s} {'reward_on':>10s} {'reward_off':>11s}")
        for r in range(NR):
            wl_on_m = wl_on[3:, r].mean()
            wl_off_m = wl_off[3:, r].mean()
            pen_delta = (1 - wl_on_m) - (1 - wl_off_m)
            eu_s_on = eu_share_on[3:, r].mean()
            eu_s_off = eu_share_off[3:, r].mean()
            marker = ""
            if r in mena: marker = " ◄ MENA"
            elif r == sa: marker = " ◄ SA"
            elif r == eu: marker = " ◄ EU"
            print(f"  [{r}] {labels[r]:16s} {wl_on_m:12.6f} {wl_off_m:13.6f} {pen_delta:11.6f}"
                  f" {eu_s_on:12.6f} {eu_s_off:13.6f} {reward_on[r]:10.4f} {reward_off[r]:11.4f}{marker}")

    # ── Step 2: PPO training grid search (conditioned: 1 model, eval twice) ──
    print(f"\n{'═'*60}")
    print(f"Step 2: PPO Training — Conditioned (τ randomized per episode)")
    print(f"  Train 1 model per config, evaluate with τ=0 and τ=CBAM_RATE.")
    print(f"{'═'*60}")

    training_configs = [
        # (config_name, env_kwargs, ppo_overrides, timesteps)
        ("A: baseline α=5 τ=0.80", dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=5.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ), dict(ent_coef=0.01, learning_rate=3e-4, num_epochs=8), 1_000_000),

        ("B: strong α=50 τ=0.80", dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=50.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ), dict(ent_coef=0.01, learning_rate=3e-4, num_epochs=8), 1_000_000),

        ("C: extreme α=100 τ=0.80", dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=100.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ), dict(ent_coef=0.01, learning_rate=3e-4, num_epochs=8), 1_000_000),

        ("D: strong + free persistence", dict(
            cbam_rate=0.80, persistence=0.0, sectoral=True,
            welfare_amp=50.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=0.0,
        ), dict(ent_coef=0.01, learning_rate=3e-4, num_epochs=8), 1_000_000),

        ("E: strong + more exploration", dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=50.0, sector_granularity="emissions-simple",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ), dict(ent_coef=0.05, learning_rate=1e-3, num_epochs=4), 1_000_000),

        ("F: full sectors α=50", dict(
            cbam_rate=0.80, persistence=0.55, sectoral=True,
            welfare_amp=50.0, sector_granularity="full",
            fixed_savings_rate=True, no_mitigation=True, baseline_decay=1.0,
        ), dict(ent_coef=0.01, learning_rate=3e-4, num_epochs=8), 1_000_000),
    ]

    all_results = {}
    for cfg_name, env_kwargs, ppo_overrides, ts in training_configs:
        print(f"\n{'─'*60}")
        print(f"  Training: {cfg_name}")
        print(f"  Timesteps: {ts:,}, PPO: {ppo_overrides}")
        print(f"{'─'*60}")

        cbam_rate = env_kwargs["cbam_rate"]

        # Build conditioned training env (τ randomized per episode)
        train_env = jym.LogWrapper(_make_env(
            setup, **env_kwargs,
            cbam_randomize=True, cbam_tariff_rates=(0.0, cbam_rate),
        ))

        ppo_kwargs = dict(
            total_timesteps=ts,
            learning_rate=ppo_overrides.get("learning_rate", 3e-4),
            num_steps=NUM_STEPS,
            num_envs=8,
            num_minibatches=4,
            num_epochs=ppo_overrides.get("num_epochs", 8),
            ent_coef=ppo_overrides.get("ent_coef", 0.01),
            anneal_ent_coef=0.0,
            gamma=0.99,
            gae_lambda=0.95,
            max_grad_norm=1.0,
            clip_coef=0.2,
            clip_coef_vf=0.5,
            vf_coef=0.5,
            normalize_observations=True,
            normalize_rewards=True,
        )

        print(f"    Training conditioned model (τ ∈ {{0, {cbam_rate}}})...")
        ppo = PPO(**ppo_kwargs)
        trained = ppo.train(jax.random.PRNGKey(SEED), train_env)

        # Eval under both conditions with the SAME trained agent
        results_pair = {}
        for cbam_label, eval_rate in [("CBAM-off", 0.0), ("CBAM-on", cbam_rate)]:
            ek = dict(env_kwargs)
            ek["cbam_rate"] = eval_rate
            eval_env = _make_env(setup, **ek, log_fn=full_state_info_log_fn)

            print(f"    Evaluating {cbam_label} (τ={eval_rate})...")
            eval_infos = []
            for ep in range(N_EVAL):
                info = run_single_episode(
                    jax.random.PRNGKey(SEED + ep + 100), eval_env, trained
                )
                eval_infos.append(info)

            # Extract mean trajectories
            def _mean_array(info_list, key):
                arrs = []
                for i in info_list:
                    raw = i[key]
                    arrs.append(_dict_to_array(raw) if isinstance(raw, dict) else np.array(raw))
                return np.stack(arrs).mean(0)

            utility = _mean_array(eval_infos, "utility_all_regions")
            utw = _mean_array(eval_infos, "utility_times_welfloss_all_regions")
            welfloss = utw / (utility + 1e-10)

            go = _mean_array(eval_infos, "gross_output_all_regions")
            eu_bound_list = []
            for i in eval_infos:
                tf = np.array(i["trade_flows"])
                eu_bound_list.append(tf[:, :, eu, :].sum(axis=-1))
            eu_bound = np.stack(eu_bound_list).mean(0)
            eu_share = eu_bound / (go + 1e-10)

            results_pair[cbam_label] = {
                "utility": utility[3:].mean(axis=0),
                "welfloss": welfloss[3:].mean(axis=0),
                "eu_share": eu_share[3:].mean(axis=0),
                "reward": utw[3:].mean(axis=0),
                "welfloss_ts": welfloss,
                "eu_share_ts": eu_share,
            }

        all_results[cfg_name] = results_pair

        # Print comparison for this config
        on = results_pair["CBAM-on"]
        off = results_pair["CBAM-off"]
        print(f"\n  Results for {cfg_name}:")
        print(f"  {'Region':20s} {'EU share on':>12s} {'EU share off':>13s} {'Δ':>8s}"
              f" {'welfloss on':>12s} {'reward Δ':>9s}")
        for r in range(NR):
            delta_eu = on["eu_share"][r] - off["eu_share"][r]
            delta_rew = on["reward"][r] - off["reward"][r]
            marker = ""
            if r in mena: marker = " ◄ MENA"
            elif r == sa: marker = " ◄ SA"
            elif r == eu: marker = " ◄ EU"
            print(f"  [{r}] {labels[r]:16s} {on['eu_share'][r]:12.6f} {off['eu_share'][r]:13.6f}"
                  f" {delta_eu:8.5f} {on['welfloss'][r]:12.6f} {delta_rew:9.4f}{marker}")

    # ── Step 3: Summary and verdict ──
    print(f"\n{'█'*80}")
    print(f"LEVEL 1 SUMMARY: MECHANISM VERDICT")
    print(f"{'█'*80}")

    print(f"\n  {'Config':40s} {'SA ΔEU':>10s} {'MENA ΔEU':>10s} {'SA Δwl':>10s} {'MENA Δwl':>10s}")
    print(f"  {'─'*40} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    mechanism_works = False
    for cfg_name, pair in all_results.items():
        on = pair["CBAM-on"]
        off = pair["CBAM-off"]
        sa_deu = on["eu_share"][sa] - off["eu_share"][sa]
        mena_deu = np.mean([on["eu_share"][m] - off["eu_share"][m] for m in mena])
        sa_dwl = (1 - on["welfloss"][sa]) - (1 - off["welfloss"][sa])
        mena_dwl = np.mean([(1 - on["welfloss"][m]) - (1 - off["welfloss"][m]) for m in mena])

        sa_diverts = sa_deu < -0.001  # at least 0.1% diversion
        mena_diverts = mena_deu < -0.001
        diverts = sa_diverts or mena_diverts

        flag = " ✓ WORKS" if diverts else " ✗"
        if diverts:
            mechanism_works = True

        # Short name for display
        short = cfg_name[:40]
        print(f"  {short:40s} {sa_deu:10.5f} {mena_deu:10.5f} {sa_dwl:10.5f} {mena_dwl:10.5f}{flag}")

    print()
    if mechanism_works:
        print("  ▶ VERDICT: Mechanism CAN produce diversion under exaggerated conditions.")
        print("    Next step: calibrate α and τ to find minimal signal for reliable learning.")
    else:
        print("  ▶ VERDICT: Mechanism DOES NOT produce diversion even under extreme conditions.")
        print("    The export_reallocation action may be insufficiently expressive, or")
        print("    the welfloss penalty may not propagate through the reward properly.")
        print("    Consider alternative mechanisms (e.g., direct export tax, quota).")

    # ── Step 4: Plot summary ──
    n_configs = len(all_results)
    fig, axes = plt.subplots(2, n_configs, figsize=(5 * n_configs, 8), squeeze=False)
    fig.suptitle(f"Level 1 Exploration: EU Export Share SA & MENA\n"
                 f"CBAM-on (red) vs CBAM-off (blue)", fontsize=13)

    for ci, (cfg_name, pair) in enumerate(all_results.items()):
        on = pair["CBAM-on"]
        off = pair["CBAM-off"]
        T = on["eu_share_ts"].shape[0]
        ts = np.arange(T)

        # SA
        axes[0, ci].plot(ts, off["eu_share_ts"][:, sa], color="blue", label="off", alpha=0.8)
        axes[0, ci].plot(ts, on["eu_share_ts"][:, sa], color="red", label="on", alpha=0.8)
        axes[0, ci].set_title(f"SA: {cfg_name[:25]}", fontsize=8)
        if ci == 0:
            axes[0, ci].set_ylabel("EU export share")
            axes[0, ci].legend(fontsize=7)

        # MENA (average if multiple)
        mena_off = np.mean([off["eu_share_ts"][:, m] for m in mena], axis=0)
        mena_on = np.mean([on["eu_share_ts"][:, m] for m in mena], axis=0)
        axes[1, ci].plot(ts, mena_off, color="blue", label="off", alpha=0.8)
        axes[1, ci].plot(ts, mena_on, color="red", label="on", alpha=0.8)
        axes[1, ci].set_title(f"MENA: {cfg_name[:25]}", fontsize=8)
        if ci == 0:
            axes[1, ci].set_ylabel("EU export share")
            axes[1, ci].legend(fontsize=7)

    outpath = f"plots/level1_exploration_{setup_name}_{TIMESTAMP}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"\n  ✓ Saved Level 1 plot: {outpath}")
    plt.close(fig)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Investigate CBAM anomalies")
    parser.add_argument("--phase", type=str, default="1,1b",
                        help="Which phases to run: '1', '1b', '2', '3', '2b', 'L1', 'all' (default: '1,1b')")
    parser.add_argument("--setup", type=str, default="7-region",
                        help="Setup to use: '7-region' (default: '7-region')")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help=f"Training timesteps for Phase 2B (default: {TOTAL_TIMESTEPS:,})")
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    phases = args.phase.lower().split(",")

    for setup_name, setup in SETUPS.items():
        if args.setup != "all" and args.setup != setup_name:
            continue

        if "1" in phases or "all" in phases:
            phase1_static_audit(setup_name, setup)

        if "1b" in phases or "all" in phases:
            phase1b_signal_comparison(setup_name, setup)

        if "2" in phases or "all" in phases:
            phase2_trajectory_diagnostics(setup_name, setup)

        if "3" in phases or "all" in phases:
            phase3_mechanism_isolation(setup_name, setup)

        if "2b" in phases or "all" in phases:
            phase2b_training_comparison(setup_name, setup)

        if "l1" in phases or "all" in phases:
            level1_deep_exploration(setup_name, setup)

    print("\n" + "=" * 80)
    print("Investigation complete.")
    print("=" * 80)
