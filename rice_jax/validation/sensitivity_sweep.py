"""sensitivity_sweep.py
======================
Sweeps ``welfare_loss_per_unit_tariff`` (α) from the working exaggerated
value (α=5) down to the Nordhaus-calibrated realistic value (α=0.4) and
measures the CBAM effect size at each step.

All other parameters are held at the known-working exaggerated baseline
(τ=0.80, sector_granularity="emissions-simple", sectoral_welfloss=True,
ρ=0.55, baseline_decay=1.0, fixed_savings_rate=True, no_mitigation=True).

For each α a single conditioned PPO model is trained and then evaluated
under τ=0 (CBAM-off) and τ=0.80 (CBAM-on).  The primary metric is:

    Δ_dirty[α, r] = dirty_EU_share(CBAM-on) − dirty_EU_share(CBAM-off)

A negative value means the policy learned to divert dirty-sector exports
away from the EU when CBAM is active — i.e. the mechanism is working.

Output
------
  - Console table: Δ_dirty and Δ_clean per (α, region)
  - plots/sensitivity_sweep_alpha_<TIMESTAMP>.png

Usage (from rice_jax/ directory):
    python sensitivity_sweep.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from __future__ import annotations

import os
import sys
from dataclasses import replace
from datetime import datetime

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("plots", exist_ok=True)

MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls, full_state_info_log_fn
from _experiment_util import FixedActionAgent, run_single_episode
import jaxnasium as jym
from jaxnasium.algorithms import PPO

# ── Configuration ──────────────────────────────────────────────────────────────

# Standard 7-region RICE ordering (CountryClass_7.csv):
#   0 Sub-Saharan Africa  1 South Asia  2 North America
#   3 MENA  4 Latin America  5 Europe & Central Asia (EU)  6 East Asia & Pacific
NUM_REGIONS: int = 7
EU_REGION_IDX: int = 5

REGION_LABELS: list[str] = [
    "Sub-Saharan Africa",   # 0
    "South Asia",           # 1
    "North America",        # 2
    "MENA",                 # 3
    "Latin America",        # 4
    "Europe & C.Asia",      # 5  ← EU
    "East Asia & Pacific",  # 6
]

# α levels stepped from exaggerated → realistic.  Log-evenly spaced to give
# a clean dose-response curve.
ALPHA_LEVELS: list[float] = [5.0, 2.0, 0.8, 0.4]

# All other env kwargs held constant at the known-working exaggerated baseline
# (matching investigate_cbam_anomalies.py Level 1 Config A / validate_emissions_simple.py).
BASE_ENV_KWARGS: dict = dict(
    cbam_tariff_rate=0.80,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    sectoral_welfloss=True,
    sector_granularity="emissions-simple",
    fixed_savings_rate=True,
    no_mitigation=True,
)

# PPO config matching Level 1 Config A
PPO_KWARGS: dict = dict(
    total_timesteps=1_000_000,
    num_steps=100,
    num_envs=8,
    num_minibatches=4,
    num_epochs=8,
    learning_rate=3e-4,
    ent_coef=0.01,
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

N_EVAL: int = 4   # eval episodes per (α, CBAM condition)
SEED: int = 42

# ── Environment factory ────────────────────────────────────────────────────────

_REGION_PARAMS = None


def _region_params():
    global _REGION_PARAMS
    if _REGION_PARAMS is None:
        _REGION_PARAMS = load_region_yamls(NUM_REGIONS)
    return _REGION_PARAMS


def _make_env(
    alpha: float,
    cbam_rate: float,
    *,
    cbam_randomize: bool = False,
    cbam_tariff_rates: tuple | None = None,
    log_fn=None,
) -> RiceMRIO:
    """Create a 7-region RiceMRIO env with the given α and CBAM rate."""
    kwargs = dict(
        num_regions=NUM_REGIONS,
        region_params=_region_params(),
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        eu_region_idx=EU_REGION_IDX,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        welfare_loss_per_unit_tariff=alpha,
        **BASE_ENV_KWARGS,
    )
    kwargs["cbam_tariff_rate"] = cbam_rate
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

# ── Metrics helpers ────────────────────────────────────────────────────────────


def _dict_to_array(d: dict) -> np.ndarray:
    """Convert {agent_id: (T,)} dict → (T, NR) array."""
    return np.stack([np.array(d[k]) for k in sorted(d.keys())], axis=-1)


def _as_array(x) -> np.ndarray:
    if isinstance(x, dict):
        return _dict_to_array(x)
    return np.array(x)


def _extract_dirty_clean_eu_share(
    trade_flows: np.ndarray,
    eu_idx: int,
    emissions_intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sectors into dirty (top-50% intensity) and clean (bottom-50%).

    Parameters
    ----------
    trade_flows         : (E, T, NR, NR, NS)
    emissions_intensity : (NR, NS)

    Returns
    -------
    dirty_share : (E, T, NR)  fraction of each region's dirty output sent to EU
    clean_share : (E, T, NR)  fraction of each region's clean output sent to EU
    """
    mean_intensity = emissions_intensity.mean(axis=0)       # (NS,)
    median = np.median(mean_intensity)
    dirty_mask = mean_intensity >= median                   # (NS,) bool

    eu_dirty  = trade_flows[:, :, :, eu_idx, :][:, :, :, dirty_mask].sum(axis=-1)
    eu_clean  = trade_flows[:, :, :, eu_idx, :][:, :, :, ~dirty_mask].sum(axis=-1)
    tot_dirty = trade_flows[:, :, :, :, dirty_mask].sum(axis=(3, 4))
    tot_clean = trade_flows[:, :, :, :, ~dirty_mask].sum(axis=(3, 4))

    with np.errstate(divide="ignore", invalid="ignore"):
        dirty_share = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
        clean_share = np.where(tot_clean > 1e-12, eu_clean / tot_clean, np.nan)
    return dirty_share, clean_share   # (E, T, NR) each


def _theoretical_penalty(alpha: float, env: RiceMRIO) -> np.ndarray:
    """Compute static per-region CBAM penalty at baseline trade flows.

    Returns penalty (NR,) = sum_s share * tef * dab[:,s,eu] * σ * τ * α
    """
    eu = env.eu_region_idx
    tau = BASE_ENV_KWARGS["cbam_tariff_rate"]

    eu_vol_sector = (                                           # (NR, NS)
        env.sector_output_shares
        * env.total_export_frac
        * env.dest_alloc_baseline[:, :, eu]
    )
    return (eu_vol_sector * env.emissions_intensity * tau * alpha).sum(axis=1)  # (NR,)


# ── Training / evaluation ──────────────────────────────────────────────────────


def train_and_collect(alpha: float) -> dict:
    """Train one conditioned PPO model at the given α and collect eval flows.

    Returns a dict with keys:
        "trade_flows_on"   : (E, T, NR, NR, NS)  CBAM-on evaluation
        "trade_flows_off"  : (E, T, NR, NR, NS)  CBAM-off evaluation
        "emissions_intensity" : (NR, NS)
        "penalty_static"   : (NR,)  theoretical static penalty
    """
    cbam_rate = BASE_ENV_KWARGS["cbam_tariff_rate"]

    # ── Train ──
    train_env = jym.LogWrapper(
        _make_env(
            alpha, cbam_rate,
            cbam_randomize=True,
            cbam_tariff_rates=(0.0, cbam_rate),
        )
    )
    ppo = PPO(**PPO_KWARGS)
    print(f"  Training α={alpha}  (τ randomized ∈ {{0, {cbam_rate}}})...")
    trained = ppo.train(jax.random.PRNGKey(SEED), train_env)

    emissions_intensity = train_env._env.emissions_intensity
    penalty_static = _theoretical_penalty(alpha, train_env._env)

    # ── Eval ──
    result = {
        "emissions_intensity": np.array(emissions_intensity),
        "penalty_static": np.array(penalty_static),
    }
    for label, eval_rate in [("off", 0.0), ("on", cbam_rate)]:
        eval_env = _make_env(alpha, eval_rate, log_fn=full_state_info_log_fn)
        flows_list = []
        for ep in range(N_EVAL):
            info = run_single_episode(
                jax.random.PRNGKey(SEED + ep + 100), eval_env, trained
            )
            flows_list.append(np.array(info["trade_flows"]))   # (T, NR, NR, NS)
        result[f"trade_flows_{label}"] = np.stack(flows_list)  # (E, T, NR, NR, NS)

    return result


# ── Output ─────────────────────────────────────────────────────────────────────


def _print_table(results_by_alpha: dict[float, dict]) -> None:
    """Print Δ_dirty and Δ_clean per (α, region) to stdout."""
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_REGION_IDX]
    header_cols = "  ".join(f"{REGION_LABELS[r][:12]:>13s}" for r in non_eu)

    for metric_label, key in [("Δ dirty EU share (CBAM-on − CBAM-off)", "dirty"),
                               ("Δ clean EU share (effect should be ~0)", "clean")]:
        print(f"\n{'─'*80}")
        print(f"  {metric_label}")
        print(f"  {'α':>8s}  {header_cols}")
        print(f"  {'─'*8}  {'  '.join(['─'*13]*len(non_eu))}")

        for alpha, res in sorted(results_by_alpha.items(), reverse=True):
            ei = res["emissions_intensity"]
            tf_on  = res["trade_flows_on"]
            tf_off = res["trade_flows_off"]

            d_on,  c_on  = _extract_dirty_clean_eu_share(tf_on,  EU_REGION_IDX, ei)
            d_off, c_off = _extract_dirty_clean_eu_share(tf_off, EU_REGION_IDX, ei)

            if key == "dirty":
                delta = np.nanmean(d_on, axis=(0, 1)) - np.nanmean(d_off, axis=(0, 1))
            else:
                delta = np.nanmean(c_on, axis=(0, 1)) - np.nanmean(c_off, axis=(0, 1))

            vals = "  ".join(f"{delta[r]:+13.6f}" for r in non_eu)
            print(f"  {alpha:>8.1f}  {vals}")

    print(f"\n  (negative Δ_dirty = CBAM caused dirty-sector diversion away from EU)")


def _make_figure(results_by_alpha: dict[float, dict]) -> plt.Figure:
    """3-row figure:
      Row 0  Δ dirty-sector EU share vs α  — one line per non-EU region
      Row 1  Δ clean-sector EU share vs α  — selectivity check (expect ~0)
      Row 2  Static theoretical CBAM penalty vs α  — contextual bar chart
    """
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_REGION_IDX]
    alphas_sorted = sorted(results_by_alpha.keys(), reverse=True)

    # Pre-compute Δ_dirty, Δ_clean, penalty per α
    data_dirty  = np.full((len(alphas_sorted), NUM_REGIONS), np.nan)
    data_clean  = np.full((len(alphas_sorted), NUM_REGIONS), np.nan)
    data_penalty = np.zeros((len(alphas_sorted), NUM_REGIONS))

    for ai, alpha in enumerate(alphas_sorted):
        res = results_by_alpha[alpha]
        ei  = res["emissions_intensity"]
        tf_on  = res["trade_flows_on"]
        tf_off = res["trade_flows_off"]

        d_on,  c_on  = _extract_dirty_clean_eu_share(tf_on,  EU_REGION_IDX, ei)
        d_off, c_off = _extract_dirty_clean_eu_share(tf_off, EU_REGION_IDX, ei)

        data_dirty[ai]   = np.nanmean(d_on, axis=(0, 1)) - np.nanmean(d_off, axis=(0, 1))
        data_clean[ai]   = np.nanmean(c_on, axis=(0, 1)) - np.nanmean(c_off, axis=(0, 1))
        data_penalty[ai] = res["penalty_static"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), constrained_layout=True)
    fig.suptitle(
        "CBAM α Sensitivity Sweep\n"
        "Effect size vs welfare-loss amplifier (τ=0.80, sector_granularity='emissions-simple')",
        fontsize=13,
    )

    # Colour cycle: one colour per non-EU region
    cmap = plt.cm.get_cmap("tab10", len(non_eu))

    x_ticks = list(range(len(alphas_sorted)))
    x_labels = [str(a) for a in alphas_sorted]

    # Row 0 — Δ dirty EU share
    ax0 = axes[0]
    for ci, r in enumerate(non_eu):
        ax0.plot(x_ticks, data_dirty[:, r],
                 marker="o", color=cmap(ci), label=REGION_LABELS[r], lw=2)
    ax0.axhline(0, color="black", lw=0.8, ls="--")
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_labels)
    ax0.set_xlabel("α (welfare_loss_per_unit_tariff)")
    ax0.set_ylabel("Δ dirty-sector EU share\n(CBAM-on − CBAM-off)")
    ax0.set_title("Primary metric: dirty-sector diversion (negative = mechanism works)")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Row 1 — Δ clean EU share (selectivity check)
    ax1 = axes[1]
    for ci, r in enumerate(non_eu):
        ax1.plot(x_ticks, data_clean[:, r],
                 marker="s", color=cmap(ci), label=REGION_LABELS[r], lw=2)
    ax1.axhline(0, color="black", lw=0.8, ls="--")
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    ax1.set_xlabel("α (welfare_loss_per_unit_tariff)")
    ax1.set_ylabel("Δ clean-sector EU share\n(CBAM-on − CBAM-off)")
    ax1.set_title("Selectivity check: clean-sector diversion (should be ≈0)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Row 2 — static penalty bar chart
    ax2 = axes[2]
    x = np.arange(len(non_eu))
    bar_width = 0.8 / len(alphas_sorted)
    for ai, alpha in enumerate(alphas_sorted):
        offset = (ai - len(alphas_sorted) / 2 + 0.5) * bar_width
        ax2.bar(
            x + offset,
            data_penalty[ai, non_eu],
            width=bar_width,
            label=f"α={alpha}",
            alpha=0.75,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([REGION_LABELS[r] for r in non_eu], rotation=25, ha="right")
    ax2.set_ylabel("Theoretical penalty\n(baseline trade flows)")
    ax2.set_title("Static penalty context: penalty = Σ share·tef·dab·σ·τ·α (≈ welfloss overhead at baseline)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    return fig


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("CBAM α Sensitivity Sweep")
    print(f"  α levels : {ALPHA_LEVELS}")
    print(f"  τ        : {BASE_ENV_KWARGS['cbam_tariff_rate']}")
    print(f"  sector   : {BASE_ENV_KWARGS['sector_granularity']}")
    print(f"  sectoral_welfloss : {BASE_ENV_KWARGS['sectoral_welfloss']}")
    print(f"  ρ        : {BASE_ENV_KWARGS['dest_alloc_persistence']}")
    print(f"  PPO timesteps per α: {PPO_KWARGS['total_timesteps']:,}")
    print(f"  eval episodes per condition: {N_EVAL}")
    print("=" * 80)

    # ── Phase 1: Print static theoretical penalties at baseline ──
    print("\n[Phase 1] Static theoretical penalty at baseline trade flows")
    env_ref = _make_env(ALPHA_LEVELS[0], BASE_ENV_KWARGS["cbam_tariff_rate"])
    print(f"\n  {'α':>8s}  " +
          "  ".join(f"{REGION_LABELS[r][:12]:>13s}" for r in range(NUM_REGIONS) if r != EU_REGION_IDX))
    print(f"  {'─'*8}  " + "  ".join(["─"*13] * (NUM_REGIONS - 1)))
    for alpha in ALPHA_LEVELS:
        pen = _theoretical_penalty(alpha, env_ref)
        vals = "  ".join(
            f"{pen[r]:+13.6f}" for r in range(NUM_REGIONS) if r != EU_REGION_IDX
        )
        print(f"  {alpha:>8.1f}  {vals}")

    # ── Phase 2: Train + collect ──
    print("\n[Phase 2] Training PPO at each α level")
    results_by_alpha: dict[float, dict] = {}
    for alpha in ALPHA_LEVELS:
        print(f"\n{'─'*60}")
        print(f"  α = {alpha}")
        print(f"{'─'*60}")
        results_by_alpha[alpha] = train_and_collect(alpha)

    # ── Phase 3: Metrics + console table ──
    print("\n[Phase 3] Effect-size summary table")
    _print_table(results_by_alpha)

    # ── Phase 4: Figure ──
    print("\n[Phase 4] Generating figure...")
    fig = _make_figure(results_by_alpha)
    outpath = f"plots/sensitivity_sweep_alpha_{TIMESTAMP}.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
