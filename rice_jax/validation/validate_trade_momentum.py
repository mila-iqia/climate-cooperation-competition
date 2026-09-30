"""validate_trade_momentum.py

Trains PPO agents on a 2 × 5 grid of conditions:
  CBAM ∈ {off (τ=0), on (τ=0.15)} × ρ ∈ {0, 0.25, 0.5, 0.75, 1.0}

For each non-EU region the figure shows three rows:
  Row 0 — EU export share timeseries  (mean ± 1σ);
           5 colours = ρ values; solid = CBAM on, dashed = CBAM off.
  Row 1 — CBAM Δ effect: (share_on − share_off) per ρ per step;
           shows whether CBAM depresses EU-directed exports and whether
           ρ moderates that effect.
  Row 2 — Grouped bar chart: time-mean EU share, CBAM on vs off, per ρ.

Output: plots/trade_momentum_validation_<TIMESTAMP>.png

Usage (from rice_jax/ directory):
    python validate_trade_momentum.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))



import os
import sys
from dataclasses import replace
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxnasium.algorithms import PPO

import jaxnasium as jym

from _experiment_util import run_single_episode
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

# ── Configuration ─────────────────────────────────────────────────────────────

RHO_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
# CBAM conditions: 0.0 = off, positive = on (EU rate ≈ 0.15)
CBAM_VALUES = [0.0, 0.15]
CBAM_LABELS = {0.0: "CBAM off", 0.15: "CBAM on (τ=0.15)"}

NUM_REGIONS: int = 7           # 3 → fast; 7 or 20 for higher fidelity
MRIO_DATA_ROOT: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset")
EU_REGION_IDX: int = 0
CBAM_TARIFF_RATE: float = 0.15  # default on-rate (used in CBAM_VALUES above)

# Training
TOTAL_TIMESTEPS: int = 300_000
NUM_ENVS: int = 4
NUM_STEPS: int = 100

# Evaluation
NUM_EVAL_EPISODES: int = 5
SEED: int = 42

# Plot
OUTPUT_DIR: str = "plots"
DPI: int = 150

# ── Environment builder ────────────────────────────────────────────────────────


def _build_env(rho: float, cbam_tariff_rate: float, region_params) -> jym.LogWrapper:
    """Return a LogWrapper around RiceMRIO with Phase 2A trade enabled."""
    env = RiceMRIO(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        mrio_data_root=MRIO_DATA_ROOT,
        mrio_trade=True,
        dest_alloc_persistence=rho,
        cbam_tariff_rate=cbam_tariff_rate,
        eu_region_idx=EU_REGION_IDX,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
    )
    return jym.LogWrapper(env)


# ── Train + evaluate ───────────────────────────────────────────────────────────


def train_and_collect(
    rho: float,
    cbam_tariff_rate: float,
    region_params,
    seed: jax.Array,
) -> np.ndarray:
    """
    Train a PPO agent for TOTAL_TIMESTEPS then roll out NUM_EVAL_EPISODES.

    Returns
    -------
    trade_flows : np.ndarray, shape (num_eval_episodes, T, NR, NR, NS)
        Bilateral sector trade flows at every episode step.
    """
    wrapped_env = _build_env(rho, cbam_tariff_rate, region_params)

    agent = PPO(
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=2.5e-4,
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        num_minibatches=4,
        num_epochs=4,
        ent_coef=2.0,
        anneal_ent_coef=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        clip_coef=0.2,
        clip_coef_vf=0.5,
        vf_coef=0.5,
        normalize_observations=True,
        normalize_rewards=False,
        log_function="tqdm",
    )

    cbam_lbl = CBAM_LABELS.get(cbam_tariff_rate, f"τ={cbam_tariff_rate}")
    print(f"  [ρ={rho}, {cbam_lbl}] Training for {TOTAL_TIMESTEPS:,} timesteps...")
    agent = agent.train(seed, wrapped_env)

    # Unwrap inner env and attach full-state logger for evaluation rollouts.
    eval_env = replace(wrapped_env._env, log_info_fn=full_state_info_log_fn)

    all_flows = []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 10_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        # trade_flows shape after scan: (T, NR, NR, NS)
        all_flows.append(np.array(logs["trade_flows"]))

    return np.stack(all_flows, axis=0)  # (E, T, NR, NR, NS)


# ── Metric helpers ─────────────────────────────────────────────────────────────


def eu_export_share(
    trade_flows: np.ndarray, eu_idx: int
) -> np.ndarray:
    """
    Fraction of each region's total exports directed to the EU.

    Parameters
    ----------
    trade_flows : (E, T, NR, NR, NS)  [episode, step, from_r, to_r, sector]
    eu_idx      : int  index of the EU region

    Returns
    -------
    share : (E, T, NR)
    """
    eu_exports   = trade_flows[:, :, :, eu_idx, :].sum(axis=-1)  # (E, T, NR)
    total_exports = trade_flows.sum(axis=(3, 4))                   # (E, T, NR)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total_exports > 1e-12, eu_exports / total_exports, np.nan)
    return share  # (E, T, NR)


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_figure(
    results: dict[tuple[float, float], np.ndarray],
    region_labels: list[str],
) -> plt.Figure:
    """
    results : {(rho, cbam_tariff_rate): (E, T, NR, NR, NS)}

    Layout (columns = non-EU regions):
      Row 0 — EU export share timeseries; 5 ρ colours, solid=CBAM on, dashed=CBAM off
      Row 1 — CBAM Δ effect: mean(on) − mean(off) per ρ per step
      Row 2 — Grouped bar chart: time-mean EU share, CBAM on vs off, per ρ
    """
    non_eu_idxs = [r for r in range(NUM_REGIONS) if r != EU_REGION_IDX]
    n_non_eu    = len(non_eu_idxs)
    T           = next(iter(results.values())).shape[1]
    timesteps   = np.arange(T)

    colors   = plt.cm.plasma(np.linspace(0.1, 0.9, len(RHO_VALUES)))
    rho_lbls = [f"ρ={r}" for r in RHO_VALUES]
    # linestyle per CBAM condition (off=dashed, on=solid)
    cbam_ls  = {0.0: "--", CBAM_TARIFF_RATE: "-"}

    fig, axes = plt.subplots(
        3, n_non_eu,
        figsize=(6 * n_non_eu, 13),
        constrained_layout=True,
    )
    if n_non_eu == 1:
        axes = axes.reshape(3, 1)

    for col_i, r_idx in enumerate(non_eu_idxs):
        rlbl     = region_labels[r_idx] if r_idx < len(region_labels) else f"Region {r_idx}"
        ax_share = axes[0, col_i]
        ax_delta = axes[1, col_i]
        ax_bar   = axes[2, col_i]

        # ── Row 0: EU share timeseries across (ρ × CBAM) ──────────────────────
        for rho_i, rho in enumerate(RHO_VALUES):
            c = colors[rho_i]
            for cbam in CBAM_VALUES:
                flows = results[(rho, cbam)]              # (E, T, NR, NR, NS)
                share = eu_export_share(flows, EU_REGION_IDX)  # (E, T, NR)
                rs    = share[:, :, r_idx]                # (E, T)
                m     = np.nanmean(rs, axis=0)
                s     = np.nanstd(rs,  axis=0)
                ls    = cbam_ls.get(cbam, "-")
                cbam_tag = "on" if cbam > 0 else "off"
                lbl   = f"{rho_lbls[rho_i]} CBAM {cbam_tag}"
                ax_share.plot(timesteps, m, color=c, linestyle=ls,
                              linewidth=1.6, label=lbl)
                ax_share.fill_between(timesteps, m - s, m + s,
                                      alpha=0.10, color=c)

        # ── Row 1: CBAM Δ effect per ρ  (on − off) ───────────────────────────
        for rho_i, rho in enumerate(RHO_VALUES):
            c = colors[rho_i]
            flows_on  = results[(rho, CBAM_TARIFF_RATE)]
            flows_off = results[(rho, 0.0)]
            share_on  = eu_export_share(flows_on,  EU_REGION_IDX)[:, :, r_idx]  # (E,T)
            share_off = eu_export_share(flows_off, EU_REGION_IDX)[:, :, r_idx]  # (E,T)
            delta     = np.nanmean(share_on, axis=0) - np.nanmean(share_off, axis=0)
            delta_std = np.sqrt(
                np.nanstd(share_on, axis=0)**2 + np.nanstd(share_off, axis=0)**2
            ) / np.sqrt(NUM_EVAL_EPISODES)  # SE of difference
            ax_delta.plot(timesteps, delta, color=c, linewidth=1.6, label=rho_lbls[rho_i])
            ax_delta.fill_between(timesteps, delta - delta_std, delta + delta_std,
                                  alpha=0.15, color=c)

        ax_delta.axhline(0, color="gray", linewidth=0.7, ls=":")
        ax_delta.set_ylabel("Δ EU share (on − off)", fontsize=9)
        ax_delta.set_xlabel("Episode step", fontsize=9)
        ax_delta.legend(fontsize=7, loc="lower right")

        # ── Row 2: Grouped bar chart  ─────────────────────────────────────────
        bar_w  = 0.35
        x_pos  = np.arange(len(RHO_VALUES))
        cbam_bar_colors = {0.0: "#aaaaaa", CBAM_TARIFF_RATE: "#e05c2a"}

        for ci, cbam in enumerate(CBAM_VALUES):
            means = []
            for rho in RHO_VALUES:
                rs = eu_export_share(results[(rho, cbam)], EU_REGION_IDX)[:, :, r_idx]
                means.append(float(np.nanmean(rs)))
            offset = (ci - 0.5) * bar_w
            bars   = ax_bar.bar(
                x_pos + offset, means,
                width=bar_w,
                color=cbam_bar_colors[cbam],
                edgecolor="white",
                linewidth=0.5,
                label=CBAM_LABELS.get(cbam, f"τ={cbam}"),
            )
            for bar, val in zip(bars, means):
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.001,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6,
                )

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels(rho_lbls, fontsize=8)
        ax_bar.set_ylabel("Mean EU share (full episode)", fontsize=9)
        ax_bar.set_xlabel("Trade momentum (ρ)", fontsize=9)
        ax_bar.legend(fontsize=8)
        ax_bar.set_ylim(bottom=0)

        # ── Column title + row 0 decoration ───────────────────────────────────
        ax_share.set_title(rlbl, fontsize=11, fontweight="bold")
        ax_share.set_ylabel("EU export share", fontsize=9)
        ax_share.set_ylim(bottom=0)
        # Compact legend: show only the two CBAM linestyles, not all 10 combos
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color="black", ls="-",  lw=1.6, label="CBAM on"),
            Line2D([0], [0], color="black", ls="--", lw=1.6, label="CBAM off"),
        ] + [
            Line2D([0], [0], color=colors[i], lw=3, label=rho_lbls[i])
            for i in range(len(RHO_VALUES))
        ]
        ax_share.legend(handles=legend_handles, fontsize=6, loc="upper right",
                        ncol=2, framealpha=0.7)

    # ── Row labels on leftmost column ─────────────────────────────────────────
    for row_i, lbl in enumerate(["EU export\nshare", "CBAM Δ effect\n(on − off)", "Summary"]):
        axes[row_i, 0].annotate(
            lbl, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=9, rotation=90, va="center", ha="right", color="dimgray",
        )

    fig.suptitle(
        f"Trade momentum (ρ) × CBAM on/off  —  export reallocation comparison\n"
        f"n_regions={NUM_REGIONS}  |  CBAM τ={CBAM_TARIFF_RATE}  |  "
        f"{TOTAL_TIMESTEPS:,} PPO steps  |  "
        f"{NUM_EVAL_EPISODES} eval eps  |  seed={SEED}",
        fontsize=11, y=1.01,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    seed = jax.random.PRNGKey(SEED)
    region_params = load_region_yamls(NUM_REGIONS)

    print(f"=== Trade Momentum × CBAM On/Off Validation ===")
    print(f"Regions: {NUM_REGIONS}  |  ρ values: {RHO_VALUES}  |  CBAM values: {CBAM_VALUES}")
    print(f"Total training runs: {len(RHO_VALUES) * len(CBAM_VALUES)}")
    print(f"Timesteps per run: {TOTAL_TIMESTEPS:,}  |  Eval episodes: {NUM_EVAL_EPISODES}")
    print()

    # Enumerate all (ρ, cbam) conditions, each with a unique deterministic key.
    conditions = [(rho, cbam) for rho in RHO_VALUES for cbam in CBAM_VALUES]
    results: dict[tuple[float, float], np.ndarray] = {}
    for cond_i, (rho, cbam) in enumerate(conditions):
        cond_key = jax.random.fold_in(seed, cond_i)
        results[(rho, cbam)] = train_and_collect(rho, cbam, region_params, cond_key)
        print(f"  [ρ={rho}, τ={cbam}] Done. trade_flows shape: {results[(rho, cbam)].shape}")

    # Build region labels from a sample env if available.
    try:
        sample_env = _build_env(0.0, 0.0, region_params)._env
        labels = list(sample_env.mrio_region_labels)
    except Exception:
        labels = [f"Region {i}" for i in range(NUM_REGIONS)]
    print(f"\nRegion labels: {labels}")

    print("\nGenerating figure...")
    fig = make_figure(results, labels)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"trade_momentum_cbam_validation_{timestamp}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
