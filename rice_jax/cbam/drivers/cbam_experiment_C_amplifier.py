"""cbam_experiment_C_amplifier.py

Experiment C — Transfer pool amplifier sweep (Phase 2B Tier 2 external finance).

Question
--------
At what external transfer multiplier does CBAM-revenue-funded redistribution
become sufficient to suppress trade diversion?

Background
----------
Experiment A showed that transfers funded *solely* from CBAM revenue
(transfer_pool_multiplier=1) with transfer_allocation="effort" and
transfer_mode="abatement" fail to attenuate the crowd-out gap — the pool is
too small in utility space (~0.04/exporter on c≈40, giving ΔU≈0.001).

This is consistent with the literature: CBAM revenues (~€2.1B/yr) are
structurally minor relative to the capital required for industrial
decarbonisation ($235-335B cumulative to 2050 for steel alone). The NCQG
$300B/yr climate finance commitment implies a ratio of ~140× CBAM revenues.

This experiment finds the *minimum multiplier* above which attenuation becomes
positive, providing an empirical estimate of how much external climate finance
(above CBAM revenues alone) is needed to make mitigation the dominant strategy.

Design
------
Sweep: transfer_pool_multiplier ∈ MULTIPLIER_LEVELS
Fixed: revenue_share=1.0, transfer_mode="abatement", transfer_allocation="effort"
       (the winning combination from Experiment A, run 2)

For each (multiplier, seed), train two policies:
  PINNED — delta_max=0 (diversion closed; only mitigation/savings respond)
  OPEN   — delta_max=3.0 (diversion channel open)

Crowd-out gap    = μ_pinned − μ_open        (per seed, per multiplier)
Attenuation      = gap(M=1) − gap(M=X)      (per seed; positive = gap shrinks)

A multiplier M* where attenuation first becomes positive is the empirical
minimum external finance threshold.

Multiplier levels
-----------------
{1, 2, 5, 10, 20, 50} × CBAM revenue.
M=1 is the Experiment A baseline (expected FAIL).
M=10-50 maps to NCQG-scale external finance.

Total runs: 6 multipliers × 3 seeds × 2 arms = 36 training runs.

Pass criterion
--------------
There exists M* ∈ MULTIPLIER_LEVELS such that:
  attenuation(M*) > 0 on the worst seed, AND
  attenuation is monotonically non-decreasing in M from M=1 to M*.

Literature anchors
------------------
- Böhringer, Fischer & Rosendahl (2010) §4: perverse recycling at small pool
- Fischer & Fox (2012): allocation rule irrelevant at small pool
- Helm & Schmidt (2015): amplified pool + effort allocation expands coalition
- NCQG COP29 outcome: $300B/yr by 2035 (implicit multiplier ~140× CBAM revenue)
- Schrag et al. (2025): "carbon tax assets for carbon tax liabilities" —
  CBAM revenue should be treated as a co-financing lever, not the sole pool

Usage (from rice_jax/)
-----------------------
    # Full 3-seed run (36 x 2M = 72M total timesteps), managed run folder:
    python run_cbam_experiment.py cbam/drivers/cbam_experiment_C_amplifier.py \\
        --depth full

    # Smoke test (1 seed, short), unmanaged:
    python cbam/drivers/cbam_experiment_C_amplifier.py --timesteps 200000 --seeds 0

    # Replot from saved pkl:
    python cbam/drivers/cbam_experiment_C_amplifier.py --replot <pkl_path>
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import sys
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

import argparse
import os as _os
import pickle
import time
from datetime import datetime

import jax
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _experiment_util import get_log_dir, get_output_dir, run_single_episode, with_log_info_fn
from rice_jax.training import (
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn
from cbam.config import metrics as M
from cbam.config.canonical_config import (
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    EU_REGION_IDX as EU_IDX,
    EVAL_LAST_T,
    NON_EU_EXPORTER_IDXS,
    NUM_EVAL_EPISODES,
    NUM_REGIONS,
    REGION_NAMES,
    canonical_train_kwargs,
    make_canonical_env,
)
from cbam.config.registry import REGISTRY


# ── Config ─────────────────────────────────────────────────────────────────

EXPERIMENT_ID = "E_amplifier"
_REGISTRY_ENTRY = REGISTRY[EXPERIMENT_ID]

CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)

# Multiplier levels to sweep.  M=1 is the Exp-A baseline (expected FAIL).
# M=10–50 maps to NCQG-scale external finance.
MULTIPLIER_LEVELS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

# Fixed transfer config (winning combo from Experiment A)
FIXED_REVENUE_SHARE  = 1.0
FIXED_TRANSFER_MODE  = "abatement"
FIXED_TRANSFER_ALLOC = "effort"

DELTA_MAX_PINNED = 0.0
DELTA_MAX_OPEN   = 3.0

_DEFAULT_TIMESTEPS = canonical_train_kwargs()["total_timesteps"]
NUM_ENVS  = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]

OUTPUT_DIR        = get_output_dir("plots")
LOG_DIR           = get_log_dir("training_logs")
LOG_PREFIX        = "cbam_C_amp_"
CHECKPOINT_PREFIX = "cbam_C_amp_ckpt_"

_PPO_KWARGS = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
               if k != "total_timesteps"}


# ── Checkpoint helpers ──────────────────────────────────────────────────────

def _checkpoint_path(timestamp: str) -> str:
    return _os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}{timestamp}.pkl")


def _save_checkpoint(cells: list, timestamp: str) -> None:
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(_checkpoint_path(timestamp), "wb") as fh:
        pickle.dump({"timestamp": timestamp, "cells": cells}, fh)


# ── Build / train helpers ──────────────────────────────────────────────────

def _build_env(multiplier: float, pinned: bool, *, for_training: bool = True):
    """Build a canonical env with the experiment knobs."""
    return make_canonical_env(
        for_training            = for_training,
        revenue_share           = FIXED_REVENUE_SHARE,
        transfer_pool_multiplier= multiplier,
        transfer_mode           = FIXED_TRANSFER_MODE,
        transfer_allocation     = FIXED_TRANSFER_ALLOC,
        delta_max               = DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN,
    )


def _cell_label(multiplier: float, pinned: bool, seed: int) -> str:
    arm = "pinned" if pinned else "open"
    return f"amp{int(multiplier):04d}_{arm}_s{seed}"


def _make_log_fn(label: str, num_iters: int):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _SCREEN_DROP = {"action_mean", "action_var"}

    def _compact_print_fn(data: dict, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _SCREEN_DROP}, iteration)

    log_fn = make_combined_log_fn(
        _compact_print_fn,
        make_csv_log_fn(csv_path),
    )
    return log_fn, csv_path


def _train_cell(multiplier: float, pinned: bool, seed: int, total_timesteps: int):
    """Train one cell. Returns (trained_ppo, csv_path, label)."""
    label = _cell_label(multiplier, pinned, seed)
    env   = _build_env(multiplier, pinned, for_training=True)
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iters)

    ppo = RCPOMonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*60}")
    print(f"  Training C-amplifier: {label}")
    print(f"    multiplier={multiplier:.0f}×  arm={'PINNED' if pinned else 'OPEN'}  seed={seed}")
    print(f"    (rs=1.0, mode={FIXED_TRANSFER_MODE}, alloc={FIXED_TRANSFER_ALLOC})")
    print(f"{'━'*60}")
    t0  = time.perf_counter()
    agent, _metrics = ppo.train(jax.random.PRNGKey(seed), env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return agent, csv_path, label


# ── Eval ───────────────────────────────────────────────────────────────────

def _eval_cell(eval_seed: int, raw_env, agent):
    eval_env = with_log_info_fn(raw_env, full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    flows, mit = [], []
    base_key = jax.random.PRNGKey(eval_seed)
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(base_key, 90_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)
        flows.append(np.array(logs["trade_flows"]))
        mit.append(_to_arr(logs["mitigation_rates_all_regions"]))

    return {
        "trade_flows": np.stack(flows, 0),
        "mitigation":  np.stack(mit,   0),
    }


# ── Grid runner ────────────────────────────────────────────────────────────

def _run_grid(seeds, total_timesteps, multiplier_levels, *, existing_cells=None, timestamp=None):
    completed = {
        (c["seed"], c["multiplier"], c["pinned"])
        for c in (existing_cells or [])
    }
    cells = list(existing_cells or [])

    for seed in seeds:
        for mult in multiplier_levels:
            for pinned in (True, False):
                if (seed, mult, pinned) in completed:
                    print(f"  Skipping (done): {_cell_label(mult, pinned, seed)}")
                    continue

                partial_csv = _os.path.join(
                    LOG_DIR, f"{LOG_PREFIX}{_cell_label(mult, pinned, seed)}.csv"
                )
                if _os.path.exists(partial_csv):
                    print(f"  Deleting partial CSV: {partial_csv}")
                    _os.remove(partial_csv)

                ppo, csv_path, label = _train_cell(mult, pinned, seed, total_timesteps)
                raw_env = _build_env(mult, pinned, for_training=False)
                ev      = _eval_cell(seed + 10_000, raw_env, ppo)

                mu_aggregate  = M.mean_mitigation_rate(
                    ev["mitigation"], region_idxs=CBAM_PLOT_REGIONS,
                )
                mu_per_region = M.per_region_mitigation_rate(ev["mitigation"])
                eu_share_agg  = M.eu_dirty_export_share(
                    ev["trade_flows"], eu_region_idx=EU_IDX,
                    exporter_idxs=CBAM_PLOT_REGIONS,
                )

                cells.append({
                    "seed":           seed,
                    "multiplier":     mult,
                    "pinned":         pinned,
                    "delta_max":      DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN,
                    "label":          label,
                    "csv_path":       csv_path,
                    "mu_non_eu":      mu_aggregate,
                    "mu_per_region":  mu_per_region,
                    "eu_dirty_share": eu_share_agg,
                    "mitigation_raw": ev["mitigation"].astype(np.float32),
                    "trade_flows_raw":ev["trade_flows"].astype(np.float32),
                })

                if timestamp is not None:
                    _save_checkpoint(cells, timestamp)

    return cells


# ── Aggregation ────────────────────────────────────────────────────────────

def _amplifier_table(cells):
    """Build summary DataFrames.

    Returns
    -------
    df_cells : pd.DataFrame  — one row per (seed, multiplier, arm)
    df_gap   : pd.DataFrame  — one row per (seed, multiplier); crowd_out_gap
    df_attn  : pd.DataFrame  — one row per (seed, multiplier); attenuation vs M=1
    """
    df_cells = pd.DataFrame([
        {k: c[k] for k in ("seed", "multiplier", "pinned", "mu_non_eu", "eu_dirty_share")}
        for c in cells
    ])

    pivot = (df_cells
             .pivot_table(index=["seed", "multiplier"],
                          columns="pinned",
                          values="mu_non_eu")
             .rename(columns={True: "mu_pinned", False: "mu_open"})
             .reset_index())

    pivot["crowd_out_gap"] = pivot["mu_pinned"] - pivot["mu_open"]

    # Attenuation relative to M=1 baseline
    baseline_gap = (pivot.query("multiplier == 1.0")
                    .set_index("seed")["crowd_out_gap"]
                    .rename("gap_m1"))
    pivot = pivot.join(baseline_gap, on="seed")
    pivot["attenuation"] = pivot["gap_m1"] - pivot["crowd_out_gap"]

    df_gap  = pivot[["seed", "multiplier", "mu_pinned", "mu_open", "crowd_out_gap"]]
    df_attn = pivot[["seed", "multiplier", "gap_m1", "crowd_out_gap", "attenuation"]]

    return df_cells, df_gap, df_attn


def _find_threshold(df_attn: pd.DataFrame) -> float | None:
    """Return the smallest M where worst-seed attenuation > 0, or None."""
    for m in sorted(df_attn["multiplier"].unique()):
        worst = df_attn.query(f"multiplier == {m}")["attenuation"].min()
        if worst > 0:
            return m
    return None


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot(cells, out_prefix: str):
    df_cells, df_gap, df_attn = _amplifier_table(cells)
    m_levels = sorted(df_attn["multiplier"].unique())
    threshold = _find_threshold(df_attn)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    ax_gap    = fig.add_subplot(gs[0, 0])  # crowd-out gap vs multiplier
    ax_attn   = fig.add_subplot(gs[0, 1])  # attenuation vs multiplier
    ax_mu     = fig.add_subplot(gs[0, 2])  # μ_open vs multiplier
    ax_eu     = fig.add_subplot(gs[1, 0])  # EU dirty share vs multiplier
    ax_mu_reg = fig.add_subplot(gs[1, 1:]) # per-region μ_open at each M (heatmap)

    seed_colors = {s: c for s, c in zip(sorted(df_gap["seed"].unique()),
                                         ["#1f77b4", "#ff7f0e", "#2ca02c"])}

    # ── Panel 1: crowd-out gap ─────────────────────────────────────────────
    for seed, grp in df_gap.groupby("seed"):
        ax_gap.plot(grp["multiplier"], grp["crowd_out_gap"],
                    marker="o", color=seed_colors[seed], label=f"seed {seed}")
    ax_gap.axhline(0, color="k", lw=0.8, ls="--")
    if threshold is not None:
        ax_gap.axvline(threshold, color="red", lw=1.2, ls=":", label=f"M*={threshold:.0f}×")
    ax_gap.set_xscale("log")
    ax_gap.set_xlabel("Transfer multiplier (× CBAM revenue)")
    ax_gap.set_ylabel("Crowd-out gap  (μ_pinned − μ_open)")
    ax_gap.set_title("Crowd-out gap vs multiplier")
    ax_gap.legend(fontsize=8)

    # ── Panel 2: attenuation ──────────────────────────────────────────────
    attn_means = df_attn.groupby("multiplier")["attenuation"]
    means = attn_means.mean()
    stds  = attn_means.std()
    ax_attn.fill_between(m_levels,
                          [means[m] - stds[m] for m in m_levels],
                          [means[m] + stds[m] for m in m_levels],
                          alpha=0.25, color="steelblue")
    ax_attn.plot(m_levels, [means[m] for m in m_levels],
                 marker="s", color="steelblue", lw=2, label="mean ± 1 SD")
    for seed, grp in df_attn.groupby("seed"):
        ax_attn.plot(grp["multiplier"], grp["attenuation"],
                     marker=".", alpha=0.5, color=seed_colors[seed])
    ax_attn.axhline(0, color="k", lw=1.0, ls="--", label="no attenuation")
    if threshold is not None:
        ax_attn.axvline(threshold, color="red", lw=1.2, ls=":",
                        label=f"M*={threshold:.0f}×")
    ax_attn.set_xscale("log")
    ax_attn.set_xlabel("Transfer multiplier (× CBAM revenue)")
    ax_attn.set_ylabel("Attenuation (gap_M1 − gap_M)")
    ax_attn.set_title("Transfer attenuation vs multiplier")
    ax_attn.legend(fontsize=8)

    # ── Panel 3: mean μ_open vs multiplier ────────────────────────────────
    for seed, grp in df_gap.groupby("seed"):
        ax_mu.plot(grp["multiplier"], grp["mu_open"],
                   marker="o", color=seed_colors[seed], label=f"seed {seed}")
    # Dashed reference: μ_pinned mean (should be roughly flat)
    mu_pin_mean = df_gap.groupby("multiplier")["mu_pinned"].mean()
    ax_mu.plot(m_levels, [mu_pin_mean[m] for m in m_levels],
               ls="--", color="gray", lw=1.5, label="μ_pinned (mean)")
    ax_mu.set_xscale("log")
    ax_mu.set_xlabel("Transfer multiplier")
    ax_mu.set_ylabel("Mean μ (non-EU exporters, last 5 steps)")
    ax_mu.set_title("Mitigation rate (OPEN arm) vs multiplier")
    ax_mu.legend(fontsize=8)

    # ── Panel 4: EU dirty share vs multiplier ─────────────────────────────
    eu_open = (df_cells[~df_cells["pinned"]]
               .groupby("multiplier")["eu_dirty_share"])
    eu_means = eu_open.mean()
    eu_stds  = eu_open.std()
    ax_eu.fill_between(m_levels,
                       [eu_means[m] - eu_stds[m] for m in m_levels],
                       [eu_means[m] + eu_stds[m] for m in m_levels],
                       alpha=0.25, color="darkorange")
    ax_eu.plot(m_levels, [eu_means[m] for m in m_levels],
               marker="D", color="darkorange", lw=2)
    if threshold is not None:
        ax_eu.axvline(threshold, color="red", lw=1.2, ls=":",
                      label=f"M*={threshold:.0f}×")
    ax_eu.set_xscale("log")
    ax_eu.set_xlabel("Transfer multiplier")
    ax_eu.set_ylabel("EU dirty export share (OPEN arm)")
    ax_eu.set_title("EU dirty share vs multiplier")
    ax_eu.legend(fontsize=8)

    # ── Panel 5: per-region μ_open heatmap ────────────────────────────────
    reg_names = [REGION_NAMES[i] for i in CBAM_PLOT_REGIONS]
    # mean over seeds for each (multiplier, region)
    mu_heat = np.zeros((len(m_levels), len(CBAM_PLOT_REGIONS)))
    open_cells = [c for c in cells if not c["pinned"]]
    for mi, m in enumerate(m_levels):
        for ri, reg in enumerate(CBAM_PLOT_REGIONS):
            vals = [c["mu_per_region"][reg] for c in open_cells
                    if c["multiplier"] == m]
            mu_heat[mi, ri] = float(np.mean(vals)) if vals else np.nan

    im = ax_mu_reg.imshow(mu_heat, aspect="auto", cmap="RdYlGn",
                          vmin=0.0, vmax=1.0)
    ax_mu_reg.set_xticks(range(len(CBAM_PLOT_REGIONS)))
    ax_mu_reg.set_xticklabels(reg_names, rotation=30, ha="right", fontsize=8)
    ax_mu_reg.set_yticks(range(len(m_levels)))
    ax_mu_reg.set_yticklabels([f"{m:.0f}×" for m in m_levels], fontsize=8)
    ax_mu_reg.set_xlabel("Region")
    ax_mu_reg.set_ylabel("Transfer multiplier")
    ax_mu_reg.set_title("Per-region μ_open (mean over seeds)")
    plt.colorbar(im, ax=ax_mu_reg, fraction=0.03, pad=0.02, label="μ")

    # ── Suptitle ──────────────────────────────────────────────────────────
    thresh_str = (f"M* = {threshold:.0f}× CBAM revenue" if threshold is not None
                  else "No threshold found in sweep")
    fig.suptitle(
        f"Experiment C — Transfer Amplifier Sweep\n"
        f"(rs=1.0, mode=abatement, alloc=effort, {len(CANONICAL_SEEDS)} seeds × 2M steps)\n"
        f"Threshold: {thresh_str}",
        fontsize=11,
    )

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = f"{out_prefix}.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {png_path}")
    return png_path


# ── Pass/fail report ───────────────────────────────────────────────────────

def _print_report(cells):
    df_cells, df_gap, df_attn = _amplifier_table(cells)
    threshold = _find_threshold(df_attn)

    print("\n" + "═" * 65)
    print("  EXPERIMENT C — TRANSFER AMPLIFIER SWEEP — RESULTS")
    print("═" * 65)
    print(f"\n  Config: rs={FIXED_REVENUE_SHARE}, mode={FIXED_TRANSFER_MODE}, "
          f"alloc={FIXED_TRANSFER_ALLOC}")
    print(f"\n  {'Mult':>6}  {'Gap mean':>10}  {'Gap min':>10}  "
          f"{'Attn mean':>12}  {'Attn min':>10}  {'PASS?':>6}")
    print("  " + "-" * 63)

    for m in sorted(df_attn["multiplier"].unique()):
        sub = df_attn.query(f"multiplier == {m}")
        gap_sub = df_gap.query(f"multiplier == {m}")
        gap_mean = gap_sub["crowd_out_gap"].mean()
        gap_min  = gap_sub["crowd_out_gap"].min()
        attn_mean = sub["attenuation"].mean()
        attn_min  = sub["attenuation"].min()
        passes = "✅ PASS" if attn_min > 0 else "❌ FAIL"
        print(f"  {m:>5.0f}×  {gap_mean:>10.4f}  {gap_min:>10.4f}  "
              f"{attn_mean:>12.4f}  {attn_min:>10.4f}  {passes:>6}")

    print()
    if threshold is not None:
        print(f"  THRESHOLD M* = {threshold:.0f}× CBAM revenue")
        print(f"  (≈ {threshold / 1:.0f}× internal; NCQG ratio ~140×)")
        print()
        print("  PASS: attenuation > 0 on worst seed at M*")
    else:
        print("  NO THRESHOLD FOUND in sweep range — extend to higher multipliers")
        print("  FAIL")
    print("═" * 65 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int,
                    default=_DEFAULT_TIMESTEPS,
                    help="Training timesteps per cell.")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=list(CANONICAL_SEEDS),
                    help="Random seeds to train (default: CANONICAL_SEEDS).")
    ap.add_argument("--multipliers", type=float, nargs="+",
                    default=list(MULTIPLIER_LEVELS),
                    help="Transfer multiplier levels to sweep.")
    ap.add_argument("--replot", metavar="PKL",
                    help="Skip training; replot from this pkl checkpoint.")
    ap.add_argument("--resume", metavar="PKL",
                    help="Resume from this pkl checkpoint (skip done cells).")
    args = ap.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_prefix = _os.path.join(OUTPUT_DIR, f"cbam_C_amplifier_{timestamp}")

    # ── Replot mode ────────────────────────────────────────────────────────
    if args.replot:
        with open(args.replot, "rb") as fh:
            data = pickle.load(fh)
        cells = data["cells"]
        _print_report(cells)
        _plot(cells, out_prefix)
        return

    # ── Training ───────────────────────────────────────────────────────────
    multiplier_levels = tuple(args.multipliers)

    existing_cells = []
    if args.resume:
        with open(args.resume, "rb") as fh:
            existing_cells = pickle.load(fh)["cells"]
        print(f"  Resuming from {args.resume}: {len(existing_cells)} cells already done.")

    cells = _run_grid(
        seeds          = args.seeds,
        total_timesteps= args.timesteps,
        multiplier_levels = multiplier_levels,
        existing_cells = existing_cells,
        timestamp      = timestamp,
    )

    # ── Save full PKL ──────────────────────────────────────────────────────
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = f"{out_prefix}.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump({
            "timestamp":        timestamp,
            "cells":            cells,
            "multiplier_levels":MULTIPLIER_LEVELS,
            "revenue_share":    FIXED_REVENUE_SHARE,
            "transfer_mode":    FIXED_TRANSFER_MODE,
            "transfer_alloc":   FIXED_TRANSFER_ALLOC,
        }, fh)
    print(f"\n  PKL saved → {pkl_path}")

    _print_report(cells)
    _plot(cells, out_prefix)


if __name__ == "__main__":
    main()
