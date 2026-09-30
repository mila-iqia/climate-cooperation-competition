"""cbam_compare_tc_AB.py

Compare symmetric vs asymmetric transition-cost formulations (A vs B).

Purpose
-------
Symmetric TC (Grubb 1995 as-is) penalises both ramp-ups AND ramp-downs of μ.
This suppresses crowd-out because agents cannot cheaply *reduce* μ when CBAM
pressure is removed, making the diversion channel ineffective.

Asymmetric TC (Ha-Duong et al. 1997; Grubb et al. 2021 WIREs) only penalises
increases in μ — downward reversals are free (their real cost is stranded
capital, a stock phenomenon captured in Phase 2C).

This script trains PINNED (δ_max=0) and OPEN (δ_max=3) policies under both
TC formulations and reports the crowd-out gap:

    gap = μ_pinned − μ_open

If symmetric TC suppresses crowd-out (gap ≈ 0), asymmetric TC should recover
it (gap > 0), confirming the literature critique.

Arms
----
  A: transition_cost_coef=10.0, transition_cost_asymmetric=False  (symmetric)
  B: transition_cost_coef=10.0, transition_cost_asymmetric=True   (asymmetric)

For each arm: PINNED (δ_max=0) + OPEN (δ_max=3), per seed.
Total: 2 arms × 2 conditions × N seeds = 4N training runs.

Usage
-----
    python validation/cbam_compare_tc_AB.py                         # canonical 2M
    python validation/cbam_compare_tc_AB.py --timesteps 500000      # quick test
    python validation/cbam_compare_tc_AB.py --seeds 0               # single seed
    python validation/cbam_compare_tc_AB.py --replot <pkl>          # replot only
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import pickle
import time
from dataclasses import replace
from datetime import datetime

import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from training_monitor import (
    MonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn
from _experiment_util import get_output_dir, get_log_dir, run_single_episode
from validation.canonical_config import (
    NUM_REGIONS,
    EU_REGION_IDX as EU_IDX,
    REGION_NAMES,
    NON_EU_EXPORTER_IDXS,
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    NUM_EVAL_EPISODES,
    EVAL_LAST_T,
    canonical_env_kwargs,
    canonical_train_kwargs,
    make_canonical_env,
)
from validation import metrics as M


# ── Config ─────────────────────────────────────────────────────────────────

TC_COEF = 10.0
DELTA_MAX_PINNED = 0.0
DELTA_MAX_OPEN   = 3.0

ARMS = {
    "symmetric":  {"transition_cost_coef": TC_COEF, "transition_cost_asymmetric": False},
    "asymmetric": {"transition_cost_coef": TC_COEF, "transition_cost_asymmetric": True},
}

CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)

_DEFAULT_TIMESTEPS = canonical_train_kwargs()["total_timesteps"]
NUM_ENVS  = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR    = get_log_dir("training_logs")
LOG_PREFIX = "cbam_tc_AB_"

_PPO_KWARGS = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
               if k != "total_timesteps"}


# ── Build / train helpers ──────────────────────────────────────────────────

def _build_env(arm_name: str, delta_max: float, *, for_training: bool = True):
    """Build a canonical env with the TC arm settings and given delta_max."""
    arm_kw = dict(ARMS[arm_name])
    arm_kw["delta_max"] = delta_max
    return make_canonical_env(for_training=for_training, **arm_kw)


def _cell_label(arm_name: str, pinned: bool, seed: int) -> str:
    cond = "pinned" if pinned else "open"
    return f"{arm_name}_{cond}_s{seed}"


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


def _train_cell(arm_name: str, pinned: bool, seed: int, total_timesteps: int):
    """Train one cell. Returns (trained_ppo, csv_path, label)."""
    label = _cell_label(arm_name, pinned, seed)
    delta_max = DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN

    env = _build_env(arm_name, delta_max, for_training=True)
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iters)

    ppo = MonitoredPPO(
        total_timesteps=total_timesteps,
        log_function=log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*60}")
    print(f"  Training TC compare: {label}")
    print(f"    arm={arm_name}  δ_max={delta_max}  seed={seed}")
    print(f"{'━'*60}")
    t0 = time.perf_counter()
    ppo = ppo.train(jax.random.PRNGKey(seed), env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path, label


# ── Eval ───────────────────────────────────────────────────────────────────

def _eval_cell(eval_seed: int, raw_env, agent):
    """Run NUM_EVAL_EPISODES rollouts; return mitigation and trade_flows."""
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    flows, mit = [], []
    base_key = jax.random.PRNGKey(eval_seed)
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(base_key, 90_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        flows.append(np.array(logs["trade_flows"]))
        mit.append(_to_arr(logs["mitigation_rates_all_regions"]))

    return {
        "trade_flows": np.stack(flows, 0),
        "mitigation":  np.stack(mit, 0),
    }


# ── Grid runner ────────────────────────────────────────────────────────────

def _run_grid(seeds, total_timesteps):
    """Train all cells. Returns list of cell dicts."""
    cells = []

    for seed in seeds:
        for arm_name in ARMS:
            for pinned in (True, False):
                ppo, csv_path, label = _train_cell(
                    arm_name, pinned, seed, total_timesteps,
                )
                delta_max = DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN
                raw_env = _build_env(arm_name, delta_max, for_training=False)
                ev = _eval_cell(seed + 10_000, raw_env, ppo)

                mu_aggregate = M.mean_mitigation_rate(
                    ev["mitigation"], region_idxs=CBAM_PLOT_REGIONS,
                )
                eu_share = M.eu_dirty_export_share(
                    ev["trade_flows"], eu_region_idx=EU_IDX,
                    exporter_idxs=CBAM_PLOT_REGIONS,
                )

                cells.append({
                    "seed":           seed,
                    "arm":            arm_name,
                    "pinned":         pinned,
                    "delta_max":      delta_max,
                    "label":          label,
                    "csv_path":       csv_path,
                    "mu_non_eu":      mu_aggregate,
                    "eu_dirty_share": eu_share,
                    "mitigation_raw": ev["mitigation"].astype(np.float32),
                    "trade_flows_raw": ev["trade_flows"].astype(np.float32),
                })

    return cells


# ── Aggregation ────────────────────────────────────────────────────────────

def _build_table(cells):
    """Build per-arm crowd-out gap table.

    Returns (df_cells, df_gap) DataFrames.
    """
    df = pd.DataFrame([
        {k: c[k] for k in ("seed", "arm", "pinned", "mu_non_eu", "eu_dirty_share")}
        for c in cells
    ])

    pivot = (df
             .pivot_table(index=["seed", "arm"],
                          columns="pinned", values="mu_non_eu")
             .reset_index())
    pivot = pivot.rename(columns={True: "mu_pinned", False: "mu_open"})
    pivot["crowd_out_gap"] = pivot["mu_pinned"] - pivot["mu_open"]

    return df, pivot


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_results(cells, df, pivot, timestamp):
    nseeds = len(set(c["seed"] for c in cells))
    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(
        f"TC Formulation Comparison: Symmetric (A) vs Asymmetric (B)\n"
        f"c_B={TC_COEF}, 9-region vuln. setup, {nseeds} seed(s)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel 1: μ pinned vs open per arm
    ax = fig.add_subplot(gs[0, 0])
    grp = pivot.groupby("arm").agg(
        mu_pinned_mean=("mu_pinned", "mean"),
        mu_open_mean=("mu_open", "mean"),
        mu_pinned_std=("mu_pinned", "std"),
        mu_open_std=("mu_open", "std"),
    ).reset_index()
    x = np.arange(len(grp))
    ax.bar(x - 0.15, grp["mu_pinned_mean"], 0.3,
           yerr=grp["mu_pinned_std"], label="μ pinned (δ=0)",
           color="tab:blue", alpha=0.8, capsize=4)
    ax.bar(x + 0.15, grp["mu_open_mean"], 0.3,
           yerr=grp["mu_open_std"], label="μ open (δ=3)",
           color="tab:red", alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["arm"])
    ax.set_ylabel("Mean μ (non-EU exporters)")
    ax.set_title("Mitigation: pinned vs open")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")

    # Panel 2: crowd-out gap per arm (seed dots + mean bar)
    ax = fig.add_subplot(gs[0, 1])
    arms_list = sorted(ARMS.keys())
    for i, arm in enumerate(arms_list):
        vals = pivot[pivot["arm"] == arm]["crowd_out_gap"]
        ax.scatter([i] * len(vals), vals, alpha=0.7,
                   color="tab:green" if arm == "asymmetric" else "tab:gray",
                   s=60)
        ax.scatter([i], [vals.mean()], marker="_", s=300, color="black", linewidths=2)
    ax.set_xticks(range(len(arms_list)))
    ax.set_xticklabels(arms_list)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("crowd_out_gap = μ_pinned − μ_open")
    ax.set_title("Crowd-out gap by TC formulation\n(dot = seed; bar = mean)")
    ax.grid(alpha=0.3)

    # Panel 3: EU dirty-export share per arm/condition
    ax = fig.add_subplot(gs[1, 0])
    for i, arm in enumerate(arms_list):
        sub = df[df["arm"] == arm]
        for pinned in (True, False):
            vals = sub[sub["pinned"] == pinned]["eu_dirty_share"]
            offset = -0.1 if pinned else 0.1
            color = "tab:blue" if pinned else "tab:red"
            marker = "o" if pinned else "s"
            lbl = f"{'pinned' if pinned else 'open'}" if i == 0 else None
            ax.scatter([i + offset] * len(vals), vals, alpha=0.7,
                       color=color, marker=marker, s=50, label=lbl)
    ax.set_xticks(range(len(arms_list)))
    ax.set_xticklabels(arms_list)
    ax.set_ylabel("EU dirty-export share")
    ax.set_title("Diversion: EU share of dirty exports")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: summary text
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    lines = [
        "TC Comparison Summary",
        "=" * 40,
        f"transition_cost_coef = {TC_COEF}",
        f"Seeds: {sorted(set(c['seed'] for c in cells))}",
        "",
    ]
    for arm in arms_list:
        sub = pivot[pivot["arm"] == arm]
        gap_mean = sub["crowd_out_gap"].mean()
        gap_min = sub["crowd_out_gap"].min()
        gap_max = sub["crowd_out_gap"].max()
        lines.append(f"{arm:12s}: gap mean={gap_mean:+.4f}  "
                     f"[{gap_min:+.4f}, {gap_max:+.4f}]")

    # Verdict
    sym_gap = pivot[pivot["arm"] == "symmetric"]["crowd_out_gap"].mean()
    asym_gap = pivot[pivot["arm"] == "asymmetric"]["crowd_out_gap"].mean()
    lines += [
        "",
        f"Δ(gap): asymmetric − symmetric = {asym_gap - sym_gap:+.4f}",
        "",
    ]
    if asym_gap > sym_gap and asym_gap > 0.01:
        lines.append("VERDICT: Asymmetric TC recovers crowd-out signal ✓")
    elif asym_gap <= 0.01:
        lines.append("VERDICT: Neither formulation shows crowd-out")
    else:
        lines.append("VERDICT: No improvement from asymmetric TC")

    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_tc_AB_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare symmetric vs asymmetric TC")
    parser.add_argument("--timesteps", type=int, default=_DEFAULT_TIMESTEPS)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: canonical 0,1,2)")
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing PKL to replot without training")
    parser.add_argument("--tc-coef", type=float, default=TC_COEF,
                        help="Override TC coefficient for both arms")
    args = parser.parse_args()

    if args.replot:
        with open(args.replot, "rb") as fh:
            bundle = pickle.load(fh)
        cells = bundle["cells"]
        df, pivot = _build_table(cells)
        ts = bundle.get("timestamp", "replot")
        _plot_results(cells, df, pivot, ts)
        print("\nReplot complete.")
        return

    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else CANONICAL_SEEDS

    # Allow runtime override of TC coefficient
    tc_coef = args.tc_coef
    if tc_coef != TC_COEF:
        ARMS["symmetric"]["transition_cost_coef"] = tc_coef
        ARMS["asymmetric"]["transition_cost_coef"] = tc_coef

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"TC A/B comparison — {len(seeds)} seed(s) × 2 arms × 2 conditions "
          f"= {len(seeds)*4} training runs")
    print(f"  timesteps/run: {args.timesteps:,}")
    print(f"  TC coef: {tc_coef}")
    print(f"  timestamp: {timestamp}")

    cells = _run_grid(seeds, args.timesteps)
    df, pivot = _build_table(cells)

    # Print summary
    print("\n" + "═" * 60)
    print("CROWD-OUT GAP SUMMARY")
    print("═" * 60)
    for arm in sorted(ARMS.keys()):
        sub = pivot[pivot["arm"] == arm]
        print(f"  {arm:12s}: gap = {sub['crowd_out_gap'].mean():+.4f} "
              f"(seeds: {list(sub['crowd_out_gap'].round(4))})")
    print("═" * 60)

    # Plot
    fig_path = _plot_results(cells, df, pivot, timestamp)

    # Save bundle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_tc_AB_{timestamp}.pkl")
    bundle = {
        "timestamp":    timestamp,
        "cells":        cells,
        "env_kwargs":   canonical_env_kwargs(),
        "train_kwargs": {**canonical_train_kwargs(), "total_timesteps": args.timesteps},
        "arms":         ARMS,
        "seeds":        seeds,
        "tc_coef":      tc_coef,
    }
    with open(pkl_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"Bundle saved → {pkl_path}")


if __name__ == "__main__":
    main()
