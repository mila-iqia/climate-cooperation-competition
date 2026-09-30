"""cbam_experiment_A_crowdout.py

Experiment A — Direct crowd-out attenuation test (audit §6.3, registry id
"A_crowd_out_redist", claim_bucket="policy-design").

Question
--------
Does revenue redistribution reduce the mitigation lost to diversion when
both margins are open?

Design
------
For each (revenue_share, transfer_mode, seed) cell, train two policies on
the canonical 9-region differential-CBAM environment:

  PINNED — `delta_max=0`. Export reallocation is shut off; only mitigation
           and savings respond to CBAM.  Yields μ_pinned per region.
  OPEN   — `delta_max=3.0`. Diversion channel is open; mitigation must
           compete with diversion.  Yields μ_open per region.

The direct crowd-out gap is then

    crowd_out_gap   = μ_pinned − μ_open                                (per seed)
    attenuation     = gap(revenue_share=0) − gap(revenue_share=1)      (per mode, seed)

Conditions
----------
  revenue_share  ∈ {0.0, 1.0}
  transfer_mode  ∈ {"consumption", "abatement"}
  seeds          = CANONICAL_SEEDS (default (0, 1, 2))
  pinned/open    : delta_max ∈ {0.0, 3.0}

Total: 2 × 2 × 3 × 2 = 24 training runs.

Pass criterion (registry)
-------------------------
  attenuation > 0 on the worst seed for at least one transfer_mode, AND
  attenuation_abatement >= 2 × attenuation_consumption on the worst seed.

Usage
-----
    # 2M-timestep canonical run (24 × 2M = 48M total)
    python validation/cbam_experiment_A_crowdout.py

    # Cheap smoke test (1 seed, short)
    python validation/cbam_experiment_A_crowdout.py --timesteps 200000 --seeds 0

    # Replot from saved pkl
    python validation/cbam_experiment_A_crowdout.py --replot <pkl_path>
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
from validation.registry import REGISTRY


# ── Config ─────────────────────────────────────────────────────────────────

EXPERIMENT_ID = "A_crowd_out_redist"
_REGISTRY     = REGISTRY[EXPERIMENT_ID]

CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)

# Experiment grid
REVENUE_SHARES = (0.0, 1.0)
TRANSFER_MODES = ("consumption", "abatement")
DELTA_MAX_PINNED = 0.0
DELTA_MAX_OPEN   = 3.0

# Defaults derived from canonical config
_DEFAULT_TIMESTEPS = canonical_train_kwargs()["total_timesteps"]
NUM_ENVS  = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]

OUTPUT_DIR        = get_output_dir("plots")
LOG_DIR           = get_log_dir("training_logs")
LOG_PREFIX        = "cbam_A_"
CHECKPOINT_PREFIX = "cbam_A_ckpt_"

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

def _build_env(revenue_share: float, transfer_mode: str, delta_max: float,
               *, for_training: bool = True):
    """Build a canonical differential-CBAM env with the experiment knobs."""
    return make_canonical_env(
        for_training            = for_training,
        revenue_share           = revenue_share,
        transfer_mode           = transfer_mode,
        transfer_allocation     = "effort",
        delta_max               = delta_max,
    )


def _cell_label(revenue_share: float, transfer_mode: str, pinned: bool,
                seed: int) -> str:
    arm = "pinned" if pinned else "open"
    return f"rs{int(revenue_share*100):03d}_{transfer_mode[:4]}_{arm}_s{seed}"


def _make_log_fn(label: str, num_iters: int):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _SCREEN_DROP = {"action_mean", "action_var"}

    def _compact_print_fn(data: dict, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _SCREEN_DROP}, iteration)

    log_fn = make_combined_log_fn(
        _compact_print_fn,         # tqdm bar: reward + ep_ret only
        make_csv_log_fn(csv_path), # CSV: full data including actions
    )
    return log_fn, csv_path


def _train_cell(revenue_share: float, transfer_mode: str, pinned: bool,
                seed: int, total_timesteps: int):
    """Train one cell of the grid. Returns (trained_ppo, csv_path, label)."""
    label = _cell_label(revenue_share, transfer_mode, pinned, seed)
    delta_max = DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN

    env = _build_env(revenue_share, transfer_mode, delta_max, for_training=True)
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iters)

    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*60}")
    print(f"  Training A: {label}")
    print(f"    rs={revenue_share:.2f}  mode={transfer_mode}  "
          f"arm={'PINNED' if pinned else 'OPEN'}  seed={seed}")
    print(f"{'━'*60}")
    t0  = time.perf_counter()
    ppo = ppo.train(jax.random.PRNGKey(seed), env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path, label


# ── Eval ───────────────────────────────────────────────────────────────────

def _eval_cell(eval_seed: int, raw_env, agent):
    """Run NUM_EVAL_EPISODES rollouts and stack (trade_flows, mitigation)."""
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

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
        "trade_flows": np.stack(flows, 0),     # (n_ep, T, NR, NR, NS)
        "mitigation":  np.stack(mit,   0),     # (n_ep, T, NR)
    }


# ── Grid runner ────────────────────────────────────────────────────────────

def _run_grid(seeds, total_timesteps, *, existing_cells=None, timestamp=None):
    """Train all cells and return a list of dicts with raw metric arrays.

    Parameters
    ----------
    existing_cells : list or None
        Cells already completed (from a checkpoint). These are skipped.
    timestamp : str or None
        When set, a checkpoint is written after every completed cell so the
        run can be resumed after an interrupt.
    """
    completed = {
        (c["seed"], c["revenue_share"], c["transfer_mode"], c["pinned"])
        for c in (existing_cells or [])
    }
    cells = list(existing_cells or [])

    for seed in seeds:
        for rs in REVENUE_SHARES:
            for tm in TRANSFER_MODES:
                for pinned in (True, False):
                    if (seed, rs, tm, pinned) in completed:
                        print(f"  Skipping (done): {_cell_label(rs, tm, pinned, seed)}")
                        continue

                    # Delete any partial CSV from a previous interrupted run.
                    partial_csv = _os.path.join(
                        LOG_DIR, f"{LOG_PREFIX}{_cell_label(rs, tm, pinned, seed)}.csv"
                    )
                    if _os.path.exists(partial_csv):
                        print(f"  Deleting partial CSV: {partial_csv}")
                        _os.remove(partial_csv)

                    ppo, csv_path, label = _train_cell(
                        rs, tm, pinned, seed, total_timesteps,
                    )
                    raw_env = _build_env(
                        rs, tm,
                        DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN,
                        for_training=False,
                    )
                    ev = _eval_cell(seed + 10_000, raw_env, ppo)

                    mu_aggregate = M.mean_mitigation_rate(
                        ev["mitigation"], region_idxs=CBAM_PLOT_REGIONS,
                    )
                    mu_per_region = M.per_region_mitigation_rate(ev["mitigation"])
                    eu_share_agg = M.eu_dirty_export_share(
                        ev["trade_flows"], eu_region_idx=EU_IDX,
                        exporter_idxs=CBAM_PLOT_REGIONS,
                    )

                    cells.append({
                        "seed":             seed,
                        "revenue_share":    rs,
                        "transfer_mode":    tm,
                        "pinned":           pinned,
                        "delta_max":        DELTA_MAX_PINNED if pinned else DELTA_MAX_OPEN,
                        "label":            label,
                        "csv_path":         csv_path,
                        "mu_non_eu":        mu_aggregate,
                        "mu_per_region":    mu_per_region,
                        "eu_dirty_share":   eu_share_agg,
                        # Keep raw arrays for post-hoc reanalysis. Lossless PKLs
                        # so any aggregation choice can be revisited without retraining.
                        "mitigation_raw":   ev["mitigation"].astype(np.float32),
                        "trade_flows_raw":  ev["trade_flows"].astype(np.float32),
                    })

                    if timestamp is not None:
                        _save_checkpoint(cells, timestamp)

    return cells


# ── Aggregation ────────────────────────────────────────────────────────────

def _crowd_out_table(cells):
    """Build a long-form DataFrame and the headline summary.

    Returns
    -------
    df_cells : pd.DataFrame
        One row per (seed, rs, mode, arm).
    df_gap : pd.DataFrame
        One row per (seed, rs, mode) with crowd_out_gap = μ_pinned − μ_open.
    df_attn : pd.DataFrame
        One row per (seed, mode) with attenuation = gap_rs0 − gap_rs1.
    """
    df_cells = pd.DataFrame([
        {k: c[k] for k in (
            "seed", "revenue_share", "transfer_mode", "pinned",
            "mu_non_eu", "eu_dirty_share", "label",
        )} for c in cells
    ])

    # μ_pinned and μ_open per (seed, rs, mode)
    pivot = (df_cells
             .pivot_table(index=["seed", "revenue_share", "transfer_mode"],
                          columns="pinned", values="mu_non_eu")
             .reset_index())
    pivot = pivot.rename(columns={True: "mu_pinned", False: "mu_open"})
    pivot["crowd_out_gap"] = pivot.apply(
        lambda r: M.crowd_out_gap(r["mu_pinned"], r["mu_open"]),
        axis=1,
    )

    # attenuation = gap(rs=0) − gap(rs=1)  per (seed, mode)
    g0 = pivot[pivot["revenue_share"] == 0.0].set_index(["seed", "transfer_mode"])["crowd_out_gap"]
    g1 = pivot[pivot["revenue_share"] == 1.0].set_index(["seed", "transfer_mode"])["crowd_out_gap"]
    attn = (g0 - g1).rename("attenuation").reset_index()

    return df_cells, pivot, attn


def _check_pass_criterion(attn: pd.DataFrame) -> dict:
    """Apply the registry pass criterion to the attenuation table."""
    by_mode = attn.groupby("transfer_mode")["attenuation"]
    summary = {}
    for mode, vals in by_mode:
        v = np.array(vals)
        summary[mode] = {
            "mean":  float(v.mean()),
            "min":   float(v.min()),
            "max":   float(v.max()),
            "n":     int(v.shape[0]),
        }

    worst_consumption = summary.get("consumption", {}).get("min", 0.0)
    worst_abatement   = summary.get("abatement",   {}).get("min", 0.0)

    # Registry pass criterion:
    #   attenuation > 0 (worst seed) for at least one mode
    #   AND attenuation_abatement >= 2 × attenuation_consumption (worst seed)
    any_positive = max(worst_consumption, worst_abatement) > 0.0
    abatement_dominates = (
        worst_consumption is not None
        and worst_abatement >= 2.0 * max(worst_consumption, 1e-9)
    )
    passed = bool(any_positive and abatement_dominates)
    return {
        "passed":              passed,
        "any_positive":        any_positive,
        "abatement_dominates": abatement_dominates,
        "summary":             summary,
    }


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_results(cells, pivot, attn, pass_info, timestamp):
    nseeds   = len(set(c["seed"] for c in cells))
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Experiment A — Crowd-out attenuation under redistribution\n"
        f"differential CBAM, 9-region vuln. setup, {nseeds} seed(s)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel 1: μ_pinned vs μ_open per (rs, mode), mean ± seed range
    ax = fig.add_subplot(gs[0, 0])
    grp = pivot.groupby(["revenue_share", "transfer_mode"]).agg(
        mu_pinned_mean=("mu_pinned", "mean"),
        mu_open_mean  =("mu_open",   "mean"),
        mu_pinned_min =("mu_pinned", "min"),
        mu_pinned_max =("mu_pinned", "max"),
        mu_open_min   =("mu_open",   "min"),
        mu_open_max   =("mu_open",   "max"),
    ).reset_index()
    labels = [f"rs={r:.0f} | {m[:4]}" for r, m in zip(grp["revenue_share"], grp["transfer_mode"])]
    x = np.arange(len(labels))
    ax.errorbar(x - 0.1, grp["mu_pinned_mean"],
                yerr=[grp["mu_pinned_mean"] - grp["mu_pinned_min"],
                      grp["mu_pinned_max"] - grp["mu_pinned_mean"]],
                fmt="o", color="tab:blue", label="μ pinned (Δ=0)", capsize=4)
    ax.errorbar(x + 0.1, grp["mu_open_mean"],
                yerr=[grp["mu_open_mean"] - grp["mu_open_min"],
                      grp["mu_open_max"] - grp["mu_open_mean"]],
                fmt="s", color="tab:red", label="μ open (Δ=3)", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean mitigation rate μ (non-EU exporters)")
    ax.set_title("Mitigation: pinned vs open per condition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 2: crowd_out_gap per (rs, mode) — seed dots
    ax = fig.add_subplot(gs[0, 1])
    for i, (rs, mode) in enumerate(
        [(r, m) for r in REVENUE_SHARES for m in TRANSFER_MODES]
    ):
        vals = pivot[(pivot["revenue_share"] == rs)
                     & (pivot["transfer_mode"] == mode)]["crowd_out_gap"]
        ax.scatter([i] * len(vals), vals, alpha=0.7,
                   color="tab:purple" if mode == "abatement" else "tab:gray")
        ax.scatter([i], [vals.mean()], marker="_", s=200, color="black")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"rs={r:.0f}\n{m[:4]}" for r in REVENUE_SHARES for m in TRANSFER_MODES])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("crowd_out_gap = μ_pinned − μ_open")
    ax.set_title("Crowd-out gap per condition (one dot = one seed)")
    ax.grid(alpha=0.3)

    # Panel 3: attenuation per mode — seed dots
    ax = fig.add_subplot(gs[1, 0])
    for i, mode in enumerate(TRANSFER_MODES):
        vals = attn[attn["transfer_mode"] == mode]["attenuation"]
        ax.scatter([i] * len(vals), vals, alpha=0.7,
                   color="tab:green" if mode == "abatement" else "tab:gray")
        ax.scatter([i], [vals.mean()], marker="_", s=200, color="black")
    ax.set_xticks(range(len(TRANSFER_MODES)))
    ax.set_xticklabels(TRANSFER_MODES)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("attenuation = gap(rs=0) − gap(rs=1)")
    ax.set_title("Crowd-out attenuation by transfer mode\n(one dot = one seed; >0 means transfers reduce crowd-out)")
    ax.grid(alpha=0.3)

    # Panel 4: pass-criterion badge
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    lines = [
        f"Experiment: {EXPERIMENT_ID}",
        f"Claim bucket: {_REGISTRY.claim_bucket}",
        f"Seeds: {sorted(set(c['seed'] for c in cells))}",
        "",
        "Worst-seed summary (attenuation):",
    ]
    for mode, s in pass_info["summary"].items():
        lines.append(f"  {mode:12s}  min={s['min']:+.4f}  "
                     f"mean={s['mean']:+.4f}  max={s['max']:+.4f}")
    lines += [
        "",
        f"any_positive (worst-seed):     {pass_info['any_positive']}",
        f"abatement ≥ 2× consumption:   {pass_info['abatement_dominates']}",
        "",
        f"REGISTRY PASS: {'PASS' if pass_info['passed'] else 'FAIL'}",
        "",
        "Interpretation limits:",
        *[f"  {ln}" for ln in _REGISTRY.interpretation_limits.split('. ') if ln],
    ]
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, va="top", fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray"))

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_A_crowdout_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Bundle / provenance ────────────────────────────────────────────────────

def _make_bundle(cells, pivot, attn, pass_info, args):
    """Lossless PKL bundle for downstream reanalysis."""
    return {
        "experiment_id":   EXPERIMENT_ID,
        "registry_entry":  {
            "question":              _REGISTRY.question,
            "claim_bucket":          _REGISTRY.claim_bucket,
            "primary_metric":        _REGISTRY.primary_metric,
            "pass_criterion":        _REGISTRY.pass_criterion,
            "interpretation_limits": _REGISTRY.interpretation_limits,
        },
        "env_kwargs":      canonical_env_kwargs(),
        "train_kwargs":    {**canonical_train_kwargs(),
                            "total_timesteps": args.timesteps},
        "seeds":           tuple(args.seeds),
        "cells":           cells,
        "pivot":           pivot.to_dict("records"),
        "attenuation":     attn.to_dict("records"),
        "pass_info":       pass_info,
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=_DEFAULT_TIMESTEPS,
                        help=f"Per-cell training timesteps (default {_DEFAULT_TIMESTEPS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS),
                        help=f"Training seeds (default {list(CANONICAL_SEEDS)})")
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing .pkl — skip training, regenerate plot only")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl — skip completed cells and continue")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            bundle = pickle.load(f)
        cells = bundle["cells"]
        df_cells, pivot, attn = _crowd_out_table(cells)
        pass_info = _check_pass_criterion(attn)
        _plot_results(cells, pivot, attn, pass_info, timestamp)
        return

    existing_cells = None
    if args.resume:
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        existing_cells = ckpt["cells"]
        timestamp = ckpt["timestamp"]  # keep original timestamp so filenames stay consistent
        print(f"Resuming: {len(existing_cells)}/{len(args.seeds) * 8} cells already done")

    print("═" * 60)
    print(f"  Experiment A — {EXPERIMENT_ID}")
    print(f"  seeds={args.seeds}  timesteps/cell={args.timesteps:,}")
    print(f"  total cells: {len(args.seeds)} × 2 (rs) × 2 (mode) × 2 (arm) "
          f"= {len(args.seeds) * 8}")
    print("═" * 60)

    cells = _run_grid(args.seeds, args.timesteps,
                      existing_cells=existing_cells, timestamp=timestamp)
    df_cells, pivot, attn = _crowd_out_table(cells)
    pass_info = _check_pass_criterion(attn)

    bundle = _make_bundle(cells, pivot, attn, pass_info, args)
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_A_crowdout_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nPickle saved → {pkl_path}")

    # Remove checkpoint now that the final bundle is written.
    ckpt_path = _checkpoint_path(timestamp)
    if _os.path.exists(ckpt_path):
        _os.remove(ckpt_path)

    _plot_results(cells, pivot, attn, pass_info, timestamp)

    # Headline summary
    print("\n" + "═" * 60)
    print(f"  EXPERIMENT A — pass criterion: "
          f"{'✅ PASS' if pass_info['passed'] else '❌ FAIL'}")
    for mode, s in pass_info["summary"].items():
        print(f"    {mode:12s}  attenuation: "
              f"min={s['min']:+.4f}  mean={s['mean']:+.4f}  max={s['max']:+.4f}")
    print("═" * 60)


if __name__ == "__main__":
    main()
