"""cbam_experiment_climate_economic.py

Four-arm climate-economic comparison: min mitigation, BAU, CBAM, max mitigation.

Question
--------
How does the canonical CBAM policy compare to (a) the unmitigated floor,
(b) the optimised baseline without CBAM, and (c) the fully-mitigated ceiling,
in terms of global climate and aggregate economic outcomes?

Design (4 arms)
---------------
  min   — fixed action: μ_r = 0 for all regions, all time. No CBAM. EU
          mitigation schedule disabled.  No training (fixed-action agent).
  bau   — trained PPO. No CBAM (flat τ=0). EU mitigation schedule disabled
          → all agents optimise freely.
  cbam  — canonical: differential CBAM, EU mitigation schedule, RCPO penalty.
          Trained PPO.
  max   — fixed action: μ_r = 0.9 for all regions (the highest mitigation rate
          reachable by the discrete action space with D=10 levels, given
          process_actions divides by D rather than D-1).
          No CBAM. EU schedule disabled. No training.

For min/max, transition_cost_coef is set to 0 to remove the t=0 jump penalty;
these arms represent counterfactual "always-on" mitigation, not realistic
ramp paths.

Outputs
-------
  experiments/cbam_experiment_climate_economic_<TS>/
    plots/cbam_clim_econ_<TS>.pkl
    plots/cbam_clim_econ_<TS>.png             8-panel global indicators
    plots/cbam_clim_econ_<TS>_regional.png    per-region gross output
    logs/cbam_clim_econ_<arm>_s<seed>.csv     (trained arms only)

Usage
-----
    # Full canonical run (2M timesteps × 2 trained arms × 3 seeds)
    python validation/cbam_experiment_climate_economic.py

    # Cheap smoke test
    python validation/cbam_experiment_climate_economic.py --timesteps 200000 --seeds 0

    # Regenerate plots from saved pkl
    python validation/cbam_experiment_climate_economic.py --replot <path.pkl>
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
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from training_monitor import (
    MonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn, i_to_agent_str
from _experiment_util import get_output_dir, get_log_dir, run_single_episode
from validation.canonical_config import (
    NUM_REGIONS,
    EU_REGION_IDX as EU_IDX,
    REGION_NAMES,
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    NUM_EVAL_EPISODES,
    canonical_env_kwargs,
    canonical_train_kwargs,
    make_canonical_env,
)


# ── Config ─────────────────────────────────────────────────────────────────

# Order controls plot legend ordering (low → high mitigation).
ARMS = ("min", "bau", "cbam", "max")
TRAINED_ARMS = ("bau", "cbam")
FIXED_ARMS   = ("min", "max")

# Action indices (with num_discrete_action_levels=10):
#   index i → mitigation rate = i / D   ⇒   index 0 → 0.0, index 9 → 0.9
FIXED_MITIGATION_ACTION = {"min": 0, "max": 9}
# Standard DICE-style savings rate ≈ 0.22 ⇒ action index 2 (= 0.2 after /D=10).
FIXED_SAVINGS_ACTION = 2

_DEFAULT_TIMESTEPS = canonical_train_kwargs()["total_timesteps"]
NUM_ENVS  = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR    = get_log_dir("training_logs")
LOG_PREFIX = "cbam_clim_econ_"
CHECKPOINT_PREFIX = "cbam_clim_econ_ckpt_"

_PPO_KWARGS = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
               if k != "total_timesteps"}


# ── Env build ──────────────────────────────────────────────────────────────

def _build_env(arm: str, *, for_training: bool = True):
    """Build the env for one arm.

    cbam — canonical (differential CBAM + EU schedule + RCPO).
    bau  — canonical setup with CBAM disabled (flat τ=0, λ_init=0) and the
           EU schedule disabled so all agents optimise freely.
    min/max — bau setup + transition_cost_coef=0 (fixed-action eval).
    """
    if arm == "cbam":
        return make_canonical_env(for_training=for_training)

    common = dict(
        cbam_tariff_mode       = "flat",
        cbam_tariff_rate       = 0.0,
        cbam_lambda_init       = 0.0,
        eu_mitigation_schedule = None,
    )
    if arm == "bau":
        return make_canonical_env(for_training=for_training, **common)
    if arm in FIXED_ARMS:
        return make_canonical_env(
            for_training         = for_training,
            transition_cost_coef = 0.0,
            **common,
        )
    raise ValueError(f"unknown arm: {arm}")


# ── Fixed-action agent ─────────────────────────────────────────────────────

class _FixedMRIOAgent:
    """A zero-state agent that always returns the same per-region action dict.

    Designed for the MRIO env's discrete action space:
      export_reallocation : MultiDiscrete([D] * (NS*NR))  — midpoint = δ=0
      savings_rate        : Discrete(D)                   — held at action index 2
      mitigation_rate     : Discrete(D)                   — held at the configured index
    """

    def __init__(self, env, mitigation_action: int,
                 savings_action: int = FIXED_SAVINGS_ACTION,
                 export_action: int | None = None):
        self.env   = env
        self.state = 0  # so RLAlgorithm-compatible callers can do .state
        D  = env.num_discrete_action_levels
        NS = env.num_sectors
        NR = env.num_regions
        if export_action is None:
            export_action = D // 2  # midpoint of discrete grid → δ=0

        per_agent = {
            "export_reallocation": jnp.full((NS * NR,), export_action, dtype=jnp.int32),
            "savings_rate":        jnp.int32(savings_action),
            "mitigation_rate":     jnp.int32(mitigation_action),
        }
        self._action = {i_to_agent_str(i): per_agent for i in range(NR)}

    def get_action(self, key, agent_state, obs):
        return self._action


def _make_fixed_agent(arm: str, env) -> _FixedMRIOAgent:
    return _FixedMRIOAgent(env, mitigation_action=FIXED_MITIGATION_ACTION[arm])


# ── Labelling / logging ────────────────────────────────────────────────────

def _cell_label(arm: str, seed: int) -> str:
    return f"{arm}_s{seed}"


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


# ── Train / eval ───────────────────────────────────────────────────────────

def _train_cell(arm: str, seed: int, total_timesteps: int):
    """Train one trained-arm cell. Returns (agent, csv_path, label)."""
    label = _cell_label(arm, seed)
    env   = _build_env(arm, for_training=True)
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iters)

    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*60}\n  Training: {arm.upper()}  seed={seed}\n{'━'*60}")
    t0  = time.perf_counter()
    ppo = ppo.train(jax.random.PRNGKey(seed), env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path, label


def _to_per_region(d) -> np.ndarray:
    """info[key] is {region_id: 1d-array(T,)} → (T, NR)."""
    return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)


def _eval_cell(eval_seed: int, raw_env, agent) -> dict:
    """Run NUM_EVAL_EPISODES rollouts and extract global + per-region series.

    Note: with carbon_model="base" (canonical), state["global_emissions"] and
    state["global_cumulative_emissions"] are never updated — only carbon mass
    in the atmosphere evolves.  We recompute industrial emissions per step
    from intensity × (1 − μ) × production, which is the actual emissions flow
    being fed into the carbon model.
    """
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    out = {
        "global_temp_atm":          [],
        "global_carbon_atm":        [],
        "industrial_emissions":     [],
        "cumulative_emissions":     [],
        "gross_output":             [],
        "aggregate_consumption":    [],
        "mitigation":               [],
        "damages":                  [],
        "capital":                  [],
        "intensity":                [],
        "production":               [],
        "cbam_cost":                [],
    }

    base_key = jax.random.PRNGKey(eval_seed)
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(base_key, 90_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        out["global_temp_atm"].append(np.array(logs["global_temperature"]["atmosphere"]))
        out["global_carbon_atm"].append(np.array(logs["global_carbon_mass"]["atmosphere"]))

        intensity  = _to_per_region(logs["intensity_all_regions"])
        production = _to_per_region(logs["production_all_regions"])
        mitigation = _to_per_region(logs["mitigation_rates_all_regions"])
        ind_em = (intensity * (1.0 - mitigation) * production).sum(axis=-1)
        out["industrial_emissions"].append(ind_em)
        out["cumulative_emissions"].append(np.cumsum(ind_em))

        out["gross_output"].append(_to_per_region(logs["gross_output_all_regions"]))
        out["aggregate_consumption"].append(_to_per_region(logs["aggregate_consumption"]))
        out["mitigation"].append(mitigation)
        out["damages"].append(_to_per_region(logs["damages_all_regions"]))
        out["capital"].append(_to_per_region(logs["capital_all_regions"]))
        out["intensity"].append(intensity)
        out["production"].append(production)
        if "cbam_cost_all_regions" in logs:
            out["cbam_cost"].append(_to_per_region(logs["cbam_cost_all_regions"]))

    stacked = {}
    for k, v in out.items():
        if v:
            stacked[k] = np.stack(v, axis=0).astype(np.float32)
    return stacked


# ── Grid runner ────────────────────────────────────────────────────────────

def _checkpoint_path(timestamp: str) -> str:
    return _os.path.join(OUTPUT_DIR, f"{CHECKPOINT_PREFIX}{timestamp}.pkl")


def _save_checkpoint(cells: list, timestamp: str) -> None:
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(_checkpoint_path(timestamp), "wb") as fh:
        pickle.dump({"timestamp": timestamp, "cells": cells}, fh)


def _run_grid(seeds, total_timesteps, *, existing_cells=None, timestamp=None):
    completed = {(c["arm"], c["seed"]) for c in (existing_cells or [])}
    cells = list(existing_cells or [])

    for arm in ARMS:
        for seed in seeds:
            if (arm, seed) in completed:
                print(f"  Skipping (done): {_cell_label(arm, seed)}")
                continue

            label = _cell_label(arm, seed)

            if arm in TRAINED_ARMS:
                partial_csv = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
                if _os.path.exists(partial_csv):
                    print(f"  Deleting partial CSV: {partial_csv}")
                    _os.remove(partial_csv)
                agent, csv_path, _ = _train_cell(arm, seed, total_timesteps)
            else:
                print(f"\n{'━'*60}\n  Fixed-action eval: {arm.upper()}  seed={seed}\n{'━'*60}")
                agent    = _make_fixed_agent(arm, _build_env(arm, for_training=False))
                csv_path = None

            raw_env = _build_env(arm, for_training=False)
            ev = _eval_cell(seed + 10_000, raw_env, agent)

            cells.append({
                "arm":      arm,
                "seed":     seed,
                "label":    label,
                "csv_path": csv_path,
                "series":   ev,
            })

            if timestamp is not None:
                _save_checkpoint(cells, timestamp)

    return cells


# ── Aggregation ────────────────────────────────────────────────────────────

def _arm_stack(cells, arm: str, key: str) -> np.ndarray:
    """Stack series across (seed, episode) for one arm.

    Returns array of shape (n_seed * n_ep, T, ...).
    """
    arrs = [c["series"][key] for c in cells if c["arm"] == arm and key in c["series"]]
    if not arrs:
        return np.empty((0,))
    return np.concatenate(arrs, axis=0)


def _mean_band(arr: np.ndarray, axis: int = 0):
    mu = arr.mean(axis=axis)
    sd = arr.std(axis=axis)
    return mu, mu - sd, mu + sd


# ── Plotting ───────────────────────────────────────────────────────────────

# Colour scheme: brown → blue → red → green (low → high mitigation).
ARM_STYLE = {
    "min":  {"color": "#8c564b", "label": "Min mitigation (μ=0)",      "ls": "-"},
    "bau":  {"color": "#1f77b4", "label": "BAU (no CBAM, trained)",    "ls": "-"},
    "cbam": {"color": "#d62728", "label": "CBAM (canonical, trained)", "ls": "-"},
    "max":  {"color": "#2ca02c", "label": "Max mitigation (μ=0.9)",    "ls": "-"},
}


def _years_axis(T: int) -> np.ndarray:
    """RiceMRIO uses 5-year steps starting at 2015."""
    return 2015.0 + 5.0 * np.arange(T)


def _plot_global(cells, timestamp: str) -> str:
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    panels = [
        ("global_temp_atm",        "Global atmospheric temperature",     "°C above pre-industrial",  False),
        ("industrial_emissions",   "Global industrial emissions (flow)", "GtC / 5-yr step",          False),
        ("cumulative_emissions",   "Cumulative industrial emissions",    "GtC (eval-integrated)",    False),
        ("global_carbon_atm",      "Atmospheric carbon mass",            "GtC",                      False),
        ("gross_output",           "World gross output (sum of regions)","trillion USD",             True),
        ("aggregate_consumption",  "World consumption (sum of regions)", "trillion USD",             True),
        ("mitigation",             "Mean mitigation rate (all regions)", "fraction",                 False),
        ("damages",                "Mean damages (1 − damages factor)",  "fraction",                 False),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharex=True)
    axes = axes.flatten()

    for ax, (key, title, ylabel, sum_over_regions) in zip(axes, panels):
        for arm in ARMS:
            style = ARM_STYLE[arm]
            arr   = _arm_stack(cells, arm, key)
            if arr.size == 0:
                continue
            if arr.ndim == 3:
                if sum_over_regions:
                    arr = arr.sum(axis=-1)
                elif key == "damages":
                    arr = (1.0 - arr).mean(axis=-1)
                else:
                    arr = arr.mean(axis=-1)
            T = arr.shape[1]
            t = _years_axis(T)
            mu, lo, hi = _mean_band(arr, axis=0)
            ax.plot(t, mu, color=style["color"], ls=style["ls"], lw=2.0,
                    label=style["label"])
            ax.fill_between(t, lo, hi, color=style["color"], alpha=0.15, linewidth=0)

        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Year")
    axes[-2].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("Climate-economic indicators across mitigation regimes "
                 "(mean ± 1 SD across seeds × eval episodes)",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = _os.path.join(OUTPUT_DIR, f"cbam_clim_econ_{timestamp}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Global plot   → {out}")
    return out


def _plot_regional(cells, timestamp: str) -> str:
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(13, 11), sharex=True)
    axes = axes.flatten()

    for r in range(NUM_REGIONS):
        ax = axes[r]
        for arm in ARMS:
            style = ARM_STYLE[arm]
            arr   = _arm_stack(cells, arm, "gross_output")     # (S, T, NR)
            if arr.size == 0:
                continue
            series = arr[:, :, r]
            T = series.shape[1]
            t = _years_axis(T)
            mu, lo, hi = _mean_band(series, axis=0)
            ax.plot(t, mu, color=style["color"], ls=style["ls"], lw=1.6,
                    label=style["label"])
            ax.fill_between(t, lo, hi, color=style["color"], alpha=0.15, linewidth=0)

        marker = " (EU)" if r == EU_IDX else ""
        ax.set_title(f"{REGION_NAMES.get(r, f'r{r}')}{marker}", fontsize=10)
        ax.set_ylabel("gross output (T USD)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for ax in axes[-3:]:
        ax.set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.985), frameon=False)
    fig.suptitle("Per-region gross output across mitigation regimes",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = _os.path.join(OUTPUT_DIR, f"cbam_clim_econ_{timestamp}_regional.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Regional plot → {out}")
    return out


# ── Summary table ──────────────────────────────────────────────────────────

def _final_step_summary(cells) -> pd.DataFrame:
    rows = []
    last_t = 5

    def _last(arr, axis_T=-1):
        return arr.take(np.arange(arr.shape[axis_T] - last_t, arr.shape[axis_T]),
                        axis=axis_T)

    for arm in ARMS:
        rec = {"arm": arm}
        T_series = _arm_stack(cells, arm, "global_temp_atm")
        if T_series.size:
            rec["temp_final"] = float(_last(T_series).mean())
        emit = _arm_stack(cells, arm, "industrial_emissions")
        if emit.size:
            rec["emissions_final"] = float(_last(emit).mean())
        cum = _arm_stack(cells, arm, "cumulative_emissions")
        if cum.size:
            rec["cum_emissions_final"] = float(_last(cum).mean())
        y = _arm_stack(cells, arm, "gross_output")
        if y.size:
            rec["world_Y_final"] = float(_last(y, axis_T=1).sum(-1).mean())
        c = _arm_stack(cells, arm, "aggregate_consumption")
        if c.size:
            rec["world_C_final"] = float(_last(c, axis_T=1).sum(-1).mean())
        mu = _arm_stack(cells, arm, "mitigation")
        if mu.size:
            rec["mean_mitigation_final"] = float(_last(mu, axis_T=1).mean())
        rows.append(rec)

    return pd.DataFrame(rows)


# ── Bundle ─────────────────────────────────────────────────────────────────

def _make_bundle(cells, summary, args):
    return {
        "experiment_id":   "climate_economic_compare",
        "timestamp":       datetime.now().isoformat(),
        "env_kwargs":      canonical_env_kwargs(),
        "train_kwargs":    {**canonical_train_kwargs(),
                            "total_timesteps": args.timesteps},
        "seeds":           tuple(args.seeds),
        "arms":            ARMS,
        "trained_arms":    TRAINED_ARMS,
        "fixed_arms":      FIXED_ARMS,
        "cells":           cells,
        "summary":         summary.to_dict("records"),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=_DEFAULT_TIMESTEPS,
                        help=f"Per-trained-cell training timesteps "
                             f"(default {_DEFAULT_TIMESTEPS})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS),
                        help=f"Seeds for all arms (default {list(CANONICAL_SEEDS)}). "
                             f"For fixed arms only the eval RNG varies.")
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing .pkl — skip training, regenerate plots only")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pkl — skip completed cells and continue")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            bundle = pickle.load(f)
        cells = bundle["cells"]
        _plot_global(cells, timestamp)
        _plot_regional(cells, timestamp)
        summary = _final_step_summary(cells)
        print("\n" + summary.to_string(index=False))
        return

    existing_cells = None
    if args.resume:
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        existing_cells = ckpt["cells"]
        timestamp = ckpt["timestamp"]
        print(f"Resuming: {len(existing_cells)}/{len(ARMS) * len(args.seeds)} cells done")

    n_trained = len(TRAINED_ARMS) * len(args.seeds)
    n_fixed   = len(FIXED_ARMS)   * len(args.seeds)
    print("═" * 60)
    print(f"  Climate-economic comparison ({', '.join(ARMS)})")
    print(f"  seeds={args.seeds}  timesteps/trained-cell={args.timesteps:,}")
    print(f"  trained cells: {n_trained}   fixed-action cells: {n_fixed}")
    print("═" * 60)

    cells = _run_grid(args.seeds, args.timesteps,
                      existing_cells=existing_cells, timestamp=timestamp)
    summary = _final_step_summary(cells)
    bundle = _make_bundle(cells, summary, args)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_clim_econ_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\nPickle saved → {pkl_path}")

    ckpt_path = _checkpoint_path(timestamp)
    if _os.path.exists(ckpt_path):
        _os.remove(ckpt_path)

    _plot_global(cells, timestamp)
    _plot_regional(cells, timestamp)

    print("\n" + "═" * 60)
    print("  Final-step summary (mean over last 5 env steps × seeds × eps)")
    print("═" * 60)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
