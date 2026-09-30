"""C-MAC ceiling: free versus costly mitigation under differential CBAM.

Both the mitigation and diversion channels remain open. The only difference
between cells is whether abatement cost is enabled, providing a fast scoping
estimate of the mitigation gap and the financing scale behind it.

Usage (from rice_jax/)
----------------------
    python cbam/drivers/cbam_experiment_C_mac_ceiling.py --timesteps 200000
    python cbam/drivers/cbam_experiment_C_mac_ceiling.py --replot <pkl>
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jaxnasium as jym
import matplotlib.pyplot as plt
import numpy as np

jym.enable_compilation_cache()
import jax

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

from _experiment_util import (
    get_log_dir,
    get_output_dir,
    run_single_episode,
    with_log_info_fn,
)
from cbam.config import metrics as M
from cbam.config.canonical_config import (
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    EVAL_LAST_T,
    EU_REGION_IDX as EU_IDX,
    NON_EU_EXPORTER_IDXS,
    NUM_EVAL_EPISODES,
    NUM_REGIONS,
    REGION_NAMES,
    canonical_env_kwargs,
    make_canonical_env,
)
from rice_jax.training import (
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn


EXPERIMENT_ID = "C_mac_ceiling"
CELL_MODES = ("costly", "free")
OUTPUT_DIR = get_output_dir("plots")
LOG_DIR = get_log_dir("training_logs")
LOG_PREFIX = "cbam_C_mac_"
RUN_LOG_NAME = "run.log"
DEFAULT_TIMESTEPS = CANONICAL_TRAIN_KWARGS["total_timesteps"]
NUM_ENVS = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]
PPO_KWARGS = {
    key: value
    for key, value in CANONICAL_TRAIN_KWARGS.items()
    if key != "total_timesteps"
}
NUM_ENVS = 16
PPO_KWARGS["num_envs"] = NUM_ENVS


def _log(message: str) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, RUN_LOG_NAME), "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _build_env(mode: str, *, for_training: bool):
    return make_canonical_env(
        for_training=for_training,
        zero_abatement_cost=mode == "free",
    )


def _make_log_fn(label: str, num_iterations: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    print_fn = make_print_log_fn(
        num_iterations=num_iterations,
        description=f"Training {label}",
    )

    def compact_print(data, iteration):
        print_fn(
            {
                key: value
                for key, value in data.items()
                if key not in {"action_mean", "action_var"}
            },
            iteration,
        )

    return make_combined_log_fn(compact_print, make_csv_log_fn(csv_path)), csv_path


def _train_cell(mode: str, seed: int, total_timesteps: int):
    label = f"{mode}_s{seed}"
    env = _build_env(mode, for_training=True)
    num_iterations = total_timesteps // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iterations)
    ppo = RCPOMonitoredPPO(
        total_timesteps=total_timesteps,
        log_function=log_fn,
        **PPO_KWARGS,
    )
    _log(
        f"train_start cell={label} timesteps={total_timesteps} "
        f"iterations={num_iterations} num_envs={NUM_ENVS} num_steps={NUM_STEPS}"
    )
    print(f"\n{'-' * 60}")
    print(f"  Training C-MAC ceiling: {label}")
    print("    differential CBAM, both channels open, transfers disabled")
    print(f"    zero_abatement_cost={mode == 'free'}, seed={seed}")
    print(f"{'-' * 60}")
    started = time.perf_counter()
    agent, _metrics = ppo.train(jax.random.PRNGKey(seed), env)
    elapsed = time.perf_counter() - started
    _log(f"train_end cell={label} elapsed_s={elapsed:.1f}")
    return agent, csv_path, label


def _to_region_array(values):
    if isinstance(values, dict):
        return np.stack([np.asarray(values[index]) for index in range(NUM_REGIONS)], axis=-1)
    return np.asarray(values)


def _eval_cell(mode: str, seed: int, agent):
    started = time.perf_counter()
    _log(f"eval_start cell={mode}_s{seed} episodes={NUM_EVAL_EPISODES}")
    eval_env = with_log_info_fn(
        _build_env(mode, for_training=False), full_state_info_log_fn
    )
    logs_by_key = {
        "trade_flows": [],
        "mitigation": [],
        "abatement_cost": [],
        "gross_output": [],
        "cbam_revenue": [],
    }
    base_key = jax.random.PRNGKey(seed + 10_000)
    for episode_id in range(NUM_EVAL_EPISODES):
        episode_key = jax.random.fold_in(base_key, 90_000 + episode_id)
        logs = run_single_episode(episode_key, eval_env, agent)
        logs_by_key["trade_flows"].append(np.asarray(logs["trade_flows"]))
        logs_by_key["mitigation"].append(
            _to_region_array(logs["mitigation_rates_all_regions"])
        )
        logs_by_key["abatement_cost"].append(
            _to_region_array(logs["abatement_cost_all_regions"])
        )
        logs_by_key["gross_output"].append(
            _to_region_array(logs["gross_output_all_regions"])
        )
        logs_by_key["cbam_revenue"].append(
            _to_region_array(logs["cbam_revenue"])
        )
    result = {key: np.stack(value, axis=0) for key, value in logs_by_key.items()}
    _log(f"eval_end cell={mode}_s{seed} elapsed_s={time.perf_counter() - started:.1f}")
    return result


def _last_t_mean(values: np.ndarray) -> np.ndarray:
    return values[:, -EVAL_LAST_T:].mean(axis=(0, 1))


def _summarize(mode: str, seed: int, csv_path: str, eval_data: dict):
    mitigation = eval_data["mitigation"]
    trade_flows = eval_data["trade_flows"]
    abatement_cost = _last_t_mean(eval_data["abatement_cost"])
    gross_output = _last_t_mean(eval_data["gross_output"])
    cbam_revenue = _last_t_mean(eval_data["cbam_revenue"])
    cost_value = abatement_cost * gross_output
    exporter_cost = float(cost_value[list(NON_EU_EXPORTER_IDXS)].sum())
    cbam_revenue_total = float(cbam_revenue[EU_IDX])
    return {
        "mode": mode,
        "seed": seed,
        "csv_path": csv_path,
        "mu_non_eu": M.mean_mitigation_rate(
            mitigation, region_idxs=NON_EU_EXPORTER_IDXS
        ),
        "mu_per_region": M.per_region_mitigation_rate(mitigation),
        "eu_dirty_share": M.eu_dirty_export_share(
            trade_flows,
            eu_region_idx=EU_IDX,
            exporter_idxs=NON_EU_EXPORTER_IDXS,
        ),
        "abatement_cost_fraction_per_region": abatement_cost,
        "gross_output_per_region": gross_output,
        "cbam_revenue_per_region": cbam_revenue,
        "abatement_cost_value_exporters": exporter_cost,
        "cbam_revenue_total": cbam_revenue_total,
        "abatement_cost_to_cbam_revenue": exporter_cost / max(cbam_revenue_total, 1e-10),
        "mitigation_raw": mitigation.astype(np.float32),
        "trade_flows_raw": trade_flows.astype(np.float32),
    }


def _run(seed: int, total_timesteps: int):
    _log(
        f"run_start experiment={EXPERIMENT_ID} seed={seed} timesteps={total_timesteps} "
        f"cells={','.join(CELL_MODES)} differential_cbam=True delta_max=3.0"
    )
    cells = []
    for mode in CELL_MODES:
        agent, csv_path, _label = _train_cell(mode, seed, total_timesteps)
        eval_data = _eval_cell(mode, seed, agent)
        cell = _summarize(mode, seed, csv_path, eval_data)
        cells.append(cell)
        _log(
            f"cell_summary cell={mode}_s{seed} mu_non_eu={cell['mu_non_eu']:.6f} "
            f"eu_dirty_share={cell['eu_dirty_share']:.6f} "
            f"cost_to_cbam={cell['abatement_cost_to_cbam_revenue']:.6f}"
        )
    _log(f"run_end experiment={EXPERIMENT_ID} cells={len(cells)}")
    return cells


def _print_report(cells):
    by_mode = {cell["mode"]: cell for cell in cells}
    costly = by_mode["costly"]
    free = by_mode["free"]
    mu_gap = free["mu_non_eu"] - costly["mu_non_eu"]
    cost_gap = (
        free["abatement_cost_value_exporters"]
        - costly["abatement_cost_value_exporters"]
    )
    print("\n" + "=" * 64)
    print("  EXPERIMENT C-MAC CEILING")
    print("  Differential CBAM; mitigation and diversion channels open")
    print("=" * 64)
    print(f"  {'Cell':<10} {'mu_non_eu':>12} {'EU dirty share':>16} {'cost/Y':>12}")
    for cell in (costly, free):
        print(
            f"  {cell['mode']:<10} {cell['mu_non_eu']:>12.4f} "
            f"{cell['eu_dirty_share']:>16.4f} "
            f"{cell['abatement_cost_value_exporters']:>12.4f}"
        )
    print(f"\n  Free-minus-costly mitigation gap: {mu_gap:.4f}")
    print(f"  Cost difference (free minus costly): {cost_gap:.4f}")
    print(
        "  Costly-cell abatement cost / exporter CBAM revenue: "
        f"{costly['abatement_cost_to_cbam_revenue']:.2f}x"
    )
    print("=" * 64 + "\n")


def _plot(cells, output_path: str):
    by_mode = {cell["mode"]: cell for cell in cells}
    regions = list(NON_EU_EXPORTER_IDXS)
    labels = [REGION_NAMES[index] for index in regions]
    positions = np.arange(len(regions))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for offset, mode, color in ((-width / 2, "costly", "#b44c3b"), (width / 2, "free", "#2f7f75")):
        values = [by_mode[mode]["mu_per_region"][index] for index in regions]
        axes[0, 0].bar(positions + offset, values, width, label=mode, color=color)
    axes[0, 0].set_title("Mitigation by exporter")
    axes[0, 0].set_ylabel("mu")
    axes[0, 0].set_xticks(positions, labels, rotation=30, ha="right")
    axes[0, 0].legend()

    axes[0, 1].bar(
        [0, 1],
        [by_mode[mode]["eu_dirty_share"] for mode in ("costly", "free")],
        color=["#b44c3b", "#2f7f75"],
    )
    axes[0, 1].set_title("EU dirty export share")
    axes[0, 1].set_xticks([0, 1], ["costly", "free"])
    axes[0, 1].set_ylabel("share")

    axes[1, 0].bar(
        positions,
        [by_mode["costly"]["abatement_cost_fraction_per_region"][index] for index in regions],
        color="#b44c3b",
    )
    axes[1, 0].set_title("Realized costly-cell abatement cost")
    axes[1, 0].set_ylabel("fraction of gross output")
    axes[1, 0].set_xticks(positions, labels, rotation=30, ha="right")

    costly = by_mode["costly"]
    free = by_mode["free"]
    summary = (
        f"mu gap (free - costly): {free['mu_non_eu'] - costly['mu_non_eu']:.4f}\n"
        f"costly exporter cost / CBAM revenue: "
        f"{costly['abatement_cost_to_cbam_revenue']:.2f}x\n"
        f"costly exporter abatement cost: {costly['abatement_cost_value_exporters']:.4f}\n"
        f"free exporter abatement cost: {free['abatement_cost_value_exporters']:.4f}"
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.95, summary, va="top", family="monospace", fontsize=11)
    fig.suptitle("C-MAC ceiling: differential CBAM with both channels open")
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=CANONICAL_SEEDS[0])
    parser.add_argument("--replot", metavar="PKL")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = os.path.join(OUTPUT_DIR, f"cbam_C_mac_ceiling_{timestamp}")
    try:
        if args.replot:
            _log(f"replot_start source={args.replot}")
            with open(args.replot, "rb") as handle:
                payload = pickle.load(handle)
            cells = payload["cells"]
        else:
            _log(
                f"config num_envs={NUM_ENVS} num_steps={NUM_STEPS} "
                f"timesteps={args.timesteps} seed={args.seed} cache_enabled=True"
            )
            cells = _run(args.seed, args.timesteps)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            pkl_path = f"{output_prefix}.pkl"
            with open(pkl_path, "wb") as handle:
                pickle.dump(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "timestamp": timestamp,
                        "cells": cells,
                        "seed": args.seed,
                        "timesteps": args.timesteps,
                        "canonical_env": canonical_env_kwargs(),
                        "cbam_tariff_mode": "differential",
                        "delta_max": 3.0,
                        "zero_abatement_cost_cells": CELL_MODES,
                    },
                    handle,
                )
            _log(f"pkl_saved path={pkl_path}")
    except Exception:
        _log("run_failed traceback_follows")
        with open(os.path.join(LOG_DIR, RUN_LOG_NAME), "a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        raise

    _print_report(cells)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = f"{output_prefix}.png"
    _plot(cells, plot_path)
    print(f"  Plot saved -> {plot_path}")


if __name__ == "__main__":
    main()