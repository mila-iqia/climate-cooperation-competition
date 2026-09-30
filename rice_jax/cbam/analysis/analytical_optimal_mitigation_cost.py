"""Analytical abatement-cost comparison for EU exporters.

No training is performed. The script compares immediate maximum mitigation with
an analytically smoothed trajectory and the canonical EU ramp, then plots cost
distributions, exporter cost as a share of EU GDP, cumulative burden, and each
exporter's own-GDP burden.

Run from ``rice_jax/`` with::

    python cbam/analysis/analytical_optimal_mitigation_cost.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve()
_RICE_ROOT = _HERE.parents[2]
if str(_RICE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_ROOT))

from _experiment_util import unwrap_rice_env
from cbam.config.canonical_config import (
    EU_REGION_IDX,
    NON_EU_EXPORTER_IDXS,
    REGION_NAMES,
    make_canonical_env,
)

TARGET = 1.0
SAVINGS = 0.22
REGIMES = ("naive", "smooth_optimal", "canonical_ramp")


def _make_actions(mu: np.ndarray, savings: float) -> dict:
    import jax.numpy as jnp

    num_regions = mu.shape[0]
    return {
        "mitigation_rate": jnp.asarray(mu, dtype=jnp.float32),
        "savings_rate": jnp.full((num_regions,), savings),
        "export_reallocation": jnp.zeros((num_regions, 2 * num_regions)),
        "export_limit": jnp.zeros((num_regions,)),
        "import_bid": jnp.zeros((num_regions, num_regions)),
        "import_tariff": jnp.zeros((num_regions, num_regions)),
    }


def _make_env():
    wrapped = make_canonical_env(
        for_training=False,
        eu_mitigation_schedule=None,
    )
    return unwrap_rice_env(wrapped)


def _rollout(env, schedule: np.ndarray, savings: float) -> dict[str, np.ndarray]:
    import jax

    _, state = env.reset(jax.random.PRNGKey(0))
    initial_mitigation = np.asarray(
        state["mitigation_rates_all_regions"], dtype=float
    )
    initial_intensity = np.asarray(state["intensity_all_regions"], dtype=float)
    rows = {
        "mitigation": [],
        "abatement_cost": [],
        "gross_output": [],
        "production": [],
        "damages": [],
        "intensity": [],
    }
    for mu in schedule:
        state = env.step_climate_and_economy(state, _make_actions(mu, savings))
        for key in rows:
            source = {
                "mitigation": "mitigation_rates_all_regions",
                "abatement_cost": "abatement_cost_all_regions",
                "gross_output": "gross_output_all_regions",
                "production": "production_all_regions",
                "damages": "damages_all_regions",
                "intensity": "intensity_all_regions",
            }[key]
            rows[key].append(np.asarray(state[source], dtype=float))
    result = {key: np.stack(value) for key, value in rows.items()}
    result["initial_mitigation"] = initial_mitigation
    result["initial_intensity"] = initial_intensity
    return result


def _schedule_template(num_steps: int, num_regions: int) -> dict[str, np.ndarray]:
    schedules = {
        name: np.zeros((num_steps, num_regions), dtype=float) for name in REGIMES
    }
    exporters = np.asarray(NON_EU_EXPORTER_IDXS)
    schedules["naive"][:, exporters] = TARGET
    ramp = np.asarray(
        (0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00),
        dtype=float,
    )
    ramp = np.pad(ramp, (0, max(0, num_steps - ramp.size)), constant_values=TARGET)
    schedules["canonical_ramp"][:, exporters] = ramp[:num_steps, None]
    return schedules


def _objective_factory(
    coefficient: np.ndarray,
    theta: float,
    transition_cost_coef: float,
    dt: float,
    target: float,
    initial_mitigation: float,
):
    def objective(mu: np.ndarray) -> float:
        previous = np.concatenate(([initial_mitigation], mu[:-1]))
        enduring = coefficient * np.power(mu, theta)
        transition = transition_cost_coef * np.square((mu - previous) / dt)
        terminal = 1e6 * max(target - mu[-1], 0.0) ** 2
        return float(np.sum(enduring + transition) + terminal)

    return objective


def _smooth_schedule(env, baseline: dict[str, np.ndarray], target: float) -> np.ndarray:
    params = env.region_params
    num_steps, num_regions = baseline["intensity"].shape
    dt = float(np.asarray(params.xDelta))
    transition_cost_coef = float(env.transition_cost_coef)
    theta_values = np.broadcast_to(np.asarray(params.xtheta_2), (num_regions,))
    backstop_values = np.broadcast_to(np.asarray(params.xp_b), (num_regions,))
    decay_values = np.broadcast_to(np.asarray(params.xdelta_pb), (num_regions,))
    schedules = np.zeros((num_steps, num_regions), dtype=float)
    exporters = set(NON_EU_EXPORTER_IDXS)
    for region in range(num_regions):
        if region not in exporters:
            continue
        theta = float(theta_values[region])
        backstop = float(backstop_values[region])
        decay = float(decay_values[region])
        coefficient = (
            backstop
            / (1000.0 * theta)
            * np.power(1.0 - decay, np.arange(num_steps))
            * baseline["intensity"][:, region]
        )
        result = minimize(
            _objective_factory(
                coefficient,
                theta,
                transition_cost_coef,
                dt,
                target,
                baseline["initial_mitigation"][region],
            ),
            x0=np.linspace(0.0, target, num_steps),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * num_steps,
            options={"ftol": 1e-12, "maxiter": 5000, "maxfun": 100000},
        )
        if not result.success:
            raise RuntimeError(f"Optimization failed for region {region}: {result.message}")
        schedules[:, region] = result.x
        schedules[-1, region] = target
    return schedules


def _cost_usd(rollout: dict[str, np.ndarray]) -> np.ndarray:
    pre_abatement_output = rollout["damages"] * rollout["production"]
    return rollout["abatement_cost"] * pre_abatement_output


def _write_csv(path: Path, results: dict[str, dict[str, np.ndarray]], eu_gdp: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exporters = set(NON_EU_EXPORTER_IDXS)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "regime",
                "timestep",
                "year",
                "region_idx",
                "region",
                "mitigation_rate",
                "abatement_cost_usd",
                "exporter_cost_pct_eu_gdp",
                "own_gdp_cost_pct",
            ]
        )
        for regime, result in results.items():
            cost = result["cost_usd"]
            for timestep in range(cost.shape[0]):
                for region in sorted(exporters):
                    own_gdp = max(result["gross_output"][timestep, region], 1e-12)
                    writer.writerow(
                        [
                            regime,
                            timestep,
                            2020 + 5 * timestep,
                            region,
                            REGION_NAMES[region],
                            result["mitigation"][timestep, region],
                            cost[timestep, region],
                            100.0 * cost[timestep, region] / max(eu_gdp[timestep], 1e-12),
                            100.0 * cost[timestep, region] / own_gdp,
                        ]
                    )


def _plot(path: Path, results: dict[str, dict[str, np.ndarray]], eu_gdp: np.ndarray) -> None:
    exporters = np.asarray(NON_EU_EXPORTER_IDXS)
    years = 2020 + 5 * np.arange(eu_gdp.size)
    colors = {"naive": "#c44e52", "smooth_optimal": "#4c72b0", "canonical_ramp": "#55a868"}
    labels = {"naive": "Naive max", "smooth_optimal": "Smooth optimal", "canonical_ramp": "Canonical ramp"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    averages = [results[name]["cost_usd"][:, exporters].mean(axis=0) for name in REGIMES]
    axes[0, 0].boxplot(
        averages,
        tick_labels=[labels[name] for name in REGIMES],
        patch_artist=True,
    )
    axes[0, 0].set_ylabel("Mean abatement cost (model USD)")
    axes[0, 0].set_title("Exporter cost distribution")
    axes[0, 0].tick_params(axis="x", rotation=18)

    for region in exporters:
        axes[0, 1].plot(years, results["smooth_optimal"]["cost_usd"][:, region], label=REGION_NAMES[region])
    axes[0, 1].set_title("Smooth-optimal exporter costs")
    axes[0, 1].set_ylabel("Abatement cost (model USD)")
    axes[0, 1].legend(fontsize=7, ncol=2)

    for name in REGIMES:
        mean_mu = results[name]["mitigation"][:, exporters].mean(axis=1)
        axes[0, 2].plot(years, mean_mu, color=colors[name], label=labels[name])
    axes[0, 2].set_title("Mean exporter mitigation")
    axes[0, 2].set_ylabel("Mitigation rate")
    axes[0, 2].set_ylim(-0.02, 1.02)
    axes[0, 2].legend(fontsize=8)

    for name in REGIMES:
        cost = results[name]["cost_usd"][:, exporters].sum(axis=1)
        pct = 100.0 * cost / np.maximum(eu_gdp, 1e-12)
        axes[1, 0].plot(years, pct, color=colors[name], label=labels[name])
        axes[1, 1].plot(years, np.cumsum(pct), color=colors[name], label=labels[name])
    axes[1, 0].set_title("Exporter cost as % of EU GDP")
    axes[1, 0].set_ylabel("Percent of EU GDP")
    axes[1, 1].set_title("Cumulative exporter cost / EU GDP")
    axes[1, 1].set_ylabel("Cumulative percentage-points")
    axes[1, 0].legend(fontsize=8)

    for name in REGIMES:
        own_pct = 100.0 * results[name]["cost_usd"][:, exporters] / np.maximum(
            results[name]["gross_output"][:, exporters], 1e-12
        )
        axes[1, 2].plot(years, own_pct.mean(axis=1), color=colors[name], label=labels[name])
    axes[1, 2].set_title("Mean exporter cost as % of own GDP")
    axes[1, 2].set_ylabel("Percent of own GDP")
    axes[1, 2].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
        if axis is not axes[0, 0]:
            axis.set_xlabel("Year")
    fig.suptitle("Analytical mitigation-cost burden for EU exporters", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(target: float = TARGET, savings: float = SAVINGS, out_dir: Path | None = None) -> tuple[Path, Path]:
    env = _make_env()
    num_steps = int(env.episode_length)
    num_regions = int(env.num_regions)
    schedules = _schedule_template(num_steps, num_regions)
    baseline_schedule = np.zeros_like(schedules["naive"])
    baseline = _rollout(env, baseline_schedule, savings)
    schedules["smooth_optimal"] = _smooth_schedule(env, baseline, target)

    results: dict[str, dict[str, np.ndarray]] = {}
    for name, schedule in schedules.items():
        rollout = _rollout(env, schedule, savings)
        rollout["cost_usd"] = _cost_usd(rollout)
        results[name] = rollout

    params = env.region_params
    theta_values = np.broadcast_to(np.asarray(params.xtheta_2), (num_regions,))
    backstop_values = np.broadcast_to(np.asarray(params.xp_b), (num_regions,))
    decay_values = np.broadcast_to(np.asarray(params.xdelta_pb), (num_regions,))
    exporter_indices = list(NON_EU_EXPORTER_IDXS)
    first_step_enduring = (
        backstop_values[exporter_indices]
        / (1000.0 * theta_values[exporter_indices])
        * (1.0 - decay_values[exporter_indices]) ** (-1.0)
        * baseline["initial_intensity"][exporter_indices]
    )
    transition = float(env.transition_cost_coef) * np.square(
        (1.0 - baseline["initial_mitigation"][exporter_indices])
        / float(np.asarray(params.xDelta))
    )
    np.testing.assert_allclose(
        results["naive"]["abatement_cost"][0, exporter_indices],
        first_step_enduring + transition,
        rtol=1e-3,
        atol=1e-7,
    )
    if np.any(np.diff(schedules["smooth_optimal"][:, list(NON_EU_EXPORTER_IDXS)], axis=0) < -1e-6):
        print("Warning: optimized mitigation is not monotone for every exporter.")

    if out_dir is None:
        out_dir = _HERE / "plots"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure_path = out_dir / f"analytical_optimal_mitigation_cost_{stamp}.png"
    csv_path = out_dir / f"analytical_optimal_mitigation_cost_{stamp}.csv"
    eu_gdp = baseline["gross_output"][:, EU_REGION_IDX]
    _plot(figure_path, results, eu_gdp)
    _write_csv(csv_path, results, eu_gdp)
    return figure_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=TARGET)
    parser.add_argument("--savings", type=float, default=SAVINGS)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    figure_path, csv_path = run(args.target, args.savings, args.out_dir)
    print(f"Wrote {figure_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
