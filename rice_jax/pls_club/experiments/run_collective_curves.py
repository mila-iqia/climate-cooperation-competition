"""Collective curves for core Rice, BCTA, and PLS shield conditions.

Conditions:
  no_club  : plain Rice with no negotiation
  club     : original BasicClubTariffAmbition hard-mask club
  shield_1 : PLS tariff-ambition club with constant weights
  shield_2 : PLS tariff-ambition club with graded weights
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLS_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxnasium as jym  # noqa: E402
import numpy as np  # noqa: E402

from _experiment_util import get_output_dir, run_single_episode, with_log_info_fn  # noqa: E402
from pls_club.config import pls_train_kwargs  # noqa: E402
from pls_club.env import PLSClubTariffAmbition  # noqa: E402
from pls_club.shield import ShieldedCategoricalLayer  # noqa: E402
from rice_jax import BasicClubTariffAmbition, Rice  # noqa: E402
from rice_jax.training import LoggingPPO  # noqa: E402
from rice_jax.utils import i_to_agent_str, load_region_yamls  # noqa: E402


def _transpose_actions(actions):
    keys = sorted(actions)
    return {
        name: jnp.stack([actions[agent][name] for agent in keys])
        for name in actions[keys[0]]
    }


def curve_log_info(state, actions, rewards=None, **kwargs):
    """Log normalized action rates and common state trajectories."""
    action_values = _transpose_actions(actions)
    n = state["mitigation_rates_all_regions"].shape[0]
    zeros = jnp.zeros(n, dtype=jnp.float32)
    return {
        "actions_mitigation_rate": action_values["mitigation_rate"] / 10.0,
        "actions_savings_rate": action_values["savings_rate"] / 10.0,
        "global_temperature": state["global_temperature"],
        "mitigation_rates_all_regions": state["mitigation_rates_all_regions"],
        "savings_all_regions": state["savings_all_regions"],
        "utility_all_regions": state["utility_all_regions"],
        "gross_output_all_regions": state["gross_output_all_regions"],
        "club_defectors": state.get("club_defectors", zeros),
        "minimum_mitigation_rate_all_regions": state.get(
            "minimum_mitigation_rate_all_regions", zeros
        ),
        "rewards": rewards,
    }


def make_env(condition: str, num_regions: int, args):
    params = load_region_yamls(num_regions)
    common = dict(
        region_params=params,
        num_regions=num_regions,
        num_discrete_action_levels=10,
        diff_reward_mode=True,
    )
    if condition == "no_club":
        return Rice(**common, negotiation_on=False)
    if condition == "club":
        return BasicClubTariffAmbition(**common, negotiation_on=True)
    if condition in {"shield_1", "shield_2"}:
        return PLSClubTariffAmbition(
            **common,
            negotiation_on=True,
            shield_strength=args.shield_strength,
            shield_mode="graded" if condition == "shield_2" else "constant",
            shield_kappa=args.shield_kappa,
        )
    raise ValueError(condition)


def progress_logger(path: Path, condition: str, total_timesteps: int, every: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_logged = {"timestep": -1}
    with path.open("a") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} START condition={condition} "
            f"total_timesteps={total_timesteps}\n"
        )
        handle.flush()

    def log_fn(data, iteration):
        raw = data.get("timestep", iteration)
        timestep = int(np.asarray(raw).max()) if np.asarray(raw).size else int(iteration)
        if timestep >= total_timesteps or timestep - last_logged["timestep"] >= every:
            elapsed = time.perf_counter() - started
            rate = timestep / elapsed if elapsed > 0 else 0.0
            with path.open("a") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat()} PROGRESS "
                    f"condition={condition} iteration={int(np.asarray(iteration))} "
                    f"timestep={timestep}/{total_timesteps} elapsed_s={elapsed:.1f} "
                    f"steps_per_s={rate:.2f}\n"
                )
                handle.flush()
            last_logged["timestep"] = timestep

    return log_fn


def train_and_evaluate(condition: str, args):
    train_env = jym.LogWrapper(make_env(condition, args.num_regions, args))
    actor_kwargs = (
        {"discrete_output_layer": ShieldedCategoricalLayer}
        if condition in {"shield_1", "shield_2"}
        else {}
    )
    ppo = LoggingPPO(
        **pls_train_kwargs(total_timesteps=args.timesteps),
        log_function=progress_logger(
            Path(args.progress_log), condition, args.timesteps, args.progress_every
        ),
        actor_kwargs=actor_kwargs,
    )
    print(f"Training {condition}: {args.num_regions} regions, {args.timesteps:,} steps")
    start = time.perf_counter()
    agent, _ = ppo.train(jax.random.PRNGKey(args.seed), train_env)
    train_seconds = time.perf_counter() - start

    eval_env = with_log_info_fn(make_env(condition, args.num_regions, args), curve_log_info)
    keys = jax.random.split(
        jax.random.fold_in(jax.random.PRNGKey(args.seed), 1), args.eval_episodes
    )
    episodes = []
    eval_seconds = []
    for key in keys:
        start = time.perf_counter()
        episode = run_single_episode(key, eval_env, agent)
        eval_seconds.append(time.perf_counter() - start)
        episodes.append(jax.tree.map(np.asarray, episode))
    with Path(args.progress_log).open("a") as handle:
        handle.write(
            f"{datetime.now(timezone.utc).isoformat()} END condition={condition} "
            f"train_seconds={train_seconds:.1f}\n"
        )
        handle.flush()
    return {
        "condition": condition,
        "episodes": episodes,
        "train_seconds": train_seconds,
        "train_seconds_per_timestep": train_seconds / args.timesteps,
        "eval_seconds_per_step": float(np.mean(eval_seconds) / eval_env.episode_length),
    }


def _mean_series(episodes, key, region_mean=True):
    arrays = []
    for episode in episodes:
        value = np.asarray(episode[key])
        if key == "global_temperature" and value.ndim > 1:
            value = value[:, 0]
        elif region_mean and value.ndim > 1:
            value = value.mean(axis=tuple(range(1, value.ndim)))
        elif value.ndim > 1:
            value = value[:, 0]
        arrays.append(value)
    length = min(map(len, arrays))
    values = np.stack([value[:length] for value in arrays])
    return values.mean(axis=0), values.std(axis=0)


def _write_summary(runs, output):
    rows = []
    for run in runs:
        episode = run["episodes"]
        mit, _ = _mean_series(episode, "mitigation_rates_all_regions")
        savings, _ = _mean_series(episode, "savings_all_regions")
        utility, _ = _mean_series(episode, "utility_all_regions")
        output_series, _ = _mean_series(episode, "gross_output_all_regions")
        defect, _ = _mean_series(episode, "club_defectors")
        floor, _ = _mean_series(episode, "minimum_mitigation_rate_all_regions")
        shortfall = np.maximum(0.0, floor - mit)
        rows.append(
            {
                "condition": run["condition"],
                "mean_mitigation": float(mit.mean()),
                "mean_savings": float(savings.mean()),
                "mean_utility": float(utility.mean()),
                "mean_output": float(output_series.mean()),
                "defection_rate": float((defect > 1e-6).mean()),
                "floor_shortfall": float(shortfall.mean()),
                "train_seconds": run["train_seconds"],
                "train_seconds_per_timestep": run["train_seconds_per_timestep"],
                "eval_seconds_per_step": run["eval_seconds_per_step"],
            }
        )
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot(runs, output):
    colors = {
        "no_club": "#777777",
        "club": "#222222",
        "shield_1": "#d1495b",
        "shield_2": "#2a6fbb",
    }
    labels = {
        "no_club": "No Club",
        "club": "Club",
        "shield_1": "Shield 1",
        "shield_2": "Shield 2",
    }
    fig, axes = plt.subplots(5, 2, figsize=(15, 22), constrained_layout=True)

    def draw(axis, key, title, condition_names=None, transform=None):
        for run in runs:
            if condition_names and run["condition"] not in condition_names:
                continue
            mean, std = _mean_series(run["episodes"], key)
            if transform:
                mean, std = transform(mean, std, run)
            x = np.arange(len(mean))
            axis.plot(x, mean, label=labels[run["condition"]], color=colors[run["condition"]])
            axis.fill_between(x, mean - std, mean + std, color=colors[run["condition"]], alpha=0.10)
        axis.set_title(title)
        axis.set_xlabel("Environment step")
        axis.grid(alpha=0.25)
        axis.legend()

    draw(axes[0, 0], "global_temperature", "Global temperature (atmosphere)")
    draw(axes[0, 1], "actions_mitigation_rate", "Selected actions.mitigation_rate", {"shield_1", "shield_2"})
    draw(axes[1, 0], "mitigation_rates_all_regions", "Realized mitigation rate", {"no_club", "club", "shield_1", "shield_2"})
    draw(axes[1, 1], "savings_all_regions", "Savings rate", {"no_club", "club", "shield_1", "shield_2"})
    draw(axes[2, 0], "utility_all_regions", "Utility", {"no_club", "club", "shield_1", "shield_2"})
    draw(axes[2, 1], "gross_output_all_regions", "Output", {"no_club", "club", "shield_1", "shield_2"})
    draw(axes[3, 0], "club_defectors", "Defection rate", {"shield_1", "shield_2"})

    # Floor shortfall is a derived shield-only curve.
    for run in runs:
        if run["condition"] not in {"shield_1", "shield_2"}:
            continue
        floor, _ = _mean_series(run["episodes"], "minimum_mitigation_rate_all_regions")
        mitigation, _ = _mean_series(run["episodes"], "mitigation_rates_all_regions")
        shortfall = np.maximum(0.0, floor - mitigation)
        axes[3, 1].plot(shortfall, label=labels[run["condition"]], color=colors[run["condition"]])
    axes[3, 1].set_title("Floor shortfall")
    axes[3, 1].set_xlabel("Environment step")
    axes[3, 1].grid(alpha=0.25)
    axes[3, 1].legend()

    # Timing report: bars are evaluation seconds per environment step.
    timing = [run["eval_seconds_per_step"] for run in runs]
    axes[4, 0].bar([labels[run["condition"]] for run in runs], timing, color=[colors[run["condition"]] for run in runs])
    axes[4, 0].set_title("Evaluation time / step")
    axes[4, 0].set_ylabel("Seconds")
    axes[4, 0].tick_params(axis="x", rotation=30)
    axes[4, 0].grid(axis="y", alpha=0.25)
    axes[4, 1].axis("off")
    axes[4, 1].text(
        0.02,
        0.95,
        "Timing note:\nBars show steady-state evaluation seconds per environment step.\n\n"
        + "\n".join(
            f"{labels[run['condition']]} training sec/step: {run['train_seconds_per_timestep']:.5f}"
            for run in runs
        ),
        va="top",
        fontsize=11,
    )
    fig.suptitle("Core RICE / BCTA / PLS collective curves", fontsize=16)
    fig.savefig(output, dpi=160)


def main():
    parser = argparse.ArgumentParser(description="Collective core RICE and PLS curves")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--shield-strength", type=float, default=0.8)
    parser.add_argument("--shield-kappa", type=float, default=10.0)
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=Path("pls_club/plots/pls_collective_curves_progress.log"),
    )
    parser.add_argument("--progress-every", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    conditions = ["no_club", "club", "shield_1", "shield_2"]
    runs = [train_and_evaluate(condition, args) for condition in conditions]
    out_dir = Path(get_output_dir(str(PLS_ROOT / "plots")))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pls_collective_curves_{args.seed}"
    pkl_path = args.output or out_dir / f"{stem}.pkl"
    csv_path = pkl_path.with_suffix(".csv")
    png_path = pkl_path.with_suffix(".png")
    with pkl_path.open("wb") as handle:
        pickle.dump({"experiment": stem, "args": vars(args), "runs": runs}, handle)
    rows = _write_summary(runs, csv_path)
    _plot(runs, png_path)
    print(f"Artifacts: {pkl_path}, {csv_path}, {png_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
