"""Compare null, constant, and graded PLS shields on non-mediator BCTA.

All conditions use ``PLSClubTariffAmbition`` and the same PPO configuration.
The null condition has unit safety weights, while constant and graded use the
two implemented PLS weighting rules.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
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
from rice_jax.training import LoggingPPO  # noqa: E402
from rice_jax.utils import load_region_yamls  # noqa: E402


def log_info_fn(state, actions, rewards=None, **kwargs):
    floors = state["minimum_mitigation_rate_all_regions"]
    return {
        "rewards": rewards,
        "min_mitigation_floor": floors,
        "mitigation_rates_all_regions": state["mitigation_rates_all_regions"],
        "global_emissions": state["global_emissions"],
        "global_temperature": state["global_temperature"],
        "import_tariffs": state.get(
            "import_tariffs", jnp.zeros((floors.shape[0], floors.shape[0]))
        ),
    }


def make_env(mode: str, num_regions: int, args):
    strength = 0.0 if mode == "null" else args.shield_strength
    return PLSClubTariffAmbition(
        region_params=load_region_yamls(num_regions),
        num_regions=num_regions,
        num_discrete_action_levels=10,
        diff_reward_mode=True,
        negotiation_on=True,
        shield_strength=strength,
        shield_mode="graded" if mode == "graded" else "constant",
        shield_kappa=args.shield_kappa,
    )


def run_condition(mode: str, args):
    train_env = jym.LogWrapper(make_env(mode, args.num_regions, args))
    ppo = LoggingPPO(
        **pls_train_kwargs(total_timesteps=args.timesteps),
        log_function=lambda _data, _iteration: None,
        actor_kwargs={"discrete_output_layer": ShieldedCategoricalLayer},
    )
    print(f"Training {mode}: {args.num_regions} regions, {args.timesteps:,} steps")
    seed = args.seed
    agent, _ = ppo.train(jax.random.PRNGKey(seed), train_env)

    eval_env = with_log_info_fn(make_env(mode, args.num_regions, args), log_info_fn)
    keys = jax.random.split(
        jax.random.fold_in(jax.random.PRNGKey(seed), 1), args.eval_episodes
    )
    episodes = [
        jax.tree.map(np.asarray, run_single_episode(key, eval_env, agent))
        for key in keys
    ]
    return {"condition": mode, "seed": seed, "episodes": episodes}


def episode_row(run, episode_idx, episode):
    steps = np.arange(len(episode["global_emissions"]))
    climate_idx = steps[(steps + 1) % 3 == 0]
    tail = climate_idx[-(len(climate_idx) // 3) :]
    mitigation = episode["mitigation_rates_all_regions"]
    floors = episode["min_mitigation_floor"]
    shortfall = np.maximum(0.0, floors - mitigation)
    rewards = np.stack(
        [value for _, value in sorted(episode["rewards"].items())], axis=-1
    )
    return {
        "condition": run["condition"],
        "episode": episode_idx,
        "mean_mitigation": float(mitigation[tail].mean()),
        "mean_floor_shortfall": float(shortfall[tail].mean()),
        "defection_frequency": float((shortfall[tail] > 1e-6).mean()),
        "mean_global_emissions": float(episode["global_emissions"][tail].mean()),
        "cumulative_emissions": float(episode["global_emissions"][climate_idx].sum()),
        "mean_import_tariff": float(episode["import_tariffs"][tail].mean()),
        "mean_reward": float(rewards[tail].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare PLS shield modes")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--shield-strength", type=float, default=0.8)
    parser.add_argument("--shield-kappa", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.shield_strength <= 1.0:
        parser.error("--shield-strength must be between 0 and 1")
    runs = [run_condition(mode, args) for mode in ("null", "constant", "graded")]
    rows = [
        episode_row(run, idx, episode)
        for run in runs
        for idx, episode in enumerate(run["episodes"])
    ]

    out_dir = Path(get_output_dir(str(PLS_ROOT / "plots")))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pls_shield_mode_comparison_{args.seed}"
    output = args.output or out_dir / f"{stem}.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        pickle.dump({"experiment": stem, "args": vars(args), "runs": runs}, handle)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("mean_mitigation", "Mitigation"),
        ("mean_floor_shortfall", "Floor shortfall"),
        ("defection_frequency", "Defection frequency"),
        ("mean_global_emissions", "Emissions/step"),
        ("mean_import_tariff", "Import tariff"),
        ("mean_reward", "Reward"),
    ]
    conditions = [run["condition"] for run in runs]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (key, label) in zip(axes.flat, metrics):
        means = [np.mean([row[key] for row in rows if row["condition"] == condition]) for condition in conditions]
        stds = [np.std([row[key] for row in rows if row["condition"] == condition]) for condition in conditions]
        axis.bar(range(len(conditions)), means, yerr=stds, capsize=4)
        axis.set_xticks(range(len(conditions)), conditions)
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Non-mediator BCTA: null vs constant vs graded PLS")
    png_path = output.with_suffix(".png")
    fig.savefig(png_path, dpi=160)
    print(f"\nResults ({args.eval_episodes} episodes each):")
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        print(
            f"  {condition}: mitigation={np.mean([row['mean_mitigation'] for row in subset]):.3f} "
            f"shortfall={np.mean([row['mean_floor_shortfall'] for row in subset]):.4f} "
            f"defection={np.mean([row['defection_frequency'] for row in subset]):.4f} "
            f"reward={np.mean([row['mean_reward'] for row in subset]):.4f}"
        )
    print(f"Artifacts: {output}, {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
