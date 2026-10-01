"""Compare PLSClubTariffAmbition (shield) to BasicClubTariffAmbition (hard mask).

Both conditions train shared-policy PPO with identical settings; only the
mitigation-compliance mechanism differs. Evaluation rolls logged episodes per
condition and writes a side-by-side summary CSV plus a comparison plot.

Usage (from rice_jax/):

    conda run -n rice-jax python pls_club/experiments/run_tariff_ambition_comparison.py \
        --timesteps 200000 --num-regions 7 --eval-episodes 10 --seed 0
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

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))
_PLS_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxnasium as jym  # noqa: E402
import numpy as np  # noqa: E402

from _experiment_util import get_output_dir, run_single_episode, with_log_info_fn  # noqa: E402
from pls_club.config import pls_train_kwargs  # noqa: E402
from pls_club.env import PLSClubTariffAmbition  # noqa: E402
from pls_club.shield import ShieldedCategoricalLayer  # noqa: E402
from rice_jax import BasicClubTariffAmbition  # noqa: E402
from rice_jax.training import LoggingPPO  # noqa: E402
from rice_jax.utils import load_region_yamls  # noqa: E402


def ta_log_info_fn(state: dict, actions: dict, rewards=None, **kwargs) -> dict:
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


def _make_env(condition: str, num_regions: int, args) -> BasicClubTariffAmbition:
    kwargs = dict(
        region_params=load_region_yamls(num_regions),
        num_regions=num_regions,
        num_discrete_action_levels=10,
        diff_reward_mode=True,
        negotiation_on=True,
    )
    if condition == "pls_shield":
        return PLSClubTariffAmbition(
            **kwargs,
            shield_strength=args.shield_strength,
            shield_mode=args.shield_mode,
        )
    return BasicClubTariffAmbition(**kwargs)


def _run_condition(condition: str, args) -> dict:
    train_env = jym.LogWrapper(_make_env(condition, args.num_regions, args))
    actor_kwargs = (
        {"discrete_output_layer": ShieldedCategoricalLayer}
        if condition == "pls_shield"
        else {}
    )
    ppo = LoggingPPO(
        **pls_train_kwargs(total_timesteps=args.timesteps),
        log_function=lambda _d, _i: None,
        actor_kwargs=actor_kwargs,
    )
    seed = args.seed + (10_000 if condition == "pls_shield" else 0)
    print(f"Training {condition}: {args.num_regions} regions, {args.timesteps:,} steps")
    agent, _ = ppo.train(jax.random.PRNGKey(seed), train_env)

    eval_env = with_log_info_fn(_make_env(condition, args.num_regions, args), ta_log_info_fn)
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 1), args.eval_episodes)
    episodes = [
        jax.tree.map(np.asarray, run_single_episode(k, eval_env, agent)) for k in keys
    ]
    return {"condition": condition, "seed": seed, "episodes": episodes}


def _episode_row(run: dict, ep_idx: int, episode: dict) -> dict:
    # climate steps: current_timestep = i + 1, climate when (i + 1) % 3 == 0
    steps = np.arange(len(episode["global_emissions"]))
    idx = steps[(steps + 1) % 3 == 0]
    tail = idx[-(len(idx) // 3):]
    mit = episode["mitigation_rates_all_regions"]
    floors = episode["min_mitigation_floor"]
    shortfall = np.maximum(0.0, floors - mit)
    rewards = np.stack(
        [v for k, v in sorted(episode["rewards"].items())], axis=-1
    )
    return {
        "condition": run["condition"],
        "episode": ep_idx,
        "mean_mitigation": float(mit[tail].mean()),
        "mean_floor": float(floors[tail].mean()),
        "mean_shortfall": float(shortfall[tail].mean()),
        "defection_frequency": float((shortfall[tail] > 1e-6).mean()),
        "mean_global_emissions": float(episode["global_emissions"][tail].mean()),
        "cumulative_emissions": float(episode["global_emissions"][idx].sum()),
        "mean_import_tariff": float(episode["import_tariffs"][tail].mean()),
        "mean_reward": float(rewards[tail].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shield vs hard-mask tariff ambition")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--shield-strength", type=float, default=0.8)
    parser.add_argument("--shield-mode", choices=["constant", "graded"], default="constant")
    args = parser.parse_args()

    runs = [_run_condition(c, args) for c in ("hard_mask", "pls_shield")]

    out_dir = Path(get_output_dir(str(_PLS_ROOT / "plots")))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pls_tariff_ambition_comparison_{args.seed}"

    with (out_dir / f"{stem}.pkl").open("wb") as f:
        pickle.dump({"experiment": "pls_tariff_ambition_comparison", "args": vars(args), "runs": runs}, f)

    rows = [
        _episode_row(run, i, ep)
        for run in runs
        for i, ep in enumerate(run["episodes"])
    ]
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = [
        ("mean_mitigation", "Mitigation"),
        ("mean_shortfall", "Floor shortfall"),
        ("defection_frequency", "Defection freq."),
        ("mean_global_emissions", "Emissions/step"),
        ("mean_import_tariff", "Import tariff"),
        ("mean_reward", "Reward"),
    ]
    conditions = [r["condition"] for r in runs]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, metrics):
        means = [
            np.mean([row[key] for row in rows if row["condition"] == c])
            for c in conditions
        ]
        stds = [
            np.std([row[key] for row in rows if row["condition"] == c])
            for c in conditions
        ]
        ax.bar(range(len(conditions)), means, yerr=stds, capsize=4,
               color=["#657786", "#d1495b"])
        ax.set_xticks(range(len(conditions)), conditions)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Tariff-ambition club: hard mask vs PLS shield (last-third climate steps)")
    fig.savefig(out_dir / f"{stem}.png", dpi=160)

    print(f"\nResults ({args.eval_episodes} episodes each):")
    for c in conditions:
        sub = [r for r in rows if r["condition"] == c]
        print(f"  {c}: mitigation={np.mean([r['mean_mitigation'] for r in sub]):.3f} "
              f"shortfall={np.mean([r['mean_shortfall'] for r in sub]):.4f} "
              f"emissions={np.mean([r['mean_global_emissions'] for r in sub]):.3f} "
              f"reward={np.mean([r['mean_reward'] for r in sub]):.4f}")
    print(f"Artifacts: {out_dir / stem}.{{pkl,csv,png}}")


if __name__ == "__main__":
    main()
