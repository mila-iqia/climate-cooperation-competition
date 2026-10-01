"""Train and evaluate a PLS shield ablation for the core RICE environment.

The experiment trains one policy with the shield disabled and one with the
configured PLS shield, then evaluates each policy on logged episodes.

Usage (from rice_jax/):

    conda run -n rice-jax python pls_club/experiments/run_shield_ablation.py \
        --variant naive --timesteps 200000 --seed 0

The resulting pickle is consumed by ``pls_club/posthoc/analyze_shield_ablation.py``.
By default, it is written under ``pls_club/plots/``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))
_PLS_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")

import jax  # noqa: E402
import jaxnasium as jym  # noqa: E402
import numpy as np  # noqa: E402

from _experiment_util import get_output_dir, run_single_episode  # noqa: E402
from pls_club.config import (  # noqa: E402
    make_pls_env,
    pls_log_info_fn,
    pls_train_kwargs,
)
from pls_club.shield import ShieldedCategoricalLayer  # noqa: E402
from rice_jax.training import LoggingPPO  # noqa: E402


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {key: _to_numpy(value) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_numpy(value) for value in tree)
    return np.asarray(tree)


def _evaluate_policy(agent, env, seed: int, episodes: int) -> list[dict]:
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), 1000), episodes)
    return [
        _to_numpy(run_single_episode(episode_key, env, agent))
        for episode_key in keys
    ]


def _run_condition(
    *,
    condition: str,
    shield_strength: float,
    args: argparse.Namespace,
) -> dict:
    train_env = jym.LogWrapper(
        make_pls_env(
            variant=args.variant,
            num_regions=args.num_regions,
            shield_strength=shield_strength,
            shield_mode=args.shield_mode,
        )
    )
    train_kwargs = pls_train_kwargs(total_timesteps=args.timesteps)
    ppo = LoggingPPO(
        **train_kwargs,
        log_function=lambda _data, _iteration: None,
        actor_kwargs={"discrete_output_layer": ShieldedCategoricalLayer},
    )

    condition_seed = args.seed + (0 if condition == "unshielded" else 10_000)
    print(
        f"Training {args.variant}/{condition}: {args.num_regions} regions, "
        f"shield={args.shield_mode}@{shield_strength}, "
        f"{args.timesteps:,} timesteps, seed={condition_seed}"
    )
    agent, _metrics = ppo.train(jax.random.PRNGKey(condition_seed), train_env)

    eval_env = make_pls_env(
        variant=args.variant,
        num_regions=args.num_regions,
        shield_strength=shield_strength,
        shield_mode=args.shield_mode,
        log_info_fn=pls_log_info_fn,
    )
    episodes = _evaluate_policy(
        agent,
        eval_env,
        seed=condition_seed,
        episodes=args.eval_episodes,
    )
    return {
        "condition": condition,
        "shield_strength": shield_strength,
        "shield_mode": args.shield_mode,
        "seed": condition_seed,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PLS shield ablation experiment")
    parser.add_argument("--variant", choices=["naive", "mediator"], default="naive")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--shield-strength", type=float, default=0.8)
    parser.add_argument(
        "--shield-mode", choices=["constant", "graded"], default="constant"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be positive")
    if not 0.0 <= args.shield_strength <= 1.0:
        parser.error("--shield-strength must be between 0 and 1")

    runs = [
        _run_condition(condition="unshielded", shield_strength=0.0, args=args),
        _run_condition(
            condition="pls_shield",
            shield_strength=args.shield_strength,
            args=args,
        ),
    ]
    output = args.output or Path(get_output_dir(str(_PLS_ROOT / "plots"))) / (
        f"pls_club_shield_ablation_{args.variant}_{args.seed}.pkl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "experiment": "pls_club_shield_ablation",
        "variant": args.variant,
        "num_regions": args.num_regions,
        "timesteps": args.timesteps,
        "base_seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "shield_mode": args.shield_mode,
        "runs": runs,
    }
    with output.open("wb") as file:
        pickle.dump(artifact, file)
    print(f"Evaluation artifact written to {output}")


if __name__ == "__main__":
    main()
