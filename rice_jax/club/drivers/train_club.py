"""Train shared-policy PPO (regions + learned mediator) on the club env.

Usage (from rice_jax/):

    conda run -n rice-jax python club/drivers/train_club.py --timesteps 200000 --seed 0

Outputs:
    club/training_logs/club_train_<reward_mode>_<seed>.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))
_CLUB_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")

import jax  # noqa: E402
import jaxnasium as jym  # noqa: E402

from _experiment_util import get_log_dir  # noqa: E402
from club.config import club_train_kwargs, make_club_env  # noqa: E402
from rice_jax.training import (  # noqa: E402
    LoggingPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Club mediator PPO training")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument(
        "--mediator-reward", choices=["emissions", "members"], default="emissions"
    )
    args = parser.parse_args()

    log_dir = get_log_dir(str(_CLUB_ROOT / "training_logs"))
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(
        log_dir, f"club_train_{args.mediator_reward}_{args.seed}.csv"
    )

    env = make_club_env(
        num_regions=args.num_regions, mediator_reward_mode=args.mediator_reward
    )
    # No StackActionSpaceWrapper: action dict mixes Discrete(2) and Discrete(L)
    env = jym.LogWrapper(env)

    train_kw = club_train_kwargs(total_timesteps=args.timesteps)
    num_iters = (
        train_kw["total_timesteps"] // train_kw["num_steps"] // train_kw["num_envs"]
    )
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    ppo = LoggingPPO(**train_kw, log_function=log_fn)

    print(
        f"Training club env: {args.num_regions} regions + mediator "
        f"({args.mediator_reward} reward), {args.timesteps:,} timesteps, seed={args.seed}"
    )
    key = jax.random.PRNGKey(args.seed)
    _agent, _metrics = ppo.train(key, env)
    print(f"Done. Log: {csv_path}")


if __name__ == "__main__":
    main()
