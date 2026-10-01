"""Train shared-policy PPO with a PLS-shielded actor on the PLS club envs.

The shield lives in the policy: ``ShieldedCategoricalLayer`` is injected via
``actor_kwargs`` and consumes the safety weights the env emits through the
action-mask channel. No hard mitigation mask exists — compliance is soft.

Usage (from rice_jax/):

    conda run -n rice-jax python pls_club/drivers/train_pls_club.py \
        --variant naive --timesteps 200000 --seed 0

Outputs:
    pls_club/training_logs/pls_club_<variant>_<seed>.csv
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
_PLS_ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("JAXNASIUM_MULTI_AGENT_BATCH_SIZE", "false")

import jax  # noqa: E402
import jaxnasium as jym  # noqa: E402

from _experiment_util import get_log_dir  # noqa: E402
from pls_club.config import make_pls_env, pls_train_kwargs  # noqa: E402
from pls_club.shield import ShieldedCategoricalLayer  # noqa: E402
from rice_jax.training import (  # noqa: E402
    LoggingPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PLS club PPO training")
    parser.add_argument("--variant", choices=["naive", "mediator"], default="naive")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument("--shield-strength", type=float, default=0.8)
    parser.add_argument(
        "--shield-mode", choices=["constant", "graded"], default="constant"
    )
    args = parser.parse_args()

    log_dir = get_log_dir(str(_PLS_ROOT / "training_logs"))
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"pls_club_{args.variant}_{args.seed}.csv")

    env = make_pls_env(
        variant=args.variant,
        num_regions=args.num_regions,
        shield_strength=args.shield_strength,
        shield_mode=args.shield_mode,
    )
    # No StackActionSpaceWrapper: action dict mixes Discrete(2) and Discrete(L)
    env = jym.LogWrapper(env)

    train_kw = pls_train_kwargs(total_timesteps=args.timesteps)
    num_iters = (
        train_kw["total_timesteps"] // train_kw["num_steps"] // train_kw["num_envs"]
    )
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    ppo = LoggingPPO(
        **train_kw,
        log_function=log_fn,
        # PLS: the actor's discrete heads interpret mask values as P(safe|a)
        actor_kwargs={"discrete_output_layer": ShieldedCategoricalLayer},
    )

    print(
        f"Training PLS club ({args.variant}): {args.num_regions} regions, "
        f"shield={args.shield_mode}@{args.shield_strength}, "
        f"{args.timesteps:,} timesteps, seed={args.seed}"
    )
    key = jax.random.PRNGKey(args.seed)
    _agent, _metrics = ppo.train(key, env)
    print(f"Done. Log: {csv_path}")


if __name__ == "__main__":
    main()
