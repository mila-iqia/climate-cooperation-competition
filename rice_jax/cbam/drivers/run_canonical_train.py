"""Minimal entrypoint: train RCPOMonitoredPPO on the canonical 9-region CBAM env.

Usage (from rice_jax/):

    uv run python cbam/drivers/run_canonical_train.py --timesteps 50000 --seed 0

Outputs:
    training_logs/canonical_train_<seed>.csv
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

import jax  # noqa: E402

from _experiment_util import get_log_dir  # noqa: E402
from rice_jax.training import (  # noqa: E402
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from cbam.config.canonical_config import (  # noqa: E402
    MRIO_DATA_ROOT,
    canonical_train_kwargs,
    ensure_mrio_data_layout,
    make_canonical_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical 9-region CBAM PPO training")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_mrio_data_layout(MRIO_DATA_ROOT)
    print(f"MRIO data root: {MRIO_DATA_ROOT}")
    print(f"Training {args.timesteps:,} timesteps, seed={args.seed}")

    log_dir = get_log_dir("training_logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"canonical_train_{args.seed}.csv")

    env = make_canonical_env(for_training=True)
    train_kw = canonical_train_kwargs(total_timesteps=args.timesteps)

    num_iters = train_kw["total_timesteps"] // train_kw["num_steps"] // train_kw["num_envs"]
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    ppo = RCPOMonitoredPPO(**train_kw, log_function=log_fn)

    key = jax.random.PRNGKey(args.seed)
    ppo = ppo.train(key, env)
    print(f"Done. Log: {csv_path}")


if __name__ == "__main__":
    main()
