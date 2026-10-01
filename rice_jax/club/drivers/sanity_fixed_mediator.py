"""Sanity check: fixed-mediator clubs with trained region policies.

Expectations (see club/tests for the fixed-action counterparts):
  * low-bar club  (min mitigation 0.1, tariff 0.9) -> joining is cheap and
    staying out is punished, membership should approach all regions
  * high-bar club (min mitigation 0.9, tariff 0.0) -> joining is costly and
    free-riding is unpunished, membership should stay low

Usage (from rice_jax/):

    conda run -n rice-jax python club/drivers/sanity_fixed_mediator.py --timesteps 200000

Outputs:
    club/training_logs/club_sanity_<scenario>_<seed>.csv (training curves)
    club/plots/club_sanity_membership_<seed>.csv (per-scenario membership summary)
"""

from __future__ import annotations

import argparse
import csv
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
import numpy as np  # noqa: E402

from _experiment_util import get_log_dir, get_output_dir, run_single_episode  # noqa: E402
from club.config import club_log_info_fn, club_train_kwargs, make_club_env  # noqa: E402
from rice_jax.training import (  # noqa: E402
    LoggingPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)

SCENARIOS = {
    # (min mitigation, non-member tariff)
    "low_bar": (0.1, 0.9),
    "high_bar": (0.9, 0.0),
}


def run_scenario(
    name: str, club_params: tuple[float, float], args
) -> dict[str, float]:
    print(f"\n=== scenario {name}: fixed club {club_params} ===")
    log_dir = get_log_dir(str(_CLUB_ROOT / "training_logs"))
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"club_sanity_{name}_{args.seed}.csv")

    env_kwargs = dict(num_regions=args.num_regions, fixed_club_params=club_params)
    train_env = jym.LogWrapper(make_club_env(**env_kwargs))

    train_kw = club_train_kwargs(total_timesteps=args.timesteps)
    num_iters = (
        train_kw["total_timesteps"] // train_kw["num_steps"] // train_kw["num_envs"]
    )
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    ppo = LoggingPPO(**train_kw, log_function=log_fn)
    agent, _metrics = ppo.train(jax.random.PRNGKey(args.seed), train_env)

    eval_env = make_club_env(**env_kwargs, log_info_fn=club_log_info_fn)
    info = run_single_episode(jax.random.PRNGKey(args.seed + 1000), eval_env, agent)
    membership = np.asarray(info["club_membership"])  # (T, NR)
    last_third = membership[-(membership.shape[0] // 3) :]
    result = {
        "scenario": name,
        "club_min_mitigation": club_params[0],
        "club_tariff": club_params[1],
        "mean_membership_last_third": float(last_third.mean()),
        "final_membership": float(membership[-1].mean()),
    }
    print(
        f"{name}: mean membership (last third) = "
        f"{result['mean_membership_last_third']:.3f}, "
        f"final = {result['final_membership']:.3f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-mediator club sanity checks")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-regions", type=int, default=7)
    parser.add_argument(
        "--scenario", choices=[*SCENARIOS, "both"], default="both"
    )
    args = parser.parse_args()

    names = list(SCENARIOS) if args.scenario == "both" else [args.scenario]
    results = [run_scenario(n, SCENARIOS[n], args) for n in names]

    out_dir = get_output_dir(str(_CLUB_ROOT / "plots"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"club_sanity_membership_{args.seed}.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary written to {out_csv}")

    if len(results) == 2:
        low, high = results[0], results[1]
        ok = (
            low["mean_membership_last_third"]
            > high["mean_membership_last_third"]
        )
        print(
            "SANITY "
            + ("PASS" if ok else "FAIL")
            + ": low-bar membership should exceed high-bar membership"
        )


if __name__ == "__main__":
    main()
