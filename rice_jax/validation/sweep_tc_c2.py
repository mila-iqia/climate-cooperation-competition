"""sweep_tc_c2.py — Sweep transition_cost_coef to find C2 recovery threshold.

Runs ONLY the C2b sub-test (costly abatement, pinned exports, differential CBAM)
at a range of transition_cost_coef values to identify where the μ conditioning
gap (μ_on - μ_off) crosses the "weak" threshold (0.01).

Known results:
  TC=0,  AW=0  → C2 PASS  (gap ~0.05–0.10)
  TC=10, AW=0  → C2 FAIL  (gap ~0.000)

This script finds where the transition occurs.

Usage (from rice_jax/, rice-jax conda env):
    python validation/sweep_tc_c2.py [--timesteps 2000000] [--seed 42]
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time
from dataclasses import replace
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxnasium as jym
from training_monitor import (
    MonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
    rcpo_cbam_log_info_fn,
)
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls
from _experiment_util import get_output_dir, run_single_episode
from validation.canonical_config import (
    NUM_REGIONS,
    EU_REGION_IDX as EU_IDX,
    REGION_NAMES,
    NON_EU_IDXS,
    NON_EU_EXPORTER_IDXS,
    CANONICAL_TRAIN_KWARGS,
    NUM_EVAL_EPISODES,
    make_canonical_env,
    canonical_env_kwargs as _canonical_env_kwargs,
)


# ── Config ─────────────────────────────────────────────────────────────────

NON_EU = list(NON_EU_IDXS)
CBAM_LAMBDA_INIT = _canonical_env_kwargs()["cbam_lambda_init"]

NUM_ENVS  = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]
_PPO_KWARGS = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
               if k != "total_timesteps"}

OUTPUT_DIR = get_output_dir("plots")

# Sweep grid
TC_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]


# ── Build / train / eval helpers ───────────────────────────────────────────

def _build_env(tc_coef: float, *, force_rate: float | None = None,
               for_training: bool = True):
    """Build a C2b-style env: differential CBAM, costly abatement, pinned exports."""
    overrides = dict(
        cbam_tariff_mode      = "differential",
        zero_abatement_cost   = False,
        no_mitigation         = False,
        fixed_savings_rate    = False,
        delta_max             = 0.0,        # pinned exports
        action_window_size    = 0,
        transition_cost_coef  = tc_coef,
        cbam_lambda_init      = CBAM_LAMBDA_INIT,
    )
    if force_rate is not None:
        overrides["cbam_tariff_rate"] = force_rate
        overrides["cbam_randomize"]   = False
    else:
        overrides["cbam_randomize"]    = True
        overrides["cbam_tariff_rates"] = (0.0, 1.0)

    return make_canonical_env(for_training=for_training, **overrides)


def _train(label: str, env, key, total_timesteps: int):
    num_iters = total_timesteps // NUM_ENVS // NUM_STEPS
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _DROP = {"action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _DROP}, iteration)

    ppo = MonitoredPPO(
        total_timesteps=total_timesteps,
        log_function=_compact,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*50}")
    print(f"  Training: {label}")
    print(f"{'━'*50}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")
    return ppo, elapsed


def _eval(key, raw_env, agent):
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)
    mitigation_all = []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 90_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        mit = np.stack([np.array(logs["mitigation_rates_all_regions"][i])
                        for i in range(NUM_REGIONS)], axis=-1)
        mitigation_all.append(mit)
    return np.stack(mitigation_all, 0)  # (n_ep, T, NR)


def _mean_mu_non_eu(mitigation, last_t=5):
    return float(mitigation[:, -last_t:, :][:, :, NON_EU].mean())


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sweep TC values for C2 recovery")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tc-values", type=str, default=None,
                        help="Comma-separated TC values (default: built-in grid)")
    args = parser.parse_args()

    tc_vals = TC_VALUES
    if args.tc_values:
        tc_vals = [float(x.strip()) for x in args.tc_values.split(",")]

    key = jax.random.PRNGKey(args.seed)
    total_timesteps = args.timesteps

    print(f"{'═'*60}")
    print(f"  TC SWEEP for C2 conditioning (seed={args.seed}, T={total_timesteps:,})")
    print(f"  TC values: {tc_vals}")
    print(f"{'═'*60}")

    results = []
    for i, tc in enumerate(tc_vals):
        print(f"\n{'═'*60}")
        print(f"  [{i+1}/{len(tc_vals)}]  transition_cost_coef = {tc}")
        print(f"{'═'*60}")

        # Train with randomized CBAM
        train_key = jax.random.fold_in(key, int(tc * 100))
        env = _build_env(tc, for_training=True)
        agent, elapsed = _train(f"c2b_tc{tc:.1f}", env, train_key, total_timesteps)

        # Eval at cbam=0 and cbam=1
        eval_key = jax.random.fold_in(key, 20)
        raw_on  = _build_env(tc, force_rate=1.0, for_training=False)
        raw_off = _build_env(tc, force_rate=0.0, for_training=False)

        mit_on  = _eval(eval_key, raw_on,  agent)
        mit_off = _eval(eval_key, raw_off, agent)

        mu_on  = _mean_mu_non_eu(mit_on)
        mu_off = _mean_mu_non_eu(mit_off)
        gap    = mu_on - mu_off

        grade = "STRONG" if gap > 0.05 else ("WEAK" if gap > 0.01 else "NONE")
        results.append({
            "tc": tc, "mu_on": mu_on, "mu_off": mu_off,
            "gap": gap, "grade": grade, "elapsed": elapsed,
        })

        print(f"\n  TC={tc:5.1f}  μ_on={mu_on:.4f}  μ_off={mu_off:.4f}  "
              f"gap={gap:+.4f}  [{grade}]")

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  SUMMARY — C2 μ conditioning gap vs transition_cost_coef")
    print(f"{'═'*60}")
    print(f"  {'TC':>6s}  {'μ_on':>7s}  {'μ_off':>7s}  {'gap':>8s}  {'grade':>7s}  {'time':>5s}")
    print(f"  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*5}")
    for r in results:
        print(f"  {r['tc']:6.1f}  {r['mu_on']:7.4f}  {r['mu_off']:7.4f}  "
              f"{r['gap']:+8.4f}  {r['grade']:>7s}  {r['elapsed']/60:5.1f}m")

    # Find threshold
    passing = [r for r in results if r["gap"] > 0.01]
    failing = [r for r in results if r["gap"] <= 0.01]
    if passing and failing:
        highest_pass = max(r["tc"] for r in passing)
        lowest_fail  = min(r["tc"] for r in failing)
        print(f"\n  C2 threshold: passes at TC≤{highest_pass:.1f}, "
              f"fails at TC≥{lowest_fail:.1f}")
    elif not failing:
        print(f"\n  All TC values pass C2!")
    else:
        print(f"\n  All TC values fail C2!")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"C2 Mitigation Conditioning vs Transition Cost Coefficient\n"
        f"seed={args.seed}, {total_timesteps//1_000_000}M steps, "
        f"differential CBAM, pinned exports, costly abatement",
        fontsize=10, fontweight="bold",
    )

    tcs  = [r["tc"]     for r in results]
    mons = [r["mu_on"]  for r in results]
    moffs= [r["mu_off"] for r in results]
    gaps = [r["gap"]    for r in results]

    # Panel 1: μ_on and μ_off vs TC
    ax1.plot(tcs, mons, "o-", color="tab:blue", label="μ (cbam=1)", lw=2)
    ax1.plot(tcs, moffs, "s--", color="tab:red", label="μ (cbam=0)", lw=2)
    ax1.set_xlabel("transition_cost_coef")
    ax1.set_ylabel("Mean mitigation rate (non-EU)")
    ax1.set_title("Mitigation level at CBAM on vs off")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    # Panel 2: gap vs TC with threshold lines
    colors = ["tab:green" if g > 0.05 else ("tab:orange" if g > 0.01 else "tab:red")
              for g in gaps]
    ax2.bar(range(len(tcs)), gaps, color=colors, alpha=0.8, width=0.7)
    ax2.axhline(0.05, color="green", ls="--", lw=1, label="strong (0.05)")
    ax2.axhline(0.01, color="orange", ls="--", lw=1, label="weak (0.01)")
    ax2.axhline(0.0, color="black", ls="-", lw=0.5)
    ax2.set_xticks(range(len(tcs)))
    ax2.set_xticklabels([f"{tc:.1f}" for tc in tcs], rotation=45)
    ax2.set_xlabel("transition_cost_coef")
    ax2.set_ylabel("μ_on − μ_off (conditioning gap)")
    ax2.set_title("C2 conditioning gap")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _os.path.join(OUTPUT_DIR, f"sweep_tc_c2_{ts}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


if __name__ == "__main__":
    main()
