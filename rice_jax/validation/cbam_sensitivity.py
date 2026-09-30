"""cbam_sensitivity.py

CBAM sensitivity analysis (AAAI appendix / robustness).

One-at-a-time (OAT) "tornado" sweep around the canonical differential-CBAM
operating point. The canonical defaults are the center; each sensitivity-
eligible parameter is perturbed to a low and a high arm while everything else
is held at the canonical value. The center is trained once per seed and reused
across all parameters.

Swept parameters (all in canonical_config.SENSITIVITY_PARAMS; center = canonical
default):
  welfare_loss_per_unit_tariff (alpha)  5.0  -> {0.4, 2.0}   Nordhaus-calibrated vs exaggerated
  dest_alloc_persistence       (rho)    0.55 -> {0.30, 0.80} trade stickiness / diversion inertia
  transition_cost_coef         (Grubb)  10.0 -> {0.0, 5.0}   smoothness mechanism strength
  delta_max                             3.0  -> {1.0, 5.0}   max per-step export reallocation
  cbam_lambda_init                      1.0  -> {0.5, 2.0}   RCPO penalty initial value

Headline metric (metrics.eu_dirty_export_share, non-EU exporters, lower = more
diversion) and mean mitigation rate are reported per arm with cross-seed spread.

Outputs (into plots/ + training_logs/, or the CBAM_EXPERIMENT_DIR override):
  - <prefix>results_<TIMESTAMP>.csv        one row per (param, value, seed)
  - <prefix>summary_<TIMESTAMP>.pkl        raw arrays + seed summaries (for --replot)
  - <prefix>tornado_<TIMESTAMP>.png        tornado plot of both metrics

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_sensitivity.py                       # full: 5 params, 3 seeds, 2M steps
    python validation/cbam_sensitivity.py --quick               # smoke: 1 seed, 1 param, small budget
    python validation/cbam_sensitivity.py --params rho alpha    # subset of parameters
    python validation/cbam_sensitivity.py --timesteps 1000000 --seeds 0 1
    python validation/cbam_sensitivity.py --replot <summary.pkl>
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import pickle
import time
from dataclasses import replace
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training_monitor import (
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn
from _experiment_util import run_single_episode, get_output_dir, get_log_dir
from validation.canonical_config import (
    NUM_REGIONS,
    REGION_NAMES,
    NON_EU_EXPORTER_IDXS,
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    NUM_EVAL_EPISODES,
    SENSITIVITY_PARAMS,
    canonical_env_kwargs,
    make_canonical_env,
)
from validation import metrics as _metrics


# ── Sweep definition ──────────────────────────────────────────────────────────
# Short CLI alias -> (canonical env kwarg, [low_arm, high_arm]). Center value is
# read from canonical_env_kwargs() so this file never drifts from the paper-frozen
# defaults.

_SWEEP: dict[str, tuple[str, list[float]]] = {
    "alpha": ("welfare_loss_per_unit_tariff", [0.4, 2.0]),
    "rho":   ("dest_alloc_persistence",       [0.30, 0.80]),
    "tc":    ("transition_cost_coef",         [0.0, 5.0]),
    "delta": ("delta_max",                    [1.0, 5.0]),
    "lam":   ("cbam_lambda_init",             [0.5, 2.0]),
}

CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)   # non-EU, non-RoW exporters
LOG_PREFIX = "cbam_sensitivity_"

_NUM_ENVS = CANONICAL_TRAIN_KWARGS["num_envs"]
_NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]


# ── Training ──────────────────────────────────────────────────────────────────

def _make_log_fn(label: str, num_iters: int, log_dir: str):
    _os.makedirs(log_dir, exist_ok=True)
    csv_path = _os.path.join(log_dir, f"{LOG_PREFIX}{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _drop = {"action_mean", "action_var"}   # keep console output compact
    return make_combined_log_fn(
        lambda data, iteration: _print_fn(
            {k: v for k, v in data.items() if k not in _drop}, iteration
        ),
        make_csv_log_fn(csv_path),
    )


def _train(overrides: dict, label: str, key, total_timesteps: int, log_dir: str):
    """Train one canonical differential-CBAM model with the given overrides."""
    env = make_canonical_env(for_training=True, **overrides)
    num_iters = total_timesteps // (_NUM_ENVS * _NUM_STEPS)
    log_fn = _make_log_fn(label, num_iters, log_dir)
    ppo_kwargs = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
                  if k != "total_timesteps"}
    ppo = RCPOMonitoredPPO(
        total_timesteps=total_timesteps,
        log_function=log_fn,
        **ppo_kwargs,
    )
    print(f"\n{'━' * 55}\n  Training [{label}]  overrides={overrides or '(center)'}\n{'━' * 55}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo


# ── Evaluation ────────────────────────────────────────────────────────────────

def _collect_eval_arrays(key, eval_env, agent, n_eval: int):
    """Roll out n_eval episodes; stack into the shapes metrics.py expects.

    Returns
    -------
    trade_flows : (n_ep, T, NR, NR, NS)
    mitigation  : (n_ep, T, NR)
    """
    tf_eps, mit_eps = [], []
    for ep_id in range(n_eval):
        ep_key = jax.random.fold_in(key, 50_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        tf_eps.append(np.array(logs["trade_flows"]))                 # (T, NR, NR, NS)
        mit = logs["mitigation_rates_all_regions"]                   # {r: (T,)}
        mit_eps.append(np.stack([np.array(mit[r]) for r in range(NUM_REGIONS)], axis=-1))
    return np.stack(tf_eps), np.stack(mit_eps)


def _evaluate(overrides: dict, agent, key, n_eval: int) -> dict:
    """Compute headline metrics for a trained agent under its own env config."""
    eval_env = make_canonical_env(for_training=False, **overrides)
    eval_env = replace(eval_env, log_info_fn=full_state_info_log_fn)
    trade_flows, mitigation = _collect_eval_arrays(key, eval_env, agent, n_eval)
    return {
        "eu_dirty_share": _metrics.eu_dirty_export_share(trade_flows),
        "mean_mitigation": _metrics.mean_mitigation_rate(mitigation),
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def _run(params: list[str], seeds: list[int], total_timesteps: int, n_eval: int) -> dict:
    center = canonical_env_kwargs()
    rows: list[dict] = []

    # Train + eval the center once per seed and reuse across all parameters.
    center_by_seed = {}
    for seed in seeds:
        agent = _train({}, f"center_s{seed}", jax.random.PRNGKey(seed),
                       total_timesteps, get_log_dir("training_logs"))
        metric = _evaluate({}, agent, jax.random.PRNGKey(1000 + seed), n_eval)
        center_by_seed[seed] = metric
        rows.append({"param": "center", "alias": "center",
                     "value": np.nan, "seed": seed, **metric})

    # Off-center arms.
    for alias in params:
        kwarg, arms = _SWEEP[alias]
        for value in arms:
            for seed in seeds:
                label = f"{alias}_{value}_s{seed}"
                agent = _train({kwarg: value}, label, jax.random.PRNGKey(seed),
                               total_timesteps, get_log_dir("training_logs"))
                metric = _evaluate({kwarg: value}, agent,
                                   jax.random.PRNGKey(1000 + seed), n_eval)
                rows.append({"param": kwarg, "alias": alias,
                             "value": value, "seed": seed, **metric})

    return {"rows": rows, "center_value": {a: center[_SWEEP[a][0]] for a in params},
            "params": params, "seeds": seeds}


def _summarize(rows: list[dict], metric_key: str) -> dict:
    """Cross-seed summary per (alias, value), plus the center."""
    df = pd.DataFrame(rows)
    summary: dict = {"center": _metrics.seed_summary(
        df[df["param"] == "center"][metric_key].tolist())}
    for alias in df[df["param"] != "center"]["alias"].unique():
        sub = df[df["alias"] == alias]
        summary[alias] = {
            float(v): _metrics.seed_summary(g[metric_key].tolist())
            for v, g in sub.groupby("value")
        }
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

_METRICS = [
    ("eu_dirty_share", "EU dirty export share\n(lower = more diversion)"),
    ("mean_mitigation", "Mean mitigation rate\n(non-EU exporters)"),
]


def _make_tornado(result: dict, plot_path: str) -> None:
    rows = result["rows"]
    params = result["params"]
    fig, axes = plt.subplots(1, len(_METRICS), figsize=(6.5 * len(_METRICS), 1.1 + 0.7 * len(params)))
    axes = np.atleast_1d(axes)

    for ax, (metric_key, metric_label) in zip(axes, _METRICS):
        summary = _summarize(rows, metric_key)
        center_mean = summary["center"]["mean"]
        ax.axvline(center_mean, color="k", ls="--", lw=1.2, alpha=0.7, label="canonical center")

        y = np.arange(len(params))
        for yi, alias in zip(y, params):
            kwarg, arms = _SWEEP[alias]
            for value, color, marker in zip(arms, ("tab:blue", "tab:red"), ("o", "s")):
                s = summary[alias][float(value)]
                ax.errorbar(s["mean"], yi, xerr=s["std"], fmt=marker, color=color,
                            capsize=3, ms=7, lw=1.4,
                            label=f"{kwarg.split('_')[0]}={value}" if yi == 0 else None)
                ax.annotate(f"{value:g}", (s["mean"], yi), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{a}\n({_SWEEP[a][0]})" for a in params], fontsize=8)
        ax.set_xlabel(metric_label, fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:1], labels[:1], loc="upper right", fontsize=8)
    fig.suptitle(
        f"CBAM sensitivity (9-region differential, seeds={result['seeds']})\n"
        f"blue = low arm, red = high arm, dashed = canonical center",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _os.makedirs(_os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Tornado plot saved: {plot_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=CANONICAL_TRAIN_KWARGS["total_timesteps"],
                   help="Total training timesteps per run (canonical default: 2M)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(CANONICAL_SEEDS),
                   help="Training seeds (canonical default: 0 1 2)")
    p.add_argument("--params", nargs="+", choices=sorted(_SWEEP), default=sorted(_SWEEP),
                   help="Which parameters to sweep (default: all)")
    p.add_argument("--n-eval", type=int, default=NUM_EVAL_EPISODES,
                   help="Eval episodes per run")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 1 seed, 1 param (rho), 20k timesteps, 2 eval episodes")
    p.add_argument("--replot", metavar="PKL", default=None,
                   help="Skip training; regenerate the tornado plot from a saved summary pkl")
    return p.parse_args()


def main():
    args = _parse_args()
    plot_dir, log_dir = get_output_dir("plots"), get_log_dir("training_logs")
    _os.makedirs(plot_dir, exist_ok=True)

    if args.replot:
        with open(args.replot, "rb") as fh:
            result = pickle.load(fh)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _make_tornado(result, _os.path.join(plot_dir, f"{LOG_PREFIX}tornado_{ts}.png"))
        return

    if args.quick:
        params, seeds, timesteps, n_eval = ["rho"], [0], 20_000, 2
    else:
        params, seeds, timesteps, n_eval = args.params, args.seeds, args.timesteps, args.n_eval

    assert all(_SWEEP[a][0] in SENSITIVITY_PARAMS for a in params), \
        "every swept kwarg must be in canonical_config.SENSITIVITY_PARAMS"

    n_runs = len(seeds) * (1 + 2 * len(params))
    print("=== CBAM sensitivity analysis ===")
    print(f"  params    : {params}")
    print(f"  seeds     : {seeds}")
    print(f"  timesteps : {timesteps:,} per run")
    print(f"  runs      : {n_runs} (1 center + 2 arms per param, x {len(seeds)} seeds)")

    result = _run(params, seeds, timesteps, n_eval)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = _os.path.join(log_dir, f"{LOG_PREFIX}results_{ts}.csv")
    pkl_path = _os.path.join(log_dir, f"{LOG_PREFIX}summary_{ts}.pkl")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
    with open(pkl_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"\nResults CSV : {csv_path}\nSummary PKL : {pkl_path}")

    _make_tornado(result, _os.path.join(plot_dir, f"{LOG_PREFIX}tornado_{ts}.png"))


if __name__ == "__main__":
    main()
