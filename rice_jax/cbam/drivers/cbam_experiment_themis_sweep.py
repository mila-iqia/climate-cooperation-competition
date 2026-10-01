"""cbam_experiment_themis_sweep.py

Themis price sweep (Rasmussen 2025 concept note).

Trains independent policies at a ladder of Themis carbon prices p ($/tCO2e)
with endogenous per-step membership (themis_join action), then evaluates:

  - decarbonisation: mitigation rates and atmospheric temperature vs p
  - membership dynamics: who joins, and how persistently, at each p
  - redistribution: per-region net payments (cost-neutrality residual ≈ 0)

CBAM channels are OFF (flat τ=0, welfloss reward) so the Themis payment is
the only policy treatment relative to the p=0 control.

Usage (from rice_jax/, rice-jax conda env):
    python cbam/drivers/cbam_experiment_themis_sweep.py \
        [--timesteps 1000000] [--prices 0,20,50,100] [--env mrio|core] \
        [--membership action|all] [--seed 42]
"""

import argparse
import os as _os
import sys
import time
from datetime import datetime
from pathlib import Path

import cloudpickle
import jax
import matplotlib
import numpy as np

from _experiment_util import (
    get_log_dir,
    get_output_dir,
    run_single_episode,
    with_log_info_fn,
)
from cbam.config.canonical_config import (
    CANONICAL_TRAIN_KWARGS,
    CANONICAL_YAML_DIR,
    NUM_EVAL_EPISODES,
    NUM_REGIONS as MRIO_NUM_REGIONS,
    REGION_NAMES,
    canonical_env_kwargs,
    canonical_train_kwargs,
)
from rice_jax.training import (
    LoggingPPO,
    episode_return_curve,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after Agg)

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR = get_log_dir("training_logs")

CORE_NUM_REGIONS = 7  # default region_yamls set for the core arm


# ── Env factory ──────────────────────────────────────────────────────────────


def _build_env(price, env_kind, membership_mode):
    """Raw (unwrapped) Themis env with CBAM channels disabled.

    No StackActionSpaceWrapper anywhere in this experiment: the action dict
    mixes Discrete(2) themis_join with Discrete(L) — same constraint as the
    club envs.
    """
    themis_kwargs = dict(
        themis_price_schedule=(float(price),),
        themis_membership_mode=membership_mode,
    )
    if env_kind == "mrio":
        from rice_jax import ThemisRiceMRIO

        kwargs = canonical_env_kwargs()
        kwargs.update(
            cbam_tariff_mode="flat",
            cbam_tariff_rate=0.0,
            revenue_share=0.0,
            reward_mode="welfloss",     # no RCPO channel needed without CBAM
            eu_mitigation_schedule=None,  # EU learns freely; Themis is global
        )
        region_params = load_region_yamls(
            MRIO_NUM_REGIONS, yaml_dir=CANONICAL_YAML_DIR
        )
        return ThemisRiceMRIO(region_params=region_params, **kwargs, **themis_kwargs)

    from rice_jax import ThemisRice

    region_params = load_region_yamls(CORE_NUM_REGIONS)
    return ThemisRice(
        region_params=region_params,
        num_regions=CORE_NUM_REGIONS,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        **themis_kwargs,
    )


def _wrap_for_training(raw_env):
    import jaxnasium as jym

    return jym.LogWrapper(raw_env)


# ── Training ────────────────────────────────────────────────────────────────


def _make_log_fn(label, num_iters):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"themis_sweep_{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _drop = {"actions", "action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _drop}, iteration)

    return make_combined_log_fn(_compact, make_csv_log_fn(csv_path)), csv_path


def _train(label, raw_env, key, total_timesteps):
    train_kwargs = canonical_train_kwargs(total_timesteps=total_timesteps)
    num_iters = (
        total_timesteps // train_kwargs["num_envs"] // train_kwargs["num_steps"]
    )
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = LoggingPPO(
        log_function=log_fn,
        **{k: v for k, v in train_kwargs.items() if k != "total_timesteps"},
        total_timesteps=total_timesteps,
    )
    print(f"\n{'━' * 55}\n  Training: {label}\n{'━' * 55}")
    t0 = time.perf_counter()
    agent, metrics = ppo.train(key, _wrap_for_training(raw_env))
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return agent, episode_return_curve(metrics), csv_path


# ── Evaluation ──────────────────────────────────────────────────────────────


def _to_arr(per_region_dict, num_regions):
    return np.stack(
        [np.array(per_region_dict[i]) for i in range(num_regions)], axis=-1
    )


def _eval(key, raw_env, agent):
    import equinox as eqx

    eval_env = with_log_info_fn(raw_env, full_state_info_log_fn)

    # Close env/agent over a per-arm jitted rollout: passing sweep-arm envs
    # through a shared jit boundary makes jax __eq__-compare their static
    # np.ndarray MRIO fields (rebuilt on every construction) → ValueError.
    @eqx.filter_jit
    def _episode(ep_key):
        return run_single_episode.__wrapped__(ep_key, eval_env, agent)
    nr = raw_env.num_regions

    keys_per_region = {
        "mitigation": "mitigation_rates_all_regions",
        "consumption": "aggregate_consumption",
        "labor": "labor_all_regions",
        "utility": "utility_all_regions",
        "membership": "themis_membership_all_regions",
        "payments": "themis_payments_all_regions",
        "emissions": "themis_emissions_all_regions",
    }
    out = {k: [] for k in keys_per_region}
    out["temp_atmosphere"] = []
    out["themis_price"] = []

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 90_000 + ep_id)
        logs = _episode(ep_key)
        for k, state_key in keys_per_region.items():
            out[k].append(_to_arr(logs[state_key], nr))
        out["temp_atmosphere"].append(np.array(logs["global_temperature"]["atmosphere"]))
        out["themis_price"].append(np.array(logs["themis_price"]))

    return {k: np.stack(v, 0) for k, v in out.items()}  # (E, T[, NR])


# ── Summary plot ────────────────────────────────────────────────────────────


def _region_label(env_kind, idx):
    if env_kind == "mrio":
        return REGION_NAMES.get(idx, str(idx))
    return f"region {idx}"


def _summary_plot(prices, results, env_kind, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cmap = plt.get_cmap("viridis")
    colors = {p: cmap(i / max(len(prices) - 1, 1)) for i, p in enumerate(prices)}

    for p in prices:
        ev = results[p]["eval"]
        c = colors[p]
        axes[0, 0].plot(ev["temp_atmosphere"].mean(0), color=c, label=f"p={p:g}")
        axes[0, 1].plot(ev["mitigation"].mean(0).mean(-1), color=c, label=f"p={p:g}")
        axes[1, 0].plot(ev["membership"].mean(0).mean(-1), color=c, label=f"p={p:g}")

    axes[0, 0].set_title("Atmospheric temperature (°C)")
    axes[0, 1].set_title("Mean mitigation rate μ")
    axes[1, 0].set_title("Themis membership rate")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("step")
        ax.legend(fontsize=8)

    # Net payments per region at the highest price (mean over episodes/steps)
    p_hi = prices[-1]
    pay = results[p_hi]["eval"]["payments"].mean(0).mean(0)  # (NR,)
    nr = pay.shape[0]
    labels = [_region_label(env_kind, i) for i in range(nr)]
    bar_colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in pay]
    axes[1, 1].bar(range(nr), pay, color=bar_colors)
    axes[1, 1].set_xticks(range(nr))
    axes[1, 1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].set_title(f"Mean net Themis payment ($T/step) at p={p_hi:g}")

    fig.suptitle(f"Themis price sweep ({env_kind})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  summary plot → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prices", type=str, default="0,20,50,100",
                        help="comma-separated Themis prices in $/tCO2e")
    parser.add_argument("--env", choices=["mrio", "core"], default="mrio")
    parser.add_argument("--membership", choices=["action", "all"], default="action")
    parser.add_argument("--save-agents", action="store_true", default=False)
    args = parser.parse_args()

    prices = [float(p) for p in args.prices.split(",")]
    key = jax.random.PRNGKey(args.seed)

    results = {}
    for i, p in enumerate(prices):
        label = f"{args.env}_p{p:g}_s{args.seed}"
        raw_env = _build_env(p, args.env, args.membership)
        train_key = jax.random.fold_in(key, i)
        agent, curve, csv_path = _train(label, raw_env, train_key, args.timesteps)
        eval_key = jax.random.fold_in(key, 10_000 + i)
        ev = _eval(eval_key, raw_env, agent)
        residual = np.abs(ev["payments"].sum(-1)).max()
        print(
            f"  p={p:g}: final T={ev['temp_atmosphere'].mean(0)[-1]:.2f}°C  "
            f"mean μ={ev['mitigation'].mean():.3f}  "
            f"membership={ev['membership'].mean():.2f}  "
            f"max |Σ payments|={residual:.2e} $T"
        )
        results[p] = {
            "eval": ev,
            "return_curve": np.asarray(curve),
            "csv": csv_path,
        }
        if args.save_agents:
            results[p]["agent"] = agent

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"cbam_experiment_themis_sweep_{args.env}_{ts}"
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    _summary_plot(prices, results, args.env, _os.path.join(OUTPUT_DIR, f"{stem}.png"))

    payload = {
        "prices": prices,
        "env_kind": args.env,
        "membership_mode": args.membership,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "num_regions": MRIO_NUM_REGIONS if args.env == "mrio" else CORE_NUM_REGIONS,
        "region_names": (
            {i: REGION_NAMES[i] for i in range(MRIO_NUM_REGIONS)}
            if args.env == "mrio"
            else {i: f"region {i}" for i in range(CORE_NUM_REGIONS)}
        ),
        "env_kwargs": canonical_env_kwargs() if args.env == "mrio" else {},
        "train_kwargs": canonical_train_kwargs(total_timesteps=args.timesteps),
        "results": results,
    }
    pkl_path = _os.path.join(OUTPUT_DIR, f"{stem}.pkl")
    with open(pkl_path, "wb") as fh:
        cloudpickle.dump(payload, fh)
    print(f"  results pkl → {pkl_path}")


if __name__ == "__main__":
    main()
