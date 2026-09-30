"""cbam_experiment_2b_alloc.py

Phase 2B — Transfer Allocation Policy Comparison.

Default: trains one model per allocation rule at fixed revenue_share=1.0,
transfer_mode="abatement", differential CBAM tariff (per-region MAC-based
τ_eff[r] = max(0, MAC_EU − MAC_r) / MAC_EU).  Flat τ=0.80 also available
via --tariff-mode flat.

Allocation rules compared:
  burden      — proportional to CBAM cost c_r (EU CBAM Regulation default)
  effort      — proportional to mitigation rate μ_r (REDD+ style)
  equal       — uniform split (1/(NR-1)) across non-EU exporters
  vulnerability — proportional to c_r / Y_r (exposure-per-GDP)

For each model, evaluates per-region:
  - EU dirty export share  (lower = more diversion, higher = less diversion)
  - Mean mitigation rate μ (higher = more abatement)

Research question: which allocation rule most effectively shifts exporters
toward mitigation rather than diversion?  Expected ranking:
  effort > vulnerability ≈ burden > equal   (for μ)
  effort > burden > equal ≈ vulnerability   (for EU dirty share)
with heterogeneous region-level effects driven by CBAM exposure structure.

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_2b_alloc.py [--timesteps 1000000]
    python validation/cbam_experiment_2b_alloc.py --tariff-mode flat
    python validation/cbam_experiment_2b_alloc.py --replot <pickle.pkl>
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
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from _experiment_util import get_output_dir, get_log_dir


# ── Config ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT   = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", ".."))

NUM_REGIONS = 9
EU_IDX      = 3
YAML_DIR    = _os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_ROOT   = _os.path.join(_REPO_ROOT, "csv_asset")

REGION_NAMES = {
    0: "RoW", 1: "Russia+Eur.", 2: "MENA", 3: "EU",
    4: "SSA-Mining", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
NON_EU            = [r for r in range(NUM_REGIONS) if r != EU_IDX]
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]  # drop RoW

TOTAL_TIMESTEPS   = 1_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42
CBAM_RATE         = 0.80       # used only for flat mode; differential ignores this
CBAM_TARIFF_MODE  = "differential"  # "differential" (default) or "flat"
CBAM_LAMBDA_INIT  = 1.0
WELFARE_LOSS_WEIGHT = 5.0

# EU net-zero ramp (EU Climate Law 2021/1119) — ensures MAC_EU > 0 from t=0
# so the differential tariff τ_eff = max(0,(MAC_EU−MAC_r)/MAC_EU) is non-trivial.
EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)
REVENUE_SHARE     = 1.0   # always 100% recycling in this experiment
TRANSFER_MODE     = "abatement"  # always earmarked

# Allocation rules to compare (in display order)
ALLOCATION_RULES = ["burden", "effort", "equal", "vulnerability", "hybrid"]
ALLOC_LABELS = {
    "burden":        "Burden\n(∝ CBAM cost)",
    "effort":        "Effort\n(∝ μ)",
    "equal":         "Equal\n(uniform)",
    "vulnerability": "Vulnerability\n(∝ cost/GDP)",
    "hybrid":        "Hybrid\n(μ × cost/GDP)",
}
ALLOC_COLORS = {
    "burden":        "#d62728",  # red
    "effort":        "#2ca02c",  # green
    "equal":         "#1f77b4",  # blue
    "vulnerability": "#ff7f0e",  # orange
    "hybrid":        "#9467bd",  # purple
}

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR    = get_log_dir("training_logs")
LOG_PREFIX = "cbam2b_alloc_"

_BASE_ENV = dict(
    num_regions                  = NUM_REGIONS,
    mrio_data_root               = MRIO_ROOT,
    mrio_trade                   = True,
    eu_region_idx                = EU_IDX,
    dest_alloc_persistence       = 0.55,
    dest_alloc_baseline_decay    = 0.0,
    diff_reward_mode             = True,
    num_discrete_action_levels   = 10,
    sector_granularity           = "emissions-simple",
    sectoral_welfloss            = True,
    welfare_loss_per_unit_tariff = WELFARE_LOSS_WEIGHT,
    eu_mitigation_schedule       = EU_MITIGATION_SCHEDULE,
)

_PPO_KWARGS = dict(
    num_steps              = NUM_STEPS,
    num_envs               = NUM_ENVS,
    learning_rate          = 3e-4,
    num_minibatches        = 4,
    num_epochs             = 8,
    ent_coef               = 0.01,
    anneal_ent_coef        = 0.0,
    gamma                  = 0.99,
    gae_lambda             = 0.95,
    max_grad_norm          = 1.0,
    clip_coef              = 0.2,
    clip_coef_vf           = 0.5,
    vf_coef                = 0.5,
    normalize_observations = True,
    normalize_rewards      = True,
    log_interval           = 50,
)


# ── Build / train helpers ─────────────────────────────────────────────────────

def _build_env(allocation, for_training=True, tariff_mode=None):
    mode = tariff_mode or CBAM_TARIFF_MODE
    env = RiceMRIO(
        region_params         = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_rate      = CBAM_RATE,
        cbam_tariff_mode      = mode,
        cbam_lambda_init      = CBAM_LAMBDA_INIT,
        revenue_share         = REVENUE_SHARE,
        transfer_mode         = TRANSFER_MODE,
        transfer_allocation   = allocation,
        reward_mode           = "additive_cbam",
        log_info_fn           = rcpo_cbam_log_info_fn,
        **_BASE_ENV,
    )
    return jym.LogWrapper(env) if for_training else env


def _make_log_fn(label, num_iters):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _DROP = {"action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _DROP}, iteration)

    return make_combined_log_fn(
        _compact,
        make_csv_log_fn(csv_path),
    ), csv_path


def _train(allocation, key, total_timesteps, tariff_mode=None):
    mode = tariff_mode or CBAM_TARIFF_MODE
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    env       = _build_env(allocation, for_training=True, tariff_mode=mode)
    log_fn, csv_path = _make_log_fn(allocation, num_iters)
    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*60}")
    print(f"  Training allocation={allocation!r}  tariff={mode}  "
          f"(rs={REVENUE_SHARE}, mode={TRANSFER_MODE})")
    print(f"{'━'*60}")
    t0  = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval(key, raw_env, agent):
    """Return (dirty_share, transfer_mean, mit_rate) each shape (NR,)."""
    from _experiment_util import run_single_episode

    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    dirty_shares, transfer_means, mit_rates = [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 60_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        tf    = np.array(logs["trade_flows"])          # (T, NR, NR, NS)
        T     = tf.shape[0]
        dirty = tf[:, :, EU_IDX, 0]                    # (T, NR)
        clean = tf[:, :, EU_IDX, 1]
        share = (dirty / (dirty + clean + 1e-10)).mean(axis=0)
        dirty_shares.append(share)

        tr = np.array(logs.get(
            "transfer_received", np.zeros((T, NUM_REGIONS))
        ))
        transfer_means.append(tr.mean(axis=0))

        mit = _to_arr(logs["mitigation_rates_all_regions"])  # (T, NR)
        mit_rates.append(mit.mean(axis=0))

    return (
        np.mean(dirty_shares,   axis=0),
        np.mean(transfer_means, axis=0),
        np.mean(mit_rates,      axis=0),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(results, timestamp):
    """
    results: list of dicts with keys:
      allocation, dirty_share (NR,), transfer_mean (NR,), mit_rate (NR,),
      csv_path, label
    """
    n_alloc = len(results)
    x       = np.arange(len(CBAM_PLOT_REGIONS))
    width   = 0.8 / n_alloc

    fig = plt.figure(figsize=(22, 16))
    tariff_mode = results[0].get("tariff_mode", CBAM_TARIFF_MODE)
    tariff_label = (
        "differential (MAC-based)" if tariff_mode == "differential"
        else f"flat τ={CBAM_RATE:.0%}"
    )
    fig.suptitle(
        f"Phase 2B — Transfer Allocation Policy Comparison\n"
        f"tariff={tariff_label}, rs={REVENUE_SHARE:.0%}, mode={TRANSFER_MODE}, "
        f"9-region vuln, {TOTAL_TIMESTEPS//1_000_000:.0f}M steps",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

    ax_conv   = fig.add_subplot(gs[0, 0])   # training convergence
    ax_dirty  = fig.add_subplot(gs[0, 1])   # EU dirty share
    ax_mit    = fig.add_subplot(gs[0, 2])   # mitigation rate
    ax_trans  = fig.add_subplot(gs[1, 0])   # transfer received
    ax_scatter= fig.add_subplot(gs[1, 1])   # μ vs dirty share scatter
    ax_delta  = fig.add_subplot(gs[1, 2])   # Δ vs burden baseline
    ax_badge  = fig.add_subplot(gs[2, :])   # summary table

    alloc_colors = [ALLOC_COLORS[r["allocation"]] for r in results]

    # ── Convergence ──────────────────────────────────────────────────────────
    for res, col in zip(results, alloc_colors):
        try:
            df = pd.read_csv(res["csv_path"])
            if "ep_return_mean" in df.columns:
                ax_conv.plot(
                    df["ep_return_mean"].rolling(10).mean(),
                    color=col, linewidth=1.4,
                    label=res["allocation"],
                )
        except Exception:
            pass
    ax_conv.set_xlabel("PPO iteration")
    ax_conv.set_ylabel("ep_return_mean (rolling-10)")
    ax_conv.set_title("Training convergence")
    ax_conv.legend(fontsize=8)

    # ── EU dirty share ────────────────────────────────────────────────────────
    for i, (res, col) in enumerate(zip(results, alloc_colors)):
        shares = [res["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        ax_dirty.bar(
            x + (i - n_alloc/2 + 0.5) * width, shares,
            width=width, color=col, alpha=0.85,
            label=res["allocation"],
        )
    ax_dirty.set_xticks(x)
    ax_dirty.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                              rotation=25, ha="right")
    ax_dirty.set_ylabel("EU dirty export share (mean)")
    ax_dirty.set_title("EU dirty share by region\n(higher = less diversion)")
    ax_dirty.legend(fontsize=8)

    # ── Mitigation rate ───────────────────────────────────────────────────────
    for i, (res, col) in enumerate(zip(results, alloc_colors)):
        mit = [res["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        ax_mit.bar(
            x + (i - n_alloc/2 + 0.5) * width, mit,
            width=width, color=col, alpha=0.85,
            label=res["allocation"],
        )
    ax_mit.set_xticks(x)
    ax_mit.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                            rotation=25, ha="right")
    ax_mit.set_ylabel("Mean mitigation rate μ")
    ax_mit.set_title("Mitigation rate by region\n(higher = more abatement)")
    ax_mit.legend(fontsize=8)

    # ── Transfer received ─────────────────────────────────────────────────────
    for i, (res, col) in enumerate(zip(results, alloc_colors)):
        transfers = [res["transfer_mean"][r] for r in CBAM_PLOT_REGIONS]
        ax_trans.bar(
            x + (i - n_alloc/2 + 0.5) * width, transfers,
            width=width, color=col, alpha=0.85,
            label=res["allocation"],
        )
    ax_trans.set_xticks(x)
    ax_trans.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                              rotation=25, ha="right")
    ax_trans.set_ylabel("Mean transfer received (per step)")
    ax_trans.set_title("Transfer received — how the pool is split\nby allocation rule")
    ax_trans.legend(fontsize=8)

    # ── μ vs EU dirty share scatter ───────────────────────────────────────────
    markers = {"burden": "o", "effort": "^", "equal": "s", "vulnerability": "D", "hybrid": "P"}
    for res, col in zip(results, alloc_colors):
        alloc = res["allocation"]
        for r in CBAM_PLOT_REGIONS:
            ax_scatter.scatter(
                res["mit_rate"][r], res["dirty_share"][r],
                color=col, marker=markers[alloc], s=70, alpha=0.85,
                label=alloc if r == CBAM_PLOT_REGIONS[0] else None,
            )
            ax_scatter.annotate(
                REGION_NAMES[r][:3],
                (res["mit_rate"][r], res["dirty_share"][r]),
                fontsize=6, alpha=0.7,
            )
    handles, labels = ax_scatter.get_legend_handles_labels()
    ax_scatter.legend(dict(zip(labels, handles)).values(),
                      dict(zip(labels, handles)).keys(), fontsize=7)
    ax_scatter.set_xlabel("Mean mitigation rate μ")
    ax_scatter.set_ylabel("EU dirty export share")
    ax_scatter.set_title("μ vs EU dirty share per region × rule\n"
                         "Ideal: effort → up-left vs burden")

    # ── Δ vs burden baseline (per region) ────────────────────────────────────
    burden_res = next((r for r in results if r["allocation"] == "burden"), results[0])
    for res, col in zip(results, alloc_colors):
        if res["allocation"] == "burden":
            continue
        delta_mit   = [res["mit_rate"][r]    - burden_res["mit_rate"][r]
                       for r in CBAM_PLOT_REGIONS]
        delta_dirty = [res["dirty_share"][r] - burden_res["dirty_share"][r]
                       for r in CBAM_PLOT_REGIONS]
        ax_delta.bar(
            x + (ALLOCATION_RULES.index(res["allocation"]) - 1.5) * width * 0.9,
            delta_mit,
            width=width * 0.9, color=col, alpha=0.85,
            label=f"Δμ [{res['allocation']}]",
        )
        # Overlay dirty share delta as a step line
        ax_delta.step(
            x + (ALLOCATION_RULES.index(res["allocation"]) - 1.5) * width * 0.9,
            delta_dirty,
            color=col, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Δdirty [{res['allocation']}]",
            where="mid",
        )
    ax_delta.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                              rotation=25, ha="right")
    ax_delta.set_ylabel("Δ vs burden baseline")
    ax_delta.set_title("Bars: Δμ  Dashed: Δdirty share\nvs burden allocation")
    ax_delta.legend(fontsize=6, ncol=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_badge.axis("off")
    col_headers = ["Rule"] + [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS] + ["Mean"]
    table_data  = []
    for section, key, fmt in [("μ (mitigation)", "mit_rate", ".3f"),
                               ("EU dirty share", "dirty_share", ".3f")]:
        table_data.append([f"── {section} ──"] + [""] * (len(CBAM_PLOT_REGIONS) + 1))
        for res in results:
            vals  = [res[key][r] for r in CBAM_PLOT_REGIONS]
            row   = [res["allocation"]] + [f"{v:{fmt}}" for v in vals] + [f"{np.mean(vals):{fmt}}"]
            table_data.append(row)

    tbl = ax_badge.table(
        cellText    = table_data,
        colLabels   = col_headers,
        loc         = "center",
        cellLoc     = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    ax_badge.set_title("Summary: mitigation rate μ and EU dirty share per rule",
                       fontsize=9, pad=4)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_2b_alloc_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing .pkl — regenerate plot without retraining")
    parser.add_argument(
        "--rules", nargs="+", default=ALLOCATION_RULES,
        choices=ALLOCATION_RULES,
        help="Allocation rules to run (default: all five)",
    )
    parser.add_argument(
        "--tariff-mode", default=CBAM_TARIFF_MODE,
        choices=["differential", "flat"],
        help="CBAM tariff mode (default: differential)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} models from {args.replot}")
        _plot_results(results, timestamp)
        return

    root_key = jax.random.PRNGKey(args.seed)
    results  = []

    for i, rule in enumerate(args.rules):
        train_key = jax.random.fold_in(root_key, i)
        ppo, csv_path = _train(rule, train_key, args.timesteps,
                               tariff_mode=args.tariff_mode)

        eval_key = jax.random.fold_in(root_key, 200 + i)
        raw_env  = _build_env(rule, for_training=False,
                              tariff_mode=args.tariff_mode)
        dirty_share, transfer_mean, mit_rate = _eval(eval_key, raw_env, ppo)

        results.append({
            "allocation":    rule,
            "dirty_share":   dirty_share,
            "transfer_mean": transfer_mean,
            "mit_rate":      mit_rate,
            "csv_path":      csv_path,
            "label":         rule,
            "tariff_mode":   args.tariff_mode,
        })
        mean_mit   = np.mean([mit_rate[r]    for r in CBAM_PLOT_REGIONS])
        mean_dirty = np.mean([dirty_share[r] for r in CBAM_PLOT_REGIONS])
        print(f"\n  {rule:14s}  μ(CBAM)={mean_mit:.4f}  dirty(CBAM)={mean_dirty:.4f}")

    # ── Save ────────────────────────────────────────────────────────────────
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_2b_alloc_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nPickle saved → {pkl_path}")

    _plot_results(results, timestamp)

    # ── Console summary ──────────────────────────────────────────────────────
    burden_res = next(r for r in results if r["allocation"] == "burden")
    print("\n" + "═" * 65)
    print("  ALLOCATION POLICY COMPARISON — Δ vs burden baseline")
    print("  (+ = better outcome, - = worse)")
    print("─" * 65)
    for res in results:
        if res["allocation"] == "burden":
            continue
        dm = np.mean([res["mit_rate"][r] - burden_res["mit_rate"][r]
                      for r in CBAM_PLOT_REGIONS])
        dd = np.mean([res["dirty_share"][r] - burden_res["dirty_share"][r]
                      for r in CBAM_PLOT_REGIONS])
        print(f"  {res['allocation']:14s}  Δμ={dm:+.4f}  Δdirty={dd:+.4f}  "
              f"({'✅' if dm > 0 else '—'}  {'✅' if dd > 0 else '—'})")
    print("═" * 65)


if __name__ == "__main__":
    main()
