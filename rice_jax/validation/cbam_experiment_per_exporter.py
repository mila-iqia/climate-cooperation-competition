"""cbam_experiment_per_exporter.py

Per-exporter breakdown of revenue transfer effect under differential CBAM.

Structure
---------
Trains one model per revenue_share level ∈ {0.0, 0.25, 0.50, 0.75, 1.0}.
All models use:
  - cbam_tariff_mode="differential"  (replaces cbam_randomize)
  - transfer_mode="abatement", transfer_allocation="effort"
    (best settings from Phase 2B allocation comparison)
  - 2 M timesteps, 9-region vulnerability setup

Produces two panels:
  Panel A — Aggregate: EU dirty share vs revenue_share level, and aggregate
             mitigation rate vs revenue_share level. Confirms reallocation
             reduces diversion and raises mitigation in aggregate.

  Panel B — Per-exporter: for each CBAM-relevant region, EU dirty share and
             mitigation rate across all revenue_share levels.  Shows that
             exporters respond heterogeneously — some are diversion-elastic,
             others are mitigation-elastic.

Research question: which exporters need the most transfer to flip from
diversion to decarbonisation, and does any single allocation rule get
them all?

Exit criterion: with revenue_share=1.0, aggregate EU dirty share is lower
(or aggregate mitigation rate is higher) than with revenue_share=0.0.

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_per_exporter.py [--timesteps 2000000]
    python validation/cbam_experiment_per_exporter.py --replot <pickle.pkl>
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
import matplotlib.patches as mpatches
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


# ── Config ─────────────────────────────────────────────────────────────────

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
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]   # drop RoW

REVENUE_SHARE_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
_RS_COLORS = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(REVENUE_SHARE_LEVELS)))

TOTAL_TIMESTEPS   = 2_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42

CBAM_LAMBDA_INIT    = 1.0
WELFARE_LOSS_WEIGHT = 5.0
TRANSFER_MODE       = "abatement"

# EU net-zero ramp (EU Climate Law 2021/1119) — ensures MAC_EU > 0 from t=0
# so the differential tariff τ_eff = max(0,(MAC_EU−MAC_r)/MAC_EU) is non-trivial.
EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)
TRANSFER_ALLOC      = "effort"   # best from Phase 2B alloc comparison

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "per_exp_"

# ── Environment + PPO defaults ──────────────────────────────────────────────

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
    cbam_tariff_mode             = "differential",
    cbam_tariff_rate             = 0.0,   # not used in differential mode
    cbam_lambda_init             = CBAM_LAMBDA_INIT,
    reward_mode                  = "additive_cbam",
    log_info_fn                  = rcpo_cbam_log_info_fn,
    transfer_mode                = TRANSFER_MODE,
    transfer_allocation          = TRANSFER_ALLOC,
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


# ── Build / train helpers ───────────────────────────────────────────────────

def _build_env(revenue_share, for_training=True):
    env = RiceMRIO(
        region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        revenue_share = revenue_share,
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


def _train_one(revenue_share, key, total_timesteps):
    label     = f"rs{int(revenue_share * 100):03d}"
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    env       = _build_env(revenue_share, for_training=True)
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = MonitoredPPO(
        total_timesteps=total_timesteps,
        log_function=log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*55}")
    print(f"  Training: revenue_share={revenue_share:.2f}  [{label}]")
    print(f"{'━'*55}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path, label


# ── Evaluation helpers ──────────────────────────────────────────────────────

def _eval_agent(key, revenue_share, agent):
    """Evaluate agent for NUM_EVAL_EPISODES.

    Returns per-region arrays averaged over episodes:
      dirty_share : (NR,) — mean EU dirty export share
      mit_rate    : (NR,) — mean mitigation rate
      transfer    : (NR,) — mean transfer received per step
      util        : (NR,) — mean utility per step
    """
    from _experiment_util import run_single_episode

    raw_env  = _build_env(revenue_share, for_training=False)
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    dirty_shares   = []
    mit_rates_list = []
    transfer_list  = []
    util_list      = []

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 70_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        tf    = np.array(logs["trade_flows"])   # (T, NR, NR, NS)
        dirty = tf[:, :, EU_IDX, 0]             # (T, NR)
        clean = tf[:, :, EU_IDX, 1]             # (T, NR)
        total = dirty + clean + 1e-10
        dirty_shares.append((dirty / total).mean(axis=0))   # (NR,)

        mit = _to_arr(logs["mitigation_rates_all_regions"])  # (T, NR)
        mit_rates_list.append(mit.mean(axis=0))              # (NR,)

        T = tf.shape[0]
        tr = np.array(logs.get("transfer_received",
                               np.zeros((T, NUM_REGIONS))))
        transfer_list.append(tr.mean(axis=0))  # (NR,)

        util = _to_arr(logs["utility_all_regions"])  # (T, NR)
        util_list.append(util.mean(axis=0))          # (NR,)

    return dict(
        dirty_share = np.mean(dirty_shares,   axis=0),
        mit_rate    = np.mean(mit_rates_list, axis=0),
        transfer    = np.mean(transfer_list,  axis=0),
        util        = np.mean(util_list,      axis=0),
    )


# ── Plot ────────────────────────────────────────────────────────────────────

def _read_csv(path):
    if not path or not _os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def plot_results(results_list, timestamp, total_timesteps):
    """
    results_list: list of dicts with keys:
      revenue_share, csv_path, eval (dict from _eval_agent), label
    """
    n_rs = len(results_list)

    # ── Summary lines for aggregate panels ───────────────────────────────────
    agg_dirty = [
        np.mean([r["eval"]["dirty_share"][reg] for reg in CBAM_PLOT_REGIONS])
        for r in results_list
    ]
    agg_mu = [
        np.mean([r["eval"]["mit_rate"][reg] for reg in CBAM_PLOT_REGIONS])
        for r in results_list
    ]
    rs_vals = [r["revenue_share"] for r in results_list]
    rs_colors = {r["revenue_share"]: _RS_COLORS[i] for i, r in enumerate(results_list)}

    fig = plt.figure(figsize=(22, 26))
    fig.suptitle(
        f"Per-Exporter Response to Revenue Transfer — Differential CBAM\n"
        f"9-region vulnerability setup, {total_timesteps//1_000_000}M steps, "
        f"transfer_mode={TRANSFER_MODE}, allocation={TRANSFER_ALLOC}",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: Training convergence ──────────────────────────────────────────
    ax_conv = fig.add_subplot(gs[0, :])
    for res in results_list:
        df  = _read_csv(res["csv_path"])
        col = rs_colors[res["revenue_share"]]
        if not df.empty and "ep_return_mean" in df.columns:
            iters = df.get("iteration", pd.Series(range(len(df))))
            mean  = df["ep_return_mean"].rolling(10, min_periods=1).mean()
            ax_conv.plot(iters, mean, lw=1.5, color=col,
                         label=f"rs={res['revenue_share']:.2f}")
    ax_conv.set_xlabel("PPO iteration", fontsize=9)
    ax_conv.set_ylabel("ep_return_mean (rolling-10)", fontsize=9)
    ax_conv.set_title("Training convergence — all revenue_share levels", fontsize=10)
    ax_conv.legend(fontsize=8)
    ax_conv.tick_params(labelsize=8)

    # ── Row 1: Aggregate EU dirty share + aggregate mitigation ───────────────
    ax_agg_dirty = fig.add_subplot(gs[1, 0:2])
    ax_agg_mu    = fig.add_subplot(gs[1, 2:4])

    ax_agg_dirty.bar(range(n_rs), agg_dirty, color=[_RS_COLORS[i] for i in range(n_rs)],
                     edgecolor="k", linewidth=0.5)
    ax_agg_dirty.set_xticks(range(n_rs))
    ax_agg_dirty.set_xticklabels([f"{r['revenue_share']:.2f}" for r in results_list])
    ax_agg_dirty.set_xlabel("Revenue share", fontsize=9)
    ax_agg_dirty.set_ylabel("Aggregate EU dirty export share", fontsize=9)
    ax_agg_dirty.set_title("Aggregate: transfer reduces export diversion\n"
                            "(higher rs → lower dirty EU share)", fontsize=9)
    ax_agg_dirty.tick_params(labelsize=8)

    ax_agg_mu.bar(range(n_rs), agg_mu, color=[_RS_COLORS[i] for i in range(n_rs)],
                  edgecolor="k", linewidth=0.5)
    ax_agg_mu.set_xticks(range(n_rs))
    ax_agg_mu.set_xticklabels([f"{r['revenue_share']:.2f}" for r in results_list])
    ax_agg_mu.set_xlabel("Revenue share", fontsize=9)
    ax_agg_mu.set_ylabel("Aggregate mitigation rate μ", fontsize=9)
    ax_agg_mu.set_title("Aggregate: transfer increases mitigation\n"
                         "(higher rs → higher μ)", fontsize=9)
    ax_agg_mu.tick_params(labelsize=8)

    # ── Rows 2–3: Per-exporter EU dirty share and mitigation rate ─────────────
    # Grouped bars: x = exporters, bars = revenue share levels
    x       = np.arange(len(CBAM_PLOT_REGIONS))
    width   = 0.8 / n_rs
    xticklabels = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]

    ax_pr_dirty = fig.add_subplot(gs[2, :])
    ax_pr_mu    = fig.add_subplot(gs[3, :])

    for i, res in enumerate(results_list):
        col = rs_colors[res["revenue_share"]]
        dirty_vals = [res["eval"]["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        mu_vals    = [res["eval"]["mit_rate"][r]    for r in CBAM_PLOT_REGIONS]
        offset = (i - n_rs / 2 + 0.5) * width

        ax_pr_dirty.bar(x + offset, dirty_vals, width=width, color=col,
                        alpha=0.87, edgecolor="k", linewidth=0.3,
                        label=f"rs={res['revenue_share']:.2f}")
        ax_pr_mu.bar(x + offset, mu_vals, width=width, color=col,
                     alpha=0.87, edgecolor="k", linewidth=0.3,
                     label=f"rs={res['revenue_share']:.2f}")

    for ax, ylabel, title in [
        (ax_pr_dirty, "EU dirty export share",
         "Per-exporter EU dirty export share × revenue_share\n"
         "(heterogeneous diversion response: some exporters highly elastic)"),
        (ax_pr_mu, "Mitigation rate μ",
         "Per-exporter mitigation rate × revenue_share\n"
         "(heterogeneous mitigation response: not all exporters decarbonise equally)"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    # ── Row 4: Pass/fail badge + transfer received ────────────────────────────
    ax_transfer = fig.add_subplot(gs[4, 0:3])
    ax_badge    = fig.add_subplot(gs[4, 3])

    for i, res in enumerate(results_list):
        col = rs_colors[res["revenue_share"]]
        tr_vals = [res["eval"]["transfer"][r] for r in CBAM_PLOT_REGIONS]
        offset  = (i - n_rs / 2 + 0.5) * width
        ax_transfer.bar(x + offset, tr_vals, width=width, color=col,
                        alpha=0.87, edgecolor="k", linewidth=0.3,
                        label=f"rs={res['revenue_share']:.2f}")
    ax_transfer.set_xticks(x)
    ax_transfer.set_xticklabels(xticklabels, rotation=25, ha="right", fontsize=8)
    ax_transfer.set_ylabel("Mean transfer received (per step)", fontsize=9)
    ax_transfer.set_title("Transfer received per exporter × revenue_share\n"
                           "(confirms budget flows correctly to CBAM-exposed regions)", fontsize=9)
    ax_transfer.legend(fontsize=7)
    ax_transfer.tick_params(labelsize=8)

    # Badge
    ax_badge.axis("off")
    rs0 = results_list[0]["eval"]["dirty_share"]
    rs1 = results_list[-1]["eval"]["dirty_share"]
    mu0 = results_list[0]["eval"]["mit_rate"]
    mu1 = results_list[-1]["eval"]["mit_rate"]
    agg_dirty_delta = float(np.mean([rs1[r] - rs0[r] for r in CBAM_PLOT_REGIONS]))
    agg_mu_delta    = float(np.mean([mu1[r] - mu0[r] for r in CBAM_PLOT_REGIONS]))
    exit_passed = (agg_dirty_delta < 0) or (agg_mu_delta > 0)
    color = "#1a6b3a" if exit_passed else "#8b1a1a"
    lines = [
        "Exit Criterion (rs=1.0 vs rs=0.0)",
        "─" * 38,
        f"  Δ agg dirty share : {agg_dirty_delta:+.3f}",
        f"  Δ agg mitigation  : {agg_mu_delta:+.3f}",
        "",
        ("✅ PASS: transfer reduces diversion"
         if exit_passed else "❌ FAIL: no aggregate improvement"),
        "",
        "Per-exporter breakdown shows heterogeneous",
        "response — not all exporters respond equally.",
    ]
    ax_badge.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
                  transform=ax_badge.transAxes, fontsize=8.5, fontfamily="monospace",
                  bbox=dict(boxstyle="round,pad=0.6", facecolor=color,
                             alpha=0.12, edgecolor=color, linewidth=2))

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_per_exporter_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)
    return out_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--replot",    type=str, default=None,
                        help="Path to existing .pkl — skip training, regenerate plot only")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            saved = pickle.load(f)
        total_ts = saved.get("total_timesteps", args.timesteps)
        plot_results(saved["results_list"], timestamp, total_ts)
        return

    total_timesteps = args.timesteps
    key  = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, len(REVENUE_SHARE_LEVELS) + 1)
    eval_key = keys[-1]

    results_list = []
    for i, rs in enumerate(REVENUE_SHARE_LEVELS):
        agent, csv_path, label = _train_one(rs, keys[i], total_timesteps)
        print(f"\n  Evaluating rs={rs:.2f} …")
        eval_data = _eval_agent(eval_key, rs, agent)
        results_list.append(dict(
            revenue_share = rs,
            csv_path      = csv_path,
            label         = label,
            eval          = eval_data,
            ppo           = agent,      # saved for post-hoc Jacobian / weight analysis
        ))

    # Save pickle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_per_exporter_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"results_list": results_list,
                     "total_timesteps": total_timesteps}, f)
    print(f"\n  Results saved: {pkl_path}")

    plot_results(results_list, timestamp, total_timesteps)

    # Print summary table
    print("\n" + "="*55)
    print("  PER-EXPORTER SUMMARY")
    print("="*55)
    header = f"  {'Region':<16}" + "".join(f"  rs={rs:.2f}" for rs in REVENUE_SHARE_LEVELS)
    print(header)
    print("  Dirty EU share:")
    for r in CBAM_PLOT_REGIONS:
        row = f"  {REGION_NAMES[r]:<16}"
        for res in results_list:
            row += f"  {res['eval']['dirty_share'][r]:.3f} "
        print(row)
    print("  Mitigation rate μ:")
    for r in CBAM_PLOT_REGIONS:
        row = f"  {REGION_NAMES[r]:<16}"
        for res in results_list:
            row += f"  {res['eval']['mit_rate'][r]:.3f} "
        print(row)


if __name__ == "__main__":
    main()
