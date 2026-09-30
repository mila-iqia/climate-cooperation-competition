"""cbam_experiment_2b_tier1.py

Phase 2B Tier 1: Fixed-fraction revenue transfer ablation.

Train 5 separate models with revenue_share ∈ {0.0, 0.25, 0.50, 0.75, 1.0},
all on the 9-region vulnerability setup with tau=0.80.

For each model, evaluate:
  - EU dirty export share per region (does transfer reduce diversion?)
  - Transfer received per region (confirms budget flows correctly)
  - Training convergence (ep_return vs iteration)

Research question: at what transfer level does diversion flip toward
decarbonisation?  Gives a "minimum transfer threshold" comparable to
CGE literature (Böhringer et al. 2010).

Exit criterion: with revenue_share=1.0, EU dirty share is higher than
revenue_share=0.0 for at least one CBAM-exposed CBAM_PLOT_REGIONS member
(i.e. transfer reduces diversion for at least one exporter).

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_2b_tier1.py [--timesteps 1000000]
    python validation/cbam_experiment_2b_tier1.py --replot <pickle.pkl>
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
from _experiment_util import get_output_dir, get_log_dir
from validation.canonical_config import (
    NUM_REGIONS,
    EU_REGION_IDX as EU_IDX,
    REGION_NAMES,
    NON_EU_IDXS,
    NON_EU_EXPORTER_IDXS,
    CANONICAL_TRAIN_KWARGS,
    NUM_EVAL_EPISODES,
    EVAL_LAST_T,
    canonical_env_kwargs as _canonical_env_kwargs,
    make_canonical_env,
)
from validation import metrics as _metrics


# ── Config ───────────────────────────────────────────────────────────────────
# Region indexing, env defaults, and PPO kwargs are sourced from canonical_config.
# Only experiment-local knobs remain here.
#
# NOTE: this is the LEGACY tier-1 experiment using flat τ=0.80. The canonical
# default is differential CBAM; this script preserves its historical flat-τ
# setting via an explicit cbam_tariff_mode="flat" override.

NON_EU            = list(NON_EU_IDXS)
CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)    # drop RoW + EU

TOTAL_TIMESTEPS   = 1_000_000                     # legacy tier-1 timestep budget
NUM_ENVS          = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS         = CANONICAL_TRAIN_KWARGS["num_steps"]
SEED              = 42
CBAM_RATE         = 0.80                          # flat tariff (legacy)
CBAM_LAMBDA_INIT  = _canonical_env_kwargs()["cbam_lambda_init"]
WELFARE_LOSS_WEIGHT = _canonical_env_kwargs()["welfare_loss_per_unit_tariff"]

# EU net-zero ramp (EU Climate Law 2021/1119) — kept for parity with the
# differential variant of this experiment; ignored under flat τ.
EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)

# Ablation levels — these are the 5 Tier 1 models
REVENUE_SHARE_LEVELS = [0.0, 0.25, 0.50, 0.75, 1.0]
TRANSFER_MODE_COMPACT_SHARES = [0.0, 0.50, 1.0]

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR    = get_log_dir("training_logs")
LOG_PREFIX = "cbam2b_t1_"

_SHARE_COLORS = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(REVENUE_SHARE_LEVELS)))
_MODE_STYLE   = {"consumption": dict(hatch="",  alpha=0.85),
                 "abatement":   dict(hatch="//", alpha=0.85)}


# ── PPO kwargs (training-only; env defaults via canonical_config) ──────────

_PPO_KWARGS = {k: v for k, v in CANONICAL_TRAIN_KWARGS.items()
               if k != "total_timesteps"}


# ── Build / train helpers ─────────────────────────────────────────────────────

def _build_env(revenue_share, transfer_mode="consumption", for_training=True):
    """Build a tier-1 env via the canonical factory.

    Legacy behavior: flat τ=0.80, no differential mode. To run the differential
    variant, swap cbam_tariff_mode="differential" and add an eu_mitigation_schedule.
    """
    return make_canonical_env(
        for_training       = for_training,
        cbam_tariff_mode   = "flat",
        cbam_tariff_rate   = CBAM_RATE,
        revenue_share      = revenue_share,
        transfer_mode      = transfer_mode,
        eu_mitigation_schedule = EU_MITIGATION_SCHEDULE,
    )


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


def _train(revenue_share, key, total_timesteps, transfer_mode="consumption"):
    label     = f"rs{int(revenue_share*100):03d}_{transfer_mode[:4]}"
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    env       = _build_env(revenue_share, transfer_mode=transfer_mode, for_training=True)
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*55}")
    print(f"  Training rs={revenue_share:.2f}  mode={transfer_mode}  [{label}]")
    print(f"{'━'*55}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path, label


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eu_dirty_share_per_region(key, raw_env, agent):
    """Return mean EU dirty export share, transfer received, and mitigation rate
    per region over NUM_EVAL_EPISODES."""
    from _experiment_util import run_single_episode

    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)
    dirty_shares   = []  # list of (NR,) arrays
    transfer_means = []
    mit_rates      = []

    def _to_arr(d):
        """dict {region_idx: array(T)} → np array (T, NR)."""
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 50_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        tf    = np.array(logs["trade_flows"])  # (T, NR, NR, NS)  [from, to, s]
        T     = tf.shape[0]
        dirty = tf[:, :, EU_IDX, 0]   # (T, NR) — dirty sector (s=0) to EU
        clean = tf[:, :, EU_IDX, 1]   # (T, NR)
        total = dirty + clean + 1e-10
        share = (dirty / total).mean(axis=0)  # (NR,) — mean over time steps
        dirty_shares.append(share)

        tr = np.array(logs.get("transfer_received", np.zeros((T, NUM_REGIONS))))
        transfer_means.append(tr.mean(axis=0))  # (NR,)

        mit = _to_arr(logs["mitigation_rates_all_regions"])  # (T, NR)
        mit_rates.append(mit.mean(axis=0))  # (NR,) — mean over episode steps

    return (
        np.mean(dirty_shares,   axis=0),
        np.mean(transfer_means, axis=0),
        np.mean(mit_rates,      axis=0),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(results, timestamp):
    """
    results: list of dicts with keys:
      revenue_share, transfer_mode, dirty_share (NR,), transfer_mean (NR,), csv_path, label
    """
    # Detect whether both transfer modes are present
    modes_present = sorted(set(r.get("transfer_mode", "consumption") for r in results))
    multi_mode    = len(modes_present) > 1
    # Layout: 3 rows × 3 cols (multi-mode) or 3 rows × 2 cols (single mode)
    # Row 0: convergence | EU dirty share
    # Row 1: transfer    | mitigation rate
    # Row 2: mode comparison (multi only) | pass/fail badge
    ncols = 3 if multi_mode else 2
    nrows = 3 if multi_mode else 3
    fig = plt.figure(figsize=(9 * ncols, 9 * nrows // 2))
    mode_label = "Consumption vs Abatement" if multi_mode else "Revenue Transfer Ablation"
    fig.suptitle(
        f"Phase 2B Tier 1 — {mode_label} (τ={CBAM_RATE:.0%})\n"
        f"9-region vulnerability setup, {TOTAL_TIMESTEPS//1_000_000:.0f}M steps",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.45, wspace=0.35)

    ax_conv   = fig.add_subplot(gs[0, 0])   # convergence
    ax_dirty  = fig.add_subplot(gs[0, 1])   # EU dirty share per region × share level
    ax_trans  = fig.add_subplot(gs[1, 0])   # transfer received per region × share level
    ax_mit    = fig.add_subplot(gs[1, 1])   # mitigation rate per region × share level
    ax_badge  = fig.add_subplot(gs[2, -1])  # pass/fail summary
    ax_mode   = fig.add_subplot(gs[2, 0]) if multi_mode else fig.add_subplot(gs[2, 0])  # mode comparison / extra

    # ── Convergence ──────────────────────────────────────────────────────────
    rs_levels_seen = sorted(set(r["revenue_share"] for r in results))
    rs_colors = {rs: plt.cm.RdYlGn(i / max(len(rs_levels_seen) - 1, 1))
                 for i, rs in enumerate(rs_levels_seen)}
    for res in results:
        try:
            df  = pd.read_csv(res["csv_path"])
            tm  = res.get("transfer_mode", "consumption")
            col = rs_colors[res["revenue_share"]]
            ls  = "--" if tm == "abatement" else "-"
            if "ep_return_mean" in df.columns:
                ax_conv.plot(
                    df["ep_return_mean"].rolling(10).mean(),
                    color=col, linestyle=ls,
                    label=f"rs={res['revenue_share']:.2f} [{tm[:4]}]",
                    linewidth=1.4,
                )
        except Exception:
            pass
    ax_conv.set_xlabel("PPO iteration")
    ax_conv.set_ylabel("ep_return_mean (rolling-10)")
    ax_conv.set_title("Training convergence\n(solid=consumption, dashed=abatement)")
    ax_conv.legend(fontsize=7)

    # ── EU dirty share bar chart ──────────────────────────────────────────────
    x     = np.arange(len(CBAM_PLOT_REGIONS))
    width = 0.8 / len(results)
    for i, res in enumerate(results):
        col = rs_colors[res["revenue_share"]]
        tm  = res.get("transfer_mode", "consumption")
        shares = [res["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        ax_dirty.bar(
            x + (i - len(results)/2 + 0.5) * width,
            shares, width=width, color=col,
            label=f"rs={res['revenue_share']:.2f} [{tm[:4]}]",
            **{k: v for k, v in _MODE_STYLE[tm].items()},
        )
    ax_dirty.set_xticks(x)
    ax_dirty.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS], rotation=25, ha="right")
    ax_dirty.set_ylabel("EU dirty export share (mean)")
    ax_dirty.set_title("EU dirty share by region × transfer level\n(solid=consumption  //=abatement)")
    ax_dirty.legend(fontsize=7)

    # ── Transfer received ─────────────────────────────────────────────────────
    for i, res in enumerate(results):
        col = rs_colors[res["revenue_share"]]
        tm  = res.get("transfer_mode", "consumption")
        transfers = [res["transfer_mean"][r] for r in CBAM_PLOT_REGIONS]
        ax_trans.bar(
            x + (i - len(results)/2 + 0.5) * width,
            transfers, width=width, color=col,
            label=f"rs={res['revenue_share']:.2f} [{tm[:4]}]",
            **{k: v for k, v in _MODE_STYLE[tm].items()},
        )
    ax_trans.set_xticks(x)
    ax_trans.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS], rotation=25, ha="right")
    ax_trans.set_ylabel("Mean transfer received (per step)")
    ax_trans.set_title("Transfer received per region × transfer level")
    ax_trans.legend(fontsize=7)

    # ── Mitigation rate bar chart ─────────────────────────────────────────────
    for i, res in enumerate(results):
        col = rs_colors[res["revenue_share"]]
        tm  = res.get("transfer_mode", "consumption")
        mit = [res["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        ax_mit.bar(
            x + (i - len(results)/2 + 0.5) * width,
            mit, width=width, color=col,
            label=f"rs={res['revenue_share']:.2f} [{tm[:4]}]",
            **{k: v for k, v in _MODE_STYLE[tm].items()},
        )
    ax_mit.set_xticks(x)
    ax_mit.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS], rotation=25, ha="right")
    ax_mit.set_ylabel("Mean mitigation rate μ")
    ax_mit.set_title("Mitigation rate per region × transfer level\n(higher rs → higher μ expected under Mode B)")
    ax_mit.legend(fontsize=7)

    # ── Mode comparison panel (only when both modes present) ─────────────────
    # ── Mode comparison / μ vs dirty scatter (row 2, col 0/1) ───────────────
    if multi_mode:
        # Scatter: mitigation rate (x) vs EU dirty share (y) per region × condition
        # Expectation: Mode B moves each region up-left (higher μ, lower dirty share)
        idx = {(r["revenue_share"], r.get("transfer_mode", "consumption")): r
               for r in results}
        markers = {"consumption": "o", "abatement": "^"}
        for res in results:
            rs  = res["revenue_share"]
            tm  = res.get("transfer_mode", "consumption")
            col = rs_colors.get(rs, "gray")
            for r in CBAM_PLOT_REGIONS:
                ax_mode.scatter(
                    res["mit_rate"][r],
                    res["dirty_share"][r],
                    color=col, marker=markers[tm],
                    s=60, alpha=0.85,
                    label=f"rs={rs:.1f} [{tm[:4]}]" if r == CBAM_PLOT_REGIONS[0] else None,
                )
                ax_mode.annotate(
                    REGION_NAMES[r][:3],
                    (res["mit_rate"][r], res["dirty_share"][r]),
                    fontsize=6, alpha=0.7,
                )
        # Deduplicate legend
        handles, labels = ax_mode.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_mode.legend(by_label.values(), by_label.keys(), fontsize=6)
        ax_mode.set_xlabel("Mean mitigation rate μ")
        ax_mode.set_ylabel("EU dirty export share")
        ax_mode.set_title("μ vs EU dirty share (circle=consumption, triangle=abatement)\n"
                          "Ideal: Mode B → up-left shift")
    else:
        # Single mode: just turn the axes off (badge fills row 2)
        ax_mode.axis("off")

    # ── Pass/fail badge ───────────────────────────────────────────────────────
    ax_badge.axis("off")
    lines = ["Exit criterion check", "─" * 36]

    # Use consumption-mode rs=0 as baseline (no-transfer reference)
    baseline_res = next(
        (r for r in results
         if r["revenue_share"] == 0.0
         and r.get("transfer_mode", "consumption") == "consumption"),
        results[0],
    )
    rs0_shares = baseline_res["dirty_share"]

    any_increase = False
    for res in results:
        if res["revenue_share"] == 0.0:
            continue
        rs   = res["revenue_share"]
        tm   = res.get("transfer_mode", "consumption")
        rs1s = res["dirty_share"]
        deltas = [rs1s[r] - rs0_shares[r] for r in CBAM_PLOT_REGIONS]
        mean_d = np.mean(deltas)
        sign   = "▲" if mean_d > 0 else "▼"
        lines.append(f"  rs={rs:.2f} [{tm[:4]}]  mean Δ={sign}{abs(mean_d):.3f}")
        if any(d > 0 for d in deltas):
            any_increase = True

    lines.append("")
    if any_increase:
        lines.append("✅  PASS: transfer raises EU dirty share")
        lines.append("       (reduces diversion) in ≥1 region")
    else:
        lines.append("❌  FAIL: no region shows reduced diversion")

    # Also report mitigation direction for Mode B
    abatement_results = [r for r in results if r.get("transfer_mode") == "abatement"]
    if abatement_results and len(abatement_results) > 1:
        lines.append("")
        lines.append("Mode B mitigation check (higher rs → higher μ?)")
        rs0_ab = min(abatement_results, key=lambda r: r["revenue_share"])
        rs1_ab = max(abatement_results, key=lambda r: r["revenue_share"])
        for r in CBAM_PLOT_REGIONS:
            dm = rs1_ab["mit_rate"][r] - rs0_ab["mit_rate"][r]
            sign = "▲" if dm > 0 else "▼"
            lines.append(f"  {REGION_NAMES[r]:14s}  μ {sign}{abs(dm):.3f}")

    ax_badge.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_badge.transAxes, va="top", fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"),
    )

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_2b_tier1_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing .pkl — skip training, regenerate plot only")
    parser.add_argument("--shares", type=float, nargs="+", default=REVENUE_SHARE_LEVELS,
                        help="Revenue share levels to run (default: 0 .25 .5 .75 1)")
    parser.add_argument(
        "--transfer-mode", type=str, default="consumption",
        choices=["consumption", "abatement", "both"],
        help=(
            "Transfer mode: 'consumption' (Mode A, default), "
            "'abatement' (Mode B — earmarked to offset abatement cost), "
            "or 'both' (runs A and B at rs ∈ {0.0, 0.5, 1.0})"
        ),
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Replot mode ──────────────────────────────────────────────────────────
    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} models from {args.replot}")
        _plot_results(results, timestamp)
        return

    # ── Train ────────────────────────────────────────────────────────────────
    root_key = jax.random.PRNGKey(args.seed)

    # Build the list of (revenue_share, transfer_mode) conditions
    if args.transfer_mode == "both":
        shares_to_run = TRANSFER_MODE_COMPACT_SHARES
        modes_to_run  = ["consumption", "abatement"]
        conditions    = [(rs, tm) for tm in modes_to_run for rs in shares_to_run]
    else:
        conditions = [(rs, args.transfer_mode) for rs in args.shares]

    results = []
    for i, (rs, tm) in enumerate(conditions):
        train_key = jax.random.fold_in(root_key, i)
        ppo, csv_path, label = _train(rs, train_key, args.timesteps, transfer_mode=tm)

        # Evaluate — trained ppo IS the agent (MonitoredPPO has get_action / .state)
        eval_key = jax.random.fold_in(root_key, 100 + i)
        raw_env  = _build_env(rs, transfer_mode=tm, for_training=False)
        dirty_share, transfer_mean, mit_rate = _eu_dirty_share_per_region(
            eval_key, raw_env, ppo
        )

        results.append({
            "revenue_share":  rs,
            "transfer_mode":  tm,
            "dirty_share":    dirty_share,
            "transfer_mean":  transfer_mean,
            "mit_rate":       mit_rate,
            "csv_path":       csv_path,
            "label":          label,
        })
        print(f"\n  rs={rs:.2f}  mode={tm}  EU dirty share (CBAM regions, mean): "
              f"{np.mean([dirty_share[r] for r in CBAM_PLOT_REGIONS]):.4f}  "
              f"mit (non-EU mean): "
              f"{np.mean([mit_rate[r] for r in CBAM_PLOT_REGIONS]):.4f}")

    # ── Save pickle ──────────────────────────────────────────────────────────
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_2b_tier1_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nPickle saved → {pkl_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    _plot_results(results, timestamp)

    # ── Exit criterion ───────────────────────────────────────────────────────
    rs0 = results[0]["dirty_share"]   # no transfer
    rs1 = results[-1]["dirty_share"]  # full transfer
    passed = any(rs1[r] > rs0[r] for r in CBAM_PLOT_REGIONS)
    print("\n" + "═"*55)
    print(f"  EXIT CRITERION: {'✅ PASS' if passed else '❌ FAIL'}")
    print(f"  Transfer (rs=1.0) raises EU dirty share vs no-transfer")
    print(f"  (i.e. reduces diversion) in ≥1 CBAM-exposed region")
    print("═"*55)


if __name__ == "__main__":
    main()
