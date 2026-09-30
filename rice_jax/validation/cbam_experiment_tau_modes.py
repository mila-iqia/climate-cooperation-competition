"""cbam_experiment_tau_modes.py

Phase 2B — CBAM Tariff Calibration and Differential Mechanism Comparison.

Two experiments in one script:

Experiment A — Flat τ sweep (Level 1: literature-grounded calibration)
    Train models at τ ∈ {0.05, 0.10, 0.15, 0.25, 0.80} with cbam_tariff_mode="flat".
    τ=0.80 is the original empirical choice; 0.05–0.25 spans the literature-
    grounded range from ad-valorem equivalents of EU ETS price on CBAM sectors
    (Böhringer, Fischer & Rosendahl 2010 Table 2; Martin et al. 2014 JIE §4).
    Research question: are the allocation-rule comparisons (effort vs burden)
    qualitatively robust to τ calibration?

Experiment B — Differential mode (Level 2: per-region carbon price differential)
    Train with cbam_tariff_mode="differential": τ_eff[r] = max(0, MAC_EU − MAC_r)/MAC_EU.
    As μ_r rises toward μ_EU, τ_eff[r] → 0, restoring full EU market access.
    This is the self-incentivising channel absent from the flat mode.
    Research question: does the differential mechanism produce higher mitigation
    than the best flat τ, and what is the per-region effective tariff trajectory?

Canonical null test (differential):
    When all regions have the same mitigation rate as EU, τ_eff[r] = 0 for all r,
    so cbam_cost = 0.  Verified analytically before any training.

All conditions use: effort allocation, rs=1.0, mode=abatement (Phase 2B best settings).

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_tau_modes.py [--timesteps 1000000]
    python validation/cbam_experiment_tau_modes.py --replot <pickle.pkl>
    python validation/cbam_experiment_tau_modes.py --null-test-only
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
CBAM_LAMBDA_INIT  = 1.0
WELFARE_LOSS_WEIGHT = 5.0
TRANSFER_MODE     = "abatement"
TRANSFER_ALLOC    = "effort"   # best rule from Phase 2B alloc comparison
REVENUE_SHARE     = 1.0

# Literature-grounded flat τ sweep (+ original 0.80 for reference)
# Böhringer et al. 2010 Table 2: effective ad-valorem on iron/steel 8-22%,
# cement 15-25%, aluminium 10-18%.  Martin et al. 2014 JIE §4 Table 3: 10-25%.
FLAT_TAU_LEVELS  = [0.05, 0.10, 0.15, 0.25, 0.80]
FLAT_COLORS      = plt.cm.plasma(np.linspace(0.15, 0.85, len(FLAT_TAU_LEVELS)))
DIFF_COLOR       = "#2ca02c"   # green — differential mode

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "cbam_tau_"

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
    revenue_share                = REVENUE_SHARE,
    transfer_mode                = TRANSFER_MODE,
    transfer_allocation          = TRANSFER_ALLOC,
    reward_mode                  = "additive_cbam",
    cbam_lambda_init             = CBAM_LAMBDA_INIT,
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


# ── Canonical null test ───────────────────────────────────────────────────────

def run_null_test():
    """
    Canonical null condition for differential mode:
    When all regions have the same mitigation rate as EU, τ_eff[r] = 0 for
    all r, so cbam_cost = 0.  Verified analytically before any training.

    Also verifies: flat mode with τ=0 → cbam_cost = 0 (existing invariant).
    """
    import jax.numpy as jnp

    print("\n" + "═" * 60)
    print("  CANONICAL NULL TEST — differential τ_eff")
    print("═" * 60)

    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR)
    env_diff = RiceMRIO(
        region_params    = region_params,
        cbam_tariff_mode = "differential",
        **_BASE_ENV,
    )

    key = jax.random.PRNGKey(0)
    _, state = env_diff.reset_env(key)

    # Build dummy trade flows (all EU exports = 1.0 per sector for every exporter)
    trade_flows = jnp.ones((NUM_REGIONS, NUM_REGIONS, env_diff.num_sectors),
                           dtype=jnp.float32)
    trade_flows = trade_flows.at[jnp.arange(NUM_REGIONS),
                                 jnp.arange(NUM_REGIONS), :].set(0.0)  # no self-trade
    gross_imports = trade_flows.sum(axis=2).T  # (NR, NR)

    # Null: all regions at the same μ as EU (= random EU value)
    # Pick a representative μ = 0.5 for EU and all others
    mu_equal = jnp.ones(NUM_REGIONS) * 0.5

    _, _, cbam_cost = env_diff._compute_cbam(
        trade_flows, gross_imports,
        mitigation_rates=mu_equal,
        activity_timestep=jnp.float32(1.0),
    )

    cbam_cost_np = np.array(cbam_cost)
    max_abs = np.abs(cbam_cost_np).max()
    passed = max_abs < 1e-5

    print(f"  All regions μ = 0.5 (equal to EU)")
    print(f"  cbam_cost: {cbam_cost_np}")
    print(f"  max |cbam_cost| = {max_abs:.2e}  →  {'PASS ✅' if passed else 'FAIL ❌'}")

    # Null 2: higher EU μ → positive cbam_cost for lower-μ regions
    mu_low   = jnp.ones(NUM_REGIONS) * 0.1
    mu_low   = mu_low.at[EU_IDX].set(0.8)  # EU has high carbon pricing
    _, _, cbam_cost_diff = env_diff._compute_cbam(
        trade_flows, gross_imports,
        mitigation_rates=mu_low,
        activity_timestep=jnp.float32(1.0),
    )
    cbam_cost_diff_np = np.array(cbam_cost_diff)
    all_non_eu_positive = all(cbam_cost_diff_np[r] > 0 for r in NON_EU)

    print(f"\n  EU μ=0.8, all others μ=0.1")
    print(f"  cbam_cost (non-EU): {cbam_cost_diff_np[[r for r in NON_EU]]}")
    print(f"  All non-EU have positive cost → {'PASS ✅' if all_non_eu_positive else 'FAIL ❌'}")

    # Null 3: flat mode τ=0 → cbam_cost = 0
    env_flat0 = RiceMRIO(
        region_params    = region_params,
        cbam_tariff_mode = "flat",
        cbam_tariff_rate = 0.0,
        **_BASE_ENV,
    )
    _, _, cbam_cost_flat0 = env_flat0._compute_cbam(
        trade_flows, gross_imports,
        activity_timestep=jnp.float32(1.0),
    )
    flat0_max = float(jnp.abs(cbam_cost_flat0).max())
    print(f"\n  Flat mode τ=0.0 → max |cbam_cost| = {flat0_max:.2e}  "
          f"→  {'PASS ✅' if flat0_max < 1e-6 else 'FAIL ❌'}")

    print("═" * 60)
    return passed and all_non_eu_positive and (flat0_max < 1e-6)


# ── Build / train helpers ─────────────────────────────────────────────────────

def _build_env(tau_mode, tau_val=None, for_training=True):
    """
    tau_mode : "flat" or "differential"
    tau_val  : float for "flat" mode (ignored for "differential")
    """
    extra = {}
    if tau_mode == "flat":
        extra["cbam_tariff_mode"] = "flat"
        extra["cbam_tariff_rate"] = float(tau_val)
    else:
        extra["cbam_tariff_mode"] = "differential"
        extra["cbam_tariff_rate"] = 0.0  # not used; present for state init

    env = RiceMRIO(
        region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        log_info_fn   = rcpo_cbam_log_info_fn,
        **_BASE_ENV,
        **extra,
    )
    return jym.LogWrapper(env) if for_training else env


def _condition_label(tau_mode, tau_val):
    if tau_mode == "differential":
        return "differential"
    return f"flat_tau{tau_val:.2f}".replace(".", "p")


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


def _train(tau_mode, tau_val, key, total_timesteps):
    label = _condition_label(tau_mode, tau_val)
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    env       = _build_env(tau_mode, tau_val, for_training=True)
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    mode_str = f"flat τ={tau_val}" if tau_mode == "flat" else "differential"
    print(f"\n{'━'*60}")
    print(f"  Training: {mode_str}  (alloc={TRANSFER_ALLOC}, rs={REVENUE_SHARE})")
    print(f"{'━'*60}")
    t0  = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def _eval(key, raw_env, agent):
    """Return (dirty_share, mit_rate, eff_tau_step1) each shape (NR,)."""
    from _experiment_util import run_single_episode

    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        """Convert per-region log dict {region_idx: (T,) array} → (T, NR) array."""
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    dirty_shares_all, mit_rates_all, eff_tau_step1_all = [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        tf     = np.array(logs["trade_flows"])             # (T, NR, NR, NS)
        mu     = _to_arr(logs["mitigation_rates_all_regions"])  # (T, NR)
        cbam_c = _to_arr(logs["cbam_cost_all_regions"])         # (T, NR)

        # Dirty (sector 0) EU export share per region
        eu_dirty    = tf[:, :, EU_IDX, 0]                       # (T, NR)
        total_dirty = tf[:, :, :, 0].sum(axis=2)                # (T, NR)
        share       = eu_dirty / (total_dirty + 1e-8)           # (T, NR)

        # Use last 5 timesteps for evaluation metrics
        dirty_shares_all.append(share[-5:].mean(axis=0))
        mit_rates_all.append(mu[-5:].mean(axis=0))

        # Effective τ at step 1: cbam_cost / eu_dirty_exports (rough proxy)
        eu_exp_step1 = eu_dirty[0]                              # (NR,) — step 0
        eff_tau = cbam_c[0] / (eu_exp_step1 + 1e-10)
        eff_tau_step1_all.append(np.clip(eff_tau, 0, 2))

    dirty_share   = np.stack(dirty_shares_all).mean(axis=0)    # (NR,)
    mit_rate      = np.stack(mit_rates_all).mean(axis=0)       # (NR,)
    eff_tau_step1 = np.stack(eff_tau_step1_all).mean(axis=0)   # (NR,) — approx

    return dirty_share, mit_rate, eff_tau_step1


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_results(results, timestamp):
    """
    6-panel figure:
      Row 1: [Training convergence] [EU dirty share vs τ level] [μ vs τ level]
      Row 2: [Effective τ at init — differential] [Differential vs best flat — dirty] [Summary table]
    """
    n_flat_runs = sum(1 for r in results if r["tau_mode"] == "flat")
    flat_results = [r for r in results if r["tau_mode"] == "flat"]
    diff_result  = next((r for r in results if r["tau_mode"] == "differential"), None)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Phase 2B — CBAM Tariff Calibration & Differential Mechanism\n"
        f"alloc=effort, rs=100%, mode=abatement, 9-region vuln, 1M steps",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_conv    = fig.add_subplot(gs[0, 0])
    ax_dirty   = fig.add_subplot(gs[0, 1])
    ax_mit     = fig.add_subplot(gs[0, 2])
    ax_eff_tau = fig.add_subplot(gs[1, 0])
    ax_compare = fig.add_subplot(gs[1, 1])
    ax_table   = fig.add_subplot(gs[1, 2])

    x = np.arange(len(CBAM_PLOT_REGIONS))
    width = 0.13

    # ── Training convergence ─────────────────────────────────────────────────
    for res, col in zip(flat_results, FLAT_COLORS):
        if res.get("csv_path") and _os.path.exists(res["csv_path"]):
            df = pd.read_csv(res["csv_path"])
            rolled = df["ep_return_mean"].rolling(10).mean()
            label  = f"τ={res['tau_val']:.2f}"
            ax_conv.plot(df["iteration"], rolled, color=col, label=label, linewidth=1.2)
    if diff_result and diff_result.get("csv_path") and _os.path.exists(diff_result["csv_path"]):
        df = pd.read_csv(diff_result["csv_path"])
        rolled = df["ep_return_mean"].rolling(10).mean()
        ax_conv.plot(df["iteration"], rolled, color=DIFF_COLOR, label="differential",
                     linewidth=1.5, linestyle="--")
    ax_conv.set_xlabel("PPO iteration")
    ax_conv.set_ylabel("ep_return mean (rolling-10)")
    ax_conv.set_title("Training convergence")
    ax_conv.legend(fontsize=7)

    # ── EU dirty share vs τ level (flat only) ────────────────────────────────
    for i, (res, col) in enumerate(zip(flat_results, FLAT_COLORS)):
        shares = [res["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        offset = (i - len(flat_results)/2 + 0.5) * width
        ax_dirty.bar(x + offset, shares, width=width, color=col,
                     alpha=0.85, label=f"τ={res['tau_val']:.2f}")
    if diff_result:
        shares_d = [diff_result["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        ax_dirty.plot(x, shares_d, "D--", color=DIFF_COLOR,
                      markersize=7, linewidth=1.5, label="differential", zorder=5)
    ax_dirty.set_xticks(x)
    ax_dirty.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                              rotation=25, ha="right")
    ax_dirty.set_ylabel("EU dirty export share (mean)")
    ax_dirty.set_title("EU dirty share by τ level\n(higher = less diversion)")
    ax_dirty.legend(fontsize=7)

    # ── Mitigation rate vs τ level ────────────────────────────────────────────
    for i, (res, col) in enumerate(zip(flat_results, FLAT_COLORS)):
        mits = [res["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        offset = (i - len(flat_results)/2 + 0.5) * width
        ax_mit.bar(x + offset, mits, width=width, color=col,
                   alpha=0.85, label=f"τ={res['tau_val']:.2f}")
    if diff_result:
        mits_d = [diff_result["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        ax_mit.plot(x, mits_d, "D--", color=DIFF_COLOR,
                    markersize=7, linewidth=1.5, label="differential", zorder=5)
    ax_mit.set_xticks(x)
    ax_mit.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                            rotation=25, ha="right")
    ax_mit.set_ylabel("Mean mitigation rate μ")
    ax_mit.set_title("Mitigation rate by τ level\n(higher = more abatement)")
    ax_mit.legend(fontsize=7)

    # ── Effective τ at step 1 — differential mode ─────────────────────────────
    if diff_result:
        eff = [diff_result["eff_tau_step1"][r] for r in CBAM_PLOT_REGIONS]
        bars = ax_eff_tau.bar(x, eff, color=DIFF_COLOR, alpha=0.85)
        # Annotate with absolute values
        for bar, v in zip(bars, eff):
            ax_eff_tau.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        # Draw reference lines for flat τ values
        for tau_val, col in zip(FLAT_TAU_LEVELS, FLAT_COLORS):
            ax_eff_tau.axhline(tau_val, color=col, linestyle=":", linewidth=1.0,
                               label=f"flat τ={tau_val:.2f}")
        ax_eff_tau.set_xticks(x)
        ax_eff_tau.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                                    rotation=25, ha="right")
        ax_eff_tau.set_ylabel("Effective τ at step 1 (approx.)")
        ax_eff_tau.set_title("Differential mode: effective tariff per region\n"
                             "at episode start (dashed = flat τ reference levels)")
        ax_eff_tau.legend(fontsize=6)
    else:
        ax_eff_tau.text(0.5, 0.5, "Differential run not available",
                        ha="center", va="center", transform=ax_eff_tau.transAxes)
        ax_eff_tau.axis("off")

    # ── Differential vs flat τ=0.15 comparison ───────────────────────────────
    flat_ref = next((r for r in flat_results if abs(r["tau_val"] - 0.15) < 0.001), flat_results[0])
    compare_items = [("flat τ=0.15", flat_ref, "#1f77b4")]
    if diff_result:
        compare_items.append(("differential", diff_result, DIFF_COLOR))
    n_compare = len(compare_items)
    w2 = 0.35
    for i, (label, res, col) in enumerate(compare_items):
        dirty_c = [res["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        mit_c   = [res["mit_rate"][r]    for r in CBAM_PLOT_REGIONS]
        offset  = (i - n_compare/2 + 0.5) * w2
        ax_compare.bar(x + offset, dirty_c, width=w2, color=col, alpha=0.7,
                       label=f"{label} dirty")
        ax_compare.plot(x + offset + w2/2, mit_c, "^",
                        color=col, markersize=7, zorder=5,
                        label=f"{label} μ")
    ax_compare.set_xticks(x)
    ax_compare.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                                rotation=25, ha="right")
    ax_compare.set_ylabel("Bars: dirty share  ▲: μ")
    ax_compare.set_title("Differential vs flat τ=0.15\n(closest literature-grounded level)")
    ax_compare.legend(fontsize=7)

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_table.axis("off")
    col_headers = ["Condition"] + [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS] + ["Mean"]
    table_data  = []
    for section, key_name, fmt in [
        ("μ (mitigation)", "mit_rate", ".3f"),
        ("EU dirty share", "dirty_share", ".3f"),
    ]:
        table_data.append([f"── {section} ──"] + [""] * (len(CBAM_PLOT_REGIONS) + 1))
        for res in results:
            label = (f"flat τ={res['tau_val']:.2f}" if res["tau_mode"] == "flat"
                     else "differential")
            vals  = [res[key_name][r] for r in CBAM_PLOT_REGIONS]
            row   = [label] + [f"{v:{fmt}}" for v in vals] + [f"{np.mean(vals):{fmt}}"]
            table_data.append(row)

    tbl = ax_table.table(
        cellText  = table_data,
        colLabels = col_headers,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.25)
    ax_table.set_title("Summary table", fontsize=9, pad=4)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_tau_modes_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps",      type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",           type=int,   default=SEED)
    parser.add_argument("--replot",         type=str,   default=None,
                        help="Path to existing .pkl — regenerate plot without retraining")
    parser.add_argument("--null-test-only", action="store_true",
                        help="Run canonical null test only, skip training")
    parser.add_argument(
        "--flat-taus",  nargs="+", type=float, default=FLAT_TAU_LEVELS,
        help="Flat τ levels to train (default: 0.05 0.10 0.15 0.25 0.80)",
    )
    parser.add_argument("--skip-differential", action="store_true",
                        help="Skip the differential mode training run")
    args = parser.parse_args()

    # Always run null test first
    null_ok = run_null_test()
    if args.null_test_only:
        return

    if not null_ok:
        print("\n⚠  Null test FAILED — investigate before running training.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        print(f"Loaded {len(results)} conditions from {args.replot}")
        _plot_results(results, timestamp)
        return

    root_key = jax.random.PRNGKey(args.seed)
    results  = []

    # ── Experiment A: flat τ sweep ────────────────────────────────────────────
    for i, tau in enumerate(args.flat_taus):
        train_key = jax.random.fold_in(root_key, i)
        ppo, csv_path = _train("flat", tau, train_key, args.timesteps)

        eval_key = jax.random.fold_in(root_key, 100 + i)
        raw_env  = _build_env("flat", tau, for_training=False)
        dirty_share, mit_rate, eff_tau_step1 = _eval(eval_key, raw_env, ppo)

        results.append({
            "tau_mode":     "flat",
            "tau_val":      tau,
            "dirty_share":  dirty_share,
            "mit_rate":     mit_rate,
            "eff_tau_step1": eff_tau_step1,
            "csv_path":     csv_path,
        })
        mean_mit   = np.mean([mit_rate[r]    for r in CBAM_PLOT_REGIONS])
        mean_dirty = np.mean([dirty_share[r] for r in CBAM_PLOT_REGIONS])
        print(f"\n  flat τ={tau:.2f}  μ(CBAM)={mean_mit:.4f}  dirty(CBAM)={mean_dirty:.4f}")

    # ── Experiment B: differential mode ──────────────────────────────────────
    if not args.skip_differential:
        diff_idx  = len(args.flat_taus)
        train_key = jax.random.fold_in(root_key, 200 + diff_idx)
        ppo, csv_path = _train("differential", None, train_key, args.timesteps)

        eval_key = jax.random.fold_in(root_key, 300 + diff_idx)
        raw_env  = _build_env("differential", None, for_training=False)
        dirty_share, mit_rate, eff_tau_step1 = _eval(eval_key, raw_env, ppo)

        results.append({
            "tau_mode":     "differential",
            "tau_val":      None,
            "dirty_share":  dirty_share,
            "mit_rate":     mit_rate,
            "eff_tau_step1": eff_tau_step1,
            "csv_path":     csv_path,
        })
        mean_mit   = np.mean([mit_rate[r]    for r in CBAM_PLOT_REGIONS])
        mean_dirty = np.mean([dirty_share[r] for r in CBAM_PLOT_REGIONS])
        print(f"\n  differential   μ(CBAM)={mean_mit:.4f}  dirty(CBAM)={mean_dirty:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_tau_modes_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nPickle saved → {pkl_path}")

    _plot_results(results, timestamp)

    # ── Console summary ───────────────────────────────────────────────────────
    ref = next((r for r in results if r["tau_mode"] == "flat" and abs(r["tau_val"] - 0.80) < 0.001),
               results[0])
    print("\n" + "═" * 65)
    print("  TAU CALIBRATION SUMMARY — Δ vs τ=0.80 baseline")
    print("─" * 65)
    for res in results:
        if res["tau_mode"] == "flat" and abs(res["tau_val"] - 0.80) < 0.001:
            label = "flat τ=0.80 (baseline)"
        elif res["tau_mode"] == "flat":
            label = f"flat τ={res['tau_val']:.2f}"
        else:
            label = "differential"
        dm = np.mean([res["mit_rate"][r] - ref["mit_rate"][r]       for r in CBAM_PLOT_REGIONS])
        dd = np.mean([res["dirty_share"][r] - ref["dirty_share"][r] for r in CBAM_PLOT_REGIONS])
        mean_mu  = np.mean([res["mit_rate"][r]    for r in CBAM_PLOT_REGIONS])
        mean_dirty = np.mean([res["dirty_share"][r] for r in CBAM_PLOT_REGIONS])
        print(f"  {label:26s}  μ={mean_mu:.3f}  dirty={mean_dirty:.3f}  "
              f"Δμ={dm:+.3f}  Δdirty={dd:+.3f}")
    print("═" * 65)


if __name__ == "__main__":
    main()
