"""cbam_experiment_9.py

Detailed CBAM experiment on the 9-region vulnerability setup.

Structure
---------
Mirrors the litmus test (L1-L4) on the 9-region vulnerability specification.
Each test collects per-region breakdowns and training convergence data,
producing a long multi-section figure (one section per experiment):

  E1  Export diversion         (additive_cbam, fixed λ_init, cbam_randomize tau in {0, 0.8})
  E2  Mitigation signal        (additive_cbam/RCPO, free abatement, exports pinned)
  E3  DICE damage avoidance    (welfloss, no CBAM, costly abatement)
  E4  Both channels            (additive_cbam/RCPO, full action space)

Figure layout per section:
  [convergence: ep_return vs iter]  [pass/fail badge]
  [dirty sector EU share / region]  [clean sector EU share / region]  <- E1, E4
  [per-region mitigation bar]                                         <- E2, E3, E4
  [utility trajectory over episode steps]

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_9.py [--timesteps 1000000] [--seed 42]
    python validation/cbam_experiment_9.py --tests e3 --timesteps 200000
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
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
    rcpo_cbam_log_info_fn,
)
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls


# ── Config ──────────────────────────────────────────────────────────────────

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT   = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", ".."))

# 9-region vulnerability ordering (CountryClass_cbam_vuln_9.csv):
#   0: Rest of World          1: Russia+Turkey+Eurasia
#   2: MENA (Gulf+N.Africa)   3: EU & Western Europe  <- EU_IDX
#   4: SSA Metals & Mining    5: Americas
#   6: SE Asia & Pacific dev  7: China   8: India
NUM_REGIONS = 9
EU_IDX      = 3
YAML_DIR    = _os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_ROOT   = _os.path.join(_REPO_ROOT, "csv_asset")

REGION_NAMES = {
    0: "RoW",
    1: "Russia+Eur.",
    2: "MENA",
    3: "EU",
    4: "SSA-Mining",
    5: "Americas",
    6: "SE Asia",
    7: "China",
    8: "India",
}
NON_EU = [r for r in range(NUM_REGIONS) if r != EU_IDX]
# RoW (idx 0) dominates the bar charts and distorts the scale; exclude it
# from per-region plots so the CBAM-relevant exporters are legible.
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]  # drop RoW

TOTAL_TIMESTEPS   = 1_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42
CBAM_RATE         = 0.80

RCPO_ETA_LAMBDA   = 5e-2
RCPO_ALPHA_TARGET = 0.001
WELFARE_LOSS_WEIGHT    = 5.0
# E1 uses a fixed calibrated λ (additive_cbam, cbam_lambda_init=1.0) rather
# than auto-tuned RCPO.  Root-cause: RCPO stores λ in the per-episode env
# state, which resets to 0 on every episode boundary.  With RICE having ~20
# steps/episode and PPO rollouts of 100 steps, ~5 episode resets occur per
# rollout.  Each reset zeroes λ, so the RCPO loop only ever sees λ≈0.0002
# (one batch-update from 0) — far too small for the 9-region EU share of 8.5%.
# Fix: set cbam_lambda_init=1.0 in RiceMRIO so the env always resets λ to 1.0
# rather than 0.  Calibration: mean cbam_cost (non-EU, τ=0.8) ≈ 0.011,
# so λ=1.0 → penalty ≈ 17% of ΔU ≈ 0.063.  Clear signal without saturation.
# [LITERATURE NEEDED: proper RCPO persistence across resets, e.g. using a
# training-level variable outside the episode state.]
E1_CBAM_LAMBDA_INIT         = 1.0
# Pass threshold: relative reduction in EU dirty share (fraction, not pp)
# A 15% relative drop (e.g. 0.085 → 0.072) is meaningful evidence of
# τ-conditioned diversion even when the absolute baseline is small.
E1_RELATIVE_DROP_THRESHOLD = 0.15

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "cbam9_"  # avoids collision with litmus_test.py CSVs

# Consistent region colours across all trajectory plots
_REGION_COLORS = [plt.cm.tab10(i / 10) for i in range(NUM_REGIONS)]

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


# ── Build / train helpers ────────────────────────────────────────────────────

def _build_env(cbam_rate, zero_abatement_cost, no_mitigation, fixed_savings,
               reward_mode="additive_cbam", delta_max=3.0,
               cbam_randomize=False, cbam_tariff_rates=None,
               for_training=True, welfare_loss_weight=None,
               cbam_lambda_init=None):
    extra = {}
    if reward_mode == "additive_cbam":
        extra["log_info_fn"] = rcpo_cbam_log_info_fn
    if welfare_loss_weight is not None:
        extra["welfare_loss_per_unit_tariff"] = welfare_loss_weight
    if cbam_lambda_init is not None:
        extra["cbam_lambda_init"] = cbam_lambda_init
    env = RiceMRIO(
        region_params      = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_rate   = cbam_rate,
        zero_abatement_cost= zero_abatement_cost,
        no_mitigation      = no_mitigation,
        fixed_savings_rate = fixed_savings,
        delta_max          = delta_max,
        reward_mode        = reward_mode,
        cbam_randomize     = cbam_randomize,
        cbam_tariff_rates  = cbam_tariff_rates if cbam_tariff_rates is not None else (cbam_rate,),
        **_BASE_ENV,
        **extra,
    )
    return jym.LogWrapper(env) if for_training else env


def _make_log_fn(label, num_iters):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    return make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    ), csv_path


def _train(label, env, key, num_iters, reward_mode, total_timesteps=None,
           eta_lambda=None, use_rcpo=True):
    ts   = total_timesteps or TOTAL_TIMESTEPS
    _eta = eta_lambda if eta_lambda is not None else RCPO_ETA_LAMBDA
    log_fn, csv_path = _make_log_fn(label, num_iters)
    if reward_mode == "additive_cbam" and use_rcpo:
        ppo = RCPOMonitoredPPO(
            total_timesteps   = ts,
            log_function      = log_fn,
            rcpo_eta_lambda   = _eta,
            rcpo_alpha_target = RCPO_ALPHA_TARGET,
            **_PPO_KWARGS,
        )
    else:
        ppo = MonitoredPPO(
            total_timesteps = ts,
            log_function    = log_fn,
            **_PPO_KWARGS,
        )
    print(f"\n{'━'*55}")
    print(f"  Training: {label}")
    print(f"{'━'*55}")
    t0  = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path


# ── Eval helper ──────────────────────────────────────────────────────────────

def _eval_episode(key, raw_rice_env, agent):
    """Run NUM_EVAL_EPISODES; return stacked numpy dicts.

    Returns
    -------
    dict:
      trade_flows : (E, T, NR, NR, NS)
      mitigation  : (E, T, NR)
      cbam_cost   : (E, T, NR)
      utility     : (E, T, NR)
    """
    from _experiment_util import run_single_episode

    eval_env = replace(raw_rice_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    flows, mit, cbam, util = [], [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 30_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)
        flows.append(np.array(logs["trade_flows"]))
        mit.append(_to_arr(logs["mitigation_rates_all_regions"]))
        cbam.append(_to_arr(logs["cbam_cost_all_regions"]))
        util.append(_to_arr(logs["utility_all_regions"]))

    return {
        "trade_flows": np.stack(flows, 0),
        "mitigation":  np.stack(mit,   0),
        "cbam_cost":   np.stack(cbam,  0),
        "utility":     np.stack(util,  0),
    }


# ── Analysis helpers ─────────────────────────────────────────────────────────

def _read_csv(csv_path):
    if not _os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _eu_dirty_share_agg(trade_flows, last_t=5):
    """Aggregate mean EU dirty-sector export share across CBAM-relevant regions.

    Uses CBAM_PLOT_REGIONS (NON_EU minus RoW) so the metric is not dominated
    by RoW's large but policy-irrelevant trade volume.  RoW is a catch-all
    that responds differently to CBAM and would otherwise mask the behaviour
    of the CBAM-vulnerable regions this experiment is designed to test.
    """
    tf        = trade_flows[:, -last_t:]           # (E, T, NR, NR, NS)
    dirty_eu  = tf[:, :, :, EU_IDX, 0]
    dirty_all = tf[:, :, :, :,      0].sum(-1)
    return float((dirty_eu[:, :, CBAM_PLOT_REGIONS] / (dirty_all[:, :, CBAM_PLOT_REGIONS] + 1e-10)).mean())


def _per_region_eu_sector_shares(trade_flows, last_t=5):
    """Per non-EU region: {'dirty': float, 'clean': float} EU export share."""
    tf = trade_flows[:, -last_t:]
    out = {}
    for r in NON_EU:
        eu_d = tf[:, :, r, EU_IDX, 0];  all_d = tf[:, :, r, :, 0].sum(-1)
        eu_c = tf[:, :, r, EU_IDX, 1];  all_c = tf[:, :, r, :, 1].sum(-1)
        out[r] = {
            "dirty": float((eu_d / (all_d + 1e-10)).mean()),
            "clean": float((eu_c / (all_c + 1e-10)).mean()),
        }
    return out


def _per_region_mu(mitigation, last_t=5):
    """Mean mitigation rate per region over last T steps -> {r: float}."""
    return {r: float(mitigation[:, -last_t:, r].mean()) for r in range(NUM_REGIONS)}


def _mean_util_traj(utility):
    """Mean utility over eval episodes -> (T, NR)."""
    return utility.mean(axis=0)


# ── Experiment runners ────────────────────────────────────────────────────────

def run_e1(key):
    """E1: Export diversion — fixed λ (additive_cbam) + cbam_randomize.

    Uses a fixed cbam_lambda_init=E1_CBAM_LAMBDA_INIT (not RCPO auto-tuning)
    because RCPO stores λ in per-episode env state, which resets to 0 on every
    episode boundary (~5 resets/rollout with 20-step RICE episodes and 100-step
    rollouts).  λ therefore never accumulates — it oscillates at ~0.0002,
    giving a penalty ~1/47000 of ΔU.  Fix: cbam_lambda_init=1.0 forces every
    reset to reinitialise λ=1.0, giving a persistent ~17% of ΔU penalty at
    τ=0.8 and no penalty at τ=0.  MonitoredPPO is used (no λ updating needed).
    [LITERATURE NEEDED: persistent RCPO λ across episode resets.]

    Pass criterion: relative reduction in EU dirty share >
    E1_RELATIVE_DROP_THRESHOLD (15%), e.g. 0.085 → 0.072.
    """
    print("\n" + "="*55)
    print(f"  E1  Export Diversion  [additive_cbam, λ_init={E1_CBAM_LAMBDA_INIT}]")
    print(f"  tau in {{0, {CBAM_RATE}}}, no_mitigation, fixed_savings")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    # Fixed λ: use MonitoredPPO (no RCPO needed since λ is constant)
    env = _build_env(0.0, False, True, True,
                     reward_mode="additive_cbam",
                     cbam_randomize=True,
                     cbam_tariff_rates=(0.0, CBAM_RATE),
                     cbam_lambda_init=E1_CBAM_LAMBDA_INIT)
    agent, csv_path = _train("e1_divert", env, key, num_iters, "additive_cbam",
                             use_rcpo=False)

    eval_key = jax.random.fold_in(key, 10)
    raw_low  = _build_env(0.0,       False, True, True,
                          reward_mode="additive_cbam", for_training=False,
                          cbam_lambda_init=E1_CBAM_LAMBDA_INIT)
    raw_high = _build_env(CBAM_RATE, False, True, True,
                          reward_mode="additive_cbam", for_training=False,
                          cbam_lambda_init=E1_CBAM_LAMBDA_INIT)
    ev_low   = _eval_episode(eval_key, raw_low,  agent)
    ev_high  = _eval_episode(eval_key, raw_high, agent)

    share_ctrl  = _eu_dirty_share_agg(ev_low["trade_flows"])
    share_treat = _eu_dirty_share_agg(ev_high["trade_flows"])
    rel_drop    = (share_ctrl - share_treat) / (share_ctrl + 1e-10)
    drop_pp     = (share_ctrl - share_treat) * 100

    passed = rel_drop > E1_RELATIVE_DROP_THRESHOLD
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  E1 {tag}: dirty EU share  tau=0: {share_ctrl:.3f}  "
          f"tau={CBAM_RATE}: {share_treat:.3f}  "
          f"drop={drop_pp:.1f} pp ({rel_drop*100:.1f}% rel)  "
          f"(>{E1_RELATIVE_DROP_THRESHOLD*100:.0f}% rel)")
    return passed, dict(
        share_ctrl=share_ctrl, share_treat=share_treat,
        drop_pp=drop_pp, rel_drop=rel_drop,
        per_region_ctrl =_per_region_eu_sector_shares(ev_low["trade_flows"]),
        per_region_treat=_per_region_eu_sector_shares(ev_high["trade_flows"]),
        util_ctrl =_mean_util_traj(ev_low["utility"]),
        util_treat=_mean_util_traj(ev_high["utility"]),
        csv_path=csv_path,
    )


def run_e2(key):
    """E2: Mitigation signal — free abatement + CBAM -> agents learn mu>0.

    additive_cbam+RCPO amplifies the CBAM signal above DeltaU noise.
    Exports pinned via delta_max=0.0 so mitigation is the only escape.
    """
    print("\n" + "="*55)
    print("  E2  Mitigation Signal  [additive_cbam/RCPO]")
    print(f"  CBAM tau={CBAM_RATE}, zero_abatement_cost, only mu free (delta_max=0)")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(CBAM_RATE, True, False, True, delta_max=0.0)
    agent, csv_path = _train("e2_mit", env, key, num_iters, "additive_cbam")

    eval_key = jax.random.fold_in(key, 20)
    raw_env  = _build_env(CBAM_RATE, True, False, True, delta_max=0.0, for_training=False)
    ev       = _eval_episode(eval_key, raw_env, agent)

    T_last   = 5
    mean_mu  = float(ev["mitigation"][:, -T_last:, :][:, :, NON_EU].mean())
    pr_mu    = _per_region_mu(ev["mitigation"], T_last)
    ut       = _mean_util_traj(ev["utility"])

    passed = mean_mu > 0.15
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  E2 {tag}: mean mu (non-EU, last {T_last}) = {mean_mu:.3f}  (>0.15)")
    return passed, dict(mean_mu=mean_mu, per_region_mu=pr_mu,
                        util_traj=ut, csv_path=csv_path)


def run_e3(key):
    """E3: DICE damage avoidance — no CBAM, realistic abatement cost.

    Classic RICE result: optimal mu ~ 0.7 by end of century.
    """
    print("\n" + "="*55)
    print("  E3  DICE Damage Avoidance  [welfloss, tau=0]")
    print("  tau=0, realistic abatement cost, only mu free")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(0.0, False, False, True, reward_mode="welfloss")
    agent, csv_path = _train("e3_dice", env, key, num_iters, "welfloss")

    eval_key = jax.random.fold_in(key, 30)
    raw_env  = _build_env(0.0, False, False, True, reward_mode="welfloss", for_training=False)
    ev       = _eval_episode(eval_key, raw_env, agent)

    T_last  = 5
    mean_mu = float(ev["mitigation"][:, -T_last:, :].mean())
    pr_mu   = _per_region_mu(ev["mitigation"], T_last)
    ut      = _mean_util_traj(ev["utility"])

    passed = 0.01 < mean_mu < 0.99
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  E3 {tag}: mean mu (all, last {T_last}) = {mean_mu:.3f}  "
          "(RICE optimum ~0.7)")
    return passed, dict(mean_mu=mean_mu, per_region_mu=pr_mu,
                        util_traj=ut, csv_path=csv_path)


def run_e4(key, e2_mu):
    """E4: Both channels — diversion + mitigation simultaneously.

    Pass: mitigation >= 50% of E2 (diversion does not crowd out mitigation).
    Also records per-region EU sector shares at tau=0.8.
    """
    print("\n" + "="*55)
    print("  E4  Both Channels  [additive_cbam/RCPO]")
    print(f"  CBAM tau={CBAM_RATE}, zero_abatement_cost, mu + export_realloc free")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(CBAM_RATE, True, False, True)
    agent, csv_path = _train("e4_both", env, key, num_iters, "additive_cbam")

    eval_key = jax.random.fold_in(key, 40)
    raw_env  = _build_env(CBAM_RATE, True, False, True, for_training=False)
    ev       = _eval_episode(eval_key, raw_env, agent)

    T_last   = 5
    mean_mu  = float(ev["mitigation"][:, -T_last:, :][:, :, NON_EU].mean())
    pr_mu    = _per_region_mu(ev["mitigation"], T_last)
    pr_trade = _per_region_eu_sector_shares(ev["trade_flows"])
    ut       = _mean_util_traj(ev["utility"])

    threshold = max(e2_mu * 0.5, 0.05)
    passed    = mean_mu >= threshold
    tag       = "PASS" if passed else "FAIL"
    print(f"\n  E4 {tag}: mean mu = {mean_mu:.3f}  "
          f"(threshold >= {threshold:.3f} = 50% of E2 mu={e2_mu:.3f})")
    return passed, dict(
        mean_mu=mean_mu, threshold=threshold,
        per_region_mu=pr_mu, per_region_trade=pr_trade,
        util_traj=ut, csv_path=csv_path,
    )


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _plot_convergence(ax, csv_path, title="Training convergence"):
    df = _read_csv(csv_path)
    if df.empty or "ep_return_mean" not in df.columns:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="grey")
        ax.set_title(title, fontsize=9)
        return
    iters = df["iteration"] if "iteration" in df.columns else np.arange(len(df))
    mean  = df["ep_return_mean"]
    std   = df["ep_return_std"] if "ep_return_std" in df.columns else np.zeros(len(df))
    ax.plot(iters, mean, lw=1.5, color="#2980b9")
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2, color="#2980b9")
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("ep return", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _plot_badge(ax, passed, lines):
    ax.axis("off")
    color = "#1a6b3a" if passed else "#8b1a1a"
    tag   = "PASS  ✓" if passed else "FAIL  ✗"
    text  = tag + "\n\n" + "\n".join(lines)
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color,
                      alpha=0.12, edgecolor=color, linewidth=2))


def _plot_eu_sector_bars(ax_dirty, ax_clean, ctrl, treat):
    """Grouped bar charts: dirty (left) and clean (right) EU export shares per region."""
    x     = np.arange(len(CBAM_PLOT_REGIONS))
    w     = 0.35
    names = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    for ax, sector, title in [
        (ax_dirty, "dirty", f"Dirty sector — EU export share  (tau=0 vs tau={CBAM_RATE}, excl. RoW)"),
        (ax_clean, "clean", f"Clean sector — EU export share  (tau=0 vs tau={CBAM_RATE}, excl. RoW)"),
    ]:
        c_vals = [ctrl[r][sector]  for r in CBAM_PLOT_REGIONS]
        t_vals = [treat[r][sector] for r in CBAM_PLOT_REGIONS]
        ax.bar(x - w/2, c_vals, w, label="tau=0",           color="#7f8c8d", edgecolor="k", lw=0.6)
        ax.bar(x + w/2, t_vals, w, label=f"tau={CBAM_RATE}", color="#e67e22", edgecolor="k", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("EU export share", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)


def _plot_region_mu(ax, pr_mu, label="", compare=None, compare_label=""):
    """Horizontal bar chart of mitigation rate per region."""
    regions = list(range(NUM_REGIONS))
    names   = [REGION_NAMES[r] for r in regions]
    vals    = [pr_mu[r] for r in regions]
    y = np.arange(len(regions))
    h = 0.35 if compare is not None else 0.55
    if compare is not None:
        cvals = [compare[r] for r in regions]
        ax.barh(y + h/2, vals,  h, label=label,         color="#2980b9", edgecolor="k", lw=0.5)
        ax.barh(y - h/2, cvals, h, label=compare_label, color="#27ae60", edgecolor="k", lw=0.5)
        ax.legend(fontsize=8)
    else:
        ax.barh(y, vals, h, color=[_REGION_COLORS[r] for r in regions],
                edgecolor="k", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("mean mitigation rate (last 5 steps)", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.tick_params(labelsize=7)
    if label and compare is None:
        ax.set_title(label, fontsize=9)


def _plot_util_traj(ax, traj, title="", traj2=None,
                    label1="tau=0", label2=None):
    """Utility trajectory lines per region; traj / traj2 shape (T, NR)."""
    if label2 is None:
        label2 = f"tau={CBAM_RATE}"
    T     = traj.shape[0]
    steps = np.arange(T)
    handles = []
    for r in range(NUM_REGIONS):
        c = _REGION_COLORS[r]
        ln, = ax.plot(steps, traj[:, r], lw=1.5, color=c)
        handles.append(ln)
        if traj2 is not None:
            ax.plot(steps, traj2[:, r], lw=1.5, color=c, ls="--", alpha=0.65)
    ax.set_xlabel("episode step", fontsize=8)
    ax.set_ylabel("utility", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    # Region legend (small, two columns)
    ax.legend(handles=handles,
              labels=[REGION_NAMES[r] for r in range(NUM_REGIONS)],
              fontsize=6, loc="upper left", ncol=2)
    if traj2 is not None:
        solid = mpatches.Patch(color="grey", label=label1)
        dash  = plt.Line2D([0], [0], color="grey", ls="--", label=label2)
        ax2   = ax.twinx()
        ax2.axis("off")
        ax2.legend(handles=[solid, dash], fontsize=7, loc="upper right")


def _plot_e4_sector_bars(ax, pr_trade):
    """Dirty vs clean EU sector share per region for E4 (single tau=0.8 condition)."""
    x     = np.arange(len(CBAM_PLOT_REGIONS))
    w     = 0.35
    names = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    dirty = [pr_trade[r]["dirty"] for r in CBAM_PLOT_REGIONS]
    clean = [pr_trade[r]["clean"] for r in CBAM_PLOT_REGIONS]
    ax.bar(x - w/2, dirty, w, label="dirty sector", color="#c0392b", edgecolor="k", lw=0.6)
    ax.bar(x + w/2, clean, w, label="clean sector", color="#27ae60", edgecolor="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("EU export share", fontsize=8)
    ax.set_title(f"E4 — dirty vs clean EU export share per region  (tau={CBAM_RATE}, excl. RoW)", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=7)


# ── Section-title axes ────────────────────────────────────────────────────────

def _add_section_title(fig, gs_row_slice, label, passed):
    ax = fig.add_subplot(gs_row_slice)
    color = "#1a6b3a" if passed else "#8b1a1a"
    ax.set_facecolor(color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.5, label, color="white", fontsize=12, fontweight="bold",
            va="center", transform=ax.transAxes)
    return ax


# ── Master plot ───────────────────────────────────────────────────────────────

def _plot_all(results, out_path):
    """Long multi-section figure; one section per experiment."""

    # Layout constants
    COLS       = 4
    TITLE_ROWS = 1   # section title bar
    CONV_ROWS  = 4   # convergence + badge
    DATA_ROWS  = 4   # per-region bars / sector shares
    TRAJ_ROWS  = 4   # utility trajectory
    SEC_ROWS   = TITLE_ROWS + CONV_ROWS + DATA_ROWS + TRAJ_ROWS  # 13
    # E1 has two DATA rows (dirty + clean side-by-side), so needs one extra data row
    E1_EXTRA   = 4   # extra rows for second sector bar

    keys_present = [k for k in ["e1", "e2", "e3", "e4"] if k in results]
    row_counts   = {
        "e1": SEC_ROWS + E1_EXTRA,
        "e2": SEC_ROWS,
        "e3": SEC_ROWS,
        "e4": SEC_ROWS + E1_EXTRA,  # E4 also has sector bars
    }
    GAP = 1
    total_rows = sum(row_counts[k] for k in keys_present) + GAP * max(0, len(keys_present) - 1)

    row_height = 2.5
    fig = plt.figure(figsize=(20, total_rows * row_height))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.99, bottom=0.01,
                        hspace=0.6, wspace=0.35)
    gs = gridspec.GridSpec(total_rows, COLS, figure=fig)

    row = 0

    def _skip(n=GAP):
        nonlocal row; row += n

    def _section(key):
        nonlocal row
        d      = results[key]
        passed = d["passed"]
        r0     = row

        # ── Title bar ──────────────────────────────────────────────────
        sec_labels = {
            "e1": f"E1  Export Diversion  [additive_cbam/RCPO, cbam_randomize tau in {{0, {CBAM_RATE}}}]",
            "e2": f"E2  Mitigation Signal  [additive_cbam/RCPO, delta_max=0]",
            "e3":  "E3  DICE Damage Avoidance  [welfloss, tau=0, costly abatement]",
            "e4": f"E4  Both Channels  [additive_cbam/RCPO, tau={CBAM_RATE}]",
        }
        _add_section_title(fig, gs[r0, 0:COLS],
                           f"{'PASS ✓' if passed else 'FAIL ✗'}  {sec_labels[key]}",
                           passed)
        row += TITLE_ROWS

        # ── Convergence + badge ─────────────────────────────────────────
        ax_conv  = fig.add_subplot(gs[row:row + CONV_ROWS, 0:2])
        ax_badge = fig.add_subplot(gs[row:row + CONV_ROWS, 2:4])
        if key == "e1":
            badge_lines = [
                f"agg EU dirty share",
                f"  tau=0  : {d['share_ctrl']:.3f}",
                f"  tau={CBAM_RATE}: {d['share_treat']:.3f}",
                f"  drop   : {d['drop_pp']:.1f} pp",
                f"  rel    : {d.get('rel_drop', 0)*100:.1f}%",
                f"  thresh : >{E1_RELATIVE_DROP_THRESHOLD*100:.0f}% rel (fixed λ={E1_CBAM_LAMBDA_INIT})",
            ]
        elif key in ("e2", "e3"):
            thresh_str = ">0.15" if key == "e2" else "0.01 < mu < 0.99"
            badge_lines = [
                f"mean mu = {d['mean_mu']:.3f}",
                f"thresh  : {thresh_str}",
            ]
        else:  # e4
            badge_lines = [
                f"mean mu  = {d['mean_mu']:.3f}",
                f"thresh   >= {d['threshold']:.3f}",
                f"(50% of E2)",
            ]
        _plot_convergence(ax_conv, d["csv_path"],
                          title=f"{key.upper()} — training convergence (ep_return ± std)")
        _plot_badge(ax_badge, passed, badge_lines)
        row += CONV_ROWS

        # ── Data rows ──────────────────────────────────────────────────
        if key == "e1":
            # Dirty sector bars (left 2 cols) + clean sector bars (right 2 cols)
            ax_d = fig.add_subplot(gs[row:row + DATA_ROWS, 0:2])
            ax_c = fig.add_subplot(gs[row:row + DATA_ROWS, 2:4])
            _plot_eu_sector_bars(ax_d, ax_c,
                                  d["per_region_ctrl"], d["per_region_treat"])
            row += DATA_ROWS
            # Second data row: per-region dirty/clean combined as simple delta bars
            ax_delta = fig.add_subplot(gs[row:row + E1_EXTRA, 0:4])
            x     = np.arange(len(CBAM_PLOT_REGIONS))
            w     = 0.35
            names = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
            d_drops = [(d["per_region_ctrl"][r]["dirty"] - d["per_region_treat"][r]["dirty"]) * 100
                       for r in CBAM_PLOT_REGIONS]
            c_drops = [(d["per_region_ctrl"][r]["clean"] - d["per_region_treat"][r]["clean"]) * 100
                       for r in CBAM_PLOT_REGIONS]
            ax_delta.bar(x - w/2, d_drops, w, label="dirty sector drop",
                         color="#c0392b", edgecolor="k", lw=0.6)
            ax_delta.bar(x + w/2, c_drops, w, label="clean sector drop",
                         color="#27ae60", edgecolor="k", lw=0.6)
            ax_delta.axhline(0, color="k", lw=0.8, ls="--")
            ax_delta.set_xticks(x)
            ax_delta.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax_delta.set_ylabel("pp drop (tau=0 -> tau=0.8)", fontsize=8)
            ax_delta.set_title("E1 — per-region EU export share drop (positive = diversion occurred)",
                                fontsize=9)
            ax_delta.legend(fontsize=7)
            ax_delta.tick_params(labelsize=7)
            row += E1_EXTRA

        elif key in ("e2", "e3"):
            ax_mu = fig.add_subplot(gs[row:row + DATA_ROWS, 0:4])
            _plot_region_mu(ax_mu, d["per_region_mu"],
                            label=f"{key.upper()} — mitigation rate per region")
            row += DATA_ROWS

        else:  # e4
            ax_mu = fig.add_subplot(gs[row:row + DATA_ROWS, 0:4])
            e2_mu = results["e2"]["per_region_mu"] if "e2" in results else None
            if e2_mu:
                _plot_region_mu(ax_mu, d["per_region_mu"], label="E4",
                                compare=e2_mu, compare_label="E2")
                ax_mu.set_title("E4 vs E2 — mitigation rate per region", fontsize=9)
            else:
                _plot_region_mu(ax_mu, d["per_region_mu"],
                                label="E4 — mitigation rate per region")
            row += DATA_ROWS
            ax_sector = fig.add_subplot(gs[row:row + E1_EXTRA, 0:4])
            _plot_e4_sector_bars(ax_sector, d["per_region_trade"])
            row += E1_EXTRA

        # ── Utility trajectory ─────────────────────────────────────────
        ax_util = fig.add_subplot(gs[row:row + TRAJ_ROWS, 0:4])
        if key == "e1":
            _plot_util_traj(ax_util, d["util_ctrl"],
                            title="E1 — utility trajectory (solid=tau=0, dashed=tau=0.8)",
                            traj2=d["util_treat"])
        else:
            tau_str = f"tau={CBAM_RATE}" if key != "e3" else "tau=0"
            _plot_util_traj(ax_util, d["util_traj"],
                            title=f"{key.upper()} — utility trajectory ({tau_str})")
        row += TRAJ_ROWS

    for i, key in enumerate(keys_present):
        _section(key)
        if i < len(keys_present) - 1:
            _skip()

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS, SEED, RCPO_ETA_LAMBDA, RCPO_ALPHA_TARGET, WELFARE_LOSS_WEIGHT

    parser = argparse.ArgumentParser(description="9-region CBAM experiment")
    parser.add_argument("--timesteps",    type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--eta-lambda",   type=float, default=RCPO_ETA_LAMBDA)
    parser.add_argument("--alpha-target", type=float, default=RCPO_ALPHA_TARGET)
    parser.add_argument("--welfare-loss-weight", type=float, default=WELFARE_LOSS_WEIGHT)
    parser.add_argument("--tests", nargs="+", choices=["e1","e2","e3","e4"],
                        default=["e1","e2","e3","e4"])
    parser.add_argument("--replot", metavar="PKL", default=None,
                        help="Skip training; reload saved results from this pickle and replot.")
    args = parser.parse_args()

    TOTAL_TIMESTEPS   = args.timesteps
    SEED              = args.seed
    RCPO_ETA_LAMBDA   = args.eta_lambda
    RCPO_ALPHA_TARGET = args.alpha_target
    WELFARE_LOSS_WEIGHT = args.welfare_loss_weight
    _BASE_ENV["welfare_loss_per_unit_tariff"] = WELFARE_LOSS_WEIGHT

    key = jax.random.PRNGKey(SEED)
    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    print(f"\n{'='*55}")
    print(f"  CBAM Experiment — 9-region vulnerability setup")
    print(f"  {TOTAL_TIMESTEPS:,} steps ({num_iters} iters) | seed={SEED}")
    print(f"  eta_lambda={RCPO_ETA_LAMBDA:.0e}  alpha_target={RCPO_ALPHA_TARGET}  "
          f"welfare_loss_wt={WELFARE_LOSS_WEIGHT}")
    print(f"  Tests: {args.tests}")
    print(f"{'='*55}")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Replot-only mode ─────────────────────────────────────────────────
    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _os.path.join(OUTPUT_DIR, f"cbam_experiment_9_{ts_str}.png")
        _plot_all(results, out_path)
        return

    results = {}
    e2_mu   = 0.0

    if "e1" in args.tests:
        passed, data = run_e1(key)
        results["e1"] = {**data, "passed": passed}

    if "e2" in args.tests:
        passed, data = run_e2(key)
        results["e2"] = {**data, "passed": passed}
        e2_mu = data["mean_mu"]

    if "e3" in args.tests:
        passed, data = run_e3(key)
        results["e3"] = {**data, "passed": passed}

    if "e4" in args.tests:
        passed, data = run_e4(key, e2_mu)
        results["e4"] = {**data, "passed": passed}

    # Summary
    print(f"\n{'='*55}")
    print("  EXPERIMENT SUMMARY  (9-region vulnerability)")
    print(f"{'='*55}")
    descs = {
        "e1": "Export diversion  (CBAM -> less dirty exports to EU)",
        "e2": "Mitigation signal (free abat. + CBAM -> mu > 0.15)",
        "e3": "DICE baseline     (no CBAM, costly -> 0.01 < mu < 0.99)",
        "e4": "Both channels     (diversion doesn't kill mitigation)",
    }
    all_passed = True
    for tid in ["e1","e2","e3","e4"]:
        if tid not in results:
            continue
        r   = results[tid]
        tag = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        print(f"  [{tag}]  {descs[tid]}")
    print(f"\n  Overall: {'ALL PASS' if all_passed else 'SOME FAILURES'}")

    # Save results for --replot
    if results:
        ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
        pkl_path  = _os.path.join(OUTPUT_DIR, f"cbam_experiment_9_{ts_str}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"  Results saved: {pkl_path}  (use --replot {pkl_path} to replot)")

    # Plot
    if results:
        out_path = _os.path.join(OUTPUT_DIR, f"cbam_experiment_9_{ts_str}.png")
        _plot_all(results, out_path)


if __name__ == "__main__":
    main()
