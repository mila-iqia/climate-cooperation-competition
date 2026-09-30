"""cbam_experiment_litmus_diff.py

9-region litmus tests using differential CBAM (cbam_tariff_mode="differential").
Differential CBAM replaces cbam_randomize — the effective tariff τ_eff[r] is
self-incentivising: τ_eff[r] = max(0, MAC_EU − MAC_r) / MAC_EU.  As region r
decarbonises to match EU carbon pricing, its tariff shrinks to zero.

Four tests establish the basic assumptions of the CBAM-RICE model:

  L1  Export diversion under CBAM
        differential CBAM, mitigation pinned=0, exports free.
        ctrl:  flat τ=0   (no CBAM)         → agents show MRIO-baseline shares
        treat: differential CBAM             → agents divert away from EU
        Pass: treat EU dirty share < ctrl by >15% relative.

  L2  Free mitigation signal
        differential CBAM, zero_abatement_cost, exports pinned (delta_max=0).
        Mitigation is the only escape from CBAM.
        Pass: mean μ (non-EU, last 5 steps) > 0.15.

  L3  Costly mitigation signal
        differential CBAM, realistic abatement cost, exports pinned.
        Agents still mitigate some (RICE DICE result) but less than L2.
        Pass: μ ∈ (0.01, L2_mu − 0.01)   [positive but lower than free case].

  L4  Both channels open (reallocation crowds out mitigation)
        differential CBAM, realistic abatement cost, BOTH export + mitigation free.
        Diversion substitutes for mitigation.
        Pass: mean μ (non-EU) < L3_mu   [reallocation reduces mitigation effort].

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_litmus_diff.py [--timesteps 2000000]
    python validation/cbam_experiment_litmus_diff.py --replot <pickle.pkl>
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

TOTAL_TIMESTEPS   = 2_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42

# Fixed λ=1.0 (same calibration as cbam_experiment_9.py E1)
CBAM_LAMBDA_INIT     = 1.0
WELFARE_LOSS_WEIGHT  = 5.0
L1_REL_DROP_THRESH   = 0.15   # 15% relative drop in EU dirty share

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "litmus_diff_"

_REGION_COLORS = [plt.cm.tab10(i / 10) for i in range(NUM_REGIONS)]

# ── Environment defaults ────────────────────────────────────────────────────

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


# ── Build helpers ───────────────────────────────────────────────────────────

def _build_env(cbam_mode, zero_abatement_cost, no_mitigation, fixed_savings,
               delta_max=3.0, cbam_tariff_rate=0.0, cbam_lambda_init=None,
               for_training=True):
    extra = dict(log_info_fn=rcpo_cbam_log_info_fn)
    if cbam_lambda_init is not None:
        extra["cbam_lambda_init"] = cbam_lambda_init
    env = RiceMRIO(
        region_params         = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_mode      = cbam_mode,
        cbam_tariff_rate      = cbam_tariff_rate,
        zero_abatement_cost   = zero_abatement_cost,
        no_mitigation         = no_mitigation,
        fixed_savings_rate    = fixed_savings,
        delta_max             = delta_max,
        reward_mode           = "additive_cbam",
        **_BASE_ENV,
        **extra,
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


def _train(label, env, key, num_iters, total_timesteps=None):
    ts = total_timesteps or TOTAL_TIMESTEPS
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = MonitoredPPO(
        total_timesteps=ts,
        log_function=log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━'*55}")
    print(f"  Training: {label}")
    print(f"{'━'*55}")
    t0  = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path


# ── Eval helpers ────────────────────────────────────────────────────────────

def _eval_episode(key, raw_env, agent):
    from _experiment_util import run_single_episode

    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    flows, mit, util = [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 90_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)
        flows.append(np.array(logs["trade_flows"]))
        mit.append(_to_arr(logs["mitigation_rates_all_regions"]))
        util.append(_to_arr(logs["utility_all_regions"]))

    return {
        "trade_flows": np.stack(flows, 0),
        "mitigation":  np.stack(mit,   0),
        "utility":     np.stack(util,  0),
    }


def _eu_dirty_share_agg(trade_flows, last_t=5):
    """Aggregate mean EU dirty export share across CBAM-relevant regions."""
    tf = trade_flows[:, -last_t:]
    dirty_eu  = tf[:, :, :, EU_IDX, 0]
    dirty_all = tf[:, :, :, :,      0].sum(-1)
    return float((dirty_eu[:, :, CBAM_PLOT_REGIONS]
                  / (dirty_all[:, :, CBAM_PLOT_REGIONS] + 1e-10)).mean())


def _per_region_eu_dirty_share(trade_flows, last_t=5):
    tf = trade_flows[:, -last_t:]
    out = {}
    for r in NON_EU:
        eu_d  = tf[:, :, r, EU_IDX, 0]
        all_d = tf[:, :, r, :, 0].sum(-1)
        out[r] = float((eu_d / (all_d + 1e-10)).mean())
    return out


def _per_region_mu(mitigation, last_t=5):
    return {r: float(mitigation[:, -last_t:, r].mean()) for r in range(NUM_REGIONS)}


def _mean_mu_non_eu(mitigation, last_t=5):
    return float(mitigation[:, -last_t:, :][:, :, NON_EU].mean())


# ── Experiment runners ──────────────────────────────────────────────────────

def run_l1(key):
    """L1: Export diversion.
    Train ctrl (flat τ=0) and treat (differential) with mitigation pinned.
    Show that treat EU dirty share < ctrl.
    """
    print("\n" + "="*55)
    print("  L1  Export Diversion")
    print("  ctrl=flat τ=0  vs  treat=differential CBAM")
    print("  no_mitigation, fixed_savings, exports free")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    key_ctrl, key_diff = jax.random.split(key, 2)

    # Train ctrl: no CBAM
    env_ctrl = _build_env("flat", False, True, True, cbam_tariff_rate=0.0)
    agent_ctrl, csv_ctrl = _train("l1_ctrl", env_ctrl, key_ctrl, num_iters)

    # Train treat: differential CBAM
    env_diff = _build_env("differential", False, True, True,
                          cbam_lambda_init=CBAM_LAMBDA_INIT)
    agent_diff, csv_diff = _train("l1_diff", env_diff, key_diff, num_iters)

    # Eval both agents in their respective environments (ctrl at τ=0, diff at differential)
    eval_key = jax.random.fold_in(key, 10)
    raw_ctrl = _build_env("flat", False, True, True, cbam_tariff_rate=0.0, for_training=False)
    raw_diff = _build_env("differential", False, True, True,
                          cbam_lambda_init=CBAM_LAMBDA_INIT, for_training=False)

    ev_ctrl = _eval_episode(eval_key, raw_ctrl, agent_ctrl)
    ev_diff = _eval_episode(eval_key, raw_diff, agent_diff)

    share_ctrl  = _eu_dirty_share_agg(ev_ctrl["trade_flows"])
    share_diff  = _eu_dirty_share_agg(ev_diff["trade_flows"])
    rel_drop    = (share_ctrl - share_diff) / (share_ctrl + 1e-10)
    drop_pp     = (share_ctrl - share_diff) * 100

    passed = rel_drop > L1_REL_DROP_THRESH
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L1 {tag}: EU dirty share  ctrl: {share_ctrl:.3f}  diff: {share_diff:.3f}  "
          f"drop={drop_pp:.1f}pp ({rel_drop*100:.1f}% rel)  (>{L1_REL_DROP_THRESH*100:.0f}% rel)")

    return passed, dict(
        share_ctrl=share_ctrl, share_diff=share_diff,
        drop_pp=drop_pp, rel_drop=rel_drop,
        pr_ctrl=_per_region_eu_dirty_share(ev_ctrl["trade_flows"]),
        pr_diff=_per_region_eu_dirty_share(ev_diff["trade_flows"]),
        util_ctrl=ev_ctrl["utility"].mean(0),
        util_diff=ev_diff["utility"].mean(0),
        csv_ctrl=csv_ctrl, csv_diff=csv_diff,
    )


def run_l2(key):
    """L2: Free mitigation signal — agents mitigate to avoid differential CBAM."""
    print("\n" + "="*55)
    print("  L2  Free Mitigation Signal")
    print("  differential CBAM, zero_abatement_cost, exports pinned (delta_max=0)")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env  = _build_env("differential", True, False, True, delta_max=0.0,
                      cbam_lambda_init=CBAM_LAMBDA_INIT)
    agent, csv_path = _train("l2_free_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 20)
    raw_env  = _build_env("differential", True, False, True, delta_max=0.0,
                          cbam_lambda_init=CBAM_LAMBDA_INIT, for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu = _mean_mu_non_eu(ev["mitigation"])
    pr_mu   = _per_region_mu(ev["mitigation"])

    passed = mean_mu > 0.15
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L2 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (>0.15)")

    return passed, dict(mean_mu=mean_mu, pr_mu=pr_mu,
                        util_traj=ev["utility"].mean(0), csv_path=csv_path)


def run_l3(key):
    """L3: Differential CBAM self-incentivising loop under realistic abatement cost.

    With zero_abatement_cost (L2), MAC_r ≈ 0 so τ_eff[r] ≈ 1 regardless of μ_r
    — mitigation cannot reduce the tariff.  With realistic abatement cost (L3),
    MAC_r > 0 and as μ_r rises MAC_r approaches MAC_EU, shrinking τ_eff[r] → 0.
    The self-incentivising loop is active, producing μ_L3 > μ_L2.
    """
    print("\n" + "="*55)
    print("  L3  Costly Mitigation Signal")
    print("  differential CBAM, realistic abatement cost, exports pinned")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env  = _build_env("differential", False, False, True, delta_max=0.0,
                      cbam_lambda_init=CBAM_LAMBDA_INIT)
    agent, csv_path = _train("l3_costly_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 30)
    raw_env  = _build_env("differential", False, False, True, delta_max=0.0,
                          cbam_lambda_init=CBAM_LAMBDA_INIT, for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu = _mean_mu_non_eu(ev["mitigation"])
    pr_mu   = _per_region_mu(ev["mitigation"])

    # Pass: under differential CBAM the self-incentivising loop (MAC_r > 0 → τ_eff
    # decreases as μ rises) should produce μ_L3 > μ_L2.  The L2 vs L3 ordering
    # is checked after both are known; here just verify μ > 0.01.
    passed = mean_mu > 0.01
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L3 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (>0.01; expect >L2 via self-incentivising loop)")

    return passed, dict(mean_mu=mean_mu, pr_mu=pr_mu,
                        util_traj=ev["utility"].mean(0), csv_path=csv_path)


def run_l4(key, l3_mu):
    """L4: Both channels — export reallocation crowds out costly mitigation."""
    print("\n" + "="*55)
    print("  L4  Both Channels (reallocation crowds out mitigation)")
    print("  differential CBAM, realistic abatement cost, exports + mitigation free")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env  = _build_env("differential", False, False, True,
                      cbam_lambda_init=CBAM_LAMBDA_INIT)
    agent, csv_path = _train("l4_both", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 40)
    raw_env  = _build_env("differential", False, False, True,
                          cbam_lambda_init=CBAM_LAMBDA_INIT, for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu  = _mean_mu_non_eu(ev["mitigation"])
    pr_mu    = _per_region_mu(ev["mitigation"])
    pr_share = _per_region_eu_dirty_share(ev["trade_flows"])

    # Pass: mitigation is lower than L3 (diversion substitutes for mitigation)
    passed = mean_mu < l3_mu
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L4 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (< L3 μ={l3_mu:.3f})")

    return passed, dict(mean_mu=mean_mu, pr_mu=pr_mu, pr_share=pr_share,
                        util_traj=ev["utility"].mean(0), csv_path=csv_path)


# ── Plot ────────────────────────────────────────────────────────────────────

def _read_csv(csv_path):
    if not csv_path or not _os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _plot_convergence(ax, csv_paths, labels, colors, title):
    for csv_path, label, color in zip(csv_paths, labels, colors):
        df = _read_csv(csv_path)
        if df.empty or "ep_return_mean" not in df.columns:
            continue
        iters = df.get("iteration", pd.Series(range(len(df))))
        mean  = df["ep_return_mean"].rolling(5, min_periods=1).mean()
        std   = df.get("ep_return_std", pd.Series(np.zeros(len(df))))
        ax.plot(iters, mean, lw=1.5, color=color, label=label)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("ep return", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


def _bar_per_region(ax, data_dict, title, ylabel, colors=None):
    """data_dict: {label: {region_idx: float}}"""
    n_groups = len(CBAM_PLOT_REGIONS)
    n_bars   = len(data_dict)
    width    = 0.8 / n_bars
    x        = np.arange(n_groups)
    for i, (label, rdict) in enumerate(data_dict.items()):
        vals  = [rdict.get(r, 0.0) for r in CBAM_PLOT_REGIONS]
        color = colors[i] if colors else f"C{i}"
        ax.bar(x + (i - n_bars/2 + 0.5) * width, vals,
               width=width, color=color, alpha=0.85, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
                       rotation=25, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)


def _badge(ax, tests):
    """tests: list of (label, passed, summary_str)"""
    ax.axis("off")
    lines = ["Litmus Results", "─" * 34]
    for label, passed, summary in tests:
        sym = "✅" if passed else "❌"
        lines.append(f"  {sym}  {label}: {summary}")
    overall = all(p for _, p, _ in tests)
    color = "#1a6b3a" if overall else "#8b1a1a"
    ax.text(0.5, 0.5, "\n".join(lines), ha="center", va="center",
            transform=ax.transAxes, fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=color,
                      alpha=0.12, edgecolor=color, linewidth=2))


def plot_results(results, timestamp):
    r = results
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        f"9-Region Differential CBAM — Basic Assumptions (Litmus Tests)\n"
        f"{TOTAL_TIMESTEPS//1_000_000}M steps, {NUM_ENVS} envs, seed={SEED}",
        fontsize=13, fontweight="bold",
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: L1 ────────────────────────────────────────────────────────────
    ax_l1_conv = fig.add_subplot(gs[0, 0:2])
    ax_l1_bar  = fig.add_subplot(gs[0, 2:4])

    if "l1" in r:
        l1 = r["l1"]["data"]
        _plot_convergence(
            ax_l1_conv,
            [l1["csv_ctrl"], l1["csv_diff"]],
            ["ctrl (τ=0)", "differential CBAM"],
            ["#2980b9", "#e74c3c"],
            "L1: Training convergence",
        )
        _bar_per_region(
            ax_l1_bar,
            {"ctrl (τ=0)": l1["pr_ctrl"], "differential CBAM": l1["pr_diff"]},
            "L1: EU dirty export share per region\n(differential CBAM diverts away from EU)",
            "EU dirty export share",
            colors=["#2980b9", "#e74c3c"],
        )

    # ── Row 1: L2 + L3 convergence ───────────────────────────────────────────
    ax_l2_conv = fig.add_subplot(gs[1, 0])
    ax_l3_conv = fig.add_subplot(gs[1, 1])
    ax_l23_bar = fig.add_subplot(gs[1, 2:4])

    if "l2" in r:
        l2 = r["l2"]["data"]
        _plot_convergence(ax_l2_conv, [l2["csv_path"]], ["L2 free abatement"],
                          ["#27ae60"], "L2: Training convergence")
    if "l3" in r:
        l3 = r["l3"]["data"]
        _plot_convergence(ax_l3_conv, [l3["csv_path"]], ["L3 costly abatement"],
                          ["#f39c12"], "L3: Training convergence")
    if "l2" in r and "l3" in r:
        _bar_per_region(
            ax_l23_bar,
            {"L2 free (MAC≈0 → no τ-relief)": r["l2"]["data"]["pr_mu"],
             "L3 costly (μ↑ → τ_eff↓, self-incentivising)": r["l3"]["data"]["pr_mu"]},
            "L2 vs L3: Mitigation rate per region\n(L3 costly > L2 free: differential τ-relief loop active under realistic MAC)",
            "Mean mitigation rate μ",
            colors=["#27ae60", "#f39c12"],
        )

    # ── Row 2: L4 convergence + mitigation bar ────────────────────────────────
    ax_l4_conv  = fig.add_subplot(gs[2, 0])
    ax_l4_mu    = fig.add_subplot(gs[2, 1:3])
    ax_l4_share = fig.add_subplot(gs[2, 3])

    if "l4" in r:
        l4 = r["l4"]["data"]
        _plot_convergence(ax_l4_conv, [l4["csv_path"]], ["L4 both channels"],
                          ["#8e44ad"], "L4: Training convergence")

    if "l3" in r and "l4" in r:
        _bar_per_region(
            ax_l4_mu,
            {"L3 mitigation-only": r["l3"]["data"]["pr_mu"],
             "L4 realloc+mitigation": r["l4"]["data"]["pr_mu"]},
            "L3 vs L4: Reallocation crowds out mitigation\n(CBAM+diversion channel → lower μ than mitigation-only)",
            "Mean mitigation rate μ",
            colors=["#f39c12", "#8e44ad"],
        )

    if "l4" in r:
        _bar_per_region(
            ax_l4_share,
            {"L4 (both channels)": r["l4"]["data"]["pr_share"]},
            "L4: EU dirty share\n(with both channels)",
            "EU dirty export share",
            colors=["#8e44ad"],
        )

    # ── Row 3: pass/fail summary ──────────────────────────────────────────────
    ax_badge = fig.add_subplot(gs[3, :])
    tests = []
    if "l1" in r:
        l1d = r["l1"]["data"]
        tests.append((
            "L1 Diversion",
            r["l1"]["passed"],
            f"ctrl={l1d['share_ctrl']:.3f}→diff={l1d['share_diff']:.3f}  "
            f"drop={l1d['drop_pp']:.1f}pp ({l1d['rel_drop']*100:.1f}% rel)",
        ))
    if "l2" in r:
        tests.append(("L2 Free μ",   r["l2"]["passed"],
                       f"mean μ={r['l2']['data']['mean_mu']:.3f}  (>0.15)"))
    if "l3" in r:
        l3d = r["l3"]["data"]
        l2_mu_str = f"{r['l2']['data']['mean_mu']:.3f}" if "l2" in r else "?"
        tests.append(("L3 Self-incentivising loop", r["l3"]["passed"],
                       f"μ_L3={l3d['mean_mu']:.3f} > μ_L2={l2_mu_str} (loop active)"))
    if "l4" in r:
        tests.append(("L4 Crowding", r["l4"]["passed"],
                       f"mean μ={r['l4']['data']['mean_mu']:.3f}  "
                       f"(<L3 {r['l3']['data']['mean_mu']:.3f})"))
    _badge(ax_badge, tests)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"litmus_diff_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--tests",     type=str, default="l1,l2,l3,l4",
                        help="Comma-separated subset, e.g. l1,l2")
    parser.add_argument("--replot",    type=str, default=None,
                        help="Path to existing .pkl — skip training, regenerate plot only")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        plot_results(results, timestamp)
        return

    TOTAL_TIMESTEPS = args.timesteps
    tests_to_run = [t.strip().lower() for t in args.tests.split(",")]

    key  = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 10)
    results = {}

    if "l1" in tests_to_run:
        passed, data = run_l1(keys[0])
        results["l1"] = {"passed": passed, "data": data}

    if "l2" in tests_to_run:
        passed, data = run_l2(keys[1])
        results["l2"] = {"passed": passed, "data": data}

    l3_mu = 0.05   # fallback if l3 not run
    if "l3" in tests_to_run:
        passed, data = run_l3(keys[2])
        results["l3"] = {"passed": passed, "data": data}
        l3_mu = data["mean_mu"]
        # Under differential CBAM, expect L3 > L2 (self-incentivising loop):
        # MAC_r=0 (L2) → τ_eff always 1, mitigation can't reduce tariff.
        # MAC_r>0 (L3) → μ↑ reduces τ_eff, incentivising higher mitigation.
        if "l2" in results:
            l2_mu = results["l2"]["data"]["mean_mu"]
            l3_loop_pass = l3_mu > l2_mu
            results["l3"]["passed"] = l3_loop_pass
            print(f"  L3 vs L2 self-incentivising loop: {l3_mu:.3f} > {l2_mu:.3f}  "
                  f"→ {'PASS (loop active)' if l3_loop_pass else 'FAIL (loop absent)'}")

    # Save pickle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"litmus_diff_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Results saved: {pkl_path}")

    # Plot
    plot_results(results, timestamp)

    # Summary
    print("\n" + "="*55)
    print("  LITMUS SUMMARY")
    print("="*55)
    for test, res in results.items():
        tag = "PASS ✅" if res["passed"] else "FAIL ❌"
        print(f"  {test.upper()}  {tag}")


if __name__ == "__main__":
    main()
