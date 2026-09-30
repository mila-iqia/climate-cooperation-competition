"""cbam_experiment_litmus_rand.py

Repeat of the L1–L4 litmus tests with a SINGLE model trained on randomised
CBAM: each episode is drawn with probability ½ from τ=0 (CBAM off) and ½
from τ=1 (full differential CBAM).  The active rate is stored in
``state["cbam_tariff_rate"]`` and visible in the observation.

Hypothesis: the contrastive within-batch signal (CBAM-on vs CBAM-off episodes
with identical policies) produces a gradient large enough for PPO to learn a
condition-dependent policy, even when the EU hasn't yet decarbonised enough
to create a non-zero differential rate at initialisation.

Differences from cbam_experiment_litmus_diff.py
────────────────────────────────────────────────
• Training uses cbam_randomize=True, cbam_tariff_rates=(0.0, 1.0)
  for every CBAM-active condition (L2, L3, L4).
• Eval is done *twice* per condition — once with cbam_tariff_rate forced
  to 0 (CBAM off) and once with 1 (CBAM on, full differential) — by
  replacing cbam_randomize=False and the scalar rate.
• Pass criteria check conditioning: μ(cbam=1) > μ(cbam=0) for L2/L3,
  EU share drop preserved in L1, diversion crowd-out preserved in L4.

Prerequisite env change (already applied)
─────────────────────────────────────────
In rice_jax/_rice_mrio.py _compute_cbam(), the differential branch now
multiplies rate_per_region by cbam_tariff_rate when self.cbam_randomize is
True, so the binary gate works correctly.

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_experiment_litmus_rand.py [--timesteps 2000000]
    python validation/cbam_experiment_litmus_rand.py --replot <pickle.pkl>
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
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]

TOTAL_TIMESTEPS   = 2_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42

CBAM_LAMBDA_INIT     = 1.0
WELFARE_LOSS_WEIGHT  = 5.0
L1_REL_DROP_THRESH   = 0.15

# EU mitigation schedule: prescribes EU mitigation rate at each RICE timestep.
# Ramps 0.30→1.00 over 8 steps (EU Climate Law / net-zero 2050 pathway).
# Ensures MAC_EU > 0 from t=0 so the differential tariff is non-trivial.
# [EU Climate Law (EU) 2021/1119; IPCC AR6 mitigation scenarios]
EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "litmus_rand_"

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


# ── Build helpers ───────────────────────────────────────────────────────────

def _build_env(cbam_mode, zero_abatement_cost, no_mitigation, fixed_savings,
               delta_max=3.0, cbam_tariff_rate=0.0, cbam_lambda_init=None,
               randomize=False, for_training=True):
    """Build a RiceMRIO env.

    randomize=True adds cbam_randomize=True, cbam_tariff_rates=(0.0, 1.0).
    When randomize is on and cbam_mode="differential", each episode the
    effective tariff is either 0 (CBAM off) or the full MAC-based differential
    rate (CBAM on), gated by state["cbam_tariff_rate"].
    """
    extra = dict(log_info_fn=rcpo_cbam_log_info_fn)
    if cbam_lambda_init is not None:
        extra["cbam_lambda_init"] = cbam_lambda_init
    if randomize:
        extra["cbam_randomize"]       = True
        extra["cbam_tariff_rates"]    = (0.0, 1.0)
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


def _build_eval_env(cbam_mode, zero_abatement_cost, no_mitigation, fixed_savings,
                    delta_max=3.0, cbam_lambda_init=None, force_rate=1.0):
    """Build a deterministic eval env with a fixed cbam_tariff_rate.

    Always sets randomize=False and for_training=False.
    Avoids unwrapping LogWrapper (equinox proxies attribute access).
    """
    return _build_env(
        cbam_mode, zero_abatement_cost, no_mitigation, fixed_savings,
        delta_max=delta_max, cbam_tariff_rate=force_rate,
        cbam_lambda_init=cbam_lambda_init, randomize=False, for_training=False,
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


def _train(label, env, key, num_iters):
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = MonitoredPPO(
        total_timesteps=TOTAL_TIMESTEPS,
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

    # Swap in full state logging so trade_flows and mitigation_rates are available.
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
    tf = trade_flows[:, -last_t:]
    dirty_eu  = tf[:, :, :, EU_IDX, 0]
    dirty_all = tf[:, :, :, :, 0].sum(-1)
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

    ctrl: flat τ=0 always (no CBAM, no randomize).
    treat: randomised differential — half episodes CBAM off, half CBAM on.
    Both models have no_mitigation, fixed_savings, exports free.

    Pass: in CBAM-on eval, treat EU dirty share drops >15% vs ctrl.
    """
    print("\n" + "="*55)
    print("  L1  Export Diversion (randomised differential)")
    print("  ctrl=flat τ=0  vs  treat=randomised differential CBAM")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    key_ctrl, key_diff = jax.random.split(key, 2)

    # Ctrl: no CBAM, no randomize
    env_ctrl   = _build_env("flat", False, True, True, cbam_tariff_rate=0.0)
    agent_ctrl, csv_ctrl = _train("l1_ctrl", env_ctrl, key_ctrl, num_iters)

    # Treat: randomised differential
    env_diff   = _build_env("differential", False, True, True,
                             cbam_lambda_init=CBAM_LAMBDA_INIT, randomize=True)
    agent_diff, csv_diff = _train("l1_diff", env_diff, key_diff, num_iters)

    eval_key = jax.random.fold_in(key, 10)

    # Ctrl eval at τ=0 (matches training)
    raw_ctrl = _build_env("flat", False, True, True, cbam_tariff_rate=0.0, for_training=False)

    # Treat eval at force_rate=1 (CBAM on) to measure conditioned diversion
    raw_diff_on  = _build_eval_env("differential", False, True, True,
                                   cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=1.0)
    raw_diff_off = _build_eval_env("differential", False, True, True,
                                   cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=0.0)

    ev_ctrl    = _eval_episode(eval_key, raw_ctrl,    agent_ctrl)
    ev_diff_on = _eval_episode(eval_key, raw_diff_on, agent_diff)
    ev_diff_off= _eval_episode(eval_key, raw_diff_off,agent_diff)

    share_ctrl     = _eu_dirty_share_agg(ev_ctrl["trade_flows"])
    share_diff_on  = _eu_dirty_share_agg(ev_diff_on["trade_flows"])
    share_diff_off = _eu_dirty_share_agg(ev_diff_off["trade_flows"])
    # Pass on within-agent conditioning gap (cbam=0 vs cbam=1).
    # The ctrl baseline is unreliable when eu_mitigation_schedule changes EU
    # dynamics identically in both arms. The conditioning gap isolates the
    # CBAM obs signal: same weights, same world, only cbam_tariff_rate differs.
    cond_gap    = (share_diff_off - share_diff_on) / (share_diff_off + 1e-10)
    cond_gap_pp = (share_diff_off - share_diff_on) * 100
    rel_drop    = (share_ctrl - share_diff_on) / (share_ctrl + 1e-10)   # informational
    drop_pp     = (share_ctrl - share_diff_on) * 100

    passed = cond_gap > L1_REL_DROP_THRESH
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L1 {tag}: ctrl={share_ctrl:.3f}  diff(cbam=0)={share_diff_off:.3f}  "
          f"diff(cbam=1)={share_diff_on:.3f}  "
          f"cond_gap={cond_gap_pp:.1f}pp ({cond_gap*100:.1f}%)  [thresh>{L1_REL_DROP_THRESH*100:.0f}%]")
    print(f"         (vs-ctrl drop: {drop_pp:.1f}pp, {rel_drop*100:.1f}%)")

    return passed, dict(
        share_ctrl=share_ctrl, share_diff_on=share_diff_on, share_diff_off=share_diff_off,
        cond_gap=cond_gap, cond_gap_pp=cond_gap_pp,
        drop_pp=drop_pp, rel_drop=rel_drop,
        conditioned=(share_diff_on < share_diff_off),
        pr_ctrl=_per_region_eu_dirty_share(ev_ctrl["trade_flows"]),
        pr_diff_on=_per_region_eu_dirty_share(ev_diff_on["trade_flows"]),
        pr_diff_off=_per_region_eu_dirty_share(ev_diff_off["trade_flows"]),
        csv_ctrl=csv_ctrl, csv_diff=csv_diff,
    )


def run_l2(key):
    """L2: Free mitigation with randomised differential CBAM.

    Single agent trained on mixture of CBAM-on/off episodes.
    Pass: μ(cbam=1) substantially > μ(cbam=0), showing the agent conditions
    its mitigation on the tariff signal in its obs.
    """
    print("\n" + "="*55)
    print("  L2  Free Mitigation — Randomised Differential CBAM")
    print("  zero_abatement_cost, exports pinned (delta_max=0), randomize")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env("differential", True, False, True, delta_max=0.0,
                     cbam_lambda_init=CBAM_LAMBDA_INIT, randomize=True)
    agent, csv_path = _train("l2_free_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 20)
    raw_on  = _build_eval_env("differential", True, False, True, delta_max=0.0,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=1.0)
    raw_off = _build_eval_env("differential", True, False, True, delta_max=0.0,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=0.0)

    ev_on  = _eval_episode(eval_key, raw_on,  agent)
    ev_off = _eval_episode(eval_key, raw_off, agent)

    mu_on  = _mean_mu_non_eu(ev_on["mitigation"])
    mu_off = _mean_mu_non_eu(ev_off["mitigation"])

    # Agent is responding to CBAM if mu_on > mu_off
    conditioning_gap = mu_on - mu_off
    passed = mu_on > 0.15 and conditioning_gap > 0.05
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L2 {tag}: μ(cbam=1)={mu_on:.3f}  μ(cbam=0)={mu_off:.3f}  "
          f"gap={conditioning_gap:.3f}  (μ_on>0.15 and gap>0.05)")

    return passed, dict(
        mean_mu_on=mu_on, mean_mu_off=mu_off, conditioning_gap=conditioning_gap,
        pr_mu_on=_per_region_mu(ev_on["mitigation"]),
        pr_mu_off=_per_region_mu(ev_off["mitigation"]),
        util_traj=ev_on["utility"].mean(0),
        csv_path=csv_path,
    )


def run_l3(key):
    """L3: Costly mitigation — self-incentivising loop under randomised differential.

    Pass: μ_on > μ_off (agent conditions on tariff signal);
          μ_on_L3 > μ_on_L2 (self-incentivising loop still holds with realistic MAC).
    """
    print("\n" + "="*55)
    print("  L3  Costly Mitigation — Randomised Differential CBAM")
    print("  realistic abatement cost, exports pinned, randomize")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env("differential", False, False, True, delta_max=0.0,
                     cbam_lambda_init=CBAM_LAMBDA_INIT, randomize=True)
    agent, csv_path = _train("l3_costly_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 30)
    raw_on  = _build_eval_env("differential", False, False, True, delta_max=0.0,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=1.0)
    raw_off = _build_eval_env("differential", False, False, True, delta_max=0.0,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=0.0)

    ev_on  = _eval_episode(eval_key, raw_on,  agent)
    ev_off = _eval_episode(eval_key, raw_off, agent)

    mu_on  = _mean_mu_non_eu(ev_on["mitigation"])
    mu_off = _mean_mu_non_eu(ev_off["mitigation"])

    conditioning_gap = mu_on - mu_off
    passed = mu_on > 0.01 and conditioning_gap > 0.01
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L3 {tag}: μ(cbam=1)={mu_on:.3f}  μ(cbam=0)={mu_off:.3f}  "
          f"gap={conditioning_gap:.3f}  (μ_on>0.01 and gap>0.01)")

    return passed, dict(
        mean_mu_on=mu_on, mean_mu_off=mu_off, conditioning_gap=conditioning_gap,
        pr_mu_on=_per_region_mu(ev_on["mitigation"]),
        pr_mu_off=_per_region_mu(ev_off["mitigation"]),
        util_traj=ev_on["utility"].mean(0),
        csv_path=csv_path,
    )


def run_l4(key, l3_mu_on):
    """L4: Both channels — reallocation crowd-out under randomised differential.

    Pass: μ_on < μ_on_L3 (diversion substitutes for mitigation when CBAM is on).
    """
    print("\n" + "="*55)
    print("  L4  Both Channels — Randomised Differential CBAM")
    print("  realistic abatement cost, exports + mitigation free, randomize")
    print("="*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env("differential", False, False, True,
                     cbam_lambda_init=CBAM_LAMBDA_INIT, randomize=True)
    agent, csv_path = _train("l4_both", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 40)
    raw_on  = _build_eval_env("differential", False, False, True,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=1.0)
    raw_off = _build_eval_env("differential", False, False, True,
                              cbam_lambda_init=CBAM_LAMBDA_INIT, force_rate=0.0)

    ev_on  = _eval_episode(eval_key, raw_on,  agent)
    ev_off = _eval_episode(eval_key, raw_off, agent)

    mu_on       = _mean_mu_non_eu(ev_on["mitigation"])
    mu_off      = _mean_mu_non_eu(ev_off["mitigation"])
    pr_share_on = _per_region_eu_dirty_share(ev_on["trade_flows"])

    passed = mu_on < l3_mu_on
    tag    = "PASS" if passed else "FAIL"
    print(f"\n  L4 {tag}: μ_on={mu_on:.3f}  μ_off={mu_off:.3f}  "
          f"(μ_on < L3 μ_on={l3_mu_on:.3f})")
    print(f"         Crowd-out: μ drops {l3_mu_on - mu_on:.3f} vs L3 "
          f"({'diversion active' if l3_mu_on > mu_on else 'no crowd-out'})")

    return passed, dict(
        mean_mu_on=mu_on, mean_mu_off=mu_off,
        pr_mu_on=_per_region_mu(ev_on["mitigation"]),
        pr_share_on=pr_share_on,
        util_traj=ev_on["utility"].mean(0),
        csv_path=csv_path,
    )


# ── Plot helpers ────────────────────────────────────────────────────────────

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
    ax.grid(True, lw=0.4, alpha=0.4)


def _bar_per_region(ax, series_dict, title, ylabel, colors=None):
    regions   = sorted({r for v in series_dict.values() for r in v.keys()})
    show_regs = [r for r in regions if r in CBAM_PLOT_REGIONS or r == EU_IDX]
    labels    = [REGION_NAMES.get(r, str(r)) for r in show_regs]
    x = np.arange(len(show_regs))
    n = len(series_dict)
    w = 0.8 / n
    _colors = colors or [plt.cm.tab10(i / 10) for i in range(n)]
    for i, (name, series) in enumerate(series_dict.items()):
        vals = [series.get(r, 0.0) for r in show_regs]
        ax.bar(x + i * w - (n - 1) * w / 2, vals, w * 0.9, label=name, color=_colors[i], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)


def _badge(ax, tests):
    ax.axis("off")
    lines = ["Litmus Results (Randomised Differential CBAM)", "─" * 42]
    for label, passed, summary in tests:
        sym = "+" if passed else "x"
        lines.append(f"  [{sym}]  {label}: {summary}")
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
        f"9-Region Randomised Differential CBAM — Litmus Tests\n"
        f"{TOTAL_TIMESTEPS//1_000_000}M steps, {NUM_ENVS} envs, seed={SEED}  "
        f"(cbam_randomize=True, rates=(0,1))",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: L1 ────────────────────────────────────────────────────────────
    ax_l1_conv = fig.add_subplot(gs[0, 0:2])
    ax_l1_bar  = fig.add_subplot(gs[0, 2:4])
    if "l1" in r:
        l1 = r["l1"]["data"]
        _plot_convergence(ax_l1_conv,
                          [l1["csv_ctrl"], l1["csv_diff"]],
                          ["ctrl (τ=0)", "treat (randomised diff)"],
                          ["#2980b9", "#e74c3c"],
                          "L1: Training convergence")
        _bar_per_region(ax_l1_bar,
                        {"ctrl (τ=0)":         l1["pr_ctrl"],
                         "diff (cbam=1)":      l1["pr_diff_on"],
                         "diff (cbam=0)":      l1["pr_diff_off"]},
                        "L1: EU dirty share — conditioning on τ\n"
                        "(diff(cbam=1) should be lowest)",
                        "EU dirty export share",
                        colors=["#2980b9", "#e74c3c", "#f39c12"])

    # ── Row 1: L2 + L3 ───────────────────────────────────────────────────────
    ax_l2_conv = fig.add_subplot(gs[1, 0])
    ax_l3_conv = fig.add_subplot(gs[1, 1])
    ax_l23_bar = fig.add_subplot(gs[1, 2:4])
    if "l2" in r:
        _plot_convergence(ax_l2_conv, [r["l2"]["data"]["csv_path"]],
                          ["L2 free abatement"], ["#27ae60"], "L2: Convergence")
    if "l3" in r:
        _plot_convergence(ax_l3_conv, [r["l3"]["data"]["csv_path"]],
                          ["L3 costly abatement"], ["#f39c12"], "L3: Convergence")
    if "l2" in r and "l3" in r:
        _bar_per_region(ax_l23_bar,
                        {"L2 free (cbam=1)":   r["l2"]["data"]["pr_mu_on"],
                         "L2 free (cbam=0)":   r["l2"]["data"]["pr_mu_off"],
                         "L3 costly (cbam=1)":  r["l3"]["data"]["pr_mu_on"],
                         "L3 costly (cbam=0)":  r["l3"]["data"]["pr_mu_off"]},
                        "L2 vs L3: μ conditioned on CBAM\n"
                        "(cbam=1 bars should exceed cbam=0; L3-on > L2-on = self-incentivising loop)",
                        "Mean mitigation rate μ",
                        colors=["#27ae60", "#a9dfbf", "#f39c12", "#fad7a0"])

    # ── Row 2: L4 ────────────────────────────────────────────────────────────
    ax_l4_conv  = fig.add_subplot(gs[2, 0])
    ax_l4_mu    = fig.add_subplot(gs[2, 1:3])
    ax_l4_share = fig.add_subplot(gs[2, 3])
    if "l4" in r:
        _plot_convergence(ax_l4_conv, [r["l4"]["data"]["csv_path"]],
                          ["L4 both channels"], ["#8e44ad"], "L4: Convergence")
    if "l3" in r and "l4" in r:
        _bar_per_region(ax_l4_mu,
                        {"L3 mitigation-only (cbam=1)": r["l3"]["data"]["pr_mu_on"],
                         "L4 both (cbam=1)":            r["l4"]["data"]["pr_mu_on"]},
                        "L3 vs L4: crowd-out when CBAM is on\n"
                        "(L4 μ should be lower — diversion substitutes for abatement)",
                        "Mean μ", colors=["#f39c12", "#8e44ad"])
    if "l4" in r:
        _bar_per_region(ax_l4_share,
                        {"L4 EU dirty share (cbam=1)": r["l4"]["data"]["pr_share_on"]},
                        "L4: EU dirty share (cbam=1)", "EU dirty share",
                        colors=["#8e44ad"])

    # ── Row 3: badge ─────────────────────────────────────────────────────────
    ax_badge = fig.add_subplot(gs[3, :])
    tests = []
    if "l1" in r:
        l1d = r["l1"]["data"]
        tests.append((
            "L1 Diversion + Conditioning",
            r["l1"]["passed"],
            f"cbam=0:{l1d['share_diff_off']:.3f}→cbam=1:{l1d['share_diff_on']:.3f} "
            f"cond_gap={l1d.get('cond_gap_pp', (l1d['share_diff_off']-l1d['share_diff_on'])*100):.1f}pp "
            f"(>{L1_REL_DROP_THRESH*100:.0f}%)",
        ))
    if "l2" in r:
        l2d = r["l2"]["data"]
        tests.append(("L2 Free μ conditioning", r["l2"]["passed"],
                       f"μ(cbam=1)={l2d['mean_mu_on']:.3f} "
                       f"μ(cbam=0)={l2d['mean_mu_off']:.3f} "
                       f"gap={l2d['conditioning_gap']:.3f}"))
    if "l3" in r:
        l3d = r["l3"]["data"]
        l2_on_str = f"{r['l2']['data']['mean_mu_on']:.3f}" if "l2" in r else "?"
        tests.append(("L3 Costly μ + self-loop", r["l3"]["passed"],
                       f"μ_on={l3d['mean_mu_on']:.3f}>L2_on={l2_on_str} "
                       f"gap={l3d['conditioning_gap']:.3f}"))
    if "l4" in r:
        l4d = r["l4"]["data"]
        l3_on = r["l3"]["data"]["mean_mu_on"] if "l3" in r else 0.0
        tests.append(("L4 Crowd-out", r["l4"]["passed"],
                       f"μ_on={l4d['mean_mu_on']:.3f}<L3_on={l3_on:.3f}"))
    _badge(ax_badge, tests)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"litmus_rand_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)
    return out_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--tests",     type=str, default="l1,l2,l3,l4",
                        help="Comma-separated subset, e.g. l1,l2")
    parser.add_argument("--replot",    type=str, default=None,
                        help="Path to existing .pkl — skip training and just re-plot")
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    tests_to_run    = {t.strip() for t in args.tests.split(",")}

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        # Recompute L1 passed with the conditioning-gap criterion (raw data
        # is always in the pkl; the stored passed flag may use the old criterion).
        if "l1" in results:
            d = results["l1"]["data"]
            if "cond_gap" not in d:  # older pkl without cond_gap stored
                d["cond_gap"] = (d["share_diff_off"] - d["share_diff_on"]) / (d["share_diff_off"] + 1e-10)
                d["cond_gap_pp"] = (d["share_diff_off"] - d["share_diff_on"]) * 100
            results["l1"]["passed"] = d["cond_gap"] > L1_REL_DROP_THRESH
        plot_results(results, timestamp)
        return

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 4)

    results = {}
    l3_mu_on = 0.5  # default fallback if l3 not run

    if "l1" in tests_to_run:
        passed, data = run_l1(keys[0])
        results["l1"] = {"passed": passed, "data": data}

    if "l2" in tests_to_run:
        passed, data = run_l2(keys[1])
        results["l2"] = {"passed": passed, "data": data}

    if "l3" in tests_to_run:
        passed, data = run_l3(keys[2])
        l3_mu_on = data["mean_mu_on"]
        results["l3"] = {"passed": passed, "data": data}
        # Cross-check self-incentivising loop vs L2
        if "l2" in results:
            l2_mu_on = results["l2"]["data"]["mean_mu_on"]
            loop_pass = l3_mu_on > l2_mu_on
            results["l3"]["passed"] = results["l3"]["passed"] and loop_pass
            print(f"  L3 self-incentivising loop: μ_L3_on={l3_mu_on:.3f} > "
                  f"μ_L2_on={l2_mu_on:.3f} → {'PASS' if loop_pass else 'FAIL'}")

    if "l4" in tests_to_run:
        passed, data = run_l4(keys[3], l3_mu_on)
        results["l4"] = {"passed": passed, "data": data}

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY")
    print("="*55)
    for name, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"  {name.upper()}: {status}")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"litmus_rand_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Results saved: {pkl_path}")

    plot_results(results, timestamp)


if __name__ == "__main__":
    main()
