"""cbam_litmus_mechanism.py

Part I of the revised CBAM litmus suite (see LITMUS_TEST_SPEC.md).

Mechanism-Identification tests using fixed-differential CBAM.
These compare *different environments or action constraints* and answer
structural questions about the CBAM-RICE model.

  M1  Diversion Exists When Mitigation Is Closed
        differential CBAM, no_mitigation=True, exports free.
        Compare against flat τ=0 control.
        Pass: EU dirty share drops >15% relative.

  M2  Mitigation Exists When Diversion Is Closed And Costless
        differential CBAM, zero_abatement_cost=True, exports pinned (delta_max=0).
        Pass: mean μ (non-EU) > 0.15.

  M3  Mitigation Survives When Diversion Is Closed And Cost Is Restored
        differential CBAM, realistic abatement cost, exports pinned.
        Pass: mean μ (non-EU) > 0.01.
        Secondary: compare descriptively to M2 (do NOT use M3 vs M2 as primary).

  M4  Diversion Crowds Out Mitigation When Both Are Open
        differential CBAM, realistic abatement cost, both channels free.
        Pass: μ_M4 < μ_M3.

  N1  Mechanical Tariff-Relief Null (fixed-action, no RL)
        Holding exports fixed, does exogenous μ_high mechanically lower τ_eff?
        Expected: τ_eff(μ_high) < τ_eff(μ_low).

Usage (from rice_jax/, rice-jax conda env):
    python cbam/drivers/cbam_litmus_mechanism.py [--timesteps 2000000]
    python cbam/drivers/cbam_litmus_mechanism.py --replot <pickle.pkl>
"""

import argparse
import os as _os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cloudpickle
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from _experiment_util import get_log_dir, get_output_dir, save_run_config
from cbam.config import metrics as _metrics
from cbam.config.canonical_config import (
    CANONICAL_TRAIN_KWARGS,
    EU_REGION_IDX as EU_IDX,
    EVAL_LAST_T,
    NON_EU_EXPORTER_IDXS,
    NON_EU_IDXS,
    NUM_EVAL_EPISODES,
    NUM_REGIONS,
    REGION_NAMES,
    canonical_env_kwargs as _canonical_env_kwargs,
    canonical_train_kwargs as _canonical_train_kwargs,
    make_canonical_env,
)
from rice_jax.training import (
    LoggingPPO,
    episode_return_curve,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax.utils import full_state_info_log_fn

matplotlib.use("Agg")


_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

# ── Config ─────────────────────────────────────────────────────────────────
# Region indexing, env defaults, and PPO kwargs are sourced from
# canonical_config. Only experiment-local knobs remain here.

NON_EU = list(NON_EU_IDXS)  # backward-compat (list)
CBAM_PLOT_REGIONS = list(NON_EU_EXPORTER_IDXS)  # excludes RoW + EU

TOTAL_TIMESTEPS = _canonical_train_kwargs()["total_timesteps"]
NUM_ENVS = CANONICAL_TRAIN_KWARGS["num_envs"]
NUM_STEPS = CANONICAL_TRAIN_KWARGS["num_steps"]
SEED = 42

CBAM_LAMBDA_INIT = _canonical_env_kwargs()["cbam_lambda_init"]
WELFARE_LOSS_WEIGHT = _canonical_env_kwargs()["welfare_loss_per_unit_tariff"]
M1_REL_DROP_THRESH = 0.15  # 15% relative drop in EU dirty share

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR = get_log_dir("training_logs")
FIXED_SAVINGS = True  # default; overridden by --free-savings

# Sensitivity-param overrides injected from the experiment runner.
# Set externally (e.g. by cbam_experiment_C_litmus.py) before calling run_m*().
# Keys must be in canonical_config.SENSITIVITY_PARAMS.
ENV_OVERRIDES: dict = {}


# ── PPO kwargs (training-only; env defaults via canonical_config) ──────────

_PPO_KWARGS = {
    k: v for k, v in CANONICAL_TRAIN_KWARGS.items() if k != "total_timesteps"
}


# ── Build helpers ───────────────────────────────────────────────────────────


def _build_env(
    cbam_mode,
    zero_abatement_cost,
    no_mitigation,
    fixed_savings,
    delta_max=3.0,
    cbam_tariff_rate=0.0,
    cbam_lambda_init=None,
    for_training=True,
):
    """Build an env via the canonical factory, layering on litmus-specific knobs."""
    overrides = dict(
        cbam_tariff_mode=cbam_mode,
        cbam_tariff_rate=cbam_tariff_rate,
        zero_abatement_cost=zero_abatement_cost,
        no_mitigation=no_mitigation,
        fixed_savings_rate=fixed_savings,
        delta_max=delta_max,
    )
    if cbam_lambda_init is not None:
        overrides["cbam_lambda_init"] = cbam_lambda_init
    # Merge sensitivity-param overrides from the experiment runner
    overrides.update(ENV_OVERRIDES)
    return make_canonical_env(for_training=for_training, **overrides)


def _config_tag():
    """Build a human-readable tag encoding the key experiment config."""
    ts = TOTAL_TIMESTEPS
    if ts >= 1_000_000:
        ts_label = f"{ts // 1_000_000}M"
    else:
        ts_label = f"{ts // 1_000}k"
    sav_label = "fixsav" if FIXED_SAVINGS else "freesav"
    return f"{ts_label}_{sav_label}"


def _make_log_fn(label, num_iters):
    _os.makedirs(LOG_DIR, exist_ok=True)
    tag = _config_tag()
    csv_path = _os.path.join(LOG_DIR, f"litmus_mech_{tag}_{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _DROP = {"actions", "action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _DROP}, iteration)

    return make_combined_log_fn(
        _compact,
        make_csv_log_fn(csv_path),
    ), csv_path


def _train(label, env, key, num_iters, total_timesteps=None):
    ts = total_timesteps or TOTAL_TIMESTEPS
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = LoggingPPO(
        total_timesteps=ts,
        log_function=log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━' * 55}")
    print(f"  Training: {label}")
    print(f"{'━' * 55}")
    t0 = time.perf_counter()
    agent, metrics = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return agent, episode_return_curve(metrics), csv_path


# ── Eval helpers ────────────────────────────────────────────────────────────


def _eval_episode(key, raw_env, agent):
    from _experiment_util import run_single_episode, with_log_info_fn

    eval_env = with_log_info_fn(raw_env, full_state_info_log_fn)

    def _to_arr(d):
        return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)

    flows, mit, util = [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 90_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        flows.append(np.array(logs["trade_flows"]))
        mit.append(_to_arr(logs["mitigation_rates_all_regions"]))
        util.append(_to_arr(logs["utility_all_regions"]))

    return {
        "trade_flows": np.stack(flows, 0),
        "mitigation": np.stack(mit, 0),
        "utility": np.stack(util, 0),
    }


def _eu_dirty_share_agg(trade_flows, last_t=EVAL_LAST_T):
    """Aggregate mean EU dirty export share across CBAM-relevant regions.

    Delegates to metrics.eu_dirty_export_share (RoW + EU excluded by default).
    """
    return _metrics.eu_dirty_export_share(
        trade_flows,
        eu_region_idx=EU_IDX,
        exporter_idxs=CBAM_PLOT_REGIONS,
        last_t=last_t,
    )


def _per_region_eu_dirty_share(trade_flows, last_t=EVAL_LAST_T):
    """Per-region EU dirty-export share (all non-EU regions, incl. RoW)."""
    return _metrics.per_region_eu_dirty_export_share(
        trade_flows,
        eu_region_idx=EU_IDX,
        last_t=last_t,
    )


def _per_region_mu(mitigation, last_t=EVAL_LAST_T):
    """Per-region mean mitigation rate (all regions)."""
    return _metrics.per_region_mitigation_rate(mitigation, last_t=last_t)


def _mean_mu_non_eu(mitigation, last_t=EVAL_LAST_T):
    """Mean mitigation rate over canonical non-EU exporter set (excludes RoW).

    NOTE: kept under its historical name; default exclusion is the canonical
    headline aggregation (RoW + EU). Pass `region_idxs=NON_EU` for the
    'include RoW' variant.
    """
    return _metrics.mean_mitigation_rate(
        mitigation,
        region_idxs=CBAM_PLOT_REGIONS,
        last_t=last_t,
    )


# ── M1: Diversion Exists When Mitigation Is Closed ─────────────────────────


def run_m1(key):
    """M1: Diversion exists when mitigation is closed.

    Claim supported if pass: diversion is a viable adaptation margin under CBAM.
    Not supported: diversion is welfare-improving; diversion dominates mitigation.
    """
    print("\n" + "=" * 55)
    print("  M1  Diversion Exists When Mitigation Is Closed")
    print("  ctrl=flat τ=0  vs  treat=differential CBAM")
    print("  no_mitigation=True, fixed_savings, exports free")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    key_ctrl, key_diff = jax.random.split(key, 2)

    # ctrl: no CBAM
    env_ctrl = _build_env("flat", False, True, FIXED_SAVINGS, cbam_tariff_rate=0.0)
    agent_ctrl, metrics_ctrl, csv_ctrl = _train("m1_ctrl", env_ctrl, key_ctrl, num_iters)

    # treat: differential CBAM
    env_diff = _build_env(
        "differential", False, True, FIXED_SAVINGS, cbam_lambda_init=CBAM_LAMBDA_INIT
    )
    agent_diff, metrics_diff, csv_diff = _train("m1_diff", env_diff, key_diff, num_iters)

    eval_key = jax.random.fold_in(key, 10)
    raw_ctrl = _build_env(
        "flat", False, True, FIXED_SAVINGS, cbam_tariff_rate=0.0, for_training=False
    )
    raw_diff = _build_env(
        "differential",
        False,
        True,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        for_training=False,
    )

    ev_ctrl = _eval_episode(eval_key, raw_ctrl, agent_ctrl)
    ev_diff = _eval_episode(eval_key, raw_diff, agent_diff)

    share_ctrl = _eu_dirty_share_agg(ev_ctrl["trade_flows"])
    share_diff = _eu_dirty_share_agg(ev_diff["trade_flows"])
    rel_drop = (share_ctrl - share_diff) / (share_ctrl + 1e-10)
    drop_pp = (share_ctrl - share_diff) * 100

    passed = rel_drop > M1_REL_DROP_THRESH
    tag = "PASS" if passed else "FAIL"
    print(
        f"\n  M1 {tag}: EU dirty share  ctrl={share_ctrl:.3f}  diff={share_diff:.3f}  "
        f"drop={drop_pp:.1f}pp ({rel_drop * 100:.1f}% rel)  (>{M1_REL_DROP_THRESH * 100:.0f}% required)"
    )

    return passed, dict(
        share_ctrl=share_ctrl,
        share_diff=share_diff,
        drop_pp=drop_pp,
        rel_drop=rel_drop,
        pr_ctrl=_per_region_eu_dirty_share(ev_ctrl["trade_flows"]),
        pr_diff=_per_region_eu_dirty_share(ev_diff["trade_flows"]),
        util_ctrl=ev_ctrl["utility"].mean(0),
        util_diff=ev_diff["utility"].mean(0),
        csv_ctrl=csv_ctrl,
        csv_diff=csv_diff,
        train_metrics={"m1_ctrl": metrics_ctrl, "m1_diff": metrics_diff},
        _agent_ctrl=agent_ctrl,
        _agent_diff=agent_diff,
    )


# ── M2: Mitigation Exists When Diversion Is Closed And Costless ────────────


def run_m2(key):
    """M2: Mitigation exists when diversion is closed and costless.

    Claim supported if pass: mitigation is a viable adaptation margin when it
    is the only escape and is not welfare-costly.
    Not supported: mitigation will remain attractive under realistic costs.
    """
    print("\n" + "=" * 55)
    print("  M2  Mitigation Exists (Costless, Diversion Closed)")
    print("  differential CBAM, zero_abatement_cost=True, delta_max=0")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(
        "differential",
        True,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
    )
    agent, train_metrics, csv_path = _train("m2_costless_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 20)
    raw_env = _build_env(
        "differential",
        True,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        for_training=False,
    )
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu = _mean_mu_non_eu(ev["mitigation"])
    pr_mu = _per_region_mu(ev["mitigation"])

    passed = mean_mu > 0.15
    tag = "PASS" if passed else "FAIL"
    print(f"\n  M2 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (>0.15 required)")

    return passed, dict(
        mean_mu=mean_mu,
        pr_mu=pr_mu,
        util_traj=ev["utility"].mean(0),
        csv_path=csv_path,
        train_metrics={"m2_costless_mu": train_metrics},
        _agent=agent,
    )


# ── M3: Mitigation Survives When Diversion Is Closed And Cost Restored ─────


def run_m3(key, m2_mu=None):
    """M3: Mitigation survives when diversion is closed and cost is restored.

    Pass criterion: mean μ (non-EU) > 0.01.
    Secondary diagnostic: compare μ_M3 to μ_M2 descriptively (do NOT use
    M3 vs M2 ordering as the primary pass condition per LITMUS_TEST_SPEC.md).

    Claim supported if pass: mitigation remains behaviorally viable under
    realistic cost when diversion is unavailable.
    Not supported: the tariff-relief mechanism alone explains the level;
    realistic-cost mitigation is stronger for structural reasons.
    """
    print("\n" + "=" * 55)
    print("  M3  Mitigation Survives (Realistic Cost, Diversion Closed)")
    print("  differential CBAM, realistic abatement cost, delta_max=0")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
    )
    agent, train_metrics, csv_path = _train("m3_costly_mu", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 30)
    raw_env = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        for_training=False,
    )
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu = _mean_mu_non_eu(ev["mitigation"])
    pr_mu = _per_region_mu(ev["mitigation"])

    passed = mean_mu > 0.01
    tag = "PASS" if passed else "FAIL"
    print(f"\n  M3 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (>0.01 required)")

    # Secondary diagnostic: M2 vs M3 gap (descriptive only)
    if m2_mu is not None:
        gap = mean_mu - m2_mu
        direction = "higher" if gap > 0 else "lower"
        print(
            f"         Secondary: μ_M3={mean_mu:.3f} vs μ_M2={m2_mu:.3f}  "
            f"(M3 is {abs(gap):.3f} {direction} — descriptive, not pass criterion)"
        )

    return passed, dict(
        mean_mu=mean_mu,
        pr_mu=pr_mu,
        util_traj=ev["utility"].mean(0),
        csv_path=csv_path,
        train_metrics={"m3_costly_mu": train_metrics},
        _agent=agent,
    )


# ── M4: Diversion Crowds Out Mitigation When Both Are Open ─────────────────


def run_m4(key, m3_mu):
    """M4: Diversion crowds out mitigation when both are open.

    This is the central mechanism result.
    Claim supported if pass: diversion crowds out mitigation when both margins
    are available.
    """
    print("\n" + "=" * 55)
    print("  M4  Diversion Crowds Out Mitigation (Both Open)")
    print("  differential CBAM, realistic abatement cost, exports+mitigation free")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    env = _build_env(
        "differential", False, False, FIXED_SAVINGS, cbam_lambda_init=CBAM_LAMBDA_INIT
    )
    agent, train_metrics, csv_path = _train("m4_both", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 40)
    raw_env = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        for_training=False,
    )
    ev = _eval_episode(eval_key, raw_env, agent)

    mean_mu = _mean_mu_non_eu(ev["mitigation"])
    pr_mu = _per_region_mu(ev["mitigation"])
    pr_share = _per_region_eu_dirty_share(ev["trade_flows"])

    passed = mean_mu < m3_mu
    tag = "PASS" if passed else "FAIL"
    print(
        f"\n  M4 {tag}: mean μ (non-EU) = {mean_mu:.3f}  (< M3 μ={m3_mu:.3f} required)"
    )

    return passed, dict(
        mean_mu=mean_mu,
        pr_mu=pr_mu,
        pr_share=pr_share,
        util_traj=ev["utility"].mean(0),
        csv_path=csv_path,
        train_metrics={"m4_both": train_metrics},
        _agent=agent,
    )


# ── N1: Mechanical Tariff-Relief Null (no RL) ──────────────────────────────


def run_n1(key):
    """N1: Mechanical tariff-relief null test.

    Fixed-action rollout (no RL). Exports pinned. Compare two exogenous
    mitigation vectors: μ_high > μ_low for a non-EU exporter.
    Expected: τ_eff(μ_high) < τ_eff(μ_low) and cbam_cost(μ_high) < cbam_cost(μ_low).

    This isolates the tariff-relief mechanism from RL optimization / reward shaping.
    """
    import optax

    from _experiment_util import FixedActionAgent, run_single_episode, with_log_info_fn

    print("\n" + "=" * 55)
    print("  N1  Mechanical Tariff-Relief Null (no RL)")
    print("  fixed-action rollout, exports pinned, compare μ_low vs μ_high")
    print("=" * 55)

    raw_env = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        for_training=False,
    )

    eval_env = with_log_info_fn(raw_env, full_state_info_log_fn)

    # Build two FixedActionAgents with different mitigation levels
    from rice_jax.utils import i_to_agent_str

    def _make_fixed_agent(mu_non_eu):
        """Create a FixedActionAgent with specified non-EU mitigation rate."""
        agent = FixedActionAgent(eval_env)
        # Override default actions with specific mitigation rates
        new_actions = {}
        for i in range(NUM_REGIONS):
            action_i = optax.tree.zeros_like(agent.default_actions[i_to_agent_str(i)])
            if i == EU_IDX:
                action_i["mitigation_rate"] = 5.0  # high EU mitigation (saturates)
            else:
                action_i["mitigation_rate"] = mu_non_eu
            action_i["savings_rate"] = 2.5
            new_actions[i_to_agent_str(i)] = action_i
        agent.default_actions = new_actions
        return agent

    agent_low = _make_fixed_agent(0.0)  # non-EU mitigation = 0
    agent_high = _make_fixed_agent(5.0)  # non-EU mitigation = high (saturates)

    logs_low = run_single_episode(key, eval_env, agent_low)
    logs_high = run_single_episode(key, eval_env, agent_high)

    # Extract cbam_cost_all_regions at step 1 (after initial dynamics settle).
    # full_state_info_log_fn stores it as dict {region_idx: array(T,)}.
    step_idx = 1
    cost_dict_low = logs_low.get("cbam_cost_all_regions", None)
    cost_dict_high = logs_high.get("cbam_cost_all_regions", None)

    if cost_dict_low is None or cost_dict_high is None:
        print("  N1 SKIP: cbam_cost_all_regions not found in info logs")
        return None, dict(skipped=True, reason="cbam_cost_all_regions not in info")

    # Sum CBAM cost across non-EU regions at the test step
    cost_low_total = float(sum(np.array(cost_dict_low[r])[step_idx] for r in NON_EU))
    cost_high_total = float(sum(np.array(cost_dict_high[r])[step_idx] for r in NON_EU))

    passed = cost_high_total < cost_low_total
    tag = "PASS" if passed else "FAIL"
    print(f"\n  N1 {tag}: CBAM cost (step {step_idx})")
    print(f"         μ_low  → cost={cost_low_total:.4f}")
    print(f"         μ_high → cost={cost_high_total:.4f}")
    print(
        f"         Higher μ {'reduces' if passed else 'does NOT reduce'} CBAM cost mechanically"
    )

    return passed, dict(
        cost_low=cost_low_total,
        cost_high=cost_high_total,
    )


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
        mean = df["ep_return_mean"].rolling(5, min_periods=1).mean()
        std = df.get("ep_return_std", pd.Series(np.zeros(len(df))))
        ax.plot(iters, mean, lw=1.5, color=color, label=label)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.15, color=color)
    ax.set_xlabel("iteration", fontsize=8)
    ax.set_ylabel("ep return", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, lw=0.4, alpha=0.4)


def _bar_per_region(ax, data_dict, title, ylabel, colors=None):
    """data_dict: {label: {region_idx: float}}"""
    n_groups = len(CBAM_PLOT_REGIONS)
    n_bars = len(data_dict)
    width = 0.8 / n_bars
    x = np.arange(n_groups)
    for i, (label, rdict) in enumerate(data_dict.items()):
        vals = [rdict.get(r, 0.0) for r in CBAM_PLOT_REGIONS]
        color = colors[i] if colors else f"C{i}"
        ax.bar(
            x + (i - n_bars / 2 + 0.5) * width,
            vals,
            width=width,
            color=color,
            alpha=0.85,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS],
        rotation=25,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)


def _badge(ax, tests):
    """tests: list of (label, passed, summary_str)"""
    ax.axis("off")
    lines = [
        "Mechanism-Identification Litmus (Fixed Differential CBAM)",
        "─" * 56,
    ]
    for label, passed, summary in tests:
        if passed is None:
            sym = "⏭️"
        elif passed:
            sym = "✅"
        else:
            sym = "❌"
        lines.append(f"  {sym}  {label}: {summary}")
    lines.append("")
    lines.append("─" * 56)
    lines.append("Claim: both diversion and mitigation are viable responses to CBAM,")
    lines.append("but when both margins are open, exporters shift toward diversion")
    lines.append("and away from costly mitigation.")
    overall = all(p for _, p, _ in tests if p is not None)
    color = "#1a6b3a" if overall else "#8b1a1a"
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=9,
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor=color,
            alpha=0.12,
            edgecolor=color,
            linewidth=2,
        ),
    )


def plot_results(results, timestamp):
    r = results
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        f"Mechanism-Identification Litmus Suite (M1–M4, N1)\n"
        f"9-Region Differential CBAM — {TOTAL_TIMESTEPS // 1_000_000}M steps, "
        f"{NUM_ENVS} envs, seed={SEED}",
        fontsize=13,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(5, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: M1 ────────────────────────────────────────────────────────────
    ax_m1_conv = fig.add_subplot(gs[0, 0:2])
    ax_m1_bar = fig.add_subplot(gs[0, 2:4])

    if "m1" in r:
        m1 = r["m1"]["data"]
        _plot_convergence(
            ax_m1_conv,
            [m1["csv_ctrl"], m1["csv_diff"]],
            ["ctrl (τ=0)", "differential CBAM"],
            ["#2980b9", "#e74c3c"],
            "M1: Training convergence",
        )
        _bar_per_region(
            ax_m1_bar,
            {"ctrl (τ=0)": m1["pr_ctrl"], "differential CBAM": m1["pr_diff"]},
            "M1: EU dirty export share per region\n"
            "(diversion exists when mitigation is closed)",
            "EU dirty export share",
            colors=["#2980b9", "#e74c3c"],
        )

    # ── Row 1: M2 + M3 convergence + bar ─────────────────────────────────────
    ax_m2_conv = fig.add_subplot(gs[1, 0])
    ax_m3_conv = fig.add_subplot(gs[1, 1])
    ax_m23_bar = fig.add_subplot(gs[1, 2:4])

    if "m2" in r:
        m2 = r["m2"]["data"]
        _plot_convergence(
            ax_m2_conv,
            [m2["csv_path"]],
            ["M2 costless abatement"],
            ["#27ae60"],
            "M2: Training convergence",
        )
    if "m3" in r:
        m3 = r["m3"]["data"]
        _plot_convergence(
            ax_m3_conv,
            [m3["csv_path"]],
            ["M3 costly abatement"],
            ["#f39c12"],
            "M3: Training convergence",
        )
    if "m2" in r and "m3" in r:
        _bar_per_region(
            ax_m23_bar,
            {
                "M2 costless (diversion closed)": r["m2"]["data"]["pr_mu"],
                "M3 costly (diversion closed)": r["m3"]["data"]["pr_mu"],
            },
            "M2 vs M3: Mitigation rate per region (descriptive)\n"
            "(M2/M3 differ in cost AND learning landscape — comparison is informative, not clean ID)",
            "Mean mitigation rate μ",
            colors=["#27ae60", "#f39c12"],
        )

    # ── Row 2: M4 ────────────────────────────────────────────────────────────
    ax_m4_conv = fig.add_subplot(gs[2, 0])
    ax_m4_mu = fig.add_subplot(gs[2, 1:3])
    ax_m4_share = fig.add_subplot(gs[2, 3])

    if "m4" in r:
        m4 = r["m4"]["data"]
        _plot_convergence(
            ax_m4_conv,
            [m4["csv_path"]],
            ["M4 both channels"],
            ["#8e44ad"],
            "M4: Training convergence",
        )

    if "m3" in r and "m4" in r:
        _bar_per_region(
            ax_m4_mu,
            {
                "M3 mitigation-only": r["m3"]["data"]["pr_mu"],
                "M4 both channels": r["m4"]["data"]["pr_mu"],
            },
            "M3 vs M4: Diversion crowds out mitigation\n(central mechanism result)",
            "Mean mitigation rate μ",
            colors=["#f39c12", "#8e44ad"],
        )

    if "m4" in r:
        _bar_per_region(
            ax_m4_share,
            {"M4 (both channels)": r["m4"]["data"]["pr_share"]},
            "M4: EU dirty share\n(with both channels)",
            "EU dirty export share",
            colors=["#8e44ad"],
        )

    # ── Row 3: N1 mechanical null ─────────────────────────────────────────────
    ax_n1 = fig.add_subplot(gs[3, 0:2])
    if "n1" in r and not r["n1"]["data"].get("skipped", False):
        n1 = r["n1"]["data"]
        bars = [n1["cost_low"], n1["cost_high"]]
        ax_n1.bar([0, 1], bars, color=["#e74c3c", "#27ae60"], alpha=0.8, width=0.5)
        ax_n1.set_xticks([0, 1])
        ax_n1.set_xticklabels(
            ["μ_low (no mitigation)", "μ_high (saturated)"], fontsize=8
        )
        ax_n1.set_ylabel("CBAM cost (step 1)", fontsize=8)
        ax_n1.set_title(
            "N1: Mechanical Tariff-Relief Null\n"
            "(higher μ → lower τ_eff → lower CBAM cost, no RL)",
            fontsize=9,
        )
        ax_n1.grid(True, axis="y", lw=0.4, alpha=0.4)
    else:
        ax_n1.axis("off")
        ax_n1.text(
            0.5,
            0.5,
            "N1 skipped or not run",
            ha="center",
            va="center",
            transform=ax_n1.transAxes,
            fontsize=10,
            color="grey",
        )

    # ── Row 3-4: badge ───────────────────────────────────────────────────────
    ax_badge = fig.add_subplot(gs[3:5, 2:4])
    tests = []
    if "m1" in r:
        m1d = r["m1"]["data"]
        tests.append(
            (
                "M1 Diversion exists",
                r["m1"]["passed"],
                f"ctrl={m1d['share_ctrl']:.3f} → diff={m1d['share_diff']:.3f}  "
                f"drop={m1d['drop_pp']:.1f}pp ({m1d['rel_drop'] * 100:.1f}% rel)",
            )
        )
    if "m2" in r:
        tests.append(
            (
                "M2 Costless μ exists",
                r["m2"]["passed"],
                f"mean μ={r['m2']['data']['mean_mu']:.3f} (>0.15)",
            )
        )
    if "m3" in r:
        m3d = r["m3"]["data"]
        m2_str = f"{r['m2']['data']['mean_mu']:.3f}" if "m2" in r else "?"
        tests.append(
            (
                "M3 Costly μ survives",
                r["m3"]["passed"],
                f"mean μ={m3d['mean_mu']:.3f} (>0.01); cf. M2={m2_str} (descriptive)",
            )
        )
    if "m4" in r:
        m3_ref = r["m3"]["data"]["mean_mu"] if "m3" in r else "?"
        tests.append(
            (
                "M4 Diversion crowds out μ",
                r["m4"]["passed"],
                f"mean μ={r['m4']['data']['mean_mu']:.3f} (< M3={m3_ref})",
            )
        )
    if "n1" in r:
        n1d = r["n1"]["data"]
        if n1d.get("skipped"):
            tests.append(("N1 Tariff-relief null", None, "skipped"))
        else:
            tests.append(
                (
                    "N1 Tariff-relief null",
                    r["n1"]["passed"],
                    f"cost(μ_low)={n1d['cost_low']:.4f} → "
                    f"cost(μ_high)={n1d['cost_high']:.4f}",
                )
            )
    _badge(ax_badge, tests)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"litmus_mech_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    global TOTAL_TIMESTEPS, FIXED_SAVINGS
    parser = argparse.ArgumentParser(
        description="Mechanism-Identification Litmus Suite (LITMUS_TEST_SPEC.md Part I)"
    )
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--tests",
        type=str,
        default="m1,m2,m3,m4,n1",
        help="Comma-separated subset, e.g. m1,m2",
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Path to existing .pkl — skip training, regenerate plot only",
    )
    sav_group = parser.add_mutually_exclusive_group()
    sav_group.add_argument(
        "--fixed-savings",
        dest="fixed_savings",
        action="store_true",
        default=True,
        help="Fix savings_rate to 0.2 (default)",
    )
    sav_group.add_argument(
        "--free-savings",
        dest="fixed_savings",
        action="store_false",
        help="Let savings_rate be a free RL action",
    )
    parser.add_argument(
        "--save-agents",
        action="store_true",
        default=False,
        help="Include trained PPO agents in pkl (larger files, needed for Layer 2 introspection)",
    )
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    FIXED_SAVINGS = args.fixed_savings
    tag = _config_tag()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_tag = f"{tag}_{timestamp}"

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        plot_results(results, file_tag)
        return

    tests_to_run = {t.strip().lower() for t in args.tests.split(",")}

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 10)
    results = {}

    if "m1" in tests_to_run:
        passed, data = run_m1(keys[0])
        results["m1"] = {"passed": passed, "data": data}

    if "m2" in tests_to_run:
        passed, data = run_m2(keys[1])
        results["m2"] = {"passed": passed, "data": data}

    m3_mu = 0.05  # fallback
    if "m3" in tests_to_run:
        m2_mu = results["m2"]["data"]["mean_mu"] if "m2" in results else None
        passed, data = run_m3(keys[2], m2_mu=m2_mu)
        results["m3"] = {"passed": passed, "data": data}
        m3_mu = data["mean_mu"]

    if "m4" in tests_to_run:
        passed, data = run_m4(keys[3], m3_mu)
        results["m4"] = {"passed": passed, "data": data}

    if "n1" in tests_to_run:
        passed, data = run_n1(keys[4])
        results["n1"] = {"passed": passed, "data": data}

    # Strip agents from pkl if not requested (they're large)
    if not args.save_agents:
        for test_id, res in results.items():
            d = res.get("data", {})
            for k in list(d.keys()):
                if k.startswith("_agent"):
                    del d[k]

    # Save pickle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"litmus_mech_{file_tag}.pkl")
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(results, f)
    print(f"\n  Results saved: {pkl_path}")
    if args.save_agents:
        print("  (includes trained agents for Layer 2 introspection)")

    # Persist frozen config into experiment dir (no-op when run standalone)
    save_run_config(
        {
            "script": "cbam/drivers/cbam_litmus_mechanism.py",
            "timesteps": TOTAL_TIMESTEPS,
            "fixed_savings": FIXED_SAVINGS,
            "seed": args.seed,
            "tests": args.tests,
            "save_agents": args.save_agents,
            "pkl_path": pkl_path,
        }
    )

    # Plot
    plot_results(results, file_tag)

    # Summary
    print("\n" + "=" * 55)
    print("  MECHANISM-IDENTIFICATION LITMUS SUMMARY")
    print("=" * 55)
    print("  (see LITMUS_TEST_SPEC.md for interpretation guidelines)")
    print()
    for test, res in results.items():
        if res["passed"] is None:
            tag = "SKIP ⏭️"
        elif res["passed"]:
            tag = "PASS ✅"
        else:
            tag = "FAIL ❌"
        print(f"  {test.upper()}  {tag}")
    print()
    print("  One-sentence summary:")
    print("  The fixed-differential litmus suite shows that both diversion and")
    print("  mitigation are viable responses to CBAM, but when both margins are")
    print("  open, exporters shift toward diversion and away from costly mitigation.")


if __name__ == "__main__":
    main()
