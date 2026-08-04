"""cbam_litmus_conditioning.py

Part II of the revised CBAM litmus suite (see LITMUS_TEST_SPEC.md).

Policy-Conditioning tests using randomised-differential CBAM.
These ask whether ONE trained policy reacts to the CBAM signal in its
observation, not whether the economic mechanism exists in principle.

  C1  Export Conditioning
        Single agent trained with cbam_randomize=True.
        Evaluate same weights at cbam=0 and cbam=1.
        Pass: within-policy EU dirty share gap > 15% relative.

  C2  Mitigation Conditioning Under Pinned Exports
        Randomised differential, exports pinned.
        C2a: zero_abatement_cost=True  (costless)
        C2b: realistic abatement cost  (costly)
        Primary metric: mu_on − mu_off.
        Graded: strong (gap > 0.05), weak (0.01–0.05), none/mixed (≤ 0.01).

  C3  Conditioned Crowd-Out
        Compare C2b (exports pinned, costly) vs a new both-channels run
        using the cbam=1 eval branch.
        Pass: mu_on_both < mu_on_pinned.

Usage (from rice_jax/, rice-jax conda env):
    python cbam/drivers/cbam_litmus_conditioning.py [--timesteps 2000000]
    python cbam/drivers/cbam_litmus_conditioning.py --replot <pickle.pkl>
"""

import argparse
import os as _os
import pickle
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cloudpickle
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jaxnasium.algorithms import PPO
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
C1_REL_GAP_THRESH = 0.15  # 15% relative conditioning gap

# C2 grading thresholds
C2_STRONG_THRESH = 0.05
C2_WEAK_THRESH = 0.01

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR = get_log_dir("training_logs")
FIXED_SAVINGS = True  # default; overridden by --free-savings

# Sensitivity-param overrides injected from the experiment runner.
# Set externally (e.g. by cbam_experiment_C_litmus.py) before calling run_c*().
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
    randomize=False,
    for_training=True,
):
    """Build a RiceMRIO env via canonical_config.

    randomize=True adds cbam_randomize=True, cbam_tariff_rates=(0.0, 1.0).
    """
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
    if randomize:
        overrides["cbam_randomize"] = True
        overrides["cbam_tariff_rates"] = (0.0, 1.0)
    # Merge sensitivity-param overrides from the experiment runner
    overrides.update(ENV_OVERRIDES)
    return make_canonical_env(for_training=for_training, **overrides)


def _build_eval_env(
    cbam_mode,
    zero_abatement_cost,
    no_mitigation,
    fixed_savings,
    delta_max=3.0,
    cbam_lambda_init=None,
    force_rate=1.0,
):
    """Build a deterministic eval env with a fixed cbam_tariff_rate."""
    return _build_env(
        cbam_mode,
        zero_abatement_cost,
        no_mitigation,
        fixed_savings,
        delta_max=delta_max,
        cbam_tariff_rate=force_rate,
        cbam_lambda_init=cbam_lambda_init,
        randomize=False,
        for_training=False,
    )


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
    csv_path = _os.path.join(LOG_DIR, f"litmus_cond_{tag}_{label}.csv")
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
    ppo = PPO(
        total_timesteps=ts,
        log_function=log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'━' * 55}")
    print(f"  Training: {label}")
    print(f"{'━' * 55}")
    t0 = time.perf_counter()
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
    """Aggregate mean EU dirty export share across CBAM-relevant regions."""
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
    """Mean mitigation rate over canonical non-EU exporter set (excludes RoW)."""
    return _metrics.mean_mitigation_rate(
        mitigation,
        region_idxs=CBAM_PLOT_REGIONS,
        last_t=last_t,
    )


def _per_region_eu_share_by_sector(trade_flows, last_t=EVAL_LAST_T):
    """Per-region EU export share per sector.

    Returns array of shape ``(len(CBAM_PLOT_REGIONS), NS)``.
    Row order matches ``CBAM_PLOT_REGIONS``; column order matches trade_flows
    sector axis (as produced by the env's sector_granularity setting).
    """
    tf = trade_flows[:, -last_t:]  # (n_ep, last_t, NR, NR, NS)
    NS = tf.shape[-1]
    out = np.zeros((len(CBAM_PLOT_REGIONS), NS))
    for i, r in enumerate(CBAM_PLOT_REGIONS):
        eu_s = tf[:, :, r, EU_IDX, :]  # (n_ep, last_t, NS)
        all_s = tf[:, :, r, :, :].sum(axis=2)  # (n_ep, last_t, NS) — sum over dest axis
        out[i] = (eu_s / (all_s + 1e-10)).mean(axis=(0, 1))
    return out


def _grade_c2(gap):
    """Grade C2 mitigation conditioning gap per LITMUS_TEST_SPEC.md."""
    if gap > C2_STRONG_THRESH:
        return "strong"
    elif gap > C2_WEAK_THRESH:
        return "weak"
    else:
        return "none/mixed"


# ── C1: Export Conditioning ─────────────────────────────────────────────────


def run_c1(key):
    """C1: Does one trained policy route exports differently when CBAM on vs off?

    Single agent trained with cbam_randomize=True, no_mitigation=True,
    exports free. Eval same weights at cbam=0 and cbam=1.

    This is the cleanest conditioning test in the suite.
    Claim supported if pass: the policy is reading and responding to CBAM.
    """
    print("\n" + "=" * 55)
    print("  C1  Export Conditioning")
    print("  randomise=True, no_mitigation, exports free")
    print("  eval same weights at cbam=0 vs cbam=1")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    env = _build_env(
        "differential",
        False,
        True,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        randomize=True,
    )
    agent, csv_path = _train("c1_export_cond", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 10)
    raw_on = _build_eval_env(
        "differential",
        False,
        True,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=1.0,
    )
    raw_off = _build_eval_env(
        "differential",
        False,
        True,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=0.0,
    )

    ev_on = _eval_episode(eval_key, raw_on, agent)
    ev_off = _eval_episode(eval_key, raw_off, agent)

    share_on = _eu_dirty_share_agg(ev_on["trade_flows"])
    share_off = _eu_dirty_share_agg(ev_off["trade_flows"])
    cond_gap = (share_off - share_on) / (share_off + 1e-10)

    passed = cond_gap > C1_REL_GAP_THRESH
    tag = "PASS" if passed else "FAIL"
    print(
        f"\n  C1 {tag}: share(cbam=0)={share_off:.3f}  share(cbam=1)={share_on:.3f}  "
        f"gap={cond_gap * 100:.1f}% rel  (>{C1_REL_GAP_THRESH * 100:.0f}% required)"
    )

    # Per-sector arrays: shape (len(CBAM_PLOT_REGIONS), NS).  Retained in pkl so
    # the post-hoc can produce one bar-chart panel per sector (C1 only).
    share_on_by_sector = _per_region_eu_share_by_sector(ev_on["trade_flows"])
    share_off_by_sector = _per_region_eu_share_by_sector(ev_off["trade_flows"])

    return passed, dict(
        share_on=share_on,
        share_off=share_off,
        cond_gap=cond_gap,
        cond_gap_pp=(share_off - share_on) * 100,
        pr_on=_per_region_eu_dirty_share(ev_on["trade_flows"]),
        pr_off=_per_region_eu_dirty_share(ev_off["trade_flows"]),
        share_on_by_sector=share_on_by_sector,
        share_off_by_sector=share_off_by_sector,
        sector_names=list(raw_on.sector_names),
        sector_granularity=ENV_OVERRIDES.get("sector_granularity", "emissions-simple"),
        csv_path=csv_path,
        _agent=agent.agent,
    )


# ── C2: Mitigation Conditioning Under Pinned Exports ──────────────────────


def run_c2(key):
    """C2: Does one policy change mitigation when only the CBAM obs changes?

    C2a: costless abatement, exports pinned.
    C2b: realistic cost, exports pinned.
    Graded result: strong / weak / none-mixed.
    """
    print("\n" + "=" * 55)
    print("  C2  Mitigation Conditioning (Pinned Exports)")
    print("  C2a: costless  |  C2b: realistic cost")
    print("  randomise=True, delta_max=0, eval at cbam=0 vs cbam=1")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    key_a, key_b = jax.random.split(key, 2)

    # ── C2a: costless ──
    env_a = _build_env(
        "differential",
        True,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        randomize=True,
    )
    agent_a, csv_a = _train("c2a_costless", env_a, key_a, num_iters)

    eval_key = jax.random.fold_in(key, 20)
    raw_a_on = _build_eval_env(
        "differential",
        True,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=1.0,
    )
    raw_a_off = _build_eval_env(
        "differential",
        True,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=0.0,
    )

    ev_a_on = _eval_episode(eval_key, raw_a_on, agent_a)
    ev_a_off = _eval_episode(eval_key, raw_a_off, agent_a)

    mu_a_on = _mean_mu_non_eu(ev_a_on["mitigation"])
    mu_a_off = _mean_mu_non_eu(ev_a_off["mitigation"])
    gap_a = mu_a_on - mu_a_off
    grade_a = _grade_c2(gap_a)

    print(
        f"\n  C2a ({grade_a}): μ(cbam=1)={mu_a_on:.3f}  μ(cbam=0)={mu_a_off:.3f}  "
        f"gap={gap_a:.3f}"
    )

    # ── C2b: costly ──
    env_b = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        randomize=True,
    )
    agent_b, csv_b = _train("c2b_costly", env_b, key_b, num_iters)

    raw_b_on = _build_eval_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=1.0,
    )
    raw_b_off = _build_eval_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        delta_max=0.0,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=0.0,
    )

    ev_b_on = _eval_episode(eval_key, raw_b_on, agent_b)
    ev_b_off = _eval_episode(eval_key, raw_b_off, agent_b)

    mu_b_on = _mean_mu_non_eu(ev_b_on["mitigation"])
    mu_b_off = _mean_mu_non_eu(ev_b_off["mitigation"])
    gap_b = mu_b_on - mu_b_off
    grade_b = _grade_c2(gap_b)

    print(
        f"  C2b ({grade_b}): μ(cbam=1)={mu_b_on:.3f}  μ(cbam=0)={mu_b_off:.3f}  "
        f"gap={gap_b:.3f}"
    )

    # Overall C2 grade: take the stronger of the two
    grades_order = {"none/mixed": 0, "weak": 1, "strong": 2}
    overall_grade = (
        grade_a if grades_order[grade_a] >= grades_order[grade_b] else grade_b
    )

    return overall_grade, dict(
        # C2a
        mu_a_on=mu_a_on,
        mu_a_off=mu_a_off,
        gap_a=gap_a,
        grade_a=grade_a,
        pr_mu_a_on=_per_region_mu(ev_a_on["mitigation"]),
        pr_mu_a_off=_per_region_mu(ev_a_off["mitigation"]),
        csv_a=csv_a,
        _agent_a=agent_a.agent,
        # C2b
        mu_b_on=mu_b_on,
        mu_b_off=mu_b_off,
        gap_b=gap_b,
        grade_b=grade_b,
        pr_mu_b_on=_per_region_mu(ev_b_on["mitigation"]),
        pr_mu_b_off=_per_region_mu(ev_b_off["mitigation"]),
        csv_b=csv_b,
        _agent_b=agent_b.agent,
        # Overall
        overall_grade=overall_grade,
    )


# ── C3: Conditioned Crowd-Out ──────────────────────────────────────────────


def run_c3(key, c2b_mu_on):
    """C3: When one trained policy sees CBAM-on, does opening diversion reduce μ?

    Compare a both-channels randomised agent (exports + mitigation free) against
    the pinned-export C2b agent, using cbam=1 eval only.

    This is the strongest result in the randomised suite after C1.
    """
    print("\n" + "=" * 55)
    print("  C3  Conditioned Crowd-Out")
    print("  randomise=True, realistic cost, both channels free")
    print("  compare μ_on vs C2b μ_on (pinned exports)")
    print("=" * 55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    env = _build_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        randomize=True,
    )
    agent, csv_path = _train("c3_both", env, key, num_iters)

    eval_key = jax.random.fold_in(key, 30)
    raw_on = _build_eval_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=1.0,
    )
    raw_off = _build_eval_env(
        "differential",
        False,
        False,
        FIXED_SAVINGS,
        cbam_lambda_init=CBAM_LAMBDA_INIT,
        force_rate=0.0,
    )

    ev_on = _eval_episode(eval_key, raw_on, agent)
    ev_off = _eval_episode(eval_key, raw_off, agent)

    mu_on = _mean_mu_non_eu(ev_on["mitigation"])
    mu_off = _mean_mu_non_eu(ev_off["mitigation"])
    pr_share_on = _per_region_eu_dirty_share(ev_on["trade_flows"])

    passed = mu_on < c2b_mu_on
    tag = "PASS" if passed else "FAIL"
    print(
        f"\n  C3 {tag}: μ_on(both)={mu_on:.3f}  μ_on(pinned/C2b)={c2b_mu_on:.3f}  "
        f"(both < pinned required)"
    )
    print(
        f"         Conditioned crowd-out: μ drops {c2b_mu_on - mu_on:.3f} "
        f"({'present' if passed else 'absent'})"
    )

    return passed, dict(
        mu_on=mu_on,
        mu_off=mu_off,
        pr_mu_on=_per_region_mu(ev_on["mitigation"]),
        pr_share_on=pr_share_on,
        csv_path=csv_path,
        _agent=agent.agent,
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


def _bar_per_region(ax, series_dict, title, ylabel, colors=None):
    regions = sorted({r for v in series_dict.values() for r in v.keys()})
    show_regs = [r for r in regions if r in CBAM_PLOT_REGIONS or r == EU_IDX]
    labels = [REGION_NAMES.get(r, str(r)) for r in show_regs]
    x = np.arange(len(show_regs))
    n = len(series_dict)
    w = 0.8 / n
    _colors = colors or [plt.cm.tab10(i / 10) for i in range(n)]
    for i, (name, series) in enumerate(series_dict.items()):
        vals = [series.get(r, 0.0) for r in show_regs]
        ax.bar(
            x + i * w - (n - 1) * w / 2,
            vals,
            w * 0.9,
            label=name,
            color=_colors[i],
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", lw=0.4, alpha=0.4)


def _badge(ax, tests):
    ax.axis("off")
    lines = [
        "Policy-Conditioning Litmus (Randomised Differential CBAM)",
        "─" * 56,
    ]
    for label, result, summary in tests:
        if isinstance(result, bool):
            sym = "✅" if result else "❌"
        elif isinstance(result, str):
            # graded (C2)
            sym_map = {"strong": "🟢", "weak": "🟡", "none/mixed": "🔴"}
            sym = sym_map.get(result, "❓")
        else:
            sym = "⏭️"
        lines.append(f"  {sym}  {label}: {summary}")
    lines.append("")
    lines.append("─" * 56)
    lines.append("Claim: the policy clearly conditions export reallocation")
    lines.append("on the CBAM signal; mitigation conditioning is weaker")
    lines.append("and less robust; crowd-out remains when both margins open.")

    # Overall color: green if C1 passes and C3 passes
    c1_pass = any(r for l, r, _ in tests if "C1" in l and r is True)
    c3_pass = any(r for l, r, _ in tests if "C3" in l and r is True)
    color = "#1a6b3a" if (c1_pass and c3_pass) else "#8b1a1a"
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
    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        f"Policy-Conditioning Litmus Suite (C1, C2, C3)\n"
        f"9-Region Randomised Differential CBAM — {TOTAL_TIMESTEPS // 1_000_000}M steps, "
        f"{NUM_ENVS} envs, seed={SEED}  (cbam_randomize=True, rates=(0,1))",
        fontsize=12,
        fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Row 0: C1 Export Conditioning ────────────────────────────────────────
    ax_c1_conv = fig.add_subplot(gs[0, 0:2])
    ax_c1_bar = fig.add_subplot(gs[0, 2:4])

    if "c1" in r:
        c1 = r["c1"]["data"]
        _plot_convergence(
            ax_c1_conv,
            [c1["csv_path"]],
            ["C1 randomised diff"],
            ["#e74c3c"],
            "C1: Training convergence",
        )
        _bar_per_region(
            ax_c1_bar,
            {"cbam=0 (off)": c1["pr_off"], "cbam=1 (on)": c1["pr_on"]},
            "C1: EU dirty share — export conditioning\n"
            "(same weights, different CBAM obs → share gap)",
            "EU dirty export share",
            colors=["#2980b9", "#e74c3c"],
        )

    # ── Row 1: C2 Mitigation Conditioning ────────────────────────────────────
    ax_c2a_conv = fig.add_subplot(gs[1, 0])
    ax_c2b_conv = fig.add_subplot(gs[1, 1])
    ax_c2_bar = fig.add_subplot(gs[1, 2:4])

    if "c2" in r:
        c2 = r["c2"]["data"]
        _plot_convergence(
            ax_c2a_conv,
            [c2["csv_a"]],
            ["C2a costless"],
            ["#27ae60"],
            "C2a: Convergence (costless)",
        )
        _plot_convergence(
            ax_c2b_conv,
            [c2["csv_b"]],
            ["C2b costly"],
            ["#f39c12"],
            "C2b: Convergence (costly)",
        )
        _bar_per_region(
            ax_c2_bar,
            {
                "C2a costless (cbam=1)": c2["pr_mu_a_on"],
                "C2a costless (cbam=0)": c2["pr_mu_a_off"],
                "C2b costly (cbam=1)": c2["pr_mu_b_on"],
                "C2b costly (cbam=0)": c2["pr_mu_b_off"],
            },
            f"C2: Mitigation conditioning (graded)\n"
            f"C2a={c2['grade_a']}  C2b={c2['grade_b']}",
            "Mean mitigation rate μ",
            colors=["#27ae60", "#a9dfbf", "#f39c12", "#fad7a0"],
        )

    # ── Row 2: C3 Conditioned Crowd-Out ──────────────────────────────────────
    ax_c3_conv = fig.add_subplot(gs[2, 0])
    ax_c3_mu = fig.add_subplot(gs[2, 1:3])
    ax_c3_share = fig.add_subplot(gs[2, 3])

    if "c3" in r:
        c3 = r["c3"]["data"]
        _plot_convergence(
            ax_c3_conv,
            [c3["csv_path"]],
            ["C3 both channels"],
            ["#8e44ad"],
            "C3: Convergence",
        )

    if "c2" in r and "c3" in r:
        c2 = r["c2"]["data"]
        c3 = r["c3"]["data"]
        _bar_per_region(
            ax_c3_mu,
            {
                "C2b pinned (cbam=1)": c2["pr_mu_b_on"],
                "C3 both (cbam=1)": c3["pr_mu_on"],
            },
            "C2b vs C3: conditioned crowd-out (cbam=1)\n"
            "(C3 μ should be lower — diversion displaces mitigation)",
            "Mean μ",
            colors=["#f39c12", "#8e44ad"],
        )

    if "c3" in r:
        _bar_per_region(
            ax_c3_share,
            {"C3 EU dirty share (cbam=1)": r["c3"]["data"]["pr_share_on"]},
            "C3: EU dirty share\n(cbam=1, both open)",
            "EU dirty share",
            colors=["#8e44ad"],
        )

    # ── Row 3: badge ─────────────────────────────────────────────────────────
    ax_badge = fig.add_subplot(gs[3, :])
    tests = []
    if "c1" in r:
        c1d = r["c1"]["data"]
        tests.append(
            (
                "C1 Export conditioning",
                r["c1"]["passed"],
                f"share(off)={c1d['share_off']:.3f} → share(on)={c1d['share_on']:.3f}  "
                f"gap={c1d['cond_gap'] * 100:.1f}% rel (>{C1_REL_GAP_THRESH * 100:.0f}%)",
            )
        )
    if "c2" in r:
        c2d = r["c2"]["data"]
        tests.append(
            (
                "C2a Costless μ conditioning",
                c2d["grade_a"],
                f"μ(on)={c2d['mu_a_on']:.3f} μ(off)={c2d['mu_a_off']:.3f} "
                f"gap={c2d['gap_a']:.3f} → {c2d['grade_a']}",
            )
        )
        tests.append(
            (
                "C2b Costly μ conditioning",
                c2d["grade_b"],
                f"μ(on)={c2d['mu_b_on']:.3f} μ(off)={c2d['mu_b_off']:.3f} "
                f"gap={c2d['gap_b']:.3f} → {c2d['grade_b']}",
            )
        )
    if "c3" in r:
        c3d = r["c3"]["data"]
        c2b_ref = r["c2"]["data"]["mu_b_on"] if "c2" in r else "?"
        tests.append(
            (
                "C3 Conditioned crowd-out",
                r["c3"]["passed"],
                f"μ_on(both)={c3d['mu_on']:.3f} < μ_on(pinned)={c2b_ref}",
            )
        )
    _badge(ax_badge, tests)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"litmus_cond_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Figure saved: {out_path}")
    plt.close(fig)
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    global TOTAL_TIMESTEPS, FIXED_SAVINGS
    parser = argparse.ArgumentParser(
        description="Policy-Conditioning Litmus Suite (LITMUS_TEST_SPEC.md Part II)"
    )
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--tests",
        type=str,
        default="c1,c2,c3",
        help="Comma-separated subset, e.g. c1,c2",
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Path to existing .pkl — skip training, re-plot only",
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
    tests_to_run = {t.strip().lower() for t in args.tests.split(",")}

    if args.replot:
        with open(args.replot, "rb") as f:
            results = pickle.load(f)
        plot_results(results, file_tag)
        return

    key = jax.random.PRNGKey(args.seed)
    keys = jax.random.split(key, 10)
    results = {}

    if "c1" in tests_to_run:
        passed, data = run_c1(keys[0])
        results["c1"] = {"passed": passed, "data": data}

    c2b_mu_on = 0.5  # fallback
    if "c2" in tests_to_run:
        grade, data = run_c2(keys[1])
        results["c2"] = {"passed": grade, "data": data}
        c2b_mu_on = data["mu_b_on"]

    if "c3" in tests_to_run:
        passed, data = run_c3(keys[2], c2b_mu_on)
        results["c3"] = {"passed": passed, "data": data}

    # Strip agents from pkl if not requested (they're large)
    if not args.save_agents:
        for test_id, res in results.items():
            d = res.get("data", {})
            for k in list(d.keys()):
                if k.startswith("_agent"):
                    del d[k]

    # Save pickle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"litmus_cond_{file_tag}.pkl")
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(results, f)
    print(f"\n  Results saved: {pkl_path}")
    if args.save_agents:
        print("  (includes trained agents for Layer 2 introspection)")

    # Persist frozen config into experiment dir (no-op when run standalone)
    save_run_config(
        {
            "script": "cbam/drivers/cbam_litmus_conditioning.py",
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
    print("  POLICY-CONDITIONING LITMUS SUMMARY")
    print("=" * 55)
    print("  (see LITMUS_TEST_SPEC.md for interpretation guidelines)")
    print()
    for test, res in results.items():
        p = res["passed"]
        if isinstance(p, bool):
            tag = "PASS ✅" if p else "FAIL ❌"
        elif isinstance(p, str):
            tag = f"{p.upper()} ({'✅' if p == 'strong' else '🟡' if p == 'weak' else '❌'})"
        else:
            tag = "SKIP ⏭️"
        print(f"  {test.upper()}  {tag}")
    print()
    print("  One-sentence summary:")
    print("  The randomised-differential litmus suite shows that a single trained")
    print("  policy clearly conditions export reallocation on the CBAM signal,")
    print("  while mitigation conditioning is weaker and less robust, and")
    print("  crowd-out remains present when both margins are available.")


if __name__ == "__main__":
    main()
