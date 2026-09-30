"""validate_rcpo_additive.py

Compare welfloss (multiplicative) vs RCPO additive CBAM reward on the
9-region vulnerability setup.

Trains two agents:
  1. welfloss mode (reward_mode="welfloss", sectoral_welfloss=True)
  2. additive_cbam mode (reward_mode="additive_cbam", RCPOMonitoredPPO)

For each, runs eval episodes with τ=0.8 and τ=0, then plots:
  Row 0: Training metrics — reward_mean over iterations (from CSV)
  Row 1: λ trajectory (RCPO only; from training CSV)
  Row 2: Per-region CBAM cost (eval, τ=0.8)
  Row 3: EU export share — dirty vs clean (eval, τ=0.8)
  Row 4: Reward curves (eval, τ=0.8 vs τ=0)

Columns: one per focus region (all except EU and RoW).

Usage:
    cd rice_jax
    python validation/validate_rcpo_additive.py [--timesteps 5_000_000]
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import os
from dataclasses import replace
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaxnasium as jym
from training_monitor import (
    MonitoredPPO,
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
    rcpo_cbam_log_info_fn,
)
from _experiment_util import run_single_episode
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

# ── Configuration ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

NUM_REGIONS = 9
EU_REGION_IDX = 3
YAML_DIR = os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_DATA_ROOT = os.path.join(_REPO_ROOT, "csv_asset")

# All except RoW (0) and EU (3)
FOCUS_REGIONS = [1, 2, 4, 5, 6, 7, 8]

TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 8
NUM_STEPS = 100
NUM_EVAL_EPISODES = 5
SEED = 42
CBAM_RATE = 0.80

OUTPUT_DIR = "plots"
LOG_DIR = "training_logs"
DPI = 150

# RCPO hyperparams (Tessler et al. 2019)
RCPO_ETA_LAMBDA = 1e-4
RCPO_ALPHA_TARGET = 0.005

# ── Shared environment kwargs ─────────────────────────────────────────────────

_BASE_MRIO_KWARGS = dict(
    num_regions=NUM_REGIONS,
    mrio_data_root=MRIO_DATA_ROOT,
    mrio_trade=True,
    eu_region_idx=EU_REGION_IDX,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    fixed_savings_rate=True,
    no_mitigation=True,
    sector_granularity="emissions-simple",
)

_PPO_KWARGS = dict(
    total_timesteps=TOTAL_TIMESTEPS,
    learning_rate=3e-4,
    num_steps=NUM_STEPS,
    num_envs=NUM_ENVS,
    num_minibatches=4,
    num_epochs=8,
    ent_coef=0.01,
    anneal_ent_coef=0.0,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    clip_coef=0.2,
    clip_coef_vf=0.5,
    vf_coef=0.5,
    normalize_observations=True,
    normalize_rewards=True,
    log_interval=50,   # absolute iters between callbacks; 0.02-fraction caused 62 syncs/run at 100k steps
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_log_fn(label: str, num_iters: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"{label}.csv")
    return make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    ), csv_path


def _build_env(reward_mode: str, cbam_rate: float, *, for_training: bool = True):
    """Build a RiceMRIO env with the given reward_mode and tariff rate."""
    extra = {}
    if reward_mode == "welfloss":
        extra["sectoral_welfloss"] = True
        extra["welfare_loss_per_unit_tariff"] = 5.0
    elif reward_mode == "additive_cbam":
        extra["log_info_fn"] = rcpo_cbam_log_info_fn

    env = RiceMRIO(
        region_params=load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_rate=cbam_rate,
        cbam_randomize=False,
        reward_mode=reward_mode,
        **_BASE_MRIO_KWARGS,
        **extra,
    )
    return jym.LogWrapper(env) if for_training else env


def _eval_agent(agent, reward_mode: str, cbam_rate: float, seed) -> dict:
    """Roll out eval episodes and return stacked arrays."""
    raw_env = _build_env(reward_mode, cbam_rate, for_training=False)
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    all_flows, all_uwl, all_util, all_cbam_cost = [], [], [], []

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 10_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)

        all_flows.append(np.array(logs["trade_flows"]))
        uwl = np.stack(
            [np.array(logs["utility_times_welfloss_all_regions"][r])
             for r in range(NUM_REGIONS)], axis=-1)
        all_uwl.append(uwl)
        util = np.stack(
            [np.array(logs["utility_all_regions"][r])
             for r in range(NUM_REGIONS)], axis=-1)
        all_util.append(util)
        cbam = np.stack(
            [np.array(logs["cbam_cost_all_regions"][r])
             for r in range(NUM_REGIONS)], axis=-1)
        all_cbam_cost.append(cbam)

    return {
        "trade_flows": np.stack(all_flows, axis=0),
        "utility_welfloss": np.stack(all_uwl, axis=0),
        "utility": np.stack(all_util, axis=0),
        "cbam_cost": np.stack(all_cbam_cost, axis=0),
    }


def _load_training_csv(csv_path: str) -> dict:
    """Load training CSV into a dict of numpy arrays."""
    import csv as csv_mod
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)
    result = {}
    for k in rows[0]:
        vals = []
        for r in rows:
            v = r[k]
            try:
                vals.append(float(v) if v != "" else np.nan)
            except ValueError:
                vals.append(np.nan)
        result[k] = np.array(vals)
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(
    welfloss_csv: dict, rcpo_csv: dict,
    welfloss_eval: dict, rcpo_eval: dict,
    welfloss_eval_nocbam: dict, rcpo_eval_nocbam: dict,
    region_labels: list[str],
) -> plt.Figure:
    n_focus = len(FOCUS_REGIONS)
    # 5 rows: training reward, λ trajectory, CBAM cost, EU export share, eval reward
    fig = plt.figure(figsize=(3.5 * n_focus, 18), constrained_layout=True)
    gs = fig.add_gridspec(5, n_focus)

    T_eval = welfloss_eval["trade_flows"].shape[1]
    ts_eval = np.arange(T_eval)

    # ── Row 0: Training reward_mean (both modes on same axes) ─────────────
    ax_train = fig.add_subplot(gs[0, :])
    iters_wl = welfloss_csv.get("iteration", np.arange(len(welfloss_csv["reward_mean"])))
    iters_rc = rcpo_csv.get("iteration", np.arange(len(rcpo_csv["reward_mean"])))
    ax_train.plot(iters_wl, welfloss_csv["reward_mean"],
                  color="#5577cc", lw=1.5, alpha=0.8, label="welfloss")
    ax_train.plot(iters_rc, rcpo_csv["reward_mean"],
                  color="#e05c2a", lw=1.5, alpha=0.8, label="RCPO additive")
    ax_train.set_ylabel("reward_mean", fontsize=10)
    ax_train.set_xlabel("iteration", fontsize=10)
    ax_train.set_title("Training Reward Curve", fontsize=12, fontweight="bold")
    ax_train.legend(fontsize=9)
    ax_train.grid(alpha=0.2)

    # ── Row 1: λ trajectory (RCPO only — spans full width) ────────────────
    ax_lam = fig.add_subplot(gs[1, :])
    if "cbam_lambda" in rcpo_csv:
        lam_vals = rcpo_csv["cbam_lambda"]
        lam_iters = rcpo_csv.get("iteration", np.arange(len(lam_vals)))
        ax_lam.plot(lam_iters, lam_vals, color="#e05c2a", lw=2)
        ax_lam.set_ylabel("λ (Lagrange multiplier)", fontsize=10)
    if "mean_cbam_cost" in rcpo_csv:
        ax_cost_train = ax_lam.twinx()
        cost_vals = rcpo_csv["mean_cbam_cost"]
        ax_cost_train.plot(lam_iters, cost_vals, color="#888888", lw=1, ls="--",
                           alpha=0.7, label="mean CBAM cost")
        ax_cost_train.set_ylabel("mean CBAM cost", fontsize=9, color="#888888")
        ax_cost_train.tick_params(axis="y", labelcolor="#888888")
        ax_cost_train.legend(fontsize=8, loc="upper right")
    ax_lam.axhline(0, color="gray", ls=":", lw=0.8)
    ax_lam.set_xlabel("iteration", fontsize=10)
    ax_lam.set_title("RCPO λ Trajectory & Mean CBAM Cost", fontsize=12, fontweight="bold")
    ax_lam.grid(alpha=0.2)

    # ── Row 2: Per-region CBAM cost (eval, τ=0.8) ─────────────────────────
    for col_i, r_idx in enumerate(FOCUS_REGIONS):
        ax = fig.add_subplot(gs[2, col_i])
        rlbl = region_labels[r_idx] if r_idx < len(region_labels) else f"R{r_idx}"

        for label, data, color in [
            ("welfloss", welfloss_eval, "#5577cc"),
            ("RCPO", rcpo_eval, "#e05c2a"),
        ]:
            costs = data["cbam_cost"][:, :, r_idx]  # (E, T)
            m = np.nanmean(costs, axis=0)
            s = np.nanstd(costs, axis=0)
            ax.plot(ts_eval, m, color=color, lw=1.5, label=label)
            ax.fill_between(ts_eval, m - s, m + s, alpha=0.12, color=color)

        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel("CBAM cost", fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.grid(alpha=0.2)

    # ── Row 3: EU export share (dirty vs clean, eval τ=0.8) ───────────────
    for col_i, r_idx in enumerate(FOCUS_REGIONS):
        ax = fig.add_subplot(gs[3, col_i])
        rlbl = region_labels[r_idx]

        for label, data, color in [
            ("welfloss", welfloss_eval, "#5577cc"),
            ("RCPO", rcpo_eval, "#e05c2a"),
        ]:
            flows = data["trade_flows"]  # (E, T, NR, NR, NS)
            eu_dirty = flows[:, :, r_idx, EU_REGION_IDX, 0]
            tot_dirty = flows[:, :, r_idx, :, 0].sum(axis=-1)
            eu_clean = flows[:, :, r_idx, EU_REGION_IDX, 1]
            tot_clean = flows[:, :, r_idx, :, 1].sum(axis=-1)

            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
                sc = np.where(tot_clean > 1e-12, eu_clean / tot_clean, np.nan)

            md = np.nanmean(sd, axis=0)
            ax.plot(ts_eval, md, color=color, ls="-", lw=2,
                    label=f"{label} dirty")
            mc = np.nanmean(sc, axis=0)
            ax.plot(ts_eval, mc, color=color, ls="--", lw=1.5,
                    label=f"{label} clean")

        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel("EU export share", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=5, loc="best")
        ax.grid(alpha=0.2)

    # ── Row 4: Eval reward comparison (τ=0.8 vs τ=0) ─────────────────────
    for col_i, r_idx in enumerate(FOCUS_REGIONS):
        ax = fig.add_subplot(gs[4, col_i])
        rlbl = region_labels[r_idx]

        for label, data_cbam, data_nocbam, color in [
            ("welfloss", welfloss_eval, welfloss_eval_nocbam, "#5577cc"),
            ("RCPO", rcpo_eval, rcpo_eval_nocbam, "#e05c2a"),
        ]:
            # Reward = utility_welfloss for welfloss mode, utility for additive
            # We plot raw utility for consistency across modes
            util_cbam = data_cbam["utility"][:, :, r_idx]
            util_nocbam = data_nocbam["utility"][:, :, r_idx]
            m_cbam = np.nanmean(util_cbam, axis=0)
            m_nocbam = np.nanmean(util_nocbam, axis=0)
            ax.plot(ts_eval, m_cbam, color=color, ls="-", lw=2,
                    label=f"{label} τ=0.8")
            ax.plot(ts_eval, m_nocbam, color=color, ls="--", lw=1.5,
                    label=f"{label} τ=0", alpha=0.6)

        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel("Utility", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.legend(fontsize=5, loc="best")
        ax.grid(alpha=0.2)

    fig.suptitle(
        f"RCPO Additive vs Welfloss — 9-Region Vulnerability Setup (τ={CBAM_RATE:.0%})\n"
        f"emissions-simple | η_λ={RCPO_ETA_LAMBDA:.0e} | α_target={RCPO_ALPHA_TARGET}\n"
        f"{TOTAL_TIMESTEPS:,} PPO steps | {NUM_EVAL_EPISODES} eval eps | seed={SEED}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS, RCPO_ETA_LAMBDA, RCPO_ALPHA_TARGET, SEED

    parser = argparse.ArgumentParser(description="RCPO additive CBAM validation (9-region)")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eta-lambda", type=float, default=RCPO_ETA_LAMBDA)
    parser.add_argument("--alpha-target", type=float, default=RCPO_ALPHA_TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    TOTAL_TIMESTEPS = args.timesteps
    RCPO_ETA_LAMBDA = args.eta_lambda
    RCPO_ALPHA_TARGET = args.alpha_target
    SEED = args.seed
    _PPO_KWARGS["total_timesteps"] = TOTAL_TIMESTEPS

    seed = jax.random.PRNGKey(SEED)
    num_iters = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS

    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR)

    # Print env info
    sample_env = _build_env("welfloss", 0.0, for_training=False)
    region_labels = list(sample_env.mrio_region_labels)
    focus_names = [region_labels[r] for r in FOCUS_REGIONS]
    print(f"=== RCPO Additive CBAM Validation (9-region vuln) ===")
    print(f"Focus regions: {focus_names}")
    print(f"Timesteps: {TOTAL_TIMESTEPS:,}  |  η_λ={RCPO_ETA_LAMBDA:.0e}  |  α_target={RCPO_ALPHA_TARGET}")
    print(f"Sectors: {list(sample_env.sector_names)}")
    print()

    # ── 1. Train welfloss agent ───────────────────────────────────────────
    print("─── Training: welfloss mode ───")
    wl_log_fn, wl_csv_path = _make_log_fn("rcpo_val_welfloss", num_iters)
    wl_env = _build_env("welfloss", CBAM_RATE, for_training=True)
    wl_agent = MonitoredPPO(**{**_PPO_KWARGS, "log_function": wl_log_fn})
    wl_agent = wl_agent.train(seed, wl_env)
    print()

    # ── 2. Train RCPO additive agent ──────────────────────────────────────
    print("─── Training: RCPO additive mode ───")
    rc_log_fn, rc_csv_path = _make_log_fn("rcpo_val_additive", num_iters)
    rc_env = _build_env("additive_cbam", CBAM_RATE, for_training=True)
    rc_agent = RCPOMonitoredPPO(
        **{**_PPO_KWARGS, "log_function": rc_log_fn},
        rcpo_eta_lambda=RCPO_ETA_LAMBDA,
        rcpo_alpha_target=RCPO_ALPHA_TARGET,
    )
    rc_agent = rc_agent.train(seed, rc_env)
    print()

    # ── 3. Eval both agents ───────────────────────────────────────────────
    print("Evaluating welfloss agent (τ=0.8)...")
    wl_eval_cbam = _eval_agent(wl_agent, "welfloss", CBAM_RATE, seed)
    print("Evaluating welfloss agent (τ=0)...")
    wl_eval_nocbam = _eval_agent(wl_agent, "welfloss", 0.0, seed)

    print("Evaluating RCPO agent (τ=0.8)...")
    rc_eval_cbam = _eval_agent(rc_agent, "additive_cbam", CBAM_RATE, seed)
    print("Evaluating RCPO agent (τ=0)...")
    rc_eval_nocbam = _eval_agent(rc_agent, "additive_cbam", 0.0, seed)
    print()

    # ── 4. Load training CSVs ─────────────────────────────────────────────
    wl_csv = _load_training_csv(wl_csv_path)
    rc_csv = _load_training_csv(rc_csv_path)

    # ── 5. Plot ───────────────────────────────────────────────────────────
    print("Generating figure...")
    fig = make_figure(
        wl_csv, rc_csv,
        wl_eval_cbam, rc_eval_cbam,
        wl_eval_nocbam, rc_eval_nocbam,
        region_labels,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"rcpo_additive_validation_{timestamp}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
