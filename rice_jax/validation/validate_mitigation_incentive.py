"""validate_mitigation_incentive.py

Phase 2B motivating experiment: do agents learn to mitigate in order to
reduce CBAM burden when abatement is free?

Research motivation
-------------------
CBAM charges on *actual* embedded emissions (EU CBAM Reg. 2023/956, Art. 7).
If mitigation reduces embedded carbon intensity — i.e. the CBAM cost formula
uses σ_{r,s} × (1-μ_r) — then agents facing a CBAM tariff have a direct
incentive to mitigate.

With realistic (costly) abatement, that incentive competes with the output
loss from abatement costs, so the equilibrium mitigation rate is below 1.
The gap between the *free-abatement optimum* and the *costly-abatement
equilibrium* is the welfare loss that EU revenue recycling (Phase 2B) needs
to bridge.

Three conditions
----------------
A  no_cbam       τ=0,   zero_abatement_cost=True   → no incentive; μ≈0
B  cbam_free     τ=0.8, zero_abatement_cost=True   → free abatement; μ→1
C  cbam_costly   τ=0.8, zero_abatement_cost=False  → costly; μ in [0,1]

All conditions:
  - reward_mode="additive_cbam" + RCPOMonitoredPPO
  - no_mitigation=False (mitigation_rate IS an action)
  - fixed_savings_rate=True (keep focus on trade + mitigation)
  - 9-region vulnerability setup (eu_region_idx=3)
  - sector_granularity="emissions-simple" (dirty=0, clean=1)

Figure layout (4 rows × N_focus columns + 2 full-width rows):
  Row 0 (full)  : Training reward_mean — all 3 conditions
  Row 1 (full)  : RCPO λ trajectory — conditions B and C
  Row 2 (cols)  : Mean mitigation rate per focus region over eval episode
  Row 3 (cols)  : CBAM cost per focus region over eval episode
  Row 4 (full)  : Phase 2B narrative — CBAM cost gap (B minus C) per region,
                  illustrating the welfare transfer required to close the gap

Exit criteria
-------------
Condition A: mean mitigation_rate across episode < 0.05 for all focus regions.
Condition B: mean mitigation_rate > 0.5 for majority of focus regions by end.
Condition C: mean mitigation_rate < that of B (costly abatement is a barrier).
CBAM cost in B < CBAM cost in C (mitigation reduces the burden).

Usage (from rice_jax/ directory, rice-jax conda env):
    python validation/validate_mitigation_incentive.py [--timesteps 5000000]
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
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

NUM_REGIONS  = 9
EU_IDX       = 3          # EU & Western Europe in 9-region vuln ordering
YAML_DIR     = os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_ROOT    = os.path.join(_REPO_ROOT, "csv_asset")

# All focus regions: every region except RoW (0) and EU (3)
FOCUS_REGIONS = [1, 2, 4, 5, 6, 7, 8]
FOCUS_LABELS  = [
    "Russia/Eurasia",  # 1
    "MENA",            # 2
    "SSA Metals",      # 4
    "Americas",        # 5
    "SE Asia",         # 6
    "China",           # 7
    "India",           # 8
]

TOTAL_TIMESTEPS   = 5_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 5
SEED              = 42
CBAM_RATE         = 0.80

RCPO_ETA_LAMBDA   = 1e-4
RCPO_ALPHA_TARGET = 0.005

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
DPI        = 150

# ── Shared env / PPO kwargs ───────────────────────────────────────────────────

_BASE_ENV_KWARGS = dict(
    num_regions              = NUM_REGIONS,
    mrio_data_root           = MRIO_ROOT,
    mrio_trade               = True,
    eu_region_idx            = EU_IDX,
    dest_alloc_persistence   = 0.55,
    dest_alloc_baseline_decay= 1.0,
    diff_reward_mode         = True,
    num_discrete_action_levels=10,
    fixed_savings_rate       = True,    # savings fixed; focus on trade + μ
    no_mitigation            = False,   # mitigation IS an action
    sector_granularity       = "emissions-simple",
    reward_mode              = "additive_cbam",
    log_info_fn              = rcpo_cbam_log_info_fn,
)

_PPO_KWARGS = dict(
    total_timesteps   = TOTAL_TIMESTEPS,
    learning_rate     = 3e-4,
    num_steps         = NUM_STEPS,
    num_envs          = NUM_ENVS,
    num_minibatches   = 4,
    num_epochs        = 8,
    ent_coef          = 0.01,
    anneal_ent_coef   = 0.0,
    gamma             = 0.99,
    gae_lambda        = 0.95,
    max_grad_norm     = 1.0,
    clip_coef         = 0.2,
    clip_coef_vf      = 0.5,
    vf_coef           = 0.5,
    normalize_observations=True,
    normalize_rewards = True,
    log_interval      = 50,   # absolute iters between callbacks; 0.02-fraction caused 62 syncs/run at 100k steps
)


# ── Condition constructors ────────────────────────────────────────────────────

def _build_env(cbam_rate: float, zero_abatement_cost: bool, for_training: bool = True):
    env = RiceMRIO(
        region_params       = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_rate    = cbam_rate,
        cbam_randomize      = False,
        zero_abatement_cost = zero_abatement_cost,
        **_BASE_ENV_KWARGS,
    )
    return jym.LogWrapper(env) if for_training else env


def _make_log_fn(label: str, num_iters: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"mit_incentive_{label}.csv")
    fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    return fn, csv_path


def _train(label: str, cbam_rate: float, zero_abatement_cost: bool, key):
    """Train one RCPOMonitoredPPO agent under the given condition."""
    env = _build_env(cbam_rate, zero_abatement_cost, for_training=True)
    num_iters = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    log_fn, csv_path = _make_log_fn(label, num_iters)

    ppo = RCPOMonitoredPPO(
        rcpo_eta_lambda   = RCPO_ETA_LAMBDA,
        rcpo_alpha_target = RCPO_ALPHA_TARGET,
        log_function      = log_fn,
        **_PPO_KWARGS,
    )
    print(f"\n{'='*60}")
    print(f"Training condition: {label}  (τ={cbam_rate}, zero_abat={zero_abatement_cost})")
    print(f"{'='*60}")
    ppo = ppo.train(key, env)
    return ppo, csv_path


def _eval(agent, cbam_rate: float, zero_abatement_cost: bool, seed) -> dict:
    """Run eval episodes; return stacked arrays."""
    eval_env = _build_env(cbam_rate, zero_abatement_cost, for_training=False)
    eval_env = replace(eval_env, log_info_fn=full_state_info_log_fn)

    all_flows, all_mit, all_cost, all_util = [], [], [], []

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 20_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        all_flows.append(np.array(logs["trade_flows"]))          # (T, NR, NR, NS)

        # mitigation_rates_all_regions: may come as per-agent dict or array
        mit_raw = logs["mitigation_rates_all_regions"]
        if isinstance(mit_raw, dict):
            mit = np.stack([np.array(mit_raw[k]) for k in sorted(mit_raw)], axis=-1)
        else:
            mit = np.array(mit_raw)                              # (T, NR)
        all_mit.append(mit)

        cbam_raw = logs["cbam_cost_all_regions"]
        if isinstance(cbam_raw, dict):
            cbam = np.stack([np.array(cbam_raw[k]) for k in sorted(cbam_raw)], axis=-1)
        else:
            cbam = np.array(cbam_raw)
        all_cost.append(cbam)

        util_raw = logs["utility_all_regions"]
        if isinstance(util_raw, dict):
            util = np.stack([np.array(util_raw[k]) for k in sorted(util_raw)], axis=-1)
        else:
            util = np.array(util_raw)
        all_util.append(util)

    return {
        "trade_flows": np.stack(all_flows, axis=0),  # (E, T, NR, NR, NS)
        "mitigation":  np.stack(all_mit,   axis=0),  # (E, T, NR)
        "cbam_cost":   np.stack(all_cost,  axis=0),  # (E, T, NR)
        "utility":     np.stack(all_util,  axis=0),  # (E, T, NR)
    }


def _load_csv(csv_path: str) -> dict:
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
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

_COND_STYLES = {
    "no_cbam":    dict(color="#5577cc", ls="-",  label="A: no CBAM (τ=0, free abat.)"),
    "cbam_free":  dict(color="#2ecc71", ls="-",  label="B: CBAM + free abatement"),
    "cbam_costly":dict(color="#e05c2a", ls="--", label="C: CBAM + costly abatement"),
}


def make_figure(csvs: dict, evals: dict) -> plt.Figure:
    n_focus = len(FOCUS_REGIONS)
    T = list(evals.values())[0]["mitigation"].shape[1]
    ts = np.arange(T)

    fig = plt.figure(figsize=(3.5 * n_focus, 22), constrained_layout=True)
    gs  = fig.add_gridspec(6, n_focus)

    # ── Row 0 (full width): Training reward curves ────────────────────────
    ax_tr = fig.add_subplot(gs[0, :])
    for cond, csv_data in csvs.items():
        s = _COND_STYLES[cond]
        iters = csv_data.get("iteration", np.arange(len(csv_data["reward_mean"])))
        ax_tr.plot(iters, csv_data["reward_mean"], color=s["color"],
                   ls=s["ls"], lw=1.5, alpha=0.85, label=s["label"])
    ax_tr.set_ylabel("reward_mean", fontsize=10)
    ax_tr.set_xlabel("iteration", fontsize=10)
    ax_tr.set_title("Training Reward (all conditions)", fontsize=12, fontweight="bold")
    ax_tr.legend(fontsize=9)
    ax_tr.grid(alpha=0.2)

    # ── Row 1 (full width): RCPO λ trajectory for B and C ─────────────────
    ax_lam = fig.add_subplot(gs[1, :])
    for cond in ("cbam_free", "cbam_costly"):
        csv_data = csvs[cond]
        s = _COND_STYLES[cond]
        if "cbam_lambda" in csv_data:
            iters = csv_data.get("iteration", np.arange(len(csv_data["cbam_lambda"])))
            ax_lam.plot(iters, csv_data["cbam_lambda"], color=s["color"],
                        ls=s["ls"], lw=2, label=f"λ — {s['label']}")
    ax_lam.set_ylabel("λ (RCPO Lagrange mult.)", fontsize=10)
    ax_lam.set_xlabel("iteration", fontsize=10)
    ax_lam.set_title(
        "RCPO λ Trajectory — B vs C\n"
        "B should converge λ→0 (μ→1 drives cost→0); C should converge to λ>0",
        fontsize=11, fontweight="bold",
    )
    ax_lam.legend(fontsize=9)
    ax_lam.axhline(0, color="gray", ls=":", lw=0.8)
    ax_lam.grid(alpha=0.2)

    # ── Row 2 (cols): Mean mitigation rate over eval episode ──────────────
    for ci, (ridx, rlbl) in enumerate(zip(FOCUS_REGIONS, FOCUS_LABELS)):
        ax = fig.add_subplot(gs[2, ci])
        for cond, ev in evals.items():
            s = _COND_STYLES[cond]
            mit = ev["mitigation"][:, :, ridx]  # (E, T)
            m = np.nanmean(mit, axis=0)
            se = np.nanstd(mit, axis=0) / np.sqrt(NUM_EVAL_EPISODES)
            ax.plot(ts, m, color=s["color"], ls=s["ls"], lw=1.8, label=s["label"])
            ax.fill_between(ts, m - se, m + se, alpha=0.10, color=s["color"])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("Mitigation rate μ", fontsize=9)
        ax.grid(alpha=0.2)
        if ci == 0:
            ax.legend(fontsize=6, loc="best")

    # ── Row 3 (cols): CBAM cost over eval episode ─────────────────────────
    for ci, (ridx, rlbl) in enumerate(zip(FOCUS_REGIONS, FOCUS_LABELS)):
        ax = fig.add_subplot(gs[3, ci])
        for cond in ("cbam_free", "cbam_costly"):
            ev = evals[cond]
            s  = _COND_STYLES[cond]
            cost = ev["cbam_cost"][:, :, ridx]  # (E, T)
            m  = np.nanmean(cost, axis=0)
            se = np.nanstd(cost, axis=0) / np.sqrt(NUM_EVAL_EPISODES)
            ax.plot(ts, m, color=s["color"], ls=s["ls"], lw=1.8, label=s["label"])
            ax.fill_between(ts, m - se, m + se, alpha=0.10, color=s["color"])
        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("CBAM cost", fontsize=9)
        ax.grid(alpha=0.2)
        if ci == 0:
            ax.legend(fontsize=6, loc="best")

    # ── Row 4 (cols): EU dirty export share (sanity — diversion still works) ─
    for ci, (ridx, rlbl) in enumerate(zip(FOCUS_REGIONS, FOCUS_LABELS)):
        ax = fig.add_subplot(gs[4, ci])
        for cond in ("cbam_free", "cbam_costly"):
            ev = evals[cond]
            s  = _COND_STYLES[cond]
            flows = ev["trade_flows"]                            # (E, T, NR, NR, NS)
            eu_dirty  = flows[:, :, ridx, EU_IDX, 0]
            tot_dirty = flows[:, :, ridx, :, 0].sum(axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
            m  = np.nanmean(share, axis=0)
            se = np.nanstd(share, axis=0) / np.sqrt(NUM_EVAL_EPISODES)
            ax.plot(ts, m, color=s["color"], ls=s["ls"], lw=1.8, label=s["label"])
            ax.fill_between(ts, m - se, m + se, alpha=0.10, color=s["color"])
        ax.set_ylim(bottom=0)
        ax.set_title(rlbl, fontsize=9, fontweight="bold")
        if ci == 0:
            ax.set_ylabel("EU dirty export share", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(alpha=0.2)
        if ci == 0:
            ax.legend(fontsize=6, loc="best")

    # ── Row 5 (full width): Phase 2B narrative bar ────────────────────────
    # Mean CBAM cost over last 5 eval steps: B vs C, per focus region.
    # The gap (C - B) is the "untapped mitigation" that revenue recycling must cover.
    ax_bar = fig.add_subplot(gs[5, :])
    T_last  = 5
    cost_B  = np.array([
        np.nanmean(evals["cbam_free"]["cbam_cost"][:, -T_last:, ridx])
        for ridx in FOCUS_REGIONS
    ])
    cost_C  = np.array([
        np.nanmean(evals["cbam_costly"]["cbam_cost"][:, -T_last:, ridx])
        for ridx in FOCUS_REGIONS
    ])
    x = np.arange(len(FOCUS_REGIONS))
    w = 0.35
    ax_bar.bar(x - w/2, cost_B, w, label="B: CBAM + free abatement (ceiling)",
               color="#2ecc71", alpha=0.85)
    ax_bar.bar(x + w/2, cost_C, w, label="C: CBAM + costly abatement (equilibrium)",
               color="#e05c2a", alpha=0.85)
    ax_bar.bar(x + w/2, np.maximum(cost_C - cost_B, 0), w,
               bottom=cost_B, label="Gap → Phase 2B: transfer needed",
               color="#f7ca18", alpha=0.9, hatch="//")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(FOCUS_LABELS, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylabel("Mean CBAM cost (last 5 steps)", fontsize=10)
    ax_bar.set_title(
        "Phase 2B Motivation: CBAM cost gap between free and costly abatement\n"
        "The yellow bars represent the welfare transfer EU must recycle to close the gap",
        fontsize=11, fontweight="bold",
    )
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Mitigation Incentive Experiment — 9-Region Vulnerability Setup\n"
        f"τ={CBAM_RATE:.0%} | η_λ={RCPO_ETA_LAMBDA:.0e} | α_target={RCPO_ALPHA_TARGET} | "
        f"{TOTAL_TIMESTEPS:,} steps | seed={SEED}",
        fontsize=12, fontweight="bold", y=1.005,
    )
    return fig


# ── Sanity checks ─────────────────────────────────────────────────────────────

def run_exit_criteria(evals: dict) -> bool:
    T_last = 5
    passed = True

    # Cond A: mean μ < 0.05 across focus regions (no incentive without CBAM)
    mit_A = np.nanmean(evals["no_cbam"]["mitigation"][:, -T_last:, :][:, :, FOCUS_REGIONS])
    if mit_A >= 0.05:
        print(f"  [FAIL] Cond A: mean μ={mit_A:.3f} ≥ 0.05 (expected near-zero without CBAM)")
        passed = False
    else:
        print(f"  [PASS] Cond A: mean μ={mit_A:.4f} < 0.05 (no CBAM → no mitigation incentive)")

    # Cond B: mean μ > 0.5 for majority of focus regions (free abatement → full mitigation)
    mit_B_per_region = [
        np.nanmean(evals["cbam_free"]["mitigation"][:, -T_last:, ridx])
        for ridx in FOCUS_REGIONS
    ]
    n_pass = sum(m > 0.5 for m in mit_B_per_region)
    if n_pass < len(FOCUS_REGIONS) // 2:
        print(f"  [FAIL] Cond B: only {n_pass}/{len(FOCUS_REGIONS)} regions reach μ>0.5")
        passed = False
    else:
        print(f"  [PASS] Cond B: {n_pass}/{len(FOCUS_REGIONS)} regions reach μ>0.5 (CBAM + free abat.)")

    # Cond C: mean μ < Cond B (costly abatement is a barrier)
    mit_C = np.nanmean(evals["cbam_costly"]["mitigation"][:, -T_last:, :][:, :, FOCUS_REGIONS])
    mit_B = np.nanmean(evals["cbam_free"]["mitigation"][:, -T_last:, :][:, :, FOCUS_REGIONS])
    if mit_C >= mit_B:
        print(f"  [FAIL] Cond C: mean μ_costly={mit_C:.3f} ≥ mean μ_free={mit_B:.3f}")
        passed = False
    else:
        print(f"  [PASS] Cond C: μ_costly={mit_C:.3f} < μ_free={mit_B:.3f} (cost is a barrier)")

    # Cost: B < C (mitigation reduces CBAM burden in B)
    cost_B = np.nanmean(evals["cbam_free"]["cbam_cost"][:, -T_last:, :][:, :, FOCUS_REGIONS])
    cost_C = np.nanmean(evals["cbam_costly"]["cbam_cost"][:, -T_last:, :][:, :, FOCUS_REGIONS])
    if cost_B >= cost_C:
        print(f"  [FAIL] Cost: cbam_cost_free={cost_B:.4f} ≥ cbam_cost_costly={cost_C:.4f}")
        passed = False
    else:
        print(f"  [PASS] Cost: cbam_cost_free={cost_B:.4f} < cbam_cost_costly={cost_C:.4f}")

    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS, RCPO_ETA_LAMBDA, RCPO_ALPHA_TARGET, SEED

    parser = argparse.ArgumentParser(
        description="Phase 2B motivating experiment: CBAM mitigation incentive"
    )
    parser.add_argument("--timesteps",    type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--eta-lambda",   type=float, default=RCPO_ETA_LAMBDA)
    parser.add_argument("--alpha-target", type=float, default=RCPO_ALPHA_TARGET)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument(
        "--plot-only", metavar="CSV_DIR",
        help="Skip training; regenerate plot from existing CSVs in this directory",
    )
    args = parser.parse_args()

    TOTAL_TIMESTEPS   = args.timesteps
    RCPO_ETA_LAMBDA   = args.eta_lambda
    RCPO_ALPHA_TARGET = args.alpha_target
    SEED              = args.seed

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    key = jax.random.PRNGKey(SEED)
    keys = jax.random.split(key, 4)

    # ── Conditions ────────────────────────────────────────────────────────
    CONDITIONS = {
        "no_cbam":     dict(cbam_rate=0.0,      zero_abatement_cost=True),
        "cbam_free":   dict(cbam_rate=CBAM_RATE, zero_abatement_cost=True),
        "cbam_costly": dict(cbam_rate=CBAM_RATE, zero_abatement_cost=False),
    }

    agents = {}
    csv_paths = {}

    if args.plot_only:
        # Reload CSVs and skip training
        for cond in CONDITIONS:
            csv_paths[cond] = os.path.join(args.plot_only, f"mit_incentive_{cond}.csv")
        print("--plot-only: loading CSVs, skipping training")
        # Agents needed for eval — cannot skip, warn user
        print("WARNING: --plot-only requires re-running eval; training is skipped but eval runs.")
        for i, (cond, kwargs) in enumerate(CONDITIONS.items()):
            _, csv_paths[cond] = _train(cond, **kwargs, key=keys[i])
            agents[cond] = None  # placeholder; eval won't work without agents
        print("Cannot regenerate eval without saved agents. Exiting after plot from CSVs only.")
        csvs = {c: _load_csv(p) for c, p in csv_paths.items() if os.path.exists(p)}
        # Minimal figure from CSVs only (rows 0-1)
    else:
        for i, (cond, kwargs) in enumerate(CONDITIONS.items()):
            agent, csv_path = _train(cond, **kwargs, key=keys[i])
            agents[cond]    = agent
            csv_paths[cond] = csv_path

    # ── Eval ──────────────────────────────────────────────────────────────
    eval_key = jax.random.PRNGKey(SEED + 999)
    evals = {}
    for cond, kwargs in CONDITIONS.items():
        print(f"\nEval: {cond} …")
        evals[cond] = _eval(agents[cond], **kwargs, seed=eval_key)

    # ── Exit criteria ─────────────────────────────────────────────────────
    print("\n── Exit criteria ──────────────────────────────────────────────")
    passed = run_exit_criteria(evals)
    print(f"Overall: {'PASS ✓' if passed else 'FAIL ✗'}")

    # ── Plot ──────────────────────────────────────────────────────────────
    csvs = {c: _load_csv(p) for c, p in csv_paths.items()}
    fig  = make_figure(csvs, evals)

    ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(OUTPUT_DIR, f"mitigation_incentive_{ts_str}.png")
    fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    main()
