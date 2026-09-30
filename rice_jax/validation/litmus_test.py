"""litmus_test.py

Minimal convergence litmus tests for the CBAM-RICE environment.

Philosophy
----------
Each test isolates ONE action channel, pins everything else, and asks a
single binary question: "did the agent learn the directionally correct
thing in a simplified setting?"

Tests
-----
L1w Export diversion (welfloss) — empirically ~27 pp contrast
L1r Export diversion (additive_cbam / RCPO) — empirically ~6 pp contrast
    3 regions, emissions-simple (2 sectors).
    ONLY export_reallocation is free (savings fixed, mitigation pinned=0).
    Both variants train ONE agent on cbam_randomize=True, τ ∈ {0, 0.8}.
    cbam_tariff_rate in the observation lets the agent learn a τ-conditioned
    routing policy without between-run variance (ctrl/treat failure mode).
    Eval: same model at τ=0 and τ=0.8.
    Question: does the EU-bound dirty-sector export SHARE decrease at τ=0.8?
    Pass: EU dirty share drops by >5 pp from τ=0 to τ=0.8.

L2  Mitigation reduces CBAM (free abatement)
    3 regions, CBAM τ=0.8, zero_abatement_cost=True.
    ONLY mitigation_rate is free; exports pinned to MRIO baseline via
    delta_max=0.0 (logit adjustments zeroed → softmax = 2016 shares).
    Question: does the agent learn μ > 0 when mitigation is the only escape?
    Pass: mean μ > 0.15 for non-EU region at episode end.

L3  DICE damage avoidance (no CBAM)
    3 regions, CBAM τ=0, realistic abatement cost.
    ONLY mitigation_rate is free.
    Question: does the agent mitigate at least a little (classic DICE result)?
    Pass: mean μ ∈ (0.01, 0.99) — RICE end-of-century optimum is ~0.7, not 0.

L4  Export diversion + mitigation together (sanity)
    Same as L2 but BOTH channels are free (the full Phase 2B action space).
    Question: does enabling trade diversion hurt the mitigation signal?
    Pass: mean μ at least 50% of L2 level.

Usage (from rice_jax/, rice-jax conda env):
    python validation/litmus_test.py [--timesteps 500000] [--seed 42]
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import time
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
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls


# ── Config ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT   = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", ".."))

# 3-region is fastest to train and has the smallest action space:
#   export_reallocation: 2 sectors × 3 destinations × 3 regions = 18 dims
# 3-region ordering (CountryClass_3.csv / eora_agg_3):
#   0: Sub-Saharan Africa,  1: Europe & Central Asia (EU),  2: North America+
NUM_REGIONS  = 3
EU_IDX       = 1
YAML_DIR     = _os.path.join(_REPO_ROOT, "cbam_yamls", "setup_3")
MRIO_ROOT    = _os.path.join(_REPO_ROOT, "csv_asset")

TOTAL_TIMESTEPS   = 500_000   # override with --timesteps
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42
CBAM_RATE         = 0.80

# RCPO: eta_lambda sized so λ reaches ~0.3 in one training run.
# With cbam_cost ≈ 0.003 (3-region) and alpha_target=0.001:
#   Δλ/iter = eta × (0.003 - 0.001) = eta × 0.002
# For λ≈0.3 after 625 iters (500k/8/100): eta = 0.3 / (625 × 0.002) = 0.24
# Use 0.05 as conservative starting point (still 100× larger than old 1e-4).
RCPO_ETA_LAMBDA   = 5e-2
RCPO_ALPHA_TARGET = 0.001
WELFARE_LOSS_WEIGHT = 5.0   # welfare_loss_per_unit_tariff
L1_TIMESTEPS = None         # if set, overrides TOTAL_TIMESTEPS for L1 only

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"

_BASE_ENV = dict(
    num_regions               = NUM_REGIONS,
    mrio_data_root            = MRIO_ROOT,
    mrio_trade                = True,
    eu_region_idx             = EU_IDX,
    dest_alloc_persistence    = 0.55,
    dest_alloc_baseline_decay = 1.0,
    diff_reward_mode          = True,
    num_discrete_action_levels= 10,
    sector_granularity        = "emissions-simple",
    sectoral_welfloss         = True,
    welfare_loss_per_unit_tariff = WELFARE_LOSS_WEIGHT,
)

_PPO_KWARGS = dict(
    num_steps           = NUM_STEPS,
    num_envs            = NUM_ENVS,
    learning_rate       = 3e-4,
    num_minibatches     = 4,
    num_epochs          = 8,
    ent_coef            = 0.01,
    anneal_ent_coef     = 0.0,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    max_grad_norm       = 1.0,
    clip_coef           = 0.2,
    clip_coef_vf        = 0.5,
    vf_coef             = 0.5,
    normalize_observations = True,
    normalize_rewards   = True,
    log_interval        = 50,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _build_env(cbam_rate: float, zero_abatement_cost: bool,
               no_mitigation: bool, fixed_savings: bool,
               reward_mode: str = "additive_cbam",
               delta_max: float = 3.0,
               cbam_randomize: bool = False,
               cbam_tariff_rates: tuple | None = None,
               for_training: bool = True):
    extra = {}
    if reward_mode == "additive_cbam":
        extra["log_info_fn"] = rcpo_cbam_log_info_fn
    env = RiceMRIO(
        region_params         = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        cbam_tariff_rate      = cbam_rate,
        zero_abatement_cost   = zero_abatement_cost,
        no_mitigation         = no_mitigation,
        fixed_savings_rate    = fixed_savings,
        delta_max             = delta_max,
        reward_mode           = reward_mode,
        cbam_randomize        = cbam_randomize,
        cbam_tariff_rates     = cbam_tariff_rates if cbam_tariff_rates is not None else (cbam_rate,),
        **_BASE_ENV,
        **extra,
    )
    return jym.LogWrapper(env) if for_training else env


def _make_log_fn(label: str, num_iters: int):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"litmus_{label}.csv")
    return make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    ), csv_path


def _train(label: str, env, key, num_iters: int, reward_mode: str,
           total_timesteps: int | None = None,
           extra_ppo_kwargs: dict | None = None,
           eta_lambda: float | None = None):
    ts = total_timesteps or TOTAL_TIMESTEPS
    ppo_kwargs = {**_PPO_KWARGS, **(extra_ppo_kwargs or {})}
    _eta = eta_lambda if eta_lambda is not None else RCPO_ETA_LAMBDA
    log_fn, csv_path = _make_log_fn(label, num_iters)
    if reward_mode == "additive_cbam":
        ppo = RCPOMonitoredPPO(
            total_timesteps   = ts,
            log_function      = log_fn,
            rcpo_eta_lambda   = _eta,
            rcpo_alpha_target = RCPO_ALPHA_TARGET,
            **ppo_kwargs,
        )
    else:
        ppo = MonitoredPPO(
            total_timesteps = ts,
            log_function    = log_fn,
            **ppo_kwargs,
        )
    print(f"\n{'━'*55}")
    print(f"  Training: {label}")
    print(f"{'━'*55}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.0f}s")
    return ppo, csv_path


def _eval_episode(key, raw_rice_env, agent) -> dict:
    """Run eval episodes via full_state_info_log_fn; return stacked logs.

    Parameters
    ----------
    raw_rice_env : RiceMRIO
        The unwrapped environment (NOT a LogWrapper). Built via
        ``_build_env(..., for_training=False)``.
    """
    from _experiment_util import run_single_episode

    eval_env = replace(raw_rice_env, log_info_fn=full_state_info_log_fn)

    # full_state_info_log_fn restructures *_all_regions keys into
    # {region_id: array(T)} dicts — convert back to (T, NR) arrays.
    def _per_region_to_array(region_dict: dict) -> np.ndarray:
        return np.stack(
            [np.array(region_dict[i]) for i in range(NUM_REGIONS)], axis=-1
        )  # (T, NR)

    all_flows, all_mit, all_cbam = [], [], []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 30_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        all_flows.append(np.array(logs["trade_flows"]))                       # (T, NR, NR, NS)
        all_mit.append(_per_region_to_array(logs["mitigation_rates_all_regions"]))  # (T, NR)
        all_cbam.append(_per_region_to_array(logs["cbam_cost_all_regions"]))        # (T, NR)

    return {
        "trade_flows": np.stack(all_flows, axis=0),  # (E, T, NR, NR, NS)
        "mitigation":  np.stack(all_mit,   axis=0),  # (E, T, NR)
        "cbam_cost":   np.stack(all_cbam,  axis=0),  # (E, T, NR)
    }


# ── Litmus tests ─────────────────────────────────────────────────────────────

def _eu_dirty_share(trade_flows: np.ndarray) -> float:
    """Mean EU-bound dirty-sector export share across non-EU regions, last 5 steps."""
    # trade_flows: (E, T, NR, NR, NS) — [ep, step, from, to, sector]
    # dirty sector = index 0 (emissions-simple: CBAM=0, non-CBAM=1)
    T_last = 5
    tf = trade_flows[:, -T_last:, :, :, :]      # (E, 5, NR, NR, NS)
    dirty_eu  = tf[:, :, :, EU_IDX, 0]          # (E, 5, NR) — dirty exports to EU
    dirty_all = tf[:, :, :, :,      0].sum(-1)  # (E, 5, NR) — total dirty exports
    # non-EU exporters only
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_IDX]
    share = (dirty_eu[:, :, non_eu] / (dirty_all[:, :, non_eu] + 1e-10)).mean()
    return float(share)


def run_l1(key, reward_mode: str = "welfloss") -> tuple[bool, dict]:
    """L1: Export diversion — single agent trained on randomized τ ∈ {0, 0.8}.

    Because cbam_tariff_rate is in every agent's observation, the model
    learns a τ-conditioned routing policy in one training run.  At eval
    time the same model is tested under two fixed conditions:
      - τ=0  : no CBAM penalty → agent routes freely (higher EU share)
      - τ=0.8: CBAM active → dirty EU exports should decrease

    This eliminates between-run variance (the failure mode of ctrl/treat
    with separate models).

    reward_mode="welfloss":
        Per-step welfloss multiplier is small but consistent. Works because
        the within-policy contrast (τ=0 vs τ=0.8 in the same batch) lets
        the advantage function attribute reward differences to τ-conditioned
        routing. Empirically gives ~27 pp contrast.

    reward_mode="additive_cbam":
        RCPO Lagrange λ equilibrates: τ=0 episodes push λ→0 (cbam_cost=0),
        τ=0.8 episodes push it up. Gives smaller contrast (~6 pp) because
        λ stabilises at ~0.001 (small additive penalty vs ΔU variance).
    """
    tag_mode = "welfloss" if reward_mode == "welfloss" else "additive_cbam/RCPO"
    train_label = "l1_divert_w" if reward_mode == "welfloss" else "l1_divert_r"
    print("\n" + "═"*55)
    print(f"  L1  Export Diversion  [{tag_mode}]")
    print(f"  cbam_randomize τ∈{{0,0.8}}, {tag_mode}, only export_realloc free")
    print("  Eval: same model at τ=0 vs τ=0.8")
    print("═"*55)

    l1_ts = L1_TIMESTEPS or TOTAL_TIMESTEPS
    num_iters = l1_ts // NUM_ENVS // NUM_STEPS

    env = _build_env(cbam_rate=0.0,  # ignored when cbam_randomize=True
                     zero_abatement_cost=False,
                     no_mitigation=True, fixed_savings=True,
                     reward_mode=reward_mode,
                     cbam_randomize=True,
                     cbam_tariff_rates=(0.0, CBAM_RATE))
    agent, _ = _train(train_label, env, key, num_iters, reward_mode,
                      total_timesteps=l1_ts)

    eval_key = jax.random.fold_in(key, 1)
    # Condition A: τ=0 — no tariff → agent routes freely
    raw_low  = _build_env(cbam_rate=0.0,       zero_abatement_cost=False,
                          no_mitigation=True,   fixed_savings=True,
                          reward_mode=reward_mode, for_training=False)
    # Condition B: τ=0.8 — tariff active → agent should divert dirty away from EU
    raw_high = _build_env(cbam_rate=CBAM_RATE, zero_abatement_cost=False,
                          no_mitigation=True,   fixed_savings=True,
                          reward_mode=reward_mode, for_training=False)
    eval_low  = _eval_episode(eval_key, raw_low,  agent)
    eval_high = _eval_episode(eval_key, raw_high, agent)

    share_ctrl  = _eu_dirty_share(eval_low["trade_flows"])
    share_treat = _eu_dirty_share(eval_high["trade_flows"])
    drop_pp = (share_ctrl - share_treat) * 100  # pp drop from τ=0 to τ=0.8

    passed = drop_pp > 5.0
    result = dict(share_ctrl=share_ctrl, share_treat=share_treat, drop_pp=drop_pp)
    tag = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  L1 [{tag_mode}] {tag}: EU dirty share  τ=0: {share_ctrl:.3f}  τ=0.8: {share_treat:.3f}  "
          f"drop={drop_pp:.1f} pp  (threshold: >5 pp)")
    return passed, result


def run_l2(key) -> tuple[bool, dict]:
    """L2: Mitigation signal — free abatement + CBAM → agent learns μ>0.

    Exports are pinned to the MRIO 2016 baseline via ``delta_max=0.0``
    (all logit adjustments collapse to zero → softmax = 2016 shares).
    This means mitigation is the *only* escape from the CBAM penalty.
    """
    print("\n" + "═"*55)
    print("  L2  Mitigation Signal (free abatement)")
    print("  CBAM τ=0.8, zero_abatement_cost=True, only μ free (exports pinned δₘₐₓ=0)")
    print("═"*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    # delta_max=0.0 collapses all export logit adjustments to zero so softmax
    # falls back to the MRIO 2016 baseline — exports are effectively frozen.
    env = _build_env(cbam_rate=CBAM_RATE, zero_abatement_cost=True,
                     no_mitigation=False, fixed_savings=True, delta_max=0.0)
    agent, _ = _train("l2_mit_free", env, key, num_iters, "additive_cbam")

    eval_key = jax.random.fold_in(key, 2)
    raw_env = _build_env(cbam_rate=CBAM_RATE, zero_abatement_cost=True,
                         no_mitigation=False, fixed_savings=True,
                         delta_max=0.0, for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    T_last = 5
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_IDX]
    mean_mu = float(ev["mitigation"][:, -T_last:, :][:, :, non_eu].mean())

    passed = mean_mu > 0.15
    tag = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  L2 {tag}: mean μ (non-EU, last {T_last} steps) = {mean_mu:.3f}  "
          f"(threshold: >0.15)")
    return passed, dict(mean_mu=mean_mu)


def run_l3(key) -> tuple[bool, dict]:
    """L3: DICE damage avoidance — no CBAM, costly abatement → small positive μ."""
    print("\n" + "═"*55)
    print("  L3  DICE Damage Avoidance")
    print("  τ=0, realistic abatement cost, only μ free")
    print("═"*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    env = _build_env(cbam_rate=0.0, zero_abatement_cost=False,
                     no_mitigation=False, fixed_savings=True,
                     reward_mode="welfloss")
    agent, _ = _train("l3_dice", env, key, num_iters, "welfloss")

    eval_key = jax.random.fold_in(key, 3)
    raw_env = _build_env(cbam_rate=0.0, zero_abatement_cost=False,
                         no_mitigation=False, fixed_savings=True,
                         reward_mode="welfloss", for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    T_last = 5
    mean_mu = float(ev["mitigation"][:, -T_last:, :].mean())

    passed = 0.01 < mean_mu < 0.99
    tag = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  L3 {tag}: mean μ (all regions, last {T_last} steps) = {mean_mu:.3f}  "
          f"(threshold: 0.01 < μ < 0.99 — classic RICE end-of-century μ≈0.7 is correct)")
    return passed, dict(mean_mu=mean_mu)


def run_l4(key, l2_mu: float) -> tuple[bool, dict]:
    """L4: Both channels free — diversion should not kill the mitigation signal."""
    print("\n" + "═"*55)
    print("  L4  Both Channels Free (diversion + mitigation)")
    print("  CBAM τ=0.8, zero_abatement_cost=True, μ + export_realloc free")
    print("═"*55)

    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    env = _build_env(cbam_rate=CBAM_RATE, zero_abatement_cost=True,
                     no_mitigation=False, fixed_savings=True)
    agent, _ = _train("l4_both", env, key, num_iters, "additive_cbam")

    eval_key = jax.random.fold_in(key, 4)
    raw_env = _build_env(cbam_rate=CBAM_RATE, zero_abatement_cost=True,
                         no_mitigation=False, fixed_savings=True, for_training=False)
    ev = _eval_episode(eval_key, raw_env, agent)

    T_last = 5
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_IDX]
    mean_mu = float(ev["mitigation"][:, -T_last:, :][:, :, non_eu].mean())

    threshold = max(l2_mu * 0.5, 0.05)
    passed = mean_mu >= threshold
    tag = "PASS ✓" if passed else "FAIL ✗"
    print(f"\n  L4 {tag}: mean μ = {mean_mu:.3f}  "
          f"(threshold: ≥{threshold:.3f} = 50% of L2 μ={l2_mu:.3f})")
    return passed, dict(mean_mu=mean_mu, threshold=threshold)


# ── Plot ─────────────────────────────────────────────────────────────────────

def _plot_results(results: dict, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    fig.suptitle("Litmus Tests — Convergence Verification", fontsize=13, fontweight="bold")

    def _color(key):
        return "#2ecc71" if results.get(key, {}).get("passed", False) else "#e74c3c"

    # ── L1w: welfloss ────────────────────────────────────────────────────
    ax = axes[0, 0]
    if "l1w" in results:
        vals = [results["l1w"]["share_ctrl"], results["l1w"]["share_treat"]]
        ax.bar(["τ=0", "τ=0.8"], vals,
               color=["#95a5a6", _color("l1w")], edgecolor="k", linewidth=0.8)
        ax.axhline(vals[0] - 0.05, color="k", ls="--", lw=0.8, label="threshold")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.legend(fontsize=8)
        drop = results["l1w"]["drop_pp"]
        ax.text(0.98, 0.97, f"Δ={drop:.1f} pp", ha="right", va="top",
                transform=ax.transAxes, fontsize=9)
    ax.set_ylabel("EU dirty export share")
    ax.set_title(f"L1w Export Diversion (welfloss)  {'✓' if results.get('l1w', {}).get('passed') else '✗'}")

    # ── L1r: additive_cbam / RCPO ────────────────────────────────────────
    ax = axes[0, 1]
    if "l1r" in results:
        vals = [results["l1r"]["share_ctrl"], results["l1r"]["share_treat"]]
        ax.bar(["τ=0", "τ=0.8"], vals,
               color=["#95a5a6", _color("l1r")], edgecolor="k", linewidth=0.8)
        ax.axhline(vals[0] - 0.05, color="k", ls="--", lw=0.8, label="threshold")
        ax.set_ylim(0, max(vals) * 1.3)
        ax.legend(fontsize=8)
        drop = results["l1r"]["drop_pp"]
        ax.text(0.98, 0.97, f"Δ={drop:.1f} pp", ha="right", va="top",
                transform=ax.transAxes, fontsize=9)
    ax.set_ylabel("EU dirty export share")
    ax.set_title(f"L1r Export Diversion (RCPO)  {'✓' if results.get('l1r', {}).get('passed') else '✗'}")

    # ── L2: mitigation signal ─────────────────────────────────────────────
    ax = axes[0, 2]
    if "l2" in results:
        mu = results["l2"]["mean_mu"]
        ax.bar(["μ (CBAM+free abat.)"], [mu], color=_color("l2"), edgecolor="k", linewidth=0.8)
        ax.axhline(0.15, color="k", ls="--", lw=0.8, label="threshold=0.15")
        ax.set_ylim(0, max(mu * 1.3, 0.3))
        ax.legend(fontsize=8)
    ax.set_ylabel("mean mitigation rate")
    ax.set_title(f"L2 Mitigation Signal  {'✓' if results.get('l2', {}).get('passed') else '✗'}")

    # ── L3: DICE damage avoidance ─────────────────────────────────────────
    ax = axes[1, 0]
    if "l3" in results:
        mu = results["l3"]["mean_mu"]
        ax.bar(["μ (no CBAM, costly)"], [mu], color=_color("l3"), edgecolor="k", linewidth=0.8)
        ax.axhline(0.01, color="k", ls="--", lw=0.8, label="lo=0.01")
        ax.axhline(0.45, color="gray", ls="--", lw=0.8, label="hi=0.45")
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=8)
    ax.set_ylabel("mean mitigation rate")
    ax.set_title(f"L3 DICE Damage Avoidance  {'✓' if results.get('l3', {}).get('passed') else '✗'}")

    # ── L4: both channels ─────────────────────────────────────────────────
    ax = axes[1, 1]
    if "l2" in results and "l4" in results:
        mu2 = results["l2"]["mean_mu"]
        mu4 = results["l4"]["mean_mu"]
        ax.bar(["L2 (μ only)", "L4 (μ + diversion)"], [mu2, mu4],
               color=["#95a5a6", _color("l4")], edgecolor="k", linewidth=0.8)
        thresh = results["l4"]["threshold"]
        ax.axhline(thresh, color="k", ls="--", lw=0.8, label=f"threshold={thresh:.2f}")
        ax.set_ylim(0, max(mu2, mu4) * 1.4)
        ax.legend(fontsize=8)
    ax.set_ylabel("mean mitigation rate")
    ax.set_title(f"L4 Both Channels  {'✓' if results.get('l4', {}).get('passed') else '✗'}")

    # ── Summary panel ─────────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    rows = [
        ("l1w", "L1w  welfloss"),
        ("l1r", "L1r  RCPO"),
        ("l2",  "L2   mitigation"),
        ("l3",  "L3   DICE"),
        ("l4",  "L4   both channels"),
    ]
    lines = []
    for k, label in rows:
        if k in results:
            sym = "✓" if results[k]["passed"] else "✗"
            extra = ""
            if k in ("l1w", "l1r"):
                extra = f"  ({results[k]['drop_pp']:.1f} pp)"
            lines.append(f"{sym}  {label}{extra}")
    ax.text(0.05, 0.92, "\n".join(lines), transform=ax.transAxes,
            fontsize=10, va="top", fontfamily="monospace")
    ax.set_title("Summary")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global TOTAL_TIMESTEPS, SEED, RCPO_ETA_LAMBDA, RCPO_ALPHA_TARGET, WELFARE_LOSS_WEIGHT, L1_TIMESTEPS

    parser = argparse.ArgumentParser(description="Convergence litmus tests")
    parser.add_argument("--timesteps",    type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--eta-lambda",   type=float, default=RCPO_ETA_LAMBDA)
    parser.add_argument("--alpha-target", type=float, default=RCPO_ALPHA_TARGET)
    parser.add_argument("--welfare-loss-weight", type=float, default=WELFARE_LOSS_WEIGHT,
                        help="welfare_loss_per_unit_tariff")
    parser.add_argument("--l1-timesteps", type=int, default=None,
                        help="Override timesteps for L1 only (default: same as --timesteps)")
    parser.add_argument("--tests", nargs="+", choices=["l1","l2","l3","l4"],
                        default=["l1","l2","l3","l4"],
                        help="Which litmus tests to run")
    args = parser.parse_args()

    TOTAL_TIMESTEPS   = args.timesteps
    SEED              = args.seed
    RCPO_ETA_LAMBDA   = args.eta_lambda
    RCPO_ALPHA_TARGET = args.alpha_target
    L1_TIMESTEPS      = args.l1_timesteps
    WELFARE_LOSS_WEIGHT = args.welfare_loss_weight
    # propagate welfare_loss_weight into _BASE_ENV
    _BASE_ENV["welfare_loss_per_unit_tariff"] = WELFARE_LOSS_WEIGHT

    key = jax.random.PRNGKey(SEED)
    num_iters = TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS

    print(f"\n{'═'*55}")
    print(f"  Convergence Litmus Tests — 3-region, emissions-simple")
    print(f"  {TOTAL_TIMESTEPS:,} steps ({num_iters} iters) | seed={SEED}")
    print(f"  η_λ={RCPO_ETA_LAMBDA:.0e}  α_target={RCPO_ALPHA_TARGET}  welfare_loss_wt={WELFARE_LOSS_WEIGHT}")
    if L1_TIMESTEPS:
        print(f"  L1 timesteps override: {L1_TIMESTEPS:,}")
    print(f"  Tests: {args.tests}")
    print(f"{'═'*55}")

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    l2_mu = 0.0  # needed for L4 threshold

    if "l1" in args.tests:
        passed_w, data_w = run_l1(key, reward_mode="welfloss")
        results["l1w"] = {**data_w, "passed": passed_w}
        passed_r, data_r = run_l1(key, reward_mode="additive_cbam")
        results["l1r"] = {**data_r, "passed": passed_r}

    if "l2" in args.tests:
        passed, data = run_l2(key)
        results["l2"] = {**data, "passed": passed}
        l2_mu = data["mean_mu"]

    if "l3" in args.tests:
        passed, data = run_l3(key)
        results["l3"] = {**data, "passed": passed}

    if "l4" in args.tests:
        passed, data = run_l4(key, l2_mu)
        results["l4"] = {**data, "passed": passed}

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print("  LITMUS TEST SUMMARY")
    print(f"{'═'*55}")
    descriptions = {
        "l1w": "Export diversion  welfloss  (τ=0 vs τ=0.8, >5 pp drop)",
        "l1r": "Export diversion  RCPO      (τ=0 vs τ=0.8, >5 pp drop)",
        "l2":  "Mitigation signal (free abat. + CBAM → μ > 0.15)",
        "l3":  "DICE baseline     (no CBAM, costly → 0.01 < μ < 0.99)",
        "l4":  "Both channels     (diversion doesn't kill mitigation)",
    }
    all_passed = True
    for tid in ["l1w", "l1r", "l2", "l3", "l4"]:
        if tid not in results:
            continue
        r = results[tid]
        tag = "PASS ✓" if r["passed"] else "FAIL ✗"
        if not r["passed"]:
            all_passed = False
        print(f"  [{tag}]  {descriptions[tid]}")
    print(f"\n  Overall: {'ALL PASS ✓' if all_passed else 'SOME FAILURES ✗'}")

    # ── Plot ──────────────────────────────────────────────────────────────
    if all(k in results for k in ["l1w", "l1r", "l2", "l3", "l4"]):
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = _os.path.join(OUTPUT_DIR, f"litmus_test_{ts_str}.png")
        _plot_results(results, out_path)


if __name__ == "__main__":
    main()
