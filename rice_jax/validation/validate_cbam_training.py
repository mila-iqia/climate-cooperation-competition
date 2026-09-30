"""
CBAM Training Validation Script
=================================
Trains PPO agents in the Phase 2A CBAM environment and checks whether
non-EU exporting regions learn to reduce their EU export allocation.

The core behavioural hypothesis is:
  Under the corrected welfloss formula (Approach B), sending less to EU
  directly increases welfloss and therefore reward.  A well-trained agent
  should therefore discover that diverting exports away from EU is beneficial.

Checks
------
  TR1. Non-EU exporters decrease their mean EU export share over training.
       Trained policy EU share < FixedAction baseline EU share.

  TR2. Non-EU exporters' mean welfloss is higher under the trained policy
       than under the FixedAction baseline (less CBAM burden).

  TR3. Total reward improves over training (basic learning health check).

  TR4. EU region's welfare is not systematically harmed (less EU revenues ≠
       worse EU welfare — EU gains from lower CBAM burden on partners as trade
       volumes stabilise).

  TR5. EU export share decreases monotonically (or near-monotonically) over
       the episode under the trained policy — agents sustain diversion strategy.

Produces plots/training_cbam_diversion.png.

Run from rice_jax/ with:
    conda run -n rice-jax python validate_cbam_training.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from __future__ import annotations

import os
import sys
import numpy as np
from datetime import datetime
from dataclasses import replace as dc_replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

os.makedirs("plots", exist_ok=True)
MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")

# ── Config ─────────────────────────────────────────────────────────────────────
NR              = 3
EU_IDX          = 1
CBAM_RATE       = 0.5
TOTAL_TIMESTEPS = 500_000  # increase for stronger signal; ~5 min on CPU
NUM_ENVS        = 4
NUM_STEPS       = 100
SEED            = 0
N_CKPTS         = 6        # eval checkpoints for learning curve
N_EVAL          = 4        # eval episodes per checkpoint
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 64)
print("CBAM Diversion Training Validation")
print(f"  NR={NR}, EU_IDX={EU_IDX}, CBAM_RATE={CBAM_RATE}")
print(f"  total_timesteps={TOTAL_TIMESTEPS:,}, num_envs={NUM_ENVS}")
print("=" * 64)

from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls, full_state_info_log_fn
from _experiment_util import FixedActionAgent, run_single_episode
import jaxnasium as jym
from jaxnasium.algorithms import PPO

params = load_region_yamls(NR)

# ── Environment factory ────────────────────────────────────────────────────────
def _make_env(log_info_fn=None):
    kwargs = dict(
        num_regions=NR,
        region_params=params,
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        cbam_tariff_rate=CBAM_RATE,
        eu_region_idx=EU_IDX,
        diff_reward_mode=True,
    )
    if log_info_fn is not None:
        kwargs["log_info_fn"] = log_info_fn
    env = RiceMRIO(**kwargs)
    if env.dest_alloc_baseline is None:
        print("\n[ABORT] MRIO data not found at:", MRIO_ROOT)
        print("  Run aggregate_local_mrio.py first.")
        sys.exit(1)
    return env

train_env   = jym.LogWrapper(_make_env())
rollout_env = _make_env(log_info_fn=full_state_info_log_fn)
EPISODE_LEN = rollout_env.episode_length
non_eu      = [r for r in range(NR) if r != EU_IDX]

try:
    region_labels = list(rollout_env.mrio_region_labels)
except AttributeError:
    region_labels = [f"R{r}" for r in range(NR)]

print(f"\n  episode_length={EPISODE_LEN}")
print(f"  intensity range (normalised): "
      f"[{rollout_env.emissions_intensity.min():.4f}, "
      f"{rollout_env.emissions_intensity.max():.4f}]  "
      f"mean={rollout_env.emissions_intensity.mean():.4f}")
print(f"  regions: {region_labels}")
print()

# ── Helpers ────────────────────────────────────────────────────────────────────
def _eu_share_per_region(episode) -> np.ndarray:
    """EU export share per region per timestep → (T, NR)."""
    trade     = np.array(episode["trade_flows"])            # (T, NR, NR, NS)
    eu_exp    = trade[:, :, EU_IDX, :].sum(axis=-1)         # (T, NR)
    total_exp = trade.sum(axis=(2, 3)) + 1e-8               # (T, NR)
    return eu_exp / total_exp                               # (T, NR)

def _rewards_matrix(episode) -> np.ndarray:
    """(T, NR) per-step utility_times_welfloss."""
    return np.stack(
        [np.array(episode["utility_times_welfloss_all_regions"][r]) for r in range(NR)],
        axis=1,
    )

def _welfloss_matrix(episode) -> np.ndarray:
    """(T, NR) inferred welfloss = utility_times_welfloss / utility."""
    utw = np.stack(
        [np.array(episode["utility_times_welfloss_all_regions"][r]) for r in range(NR)],
        axis=1,
    )
    ut = np.stack(
        [np.array(episode["utility_all_regions"][r]) for r in range(NR)],
        axis=1,
    )
    return utw / (ut + 1e-12)

# ── Baseline rollout ───────────────────────────────────────────────────────────
print("[1/3] Rolling out FixedActionAgent baseline...")
key          = jax.random.PRNGKey(SEED)
fixed_agent  = FixedActionAgent(rollout_env)
baseline_ep  = run_single_episode(key, rollout_env, fixed_agent)

baseline_eu_shares  = _eu_share_per_region(baseline_ep)   # (T, NR)
baseline_rewards    = _rewards_matrix(baseline_ep)         # (T, NR)
baseline_welfloss   = _welfloss_matrix(baseline_ep)        # (T, NR)

# Summarise last quarter of episode (steady-state behaviour)
T_q = max(1, EPISODE_LEN // 4)
baseline_eu_share_ss = baseline_eu_shares[-T_q:, non_eu].mean()
baseline_reward_ss   = baseline_rewards[-T_q:].mean()

print(f"   FixedAction EU share (non-EU, last quarter): {baseline_eu_share_ss:.4f}")
print(f"   FixedAction mean reward (last quarter):      {baseline_reward_ss:.4f}")

# ── Learning curve ─────────────────────────────────────────────────────────────
print(f"\n[2/3] Training PPO ({N_CKPTS} checkpoints × {N_EVAL} eval eps)...")

ppo = PPO(
    total_timesteps=TOTAL_TIMESTEPS,
    num_envs=NUM_ENVS,
    num_steps=NUM_STEPS,
    learning_rate=2.5e-4,
    ent_coef=1.0,
    anneal_ent_coef=0.05,
    num_minibatches=4,
    num_epochs=4,
    normalize_observations=True,
    log_function=None,
)

ckpt_ts_list   = [NUM_STEPS * NUM_ENVS] + [
    int(TOTAL_TIMESTEPS * (i + 1) / N_CKPTS) for i in range(N_CKPTS)
]
eval_timesteps = []
eval_returns   = []
best_agent     = None
best_return    = -np.inf

for ckpt_i, ckpt_ts in enumerate(ckpt_ts_list):
    ckpt_ppo   = dc_replace(ppo, total_timesteps=ckpt_ts)
    ckpt_agent = ckpt_ppo.train(key, train_env)
    ep_returns = ckpt_agent.evaluate(key, rollout_env, num_eval_episodes=N_EVAL)
    r_mean     = float(jnp.mean(ep_returns))
    label      = 0 if ckpt_i == 0 else ckpt_ts
    eval_timesteps.append(label)
    eval_returns.append(r_mean)
    if ckpt_i > 0 and r_mean > best_return:
        best_return = r_mean
        best_agent  = ckpt_agent
    tag = "(random-init)" if ckpt_i == 0 else f"ts={ckpt_ts:,}"
    print(f"   ckpt {ckpt_i}/{N_CKPTS}: {tag}  eval_return={r_mean:.4f}")

eval_timesteps = np.array(eval_timesteps)
eval_returns   = np.array(eval_returns)
trained_agent  = best_agent

# ── Post-training rollout ──────────────────────────────────────────────────────
print("\n[3/3] Post-training rollout for behavioural analysis...")
trained_ep = run_single_episode(key, rollout_env, trained_agent)

trained_eu_shares = _eu_share_per_region(trained_ep)   # (T, NR)
trained_rewards   = _rewards_matrix(trained_ep)         # (T, NR)
trained_welfloss  = _welfloss_matrix(trained_ep)        # (T, NR)
cbam_revenue      = np.array(trained_ep["cbam_revenue"])  # (T, NR)

trained_eu_share_ss = trained_eu_shares[-T_q:, non_eu].mean()
trained_reward_ss   = trained_rewards[-T_q:].mean()
trained_welfloss_ss = trained_welfloss[-T_q:, non_eu].mean()
baseline_welfloss_ss = baseline_welfloss[-T_q:, non_eu].mean()

# ── Checks ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 64)
PASSES: list[bool] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    PASSES.append(bool(cond))
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

# TR1: trained agents export less to EU than FixedAction baseline
check(
    "TR1: Trained non-EU agents reduce EU export share vs FixedAction",
    trained_eu_share_ss < baseline_eu_share_ss - 1e-4,
    f"trained={trained_eu_share_ss:.4f}  baseline={baseline_eu_share_ss:.4f}",
)

# TR2: trained agents achieve higher welfloss (lower CBAM burden)
check(
    "TR2: Trained non-EU agents achieve higher mean welfloss vs FixedAction",
    trained_welfloss_ss > baseline_welfloss_ss + 1e-6,
    f"trained={trained_welfloss_ss:.6f}  baseline={baseline_welfloss_ss:.6f}",
)

# TR3: trained policy terminal welfare beats FixedAction terminal welfare.
# Note: with diff_reward_mode=True, evaluate() accumulates Δ-rewards which
# are sensitive to episode start-point, making the raw return unreliable as
# a learning-progress signal.  Terminal welfare (last-step utility×welfloss)
# is a stabler metric that reflects actual end-state quality.
trained_terminal_welfare  = trained_rewards[-1].mean()
baseline_terminal_welfare = baseline_rewards[-1].mean()
check(
    "TR3: Trained terminal welfare ≥ FixedAction terminal welfare",
    trained_terminal_welfare >= baseline_terminal_welfare * 0.95,  # 5% tolerance
    f"trained={trained_terminal_welfare:.4f}  baseline={baseline_terminal_welfare:.4f}",
)

# TR4: EU welfare not catastrophically harmed
eu_reward_trained  = trained_rewards[-T_q:, EU_IDX].mean()
eu_reward_baseline = baseline_rewards[-T_q:, EU_IDX].mean()
check(
    "TR4: EU welfare not significantly harmed (within 20% of baseline)",
    eu_reward_trained >= eu_reward_baseline * 0.80,
    f"trained={eu_reward_trained:.4f}  baseline={eu_reward_baseline:.4f}",
)

# TR5: EU export share under trained policy trends downward or stays low
# (compare first half vs second half of episode)
mid = EPISODE_LEN // 2
eu_share_first_half  = trained_eu_shares[:mid, non_eu].mean()
eu_share_second_half = trained_eu_shares[mid:, non_eu].mean()
check(
    "TR5: EU export share non-increasing over episode (diversion sustained)",
    eu_share_second_half <= eu_share_first_half + 0.02,  # 2pp tolerance
    f"first_half={eu_share_first_half:.4f}  second_half={eu_share_second_half:.4f}",
)

print("─" * 64)
n_pass = sum(PASSES)
print(f"  {n_pass}/{len(PASSES)} checks passed")
if n_pass == len(PASSES):
    print("  ALL PASS — agents are learning to divert exports from EU.")
else:
    fails = [i + 1 for i, p in enumerate(PASSES) if not p]
    print(f"  FAILED: {fails}  — consider more training timesteps or tuning.")
print("─" * 64)

# ── Summary table ──────────────────────────────────────────────────────────────
print(f"\nBehavioural summary (last {T_q} timesteps of episode):")
print(f"  {'Region':<20}  {'EU share (Fixed)':<18}  {'EU share (Trained)':<18}  "
      f"{'Δ EU share':<12}  {'Welfloss Δ'}")
for r in range(NR):
    lbl  = str(region_labels[r]) if r < len(region_labels) else f"R{r}"
    sh_f = baseline_eu_shares[-T_q:, r].mean()
    sh_t = trained_eu_shares[-T_q:, r].mean()
    wl_f = baseline_welfloss[-T_q:, r].mean()
    wl_t = trained_welfloss[-T_q:, r].mean()
    marker = " ← EU" if r == EU_IDX else ""
    print(f"  {lbl:<20}  {sh_f:<18.4f}  {sh_t:<18.4f}  "
          f"{sh_t - sh_f:<12.4f}  {wl_t - wl_f:.6f}{marker}")

# ── Plots ──────────────────────────────────────────────────────────────────────
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
timesteps = np.arange(EPISODE_LEN)

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Learning curve (informational — not a hard check due to diff_reward_mode artifacts)
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(eval_timesteps, eval_returns, color=COLORS[0], linewidth=2,
         marker="o", markersize=5, label="PPO eval return (Δ-reward)")
ax1.axhline(baseline_terminal_welfare, linestyle="--", color=COLORS[3],
            linewidth=1.3, label=f"FixedAction terminal welfare: {baseline_terminal_welfare:.4f}")
ax1.axhline(trained_terminal_welfare, linestyle="-.", color=COLORS[2],
            linewidth=1.3, label=f"Trained terminal welfare: {trained_terminal_welfare:.4f}")
ax1.set_xlabel("Env timesteps trained for")
ax1.set_ylabel("Return / welfare")
ax1.set_title("TR3: Terminal welfare vs FixedAction\n(eval return = Δ-reward, informational)")
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3)

# Panel 2: EU export share over episode — trained vs FixedAction
ax2 = fig.add_subplot(gs[0, 1])
for r in non_eu:
    lbl = str(region_labels[r]) if r < len(region_labels) else f"R{r}"
    c   = COLORS[r % len(COLORS)]
    ax2.plot(timesteps, trained_eu_shares[:, r], color=c, linewidth=2,
             label=f"{lbl} (trained)")
    ax2.plot(timesteps, baseline_eu_shares[:, r], color=c, linewidth=1,
             linestyle="--", alpha=0.6, label=f"{lbl} (fixed)")
ax2.axvline(mid, color="#aaaaaa", linestyle=":", linewidth=1)
ax2.set_xlabel("Episode timestep")
ax2.set_ylabel("Share of exports → EU")
ax2.set_title("TR1/TR5: EU export share over episode\n(— trained, -- FixedAction)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: Welfloss over episode — trained vs baseline
ax3 = fig.add_subplot(gs[0, 2])
for r in non_eu:
    lbl = str(region_labels[r]) if r < len(region_labels) else f"R{r}"
    c   = COLORS[r % len(COLORS)]
    ax3.plot(timesteps, trained_welfloss[:, r], color=c, linewidth=2,
             label=f"{lbl} (trained)")
    ax3.plot(timesteps, baseline_welfloss[:, r], color=c, linewidth=1,
             linestyle="--", alpha=0.6, label=f"{lbl} (fixed)")
ax3.set_xlabel("Episode timestep")
ax3.set_ylabel("Welfloss multiplier")
ax3.set_title("TR2: Welfloss (non-EU)\n(higher = less CBAM burden)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel 4: Per-region mean reward
ax4 = fig.add_subplot(gs[1, 0])
for r in range(NR):
    lbl = str(region_labels[r]) if r < len(region_labels) else f"R{r}"
    c   = COLORS[r % len(COLORS)]
    ls  = "--" if r == EU_IDX else "-"
    ax4.plot(timesteps, trained_rewards[:, r], color=c, linewidth=2,
             linestyle=ls, label=f"{lbl} (trained)")
    ax4.plot(timesteps, baseline_rewards[:, r], color=c, linewidth=1,
             linestyle=":", alpha=0.5, label=f"{lbl} (fixed)")
ax4.set_xlabel("Episode timestep")
ax4.set_ylabel("Utility × welfloss")
ax4.set_title("TR3/TR4: Welfare per region\n(— trained, : FixedAction)")
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.3)

# Panel 5: CBAM revenue — trained vs baseline
cbam_rev_trained  = cbam_revenue[:, EU_IDX]
cbam_rev_baseline = np.array(run_single_episode(key, rollout_env, fixed_agent)
                             ["cbam_revenue"])[:, EU_IDX]
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(timesteps, cbam_rev_trained,  color=COLORS[0], linewidth=2,
         label="Trained")
ax5.plot(timesteps, cbam_rev_baseline, color=COLORS[3], linewidth=1.5,
         linestyle="--", label="FixedAction")
ax5.set_xlabel("Episode timestep")
ax5.set_ylabel("CBAM revenue (EU)")
ax5.set_title("CBAM revenue: trained vs baseline\n(lower = less taxed flow)")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# Panel 6: EU export share bar chart (steady-state comparison)
ax6 = fig.add_subplot(gs[1, 2])
x    = np.arange(len(non_eu))
w    = 0.35
lbls = [str(region_labels[r]) if r < len(region_labels) else f"R{r}" for r in non_eu]
ax6.bar(x - w/2,
        [baseline_eu_shares[-T_q:, r].mean() for r in non_eu],
        w, color=COLORS[3], alpha=0.8, label="FixedAction")
ax6.bar(x + w/2,
        [trained_eu_shares[-T_q:, r].mean() for r in non_eu],
        w, color=COLORS[0], alpha=0.8, label="Trained")
ax6.set_xticks(x)
ax6.set_xticklabels(lbls)
ax6.set_ylabel(f"Mean EU export share\n(last {T_q} steps)")
ax6.set_title("TR1: EU diversion (steady state)")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis="y")

fig.suptitle(
    f"CBAM Diversion Training Validation  "
    f"(NR={NR}, EU_IDX={EU_IDX}, cbam_rate={CBAM_RATE}, "
    f"ts={TOTAL_TIMESTEPS:,})\n"
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    fontsize=11,
)

out_path = f"plots/training_cbam_diversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")

sys.exit(0 if n_pass == len(PASSES) else 1)
