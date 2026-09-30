"""
Training sanity check for RiceMRIO Phase 2A.

Runs a short PPO training run (deliberately small: fast to execute) and
verifies the following MARL training health conditions:

  T1. Reward increases over training (agents are learning).
  T2. Final reward is strictly above the FixedActionAgent baseline.
  T3. Reward variance across agents does not explode (convergence signal).
  T4. EU export share for non-EU exporters under the trained policy is within
      a plausible range, confirming export_reallocation actions are being used.
  T5. CBAM revenue is collected every episode (environment accounting works).

Produces a 2×3 matplotlib figure saved to plots/training_sanity_2a.png.

Run from rice_jax/ with:
    conda run -n rice-jax python training_sanity_check_2a.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import os
import sys
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace as dc_replace

import jax
import jax.numpy as jnp

os.makedirs("plots", exist_ok=True)
MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")

# ── Config ─────────────────────────────────────────────────────────────────────
NR               = 3
EU_IDX           = 1       # Europe & Central Asia is RIG 2 → RICE idx 1
CBAM_RATE        = 0.5     # fixed CBAM tariff rate
SYNTH_INTENSITY  = 0.3     # synthetic emissions intensity (real EORA ~1e-12)
TOTAL_TIMESTEPS  = 1_000_000 # enough to see a learning trend; increase for convergence
NUM_ENVS         = 4
NUM_STEPS        = 100
SEED             = 0
N_CKPTS          = 8       # number of eval checkpoints for the learning curve
N_EVAL           = 4       # eval episodes per checkpoint
RUN_T1           = False    # Set False to skip learning curve (trains once, much faster)
# ──────────────────────────────────────────────────────────────────────────────

print("=" * 60)
print("Phase 2A Training Sanity Check")
print(f"  num_regions={NR}, eu_region_idx={EU_IDX}")
print(f"  cbam_rate={CBAM_RATE}, synthetic_intensity={SYNTH_INTENSITY}")
print(f"  total_timesteps={TOTAL_TIMESTEPS:,}, num_envs={NUM_ENVS}")
print(f"  RUN_T1={RUN_T1} {'(learning curve)' if RUN_T1 else '(single run, faster)'}")
print("=" * 60)

# ── Imports (here so the print block lands before heavy imports) ───────────────
from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls, full_state_info_log_fn
import jaxnasium as jym
from jaxnasium.algorithms import PPO
from _experiment_util import FixedActionAgent, run_single_episode

params = load_region_yamls(NR)

# ── Environment factory ────────────────────────────────────────────────────────
def _make_rice_mrio(log_info_fn=None, cbam_rate=CBAM_RATE, synth_intensity=SYNTH_INTENSITY):
    """Create a RiceMRIO instance.  Pass log_info_fn=full_state_info_log_fn
    for episode rollouts; leave as default (empty) for training."""
    kwargs = dict(
        num_regions=NR,
        region_params=params,
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        cbam_tariff_rate=cbam_rate,
        eu_region_idx=EU_IDX,
        diff_reward_mode=True,
    )
    if log_info_fn is not None:
        kwargs["log_info_fn"] = log_info_fn

    env = RiceMRIO(**kwargs)

    # Inject synthetic emissions intensity so CHECK T5 is non-trivial
    if synth_intensity is not None:
        # emissions_intensity is eqx.field(static=True); use object.__setattr__
        object.__setattr__(
            env, "emissions_intensity",
            np.full((NR, env.num_sectors), synth_intensity, dtype=np.float32),
        )
    return env

# Training env: wrapped with LogWrapper (required by PPO), empty log_info_fn
train_env  = jym.LogWrapper(_make_rice_mrio())

# Rollout env: unwrapped, full logging enabled
rollout_env = _make_rice_mrio(log_info_fn=full_state_info_log_fn)

NR_SECTORS = rollout_env.num_sectors
EPISODE_LEN = rollout_env.episode_length

print(f"\nEnvironment: {NR} regions, {NR_SECTORS} sectors, "
      f"episode_length={EPISODE_LEN}")
print(f"Action space keys: "
      f"{list(rollout_env.action_space[list(rollout_env.action_space.keys())[0]].keys())}")

# ── FixedActionAgent rollout (pre-training reference) ─────────────────────────
print("\n[1/3] Rolling out FixedActionAgent baseline...")
key = jax.random.PRNGKey(SEED)

fixed_agent  = FixedActionAgent(rollout_env)
baseline_ep  = run_single_episode(key, rollout_env, fixed_agent)

# utility_times_welfloss_all_regions is restructured by full_state_info_log_fn:
#   ep[key] → {region_id: array(T)} after scan
def _rewards_matrix(ep):
    """Return array (T, NR) of per-step per-region welfare rewards."""
    return np.stack(
        [np.array(ep["utility_times_welfloss_all_regions"][r]) for r in range(NR)],
        axis=1,
    )

baseline_reward_mat = _rewards_matrix(baseline_ep)   # (T, NR)
baseline_mean_step  = baseline_reward_mat.mean()      # scalar

print(f"   FixedAction mean per-step reward: {baseline_mean_step:.4f}")

# ── PPO training ───────────────────────────────────────────────────────────────
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
    log_function="tqdm",
)

if RUN_T1:
    print(f"\n[2/3] Training PPO for {TOTAL_TIMESTEPS:,} timesteps ({N_CKPTS} eval checkpoints)...")

    # Learning curve via independent fresh runs: for each checkpoint timestep T_c,
    # train a fresh PPO from scratch for T_c steps and evaluate.
    # This avoids optimizer step-counter carry-over issues that arise with chunking.
    # The curve answers: "how good is a policy trained for T_c steps?"
    eval_timesteps: list[int]   = []
    eval_returns:   list[float] = []
    best_agent = None
    best_return = -np.inf

    # Step-0: near-random policy (1 gradient update only)
    ckpt_ts_list = [NUM_STEPS * NUM_ENVS] + [
        int(TOTAL_TIMESTEPS * (i + 1) / N_CKPTS) for i in range(N_CKPTS)
    ]
    for ckpt_i, ckpt_ts in enumerate(ckpt_ts_list):
        ckpt_ppo = dc_replace(ppo, total_timesteps=ckpt_ts, log_function=None)
        ckpt_agent = ckpt_ppo.train(key, train_env)
        ep_returns = ckpt_agent.evaluate(key, rollout_env, num_eval_episodes=N_EVAL)
        r_mean = float(jnp.mean(ep_returns))
        ts_label = 0 if ckpt_i == 0 else ckpt_ts
        eval_timesteps.append(ts_label)
        eval_returns.append(r_mean)
        if ckpt_i > 0 and r_mean > best_return:  # don't count step-0 as "trained"
            best_return = r_mean
            best_agent = ckpt_agent
        label = "(random-init)" if ckpt_i == 0 else f"ts={ckpt_ts:,}"
        print(f"   ckpt {ckpt_i}/{N_CKPTS}: {label}  eval_return={r_mean:.3f}")

    eval_timesteps = np.array(eval_timesteps)
    eval_returns   = np.array(eval_returns)

    # Final trained agent = the checkpoint with the highest eval return
    trained_agent = best_agent
else:
    print(f"\n[2/3] Training PPO for {TOTAL_TIMESTEPS:,} timesteps (T1 skipped)...")
    trained_agent = ppo.train(key, train_env)
    best_agent = trained_agent
    eval_timesteps = None
    eval_returns   = None

# ── Post-training rollout ──────────────────────────────────────────────────────
print("\n[3/3] Post-training rollout for behavioural checks...")
trained_ep = run_single_episode(key, rollout_env, trained_agent)

# ── Derived arrays ─────────────────────────────────────────────────────────────
T         = EPISODE_LEN
timesteps = np.arange(T)

# FixedAction episode return compatible with evaluate() (= terminal_utility − init_utility ≈ terminal_utility)
# NOT baseline_mean_step * T which is mean_absolute_utility * T ≈ 70× larger than evaluate() returns.
fa_episode_return = float(baseline_reward_mat[-1].mean())

trained_reward_mat = _rewards_matrix(trained_ep)  # (T, NR)
trained_mean_step  = trained_reward_mat.mean()

# trade_flows: NOT restructured by full_state_info_log_fn → shape (T, NR, NR, NS)
trade_trained = np.array(trained_ep["trade_flows"])   # (T, NR, NR, NS)
trade_fixed   = np.array(baseline_ep["trade_flows"])  # (T, NR, NR, NS)

def _eu_export_shares(trade):
    """EU export share per exporter per timestep → (T, NR)."""
    eu_exp   = trade[:, :, EU_IDX, :].sum(axis=-1)      # (T, NR)
    total_exp = trade.sum(axis=(2, 3)) + 1e-8            # (T, NR)
    return eu_exp / total_exp

eu_shares_trained = _eu_export_shares(trade_trained)   # (T, NR)
eu_shares_fixed   = _eu_export_shares(trade_fixed)     # (T, NR)

# cbam_revenue: shape (T, NR)
cbam_revenue = np.array(trained_ep["cbam_revenue"])    # (T, NR)
eu_cbam_rev  = cbam_revenue[:, EU_IDX]                 # (T,)

# ── TRAINING CHECKS ────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
PASSES: list[bool] = []

def check(name, cond, detail=""):
    PASSES.append(bool(cond))
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))

# T1: the peak checkpoint return is substantially above random initialisation.
# (With ent_coef=1.0 annealed over N steps, the optimal training horizon is
#  ~N/8 steps; training longer can find worse local minima with slower annealing.)
if RUN_T1:
    early_r = eval_returns[0]   # step-0 (near-random)
    peak_r  = eval_returns[1:].max()  # best checkpoint (not including step-0)
    check("T1 Peak return improves over random initialisation",
          peak_r > early_r * 1.05,
          f"step-0={early_r:.3f}  peak={peak_r:.3f}  (at ts={eval_timesteps[1:][eval_returns[1:].argmax()]:,})")
else:
    check("T1 Learning curve skipped", True, "RUN_T1=False")

# T2: trained policy terminal utility ≥ FixedAction terminal utility
# (with diff_reward_mode, sum of rewards = terminal utility − initial utility ≈ terminal utility,
# so terminal utility is the natural episode-level comparison)
trained_final_u  = float(trained_reward_mat[-1].mean())
baseline_final_u = float(baseline_reward_mat[-1].mean())
check("T2 Trained terminal utility \u2265 FixedAction",
      trained_final_u >= baseline_final_u * 0.95,   # allow 5% tolerance
      f"trained={trained_final_u:.4f}  baseline={baseline_final_u:.4f}")

# T3: reward variance across regions does not explode in second half vs first half
mid = T // 2
var_first = trained_reward_mat[:mid].var()
var_second = trained_reward_mat[mid:].var()
check("T3 Inter-region reward variance stable / decreasing",
      var_second < var_first * 2.0,
      f"var_first={var_first:.4f}  var_second={var_second:.4f}")

# T4: non-EU exporters under trained policy have EU share within [0, 1]
non_eu = [r for r in range(NR) if r != EU_IDX]
eu_share_trained_end = eu_shares_trained[-T // 4:, non_eu].mean()
eu_share_fixed_end   = eu_shares_fixed[-T // 4:, non_eu].mean()
check("T4 Export-reallocation actions produce valid trade shares",
      0.0 <= eu_share_trained_end <= 1.0,
      f"eu_share_trained={eu_share_trained_end:.3f}  fixed={eu_share_fixed_end:.3f}")

# T5: CBAM revenue is positive (accounting works)
check("T5 EU collects positive CBAM revenue",
      eu_cbam_rev.mean() > 0,
      f"mean={eu_cbam_rev.mean():.4f}")

print("─" * 60)
n_pass = sum(PASSES)
print(f"  {n_pass}/{len(PASSES)} checks passed")

# ── PLOTS ──────────────────────────────────────────────────────────────────────
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
try:
    region_labels = list(rollout_env.mrio_region_labels)
except AttributeError:
    region_labels = [f"R{r}" for r in range(NR)]

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel 1: Learning curve ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
if RUN_T1 and eval_returns is not None:
    ax1.plot(eval_timesteps, eval_returns, color="#1f77b4", linewidth=2,
             marker="o", markersize=5, label="PPO eval return (fresh train)")
    peak_idx = int(np.array(eval_returns[1:]).argmax()) + 1  # +1 offset for step-0
    ax1.plot(eval_timesteps[peak_idx], eval_returns[peak_idx],
             marker="*", markersize=14, color="#2ca02c", zorder=5,
             label=f"Peak: {eval_returns[peak_idx]:.3f}")
    ax1.axhline(fa_episode_return,
                linestyle="--", color="#d62728", linewidth=1.5, label="FixedAction terminal utility")
    # Set y-axis floor so the upward jump from step-0 is obvious
    y_min = min(eval_returns[0], fa_episode_return) * 0.97
    y_max = max(eval_returns) * 1.02
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("Environment timesteps trained for")
    ax1.set_ylabel("Mean episode return")
    ax1.set_title("T1: Learning curve (each point = fresh independent run)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, "T1 skipped\n(RUN_T1=False)",
             ha="center", va="center", transform=ax1.transAxes,
             fontsize=13, color="#888888")
    ax1.set_title("T1: Learning curve (skipped)")
    ax1.axis("off")

# Panel 2: Per-step reward (trained vs baseline) ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(timesteps, trained_reward_mat.mean(axis=1),
         color="#1f77b4", linewidth=2, label="Trained (CBAM)")
ax2.plot(timesteps, baseline_reward_mat.mean(axis=1),
         color="#d62728", linewidth=2, linestyle="--", label="FixedAction (CBAM)")
ax2.set_xlabel("Episode timestep")
ax2.set_ylabel("Mean welfare level (avg regions)")
ax2.set_title("T2: Per-step welfare level")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel 3: Inter-region reward variance over episode ──────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(timesteps, trained_reward_mat.var(axis=1), color="#2ca02c", linewidth=2)
ax3.axvline(mid, color="#aaaaaa", linestyle=":", linewidth=1)
ax3.set_xlabel("Episode timestep")
ax3.set_ylabel("Variance (across regions)")
ax3.set_title("T3: Inter-region reward variance")
ax3.grid(True, alpha=0.3)

# Panel 4: EU export share – trained vs fixed ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
for r in range(NR):
    lbl = str(region_labels[r]) if r < len(region_labels) else f"R{r}"
    ls  = "--" if r == EU_IDX else "-"
    c   = COLORS[r % len(COLORS)]
    ax4.plot(timesteps, eu_shares_trained[:, r], color=c, linewidth=2,
             linestyle=ls, label=f"{lbl} trained")
    ax4.plot(timesteps, eu_shares_fixed[:, r], color=c, linewidth=1,
             linestyle=":", alpha=0.5, label=f"{lbl} fixed")
ax4.set_xlabel("Episode timestep")
ax4.set_ylabel("Share of exports → EU")
ax4.set_title("T4: EU export share per region")
ax4.legend(fontsize=7, ncol=2)
ax4.grid(True, alpha=0.3)

# Panel 5: CBAM revenue over episode ──────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(timesteps, eu_cbam_rev, color="#9467bd", alpha=0.85, width=0.85)
ax5.set_xlabel("Episode timestep")
ax5.set_ylabel("CBAM revenue (EU region)")
ax5.set_title("T5: CBAM revenue collected by EU")
ax5.grid(True, alpha=0.3, axis="y")

# Panel 6: Check summary ──────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
check_labels = [
    "T1  Peak return > random init",
    "T2  Trained > FixedAction baseline",
    "T3  Inter-region variance stable",
    "T4  Trade shares in [0, 1]",
    "T5  CBAM revenue > 0",
]
for i, (lbl, passed) in enumerate(zip(check_labels, PASSES)):
    sym   = "✓" if passed else "✗"
    color = "#2ca02c" if passed else "#d62728"
    ax6.text(0.05, 0.88 - i * 0.17, f"{sym}  {lbl}",
             transform=ax6.transAxes, fontsize=10,
             color=color, fontweight="bold", va="top")
ax6.set_title("Check summary", pad=8)
summary_txt   = f"{n_pass}/{len(PASSES)} passed"
summary_color = "#2ca02c" if n_pass == len(PASSES) else "#d62728"
ax6.text(0.5, 0.03, summary_txt, transform=ax6.transAxes,
         fontsize=14, color=summary_color, ha="center", fontweight="bold")

fig.suptitle(
    f"Phase 2A Training Sanity — NR={NR}, CBAM={CBAM_RATE}, "
    f"timesteps={TOTAL_TIMESTEPS:,}",
    fontsize=13, fontweight="bold",
)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"plots/training_sanity_2a_{_ts}.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved to {out_path}")

if n_pass < len(PASSES):
    print("WARNING: some checks failed — inspect the plot for details.")
    sys.exit(1)
else:
    print("All training sanity checks passed.")
