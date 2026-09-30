"""cbam_region_analysis.py

Post-hoc per-region analysis for CBAM experiments.

Two modes:

  Behavioural (works with any per_exporter pkl):
    - Response-type scatter: Δdirty vs Δμ across regions
    - Elasticity heatmap: region × rs-level for dirty_share and μ
    - Transfer efficiency: Δμ per unit of mean transfer received

  Policy / weight analysis (requires pkl with saved agents — re-run
  per_exporter.py after the --save-agents patch, or run --train-quick):
    - First-layer weight norms per obs group (which obs dims move the hidden layer)
    - Jacobian sensitivity heatmap: |∂action_mode/∂obs| averaged over a rollout
    - Integrated gradients for the mitigation action
    - Policy entropy per region (agent confidence)

Obs vector layout (53 dims, alphabetical key order after JAX pytree flatten):
  [0]      activity_timestep
  [1]      cbam_cost
  [2]      cbam_lambda
  [3:12]   cbam_revenue[0:9]      (all 9 regions)
  [12]     cbam_tariff_rate
  [13:31]  dest_alloc (NS=2 × NR=9, row-major)
  [31]     gross_output
  [32]     revenue_share
  [33:51]  trade_flows (NR=9 × NS=2, row-major)
  [51]     transfer_received
  [52]     utility

Usage (from rice_jax/, rice-jax conda env):

  # Behavioural analysis only (works with old per_exporter pkls):
  python validation/cbam_region_analysis.py \\
      --pkl plots/cbam_per_exporter_20260508_160427.pkl

  # Full analysis (requires pkl with 'ppo' saved per run):
  python validation/cbam_region_analysis.py \\
      --pkl plots/cbam_per_exporter_<timestamp>.pkl --jacobian

  # Train a quick single model for network analysis only (rs=0 baseline):
  python validation/cbam_region_analysis.py \\
      --train-quick --timesteps 500000
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
# Insert rice_jax/ first so `import training_monitor` resolves to
# rice_jax/training_monitor.py, not the stub in validation/.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# validation/ added later inside functions that need _experiment_util

import argparse
import pickle
import time
from dataclasses import replace
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patches as mpatches

import jax
import jax.numpy as jnp

from rice_jax.utils import load_region_yamls
from rice_jax import RiceMRIO


# ── Config ────────────────────────────────────────────────────────────────────

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
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]   # exclude RoW

EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)

NUM_EVAL_EPISODES = 4
JACOBIAN_EPISODES = 2    # episodes used for Jacobian averaging
OUTPUT_DIR = "plots"

# Obs dimension label groups (alphabetical flatten order, 53 dims total)
OBS_GROUPS = [
    ("timestep",         [0]),
    ("cbam_cost",        [1]),
    ("cbam_lambda",      [2]),
    ("cbam_revenue",     list(range(3, 12))),
    ("cbam_tariff_rate", [12]),
    ("dest_alloc",       list(range(13, 31))),
    ("gross_output",     [31]),
    ("revenue_share",    [32]),
    ("trade_flows",      list(range(33, 51))),
    ("transfer_rcvd",    [51]),
    ("utility",          [52]),
]
OBS_DIM  = 53
OBS_LABELS = [""] * OBS_DIM
for name, idxs in OBS_GROUPS:
    for idx in idxs:
        OBS_LABELS[idx] = name if len(idxs) == 1 else f"{name}[{idx - idxs[0]}]"

# Coarse group boundaries for weight norm plot
OBS_GROUP_SLICES = {name: idxs for name, idxs in OBS_GROUPS}


# ── Environment builder ───────────────────────────────────────────────────────

def _build_env(revenue_share=0.0, cbam_tariff_rate=0.0, for_training=True):
    from jaxnasium import LogWrapper
    from training_monitor import rcpo_cbam_log_info_fn
    from rice_jax.utils import full_state_info_log_fn

    env = RiceMRIO(
        region_params              = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        num_regions                = NUM_REGIONS,
        mrio_data_root             = MRIO_ROOT,
        mrio_trade                 = True,
        eu_region_idx              = EU_IDX,
        cbam_tariff_rate           = cbam_tariff_rate,
        cbam_tariff_mode           = "differential",
        dest_alloc_persistence     = 0.55,
        dest_alloc_baseline_decay  = 0.0,
        diff_reward_mode           = True,
        num_discrete_action_levels = 10,
        sector_granularity         = "emissions-simple",
        sectoral_welfloss          = True,
        welfare_loss_per_unit_tariff = 5.0,
        eu_mitigation_schedule     = EU_MITIGATION_SCHEDULE,
        cbam_lambda_init           = 1.0,
        reward_mode                = "additive_cbam",
        log_info_fn                = rcpo_cbam_log_info_fn,
        revenue_share              = revenue_share,
        transfer_mode              = "abatement",
        transfer_allocation        = "effort",
    )
    return LogWrapper(env) if for_training else env


# ── Quick training (for --train-quick mode) ───────────────────────────────────

def _train_quick(key, timesteps):
    from training_monitor import MonitoredPPO, make_print_log_fn
    env = _build_env(revenue_share=0.0, cbam_tariff_rate=0.0, for_training=True)
    num_iters = timesteps // (8 * 100)
    log_fn = make_print_log_fn(num_iterations=num_iters)
    ppo = MonitoredPPO(
        total_timesteps  = timesteps,
        num_steps        = 100,
        num_envs         = 8,
        learning_rate    = 3e-4,
        num_minibatches  = 4,
        num_epochs       = 8,
        ent_coef         = 0.01,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        max_grad_norm    = 1.0,
        clip_coef        = 0.2,
        clip_coef_vf     = 0.5,
        vf_coef          = 0.5,
        normalize_observations = True,
        normalize_rewards      = True,
        log_interval           = 50,
        log_function           = log_fn,
    )
    print(f"  Quick-training {timesteps//1000}k steps for network analysis …")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo


# ── Rollout collector ─────────────────────────────────────────────────────────

def _collect_obs_sequences(key, ppo, revenue_share=0.0, n_episodes=2):
    """Collect per-region observation trajectories from deterministic rollout.

    Returns:
      obs_seqs : dict mapping region_idx → array of shape (T*n_ep, 53)
    """
    # _experiment_util lives in validation/
    _val_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _val_dir not in _sys.path:
        _sys.path.insert(0, _val_dir)
    from _experiment_util import run_single_episode
    from rice_jax.utils import full_state_info_log_fn

    raw_env  = _build_env(revenue_share=revenue_share, for_training=False)
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    obs_seqs = {r: [] for r in CBAM_PLOT_REGIONS}

    for ep in range(n_episodes):
        ep_key = jax.random.fold_in(key, 99_000 + ep)

        # Run episode and collect observations.
        # run_single_episode returns state-level logs, but we need the obs
        # so we do a manual rollout here.
        env_state_key, act_key = jax.random.split(ep_key)
        obs_dict, env_state = eval_env.reset(env_state_key)

        T = 20  # RICE episode length
        for t in range(T):
            # Collect obs for each CBAM region
            for r in CBAM_PLOT_REGIONS:
                agent_key = f"region-{r:02d}"
                if agent_key in obs_dict:
                    ao = obs_dict[agent_key]
                    obs_vec = np.array(ao.observation)   # (53,)
                    obs_seqs[r].append(obs_vec)

            # Step with deterministic policy — pass full state dict;
            # jaxnasium's multi-agent wrapper routes per-agent internally.
            act_key, step_key = jax.random.split(act_key)
            actions = ppo.get_action(
                step_key,
                ppo.state,
                obs_dict,
                deterministic=True,
            )
            (obs_dict, *_), env_state = eval_env.step(step_key, env_state, actions)

    return {r: np.stack(obs_seqs[r], axis=0) for r in CBAM_PLOT_REGIONS}


# ── Jacobian analysis ─────────────────────────────────────────────────────────

def _compute_jacobian(ppo_state, obs_vec):
    """Gradient of summed action logits w.r.t. obs (discrete action proxy).

    For discrete (Categorical) action spaces, argmax is not differentiable,
    so we differentiate through the sum of logits across all action dims.
    Returns array of shape (1, obs_dim) for API compatibility with callers
    that do np.mean(..., axis=0).
    """
    def _forward(o):
        dist = ppo_state.actor(o)
        flat_logits, _ = jax.flatten_util.ravel_pytree(dist.logits)
        return jnp.sum(flat_logits)   # scalar

    norm_obs = ppo_state.normalizer.normalize_obs(obs_vec)
    grad = jax.grad(_forward)(norm_obs)   # (obs_dim,)
    return np.array(grad)[None, :]        # (1, obs_dim)


def _integrated_gradients(ppo_state, obs_vec, target_action_idx, n_steps=50):
    """Integrated gradients for one action dimension w.r.t. obs.

    For discrete spaces targets the mitigation-action logit block.
    NUM_DISCRETE_LEVELS=10 is assumed from the standard experiment config.
    Returns attribution vector of shape (obs_dim,).
    """
    baseline = jnp.zeros_like(obs_vec)
    NUM_CLASSES = 10   # num_discrete_action_levels (standard config)

    def _fwd_scalar(o):
        norm = ppo_state.normalizer.normalize_obs(o)
        dist = ppo_state.actor(norm)
        flat_logits, _ = jax.flatten_util.ravel_pytree(dist.logits)
        # logits for action dim `target_action_idx` occupy a contiguous block
        start = target_action_idx * NUM_CLASSES
        end   = min(start + NUM_CLASSES, flat_logits.shape[0])
        return jnp.sum(flat_logits[start:end])

    alphas = jnp.linspace(0.0, 1.0, n_steps + 1)
    grads = []
    for alpha in alphas:
        interp = baseline + alpha * (obs_vec - baseline)
        g = jax.grad(_fwd_scalar)(interp)
        grads.append(np.array(g))

    avg_grad = np.mean(grads, axis=0)                # (53,)
    ig = avg_grad * np.array(obs_vec - baseline)     # element-wise product
    return ig


def _first_layer_weight_norms(ppo_state):
    """Extract |W[:,i]| for each obs dim i from the first MLP layer.

    Returns array of shape (obs_dim,) — column norms of first weight matrix.
    """
    actor = ppo_state.actor
    # obs_processor → mlp; try mlp.layers[0]
    mlp = actor.mlp
    layers = mlp.layers if hasattr(mlp, "layers") else [mlp]
    w0 = None
    for layer in layers:
        if hasattr(layer, "weight"):
            w0 = np.array(layer.weight)   # (hidden_dim, obs_dim)
            break
    if w0 is None:
        # fallback: look through obs_processor
        for attr in ["layers", "net", "linear"]:
            sub = getattr(actor.obs_processor, attr, None)
            if sub is not None:
                for l in (sub if hasattr(sub, "__iter__") else [sub]):
                    if hasattr(l, "weight"):
                        w0 = np.array(l.weight)
                        break
    if w0 is None:
        return np.zeros(OBS_DIM)
    return np.linalg.norm(w0, axis=0)   # (obs_dim,)


def _policy_entropy(ppo_state, obs_vec):
    """Sum of action distribution entropies (scalar).

    Uses jax.flatten_util to handle pytree of per-action entropies
    from the multi-discrete Categorical distribution.
    """
    norm = ppo_state.normalizer.normalize_obs(obs_vec)
    dist = ppo_state.actor(norm)
    try:
        flat_ent, _ = jax.flatten_util.ravel_pytree(dist.entropy())
        return float(jnp.sum(flat_ent))
    except Exception:
        return float("nan")


# ── Group-level aggregation of obs-dim arrays ─────────────────────────────────

def _group_agg(vec):
    """Aggregate a per-obs-dim array into per-group mean absolute values."""
    return {name: float(np.mean(np.abs(vec[idxs])))
            for name, idxs in OBS_GROUPS}


# ── Behavioural analysis ──────────────────────────────────────────────────────

def plot_behavioural(results_list, timestamp, out_path):
    rs_vals    = [r["revenue_share"] for r in results_list]
    n_rs       = len(rs_vals)
    rs_colors  = plt.cm.RdYlGn(np.linspace(0.15, 0.85, n_rs))
    regions    = CBAM_PLOT_REGIONS
    rnames     = [REGION_NAMES[r] for r in regions]
    nr         = len(regions)

    # Per-region dirty_share and μ arrays: shape (n_rs, nr)
    dirty = np.array([[res["eval"]["dirty_share"][r] for r in regions]
                       for res in results_list])
    mu    = np.array([[res["eval"]["mit_rate"][r]    for r in regions]
                       for res in results_list])
    trns  = np.array([[res["eval"]["transfer"][r]    for r in regions]
                       for res in results_list])

    fig = plt.figure(figsize=(20, 22))
    fig.suptitle(
        f"Per-Region CBAM Response Analysis\n"
        f"9-region vulnerability, differential CBAM, {n_rs} revenue_share levels",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.40)

    # ── Panel 1: Response-type scatter (Δdirty vs Δμ) ───────────────────────
    ax_scatter = fig.add_subplot(gs[0, :2])
    delta_dirty = dirty[-1] - dirty[0]   # rs=1 minus rs=0
    delta_mu    = mu[-1]    - mu[0]
    ax_scatter.axhline(0, color="grey", lw=0.8, ls="--")
    ax_scatter.axvline(0, color="grey", lw=0.8, ls="--")
    for i, r in enumerate(regions):
        ax_scatter.scatter(delta_dirty[i], delta_mu[i], s=120, zorder=5)
        ax_scatter.annotate(
            REGION_NAMES[r], (delta_dirty[i], delta_mu[i]),
            fontsize=7, xytext=(4, 4), textcoords="offset points",
        )
    ax_scatter.set_xlabel("Δ EU dirty share  (rs=1 − rs=0)", fontsize=9)
    ax_scatter.set_ylabel("Δ mitigation rate μ  (rs=1 − rs=0)", fontsize=9)
    ax_scatter.set_title("Response-type map\n"
                          "Q2: less diversion + more mitigation (ideal)\n"
                          "Q4: more diversion + less mitigation (worst)", fontsize=8)
    ax_scatter.tick_params(labelsize=8)
    # shade quadrants
    xl, xr = ax_scatter.get_xlim()
    yb, yt = ax_scatter.get_ylim()
    ax_scatter.fill_betweenx([0, yt], xl, 0, alpha=0.05, color="green")  # Q2
    ax_scatter.fill_betweenx([yb, 0], 0, xr, alpha=0.05, color="red")   # Q4

    # ── Panel 2: Elasticity heatmap (dirty_share) ────────────────────────────
    ax_heat_d = fig.add_subplot(gs[0, 2:])
    im_d = ax_heat_d.imshow(
        dirty.T, aspect="auto",
        norm=TwoSlopeNorm(vmin=dirty.min(), vcenter=dirty.mean(), vmax=dirty.max()),
        cmap="RdYlGn_r",
    )
    ax_heat_d.set_xticks(range(n_rs))
    ax_heat_d.set_xticklabels([f"{v:.2f}" for v in rs_vals], fontsize=8)
    ax_heat_d.set_yticks(range(nr))
    ax_heat_d.set_yticklabels(rnames, fontsize=8)
    ax_heat_d.set_xlabel("Revenue share", fontsize=9)
    ax_heat_d.set_title("EU dirty export share (heatmap)\nhigher = more diversion", fontsize=8)
    plt.colorbar(im_d, ax=ax_heat_d, fraction=0.04)
    for i in range(n_rs):
        for j in range(nr):
            ax_heat_d.text(i, j, f"{dirty[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black")

    # ── Panel 3: Elasticity heatmap (μ) ─────────────────────────────────────
    ax_heat_m = fig.add_subplot(gs[1, :2])
    im_m = ax_heat_m.imshow(
        mu.T, aspect="auto",
        norm=TwoSlopeNorm(vmin=mu.min(), vcenter=mu.mean(), vmax=mu.max()),
        cmap="RdYlGn",
    )
    ax_heat_m.set_xticks(range(n_rs))
    ax_heat_m.set_xticklabels([f"{v:.2f}" for v in rs_vals], fontsize=8)
    ax_heat_m.set_yticks(range(nr))
    ax_heat_m.set_yticklabels(rnames, fontsize=8)
    ax_heat_m.set_xlabel("Revenue share", fontsize=9)
    ax_heat_m.set_title("Mitigation rate μ (heatmap)\nhigher = more abatement", fontsize=8)
    plt.colorbar(im_m, ax=ax_heat_m, fraction=0.04)
    for i in range(n_rs):
        for j in range(nr):
            ax_heat_m.text(i, j, f"{mu[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black")

    # ── Panel 4: Transfer efficiency (Δμ per unit transfer) ─────────────────
    ax_eff = fig.add_subplot(gs[1, 2:])
    # Only use runs where transfer > 0
    mean_trns = trns.mean(axis=0)          # (nr,) — mean transfer across rs levels (excl 0)
    # compute efficiency: Δμ(rs=0→1) / mean_transfer (for rs>0)
    nonzero_trns = np.where(mean_trns > 1e-6, mean_trns, np.nan)
    efficiency = delta_mu / nonzero_trns
    # bar chart
    colors = ["#2ca02c" if e > 0 else "#d62728" for e in efficiency]
    bars = ax_eff.bar(range(nr), efficiency, color=colors, edgecolor="k", linewidth=0.5)
    ax_eff.set_xticks(range(nr))
    ax_eff.set_xticklabels(rnames, rotation=30, ha="right", fontsize=8)
    ax_eff.axhline(0, color="k", lw=0.8)
    ax_eff.set_ylabel("Δμ per unit transfer received", fontsize=9)
    ax_eff.set_title("Transfer efficiency (mitigation lift per unit subsidy)\n"
                      "green = transfer raises mitigation; red = moral hazard", fontsize=8)
    ax_eff.tick_params(labelsize=8)

    # ── Panel 5: Per-region trajectory (dirty share × rs) ───────────────────
    ax_traj_d = fig.add_subplot(gs[2, :2])
    for i, r in enumerate(regions):
        ax_traj_d.plot(rs_vals, dirty[:, i], marker="o", ms=5,
                        label=REGION_NAMES[r])
    ax_traj_d.set_xlabel("Revenue share", fontsize=9)
    ax_traj_d.set_ylabel("EU dirty export share", fontsize=9)
    ax_traj_d.set_title("Per-region EU dirty share trajectory", fontsize=9)
    ax_traj_d.legend(fontsize=7, loc="upper left")
    ax_traj_d.tick_params(labelsize=8)

    # ── Panel 6: Per-region trajectory (μ × rs) ──────────────────────────────
    ax_traj_m = fig.add_subplot(gs[2, 2:])
    for i, r in enumerate(regions):
        ax_traj_m.plot(rs_vals, mu[:, i], marker="o", ms=5,
                        label=REGION_NAMES[r])
    ax_traj_m.set_xlabel("Revenue share", fontsize=9)
    ax_traj_m.set_ylabel("Mitigation rate μ", fontsize=9)
    ax_traj_m.set_title("Per-region mitigation trajectory", fontsize=9)
    ax_traj_m.legend(fontsize=7, loc="lower left")
    ax_traj_m.tick_params(labelsize=8)

    # ── Panel 7: Δdirty and Δμ bar chart side-by-side ───────────────────────
    ax_delta = fig.add_subplot(gs[3, :])
    x    = np.arange(nr)
    w    = 0.35
    ax_delta.bar(x - w/2, delta_dirty, width=w, label="Δ dirty share (rs=1−0)",
                  color="#d62728", alpha=0.8, edgecolor="k", linewidth=0.4)
    ax_delta.bar(x + w/2, delta_mu,    width=w, label="Δ mitigation μ (rs=1−0)",
                  color="#2ca02c", alpha=0.8, edgecolor="k", linewidth=0.4)
    ax_delta.axhline(0, color="k", lw=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(rnames, rotation=30, ha="right", fontsize=8)
    ax_delta.set_ylabel("Change (rs=1 minus rs=0)", fontsize=9)
    ax_delta.set_title("Net effect of full revenue transfer on diversion and mitigation\n"
                        "Ideal: Δdirty < 0 and Δμ > 0", fontsize=9)
    ax_delta.legend(fontsize=8)
    ax_delta.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Behavioural figure → {out_path}")


# ── Policy / weight analysis ──────────────────────────────────────────────────

def plot_policy_analysis(results_with_agents, timestamp, out_path):
    """Network-level analysis across rs levels.

    results_with_agents: list of dicts with 'ppo', 'revenue_share', 'eval'.
    Each ppo has ppo.state with actor + normalizer.
    """
    key = jax.random.PRNGKey(77)
    n_rs  = len(results_with_agents)
    rs_vals = [r["revenue_share"] for r in results_with_agents]

    # ── 1. First-layer weight norms ──────────────────────────────────────────
    # Average across all non-EU agents for each training condition
    group_names = [g[0] for g in OBS_GROUPS]
    n_groups     = len(group_names)
    weight_matrix = np.zeros((n_rs, n_groups))
    for i, res in enumerate(results_with_agents):
        # ppo.state is dict keyed by "region-XX"
        per_region_norms = []
        for r in CBAM_PLOT_REGIONS:
            ag_key  = f"region-{r:02d}"
            ag_st   = res["ppo"].state[ag_key]
            norms   = _first_layer_weight_norms(ag_st)   # (53,)
            per_region_norms.append(norms)
        mean_norms = np.mean(per_region_norms, axis=0)   # (53,)
        for j, (name, idxs) in enumerate(OBS_GROUPS):
            weight_matrix[i, j] = float(np.mean(np.abs(mean_norms[idxs])))

    # ── 2. Jacobian sensitivity (per region) ─────────────────────────────────
    # Use the rs=0 model for Jacobian (no transfer, baseline behaviour)
    ppo_base = results_with_agents[0]["ppo"]
    key, jac_key = jax.random.split(key)
    obs_seqs = _collect_obs_sequences(jac_key, ppo_base,
                                      revenue_share=0.0,
                                      n_episodes=JACOBIAN_EPISODES)
    jac_region  = {}
    jac_raw     = {}
    entropy_per_region = {}
    for r in CBAM_PLOT_REGIONS:
        ag_key  = f"region-{r:02d}"
        ag_st   = ppo_base.state[ag_key]   # per-agent PPOState
        seq = obs_seqs[r]   # (T, 53)
        if len(seq) == 0:
            jac_raw[r]    = np.zeros(OBS_DIM)
            jac_region[r] = np.zeros(n_groups)
            entropy_per_region[r] = float("nan")
            continue
        jacs = []
        entropies = []
        for t_idx in range(min(len(seq), 10)):
            obs_t  = jnp.array(seq[t_idx])
            j_t    = _compute_jacobian(ag_st, obs_t)           # (action_dim, 53)
            jacs.append(np.mean(np.abs(j_t), axis=0))         # (53,)
            entropies.append(_policy_entropy(ag_st, obs_t))

        mean_jac = np.mean(jacs, axis=0)                       # (53,)
        jac_raw[r]    = mean_jac
        jac_region[r] = np.array([float(np.mean(mean_jac[idxs]))
                                   for _, idxs in OBS_GROUPS])
        entropy_per_region[r] = float(np.nanmean(entropies))

    # ── 3. Integrated gradients for mitigation action ─────────────────────────
    # Use the mitigation_rate action index. In alphabetical action key order:
    # export_reallocation[0..17], mitigation_rate, savings_rate
    # → mitigation_rate index = 18 (0-indexed)
    MIT_ACTION_IDX = 18
    ig_region = {}
    for r in CBAM_PLOT_REGIONS:
        ag_key = f"region-{r:02d}"
        ag_st  = ppo_base.state[ag_key]
        seq = obs_seqs[r]
        if len(seq) == 0:
            ig_region[r] = np.zeros(OBS_DIM)
            continue
        igs = []
        for t_idx in range(min(len(seq), 5)):
            obs_t = jnp.array(seq[t_idx])
            ig_t  = _integrated_gradients(ag_st, obs_t, MIT_ACTION_IDX)
            igs.append(ig_t)
        ig_region[r] = np.mean(igs, axis=0)   # (53,)

    # ── Build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 24))
    fig.suptitle(
        "Policy / Weight Analysis — CBAM Per-Exporter\n"
        "9-region vulnerability, differential CBAM (rs=0 agent for Jacobian)",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)

    rnames_plot = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]

    # Panel 1: First-layer weight norms (rs × obs_group heatmap)
    ax_wt = fig.add_subplot(gs[0, :])
    im_wt = ax_wt.imshow(weight_matrix.T, aspect="auto", cmap="YlOrRd")
    ax_wt.set_xticks(range(n_rs))
    ax_wt.set_xticklabels([f"rs={v:.2f}" for v in rs_vals], fontsize=8)
    ax_wt.set_yticks(range(n_groups))
    ax_wt.set_yticklabels(group_names, fontsize=8)
    ax_wt.set_title("First MLP layer — mean |weight| per obs group × training condition\n"
                     "(higher = that obs group drives the hidden layer more)", fontsize=9)
    plt.colorbar(im_wt, ax=ax_wt, fraction=0.015)
    for i in range(n_rs):
        for j in range(n_groups):
            ax_wt.text(i, j, f"{weight_matrix[i, j]:.3f}",
                        ha="center", va="center", fontsize=6.5)

    # Panel 2: Jacobian sensitivity heatmap (obs_group × region)
    jac_matrix = np.array([jac_region[r] for r in CBAM_PLOT_REGIONS]).T  # (n_groups, nr)
    ax_jac = fig.add_subplot(gs[1, :])
    im_jac = ax_jac.imshow(jac_matrix, aspect="auto", cmap="Blues")
    ax_jac.set_xticks(range(len(CBAM_PLOT_REGIONS)))
    ax_jac.set_xticklabels(rnames_plot, fontsize=8)
    ax_jac.set_yticks(range(n_groups))
    ax_jac.set_yticklabels(group_names, fontsize=8)
    ax_jac.set_title("Jacobian sensitivity: mean |∂action/∂obs| × obs group × region\n"
                      "(rs=0 trained agent; averaged over episode; higher = more sensitive)", fontsize=9)
    plt.colorbar(im_jac, ax=ax_jac, fraction=0.015)
    for i in range(n_groups):
        for j in range(len(CBAM_PLOT_REGIONS)):
            ax_jac.text(j, i, f"{jac_matrix[i, j]:.3f}",
                         ha="center", va="center", fontsize=6.5)

    # Panel 3: Integrated gradients for mitigation action (bar chart per region)
    ax_ig = fig.add_subplot(gs[2, :])
    n_reg = len(CBAM_PLOT_REGIONS)
    # Aggregate IG into groups
    ig_groups = np.array([
        [float(np.sum(ig_region[r][idxs])) for r in CBAM_PLOT_REGIONS]
        for _, idxs in OBS_GROUPS
    ])  # (n_groups, n_reg)
    x    = np.arange(n_reg)
    bot  = np.zeros(n_reg)
    cmap = plt.cm.tab10
    for j, name in enumerate(group_names):
        vals = ig_groups[j]
        ax_ig.bar(x, vals, bottom=bot, label=name,
                   color=cmap(j / len(group_names)), edgecolor="k", linewidth=0.3)
        bot = bot + vals
    ax_ig.set_xticks(x)
    ax_ig.set_xticklabels(rnames_plot, rotation=30, ha="right", fontsize=8)
    ax_ig.axhline(0, color="k", lw=0.8)
    ax_ig.set_ylabel("Integrated gradient attribution", fontsize=9)
    ax_ig.set_title("Integrated gradients for mitigation_rate action per region\n"
                     "(stacked by obs group; positive = obs feature pushes μ up)", fontsize=9)
    ax_ig.legend(fontsize=7, loc="upper right", ncol=4)
    ax_ig.tick_params(labelsize=8)

    # Panel 4: Policy entropy per region (bar chart)
    ax_ent = fig.add_subplot(gs[3, :2])
    ents = [entropy_per_region.get(r, float("nan")) for r in CBAM_PLOT_REGIONS]
    finite_ents = [e for e in ents if not np.isnan(e)]
    ax_ent.bar(range(n_reg), ents,
                color=plt.cm.plasma(np.linspace(0.2, 0.8, n_reg)),
                edgecolor="k", linewidth=0.5)
    ax_ent.set_xticks(range(n_reg))
    ax_ent.set_xticklabels(rnames_plot, rotation=30, ha="right", fontsize=8)
    ax_ent.set_ylabel("Action distribution entropy", fontsize=9)
    ax_ent.set_title("Policy entropy per region (rs=0 agent)\n"
                      "high = uncertain / hedging; low = confident strategy", fontsize=9)
    ax_ent.tick_params(labelsize=8)

    # Panel 5: CBAM-obs sensitivity comparison (cbam_cost, cbam_tariff_rate, cbam_revenue)
    ax_cbam = fig.add_subplot(gs[3, 2:])
    cbam_groups = ["cbam_cost", "cbam_tariff_rate", "cbam_revenue"]
    cbam_jac = np.array([
        [float(np.mean(np.abs(jac_raw[r][OBS_GROUP_SLICES[g]])))
         for g in cbam_groups]
        for r in CBAM_PLOT_REGIONS
    ])  # (n_reg, 3)
    xg   = np.arange(n_reg)
    wg   = 0.25
    cbam_colors = ["#d62728", "#1f77b4", "#ff7f0e"]
    for gi, gname in enumerate(cbam_groups):
        ax_cbam.bar(xg + (gi - 1) * wg, cbam_jac[:, gi], width=wg,
                     label=gname, color=cbam_colors[gi],
                     alpha=0.85, edgecolor="k", linewidth=0.3)
    ax_cbam.set_xticks(xg)
    ax_cbam.set_xticklabels(rnames_plot, rotation=30, ha="right", fontsize=8)
    ax_cbam.set_ylabel("|∂action/∂obs|", fontsize=9)
    ax_cbam.set_title("CBAM-signal sensitivity per region\n"
                       "how strongly each agent conditions on CBAM observations", fontsize=9)
    ax_cbam.legend(fontsize=7)
    ax_cbam.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Policy analysis figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pkl",          type=str, default=None,
                        help="Path to per_exporter pkl (results + optionally agents).")
    parser.add_argument("--jacobian",     action="store_true",
                        help="Run Jacobian/IG/weight analysis (requires agents in pkl "
                             "or --train-quick).")
    parser.add_argument("--train-quick",  action="store_true",
                        help="Train a single rs=0 model (--timesteps steps) "
                             "for network analysis only.")
    parser.add_argument("--timesteps",   type=int, default=500_000,
                        help="Steps for --train-quick (default 500k).")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.pkl is None and not args.train_quick:
        parser.error("Provide --pkl <path> or --train-quick.")

    results_list = None
    if args.pkl:
        with open(args.pkl, "rb") as f:
            saved = pickle.load(f)
        results_list = saved["results_list"]
        print(f"  Loaded {len(results_list)} runs from {args.pkl}")

    # ── Behavioural analysis ─────────────────────────────────────────────────
    if results_list is not None:
        out_beh = _os.path.join(OUTPUT_DIR, f"cbam_region_behavioural_{timestamp}.png")
        plot_behavioural(results_list, timestamp, out_beh)

    # ── Policy / weight analysis ──────────────────────────────────────────────
    if args.jacobian or args.train_quick:
        agents_available = (
            results_list is not None and
            "ppo" in results_list[0]
        )

        if args.train_quick or not agents_available:
            if not args.train_quick and not agents_available:
                print("  WARNING: no agents in pkl — training quick model for network analysis.")
            key = jax.random.PRNGKey(args.seed)
            ppo_quick = _train_quick(key, args.timesteps)
            # Wrap as a single-run list
            results_for_net = [{
                "revenue_share": 0.0,
                "ppo": ppo_quick,
                "eval": results_list[0]["eval"] if results_list else {},
            }]
        else:
            results_for_net = results_list

        out_pol = _os.path.join(OUTPUT_DIR, f"cbam_region_policy_{timestamp}.png")
        plot_policy_analysis(results_for_net, timestamp, out_pol)


if __name__ == "__main__":
    main()
