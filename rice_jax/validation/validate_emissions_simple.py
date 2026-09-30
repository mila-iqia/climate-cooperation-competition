"""validate_emissions_simple.py

Focused validation: CBAM vs no-CBAM under ``emissions-simple`` sector
granularity (dirty = CBAM + high-emission sectors vs clean = rest).

Shows all non-EU, non-RoW regions in a wide figure.

Figure layout (3 rows × N columns, one column per focus region):
  Row 0  EU export share — dirty ("CBAM") vs clean ("non-CBAM") sector bucket
  Row 1  Reward (utility × welfloss) difference: CBAM − no-CBAM
  Row 2  Welfloss difference: CBAM − no-CBAM

Training modes:
  --conditioned (default): Train a SINGLE model with randomized τ ∈ {0, CBAM_RATE}
                           per episode.  The tariff rate is part of the observation,
                           so the policy learns to condition on it.  Both eval
                           conditions use the same weights — behavioural differences
                           are learned, not training-noise artifacts.
  --separate:              Legacy mode.  Train two independent models (one per τ).

Both use ``sector_granularity="emissions-simple"``, ``fixed_savings_rate=True``,
``no_mitigation=True``, ``dest_alloc_persistence=0.55``.

Output: plots/emissions_simple_validation_<TIMESTAMP>.png

Usage:
    python validate_emissions_simple.py [--conditioned | --separate]
"""

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
from training_monitor import MonitoredPPO, make_combined_log_fn, make_csv_log_fn, make_print_log_fn

from _experiment_util import run_single_episode
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

# ── Configuration (defaults; overridden by CLI args) ──────────────────────────

NUM_REGIONS: int = 7
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
MRIO_DATA_ROOT: str = os.path.join(_REPO_ROOT, "csv_asset")
EU_REGION_IDX: int = 5
YAML_DIR: str | None = None   # None → package default; set for 9-region vuln setup
DEST_ALLOC_PERSISTENCE: float = 0.55
DEST_ALLOC_BASELINE_DECAY: float = 1.0

TOTAL_TIMESTEPS: int = 10_000_000
NUM_ENVS: int = 8
NUM_STEPS: int = 100

NUM_EVAL_EPISODES: int = 5
SEED: int = 42

OUTPUT_DIR: str = "plots"
DPI: int = 150

CBAM_RATE: float = 0.80

# Regions of interest (0-based indices)
# Default: all regions except EU and RoW (idx 0 in 9-region)
FOCUS_REGIONS: list[int] = [0, 1, 2, 3, 4, 6]  # 7-region: all except EU (5)

# Presets for known setups
_PRESETS = {
    7: {"eu_idx": 5, "focus": [0, 1, 2, 3, 4, 6], "yaml_dir": None},
    9: {
        "eu_idx": 3,
        "focus": [1, 2, 4, 5, 6, 7, 8],  # all except RoW (0) and EU (3)
        "yaml_dir": os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9"),
    },
}

# ── Shared MRIO kwargs ────────────────────────────────────────────────────────

_MRIO_KWARGS = dict(
    num_regions=NUM_REGIONS,
    mrio_data_root=MRIO_DATA_ROOT,
    mrio_trade=True,
    dest_alloc_persistence=DEST_ALLOC_PERSISTENCE,
    dest_alloc_baseline_decay=DEST_ALLOC_BASELINE_DECAY,
    eu_region_idx=EU_REGION_IDX,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    sectoral_welfloss=True,
    fixed_savings_rate=True,
    no_mitigation=True,
    sector_granularity="emissions-simple",
    welfare_loss_per_unit_tariff=5.0,
)

# ── PPO kwargs (C5 recommended baseline) ──────────────────────────────────────

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
    log_function="tqdm",  # overridden per run via _make_ppo()
    log_interval=0.02,   # log every 2% of iterations ≈ every ~25 updates
)

# ── Training-monitor helpers ─────────────────────────────────────────────────


def _make_ppo(label: str) -> MonitoredPPO:
    """Build a MonitoredPPO with per-run CSV + print logging."""
    os.makedirs("training_logs", exist_ok=True)
    csv_path = os.path.join("training_logs", f"{label}.csv")
    # num_iterations = total_timesteps / num_steps / num_envs
    num_iters = TOTAL_TIMESTEPS // NUM_STEPS // NUM_ENVS
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    return MonitoredPPO(**{**_PPO_KWARGS, "log_function": log_fn})


# ── Environment builders ──────────────────────────────────────────────────────


def _build_train_env(region_params, *, conditioned: bool) -> jym.LogWrapper:
    """Build the training environment.

    conditioned=True  → cbam_randomize on, τ sampled from {0, CBAM_RATE}.
    conditioned=False → fixed cbam_tariff_rate (caller sets it separately).
    """
    if conditioned:
        env = RiceMRIO(
            region_params=region_params,
            cbam_tariff_rate=CBAM_RATE,        # unused when randomizing, but sets scale
            cbam_randomize=True,
            cbam_tariff_rates=(0.0, CBAM_RATE),
            **_MRIO_KWARGS,
        )
    else:
        # Should not be called without explicit rate — see _build_fixed_env
        raise ValueError("Use _build_fixed_env for separate-model training")
    return jym.LogWrapper(env)


def _build_fixed_env(cbam_tariff_rate: float, region_params) -> jym.LogWrapper:
    """Build an environment with a fixed CBAM tariff rate (for eval or separate training)."""
    env = RiceMRIO(
        region_params=region_params,
        cbam_tariff_rate=cbam_tariff_rate,
        cbam_randomize=False,
        **_MRIO_KWARGS,
    )
    return jym.LogWrapper(env)


# ── Evaluate a trained agent on a fixed-rate env ──────────────────────────────


def _eval_agent(agent, cbam_tariff_rate: float, region_params, seed: jax.Array) -> dict:
    """Roll out NUM_EVAL_EPISODES with a fixed τ and return stacked arrays."""
    wrapped_env = _build_fixed_env(cbam_tariff_rate, region_params)
    eval_env = replace(wrapped_env._env, log_info_fn=full_state_info_log_fn)

    all_flows, all_uwl, all_util = [], [], []
    NR = NUM_REGIONS

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 10_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)

        all_flows.append(np.array(logs["trade_flows"]))
        uwl = np.stack(
            [np.array(logs["utility_times_welfloss_all_regions"][r]) for r in range(NR)],
            axis=-1,
        )
        all_uwl.append(uwl)
        util = np.stack(
            [np.array(logs["utility_all_regions"][r]) for r in range(NR)],
            axis=-1,
        )
        all_util.append(util)

    return {
        "trade_flows": np.stack(all_flows, axis=0),
        "utility_welfloss": np.stack(all_uwl, axis=0),
        "utility": np.stack(all_util, axis=0),
    }


# ── Train + evaluate (separate models, legacy) ────────────────────────────────


def train_and_collect(
    cbam_tariff_rate: float,
    region_params,
    seed: jax.Array,
) -> dict[str, np.ndarray]:
    """Train an independent PPO and roll out NUM_EVAL_EPISODES."""
    wrapped_env = _build_fixed_env(cbam_tariff_rate, region_params)
    lbl = f"τ={cbam_tariff_rate}" if cbam_tariff_rate > 0 else "no-CBAM"

    safe_lbl = lbl.replace("=", "").replace("τ", "tau").replace(" ", "_")
    agent = _make_ppo(f"separate_{safe_lbl}")
    print(f"  [{lbl}] Training {TOTAL_TIMESTEPS:,} steps...")
    agent = agent.train(seed, wrapped_env)

    return _eval_agent(agent, cbam_tariff_rate, region_params, seed)


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_figure(
    no_cbam: dict[str, np.ndarray],
    cbam: dict[str, np.ndarray],
    region_labels: list[str],
    focus_regions: list[int],
    mode: str = "separate",
) -> plt.Figure:
    """
    3 rows × len(focus_regions) columns.
      Row 0: EU export share — dirty vs clean, CBAM and no-CBAM overlaid
      Row 1: Reward (utility × welfloss) — CBAM vs no-CBAM side-by-side
      Row 2: Welfloss — CBAM vs no-CBAM side-by-side
    """
    n_cols = len(focus_regions)
    T = no_cbam["trade_flows"].shape[1]
    ts = np.arange(T)

    col_width = max(3.0, min(5.0, 28.0 / n_cols))  # scale down per-col width for many regions
    fig, axes = plt.subplots(3, n_cols, figsize=(col_width * n_cols, 10), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for col_i, r_idx in enumerate(focus_regions):
        rlbl = region_labels[r_idx] if r_idx < len(region_labels) else f"Region {r_idx}"

        # ── Row 0: EU export share dirty vs clean ──────────────────────────
        ax_share = axes[0, col_i]

        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            flows = data["trade_flows"]  # (E, T, NR, NR, NS)
            eu_dirty = flows[:, :, r_idx, EU_REGION_IDX, 0]  # (E, T)
            eu_clean = flows[:, :, r_idx, EU_REGION_IDX, 1]  # (E, T)
            tot_dirty = flows[:, :, r_idx, :, 0].sum(axis=-1)  # (E, T)
            tot_clean = flows[:, :, r_idx, :, 1].sum(axis=-1)  # (E, T)

            with np.errstate(divide="ignore", invalid="ignore"):
                share_dirty = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
                share_clean = np.where(tot_clean > 1e-12, eu_clean / tot_clean, np.nan)

            md = np.nanmean(share_dirty, axis=0)
            sd = np.nanstd(share_dirty, axis=0)
            ax_share.plot(ts, md, color=color, ls="-", lw=2, label=f"{cond_label} dirty")
            ax_share.fill_between(ts, md - sd, md + sd, alpha=0.12, color=color)

            mc = np.nanmean(share_clean, axis=0)
            sc = np.nanstd(share_clean, axis=0)
            ax_share.plot(ts, mc, color=color, ls="--", lw=1.5, label=f"{cond_label} clean")
            ax_share.fill_between(ts, mc - sc, mc + sc, alpha=0.08, color=color)

        ax_share.set_title(rlbl, fontsize=11, fontweight="bold")
        if col_i == 0:
            ax_share.set_ylabel("EU export share", fontsize=10)
        ax_share.set_ylim(bottom=0)
        ax_share.legend(fontsize=7, loc="best")
        ax_share.grid(alpha=0.2)

        # ── Row 1: Reward (utility × welfloss) — side-by-side ─────────────
        ax_rew = axes[1, col_i]

        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            uwl = data["utility_welfloss"][:, :, r_idx]  # (E, T)
            md = np.nanmean(uwl, axis=0)
            sd = np.nanstd(uwl, axis=0)
            ax_rew.plot(ts, md, color=color, lw=2, label=cond_label)
            ax_rew.fill_between(ts, md - sd, md + sd, alpha=0.15, color=color)

        ax_rew.set_title(rlbl, fontsize=11, fontweight="bold")
        if col_i == 0:
            ax_rew.set_ylabel("Reward (utility × welfloss)", fontsize=10)
        ax_rew.legend(fontsize=7, loc="best")
        ax_rew.grid(alpha=0.2)

        # ── Row 2: Welfloss — side-by-side ────────────────────────────────
        ax_wl = axes[2, col_i]

        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            uwl = data["utility_welfloss"][:, :, r_idx]   # (E, T)
            util = data["utility"][:, :, r_idx]            # (E, T)
            with np.errstate(divide="ignore", invalid="ignore"):
                wl = np.where(np.abs(util) > 1e-12, uwl / util, 1.0)  # (E, T)
            md = np.nanmean(wl, axis=0)
            sd = np.nanstd(wl, axis=0)
            ax_wl.plot(ts, md, color=color, lw=2, label=cond_label)
            ax_wl.fill_between(ts, md - sd, md + sd, alpha=0.15, color=color)

        ax_wl.set_title(rlbl, fontsize=11, fontweight="bold")
        if col_i == 0:
            ax_wl.set_ylabel("Welfloss multiplier", fontsize=10)
        ax_wl.set_xlabel("Episode step", fontsize=10)
        ax_wl.legend(fontsize=7, loc="best")
        ax_wl.grid(alpha=0.2)

    # Row labels
    row_titles = [
        "EU Export Share\n(solid=dirty, dashed=clean)",
        "Reward\n(utility × welfloss)",
        "Welfloss\nmultiplier",
    ]
    for ri, lbl in enumerate(row_titles):
        axes[ri, 0].annotate(
            lbl, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=9, rotation=90, va="center", ha="right", color="dimgray",
        )

    fig.suptitle(
        f"Emissions-Simple Validation: CBAM (τ={CBAM_RATE:.0%}) vs no-CBAM\n"
        f"sector_granularity='emissions-simple'  |  sectoral_welfloss=True  |  "
        f"ρ={DEST_ALLOC_PERSISTENCE}  |  decay={DEST_ALLOC_BASELINE_DECAY}  |  fixed_savings=0.2  |  no_mitigation\n"
        f"{TOTAL_TIMESTEPS:,} PPO steps  |  {NUM_EVAL_EPISODES} eval eps  |  "
        f"normalize_rewards=True  |  seed={SEED}  |  mode={mode}",
        fontsize=11, fontweight="bold", y=1.02,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Emissions-simple CBAM validation")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--conditioned", action="store_true", default=True,
                       help="Single-model training with randomized τ (default)")
    group.add_argument("--separate", action="store_true",
                       help="Legacy: two independent models")
    parser.add_argument("--num-regions", type=int, default=None,
                       help="Number of regions (default: 7). Use 9 for vuln setup.")
    parser.add_argument("--eu-idx", type=int, default=None,
                       help="EU region index (default: from preset or 5)")
    parser.add_argument("--yaml-dir", type=str, default=None,
                       help="Path to region yamls (default: package built-in)")
    parser.add_argument("--focus-regions", type=int, nargs="+", default=None,
                       help="0-based region indices to plot (default: from preset)")
    args = parser.parse_args()
    conditioned = not args.separate

    # ── Apply preset or explicit overrides to globals ──────────────────────
    global NUM_REGIONS, EU_REGION_IDX, YAML_DIR, FOCUS_REGIONS
    if args.num_regions is not None:
        NUM_REGIONS = args.num_regions
    preset = _PRESETS.get(NUM_REGIONS, {})
    EU_REGION_IDX = args.eu_idx if args.eu_idx is not None else preset.get("eu_idx", EU_REGION_IDX)
    YAML_DIR = args.yaml_dir if args.yaml_dir is not None else preset.get("yaml_dir", YAML_DIR)
    FOCUS_REGIONS = args.focus_regions if args.focus_regions is not None else preset.get("focus", FOCUS_REGIONS)

    # Update _MRIO_KWARGS with resolved values
    _MRIO_KWARGS["num_regions"] = NUM_REGIONS
    _MRIO_KWARGS["eu_region_idx"] = EU_REGION_IDX

    seed = jax.random.PRNGKey(SEED)
    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR)

    # Resolve region labels and verify FOCUS_REGIONS
    sample_env = _build_fixed_env(0.0, region_params)._env
    region_labels = list(sample_env.mrio_region_labels)
    focus_names = [region_labels[r] for r in FOCUS_REGIONS]

    mode = "conditioned" if conditioned else "separate"
    print(f"=== Emissions-Simple Validation ({mode}) ===")
    print(f"Regions: {NUM_REGIONS}  |  Focus: {focus_names}")
    print(f"Sector granularity: emissions-simple")
    print(f"Sectors: {list(sample_env.sector_names)}")
    print(f"Emissions intensity (focus regions):")
    for r in FOCUS_REGIONS:
        emi = sample_env.emissions_intensity[r]
        print(f"  {region_labels[r]:30s}  CBAM={emi[0]:.4f}  non-CBAM={emi[1]:.4f}  ratio={emi[0]/max(emi[1],1e-10):.1f}×")
    print()

    if conditioned:
        # ── Single-model conditioned training ──────────────────────────────
        print("Training single conditioned model (τ randomized per episode)...")
        train_env = _build_train_env(region_params, conditioned=True)
        agent = _make_ppo("conditioned")
        agent = agent.train(seed, train_env)

        print("Evaluating with τ=0 (no-CBAM)...")
        no_cbam = _eval_agent(agent, 0.0, region_params, seed)
        print(f"  [no-CBAM] Done. trade_flows: {no_cbam['trade_flows'].shape}")

        print(f"Evaluating with τ={CBAM_RATE} (CBAM)...")
        cbam = _eval_agent(agent, CBAM_RATE, region_params, seed)
        print(f"  [CBAM]    Done. trade_flows: {cbam['trade_flows'].shape}")
    else:
        # ── Two separate models (legacy) ───────────────────────────────────
        seed_nocbam = jax.random.fold_in(seed, 0)
        no_cbam = train_and_collect(0.0, region_params, seed_nocbam)
        print(f"  [no-CBAM] Done. trade_flows: {no_cbam['trade_flows'].shape}")

        seed_cbam = jax.random.fold_in(seed, 1)
        cbam = train_and_collect(CBAM_RATE, region_params, seed_cbam)
        print(f"  [CBAM]    Done. trade_flows: {cbam['trade_flows'].shape}")

    print("\nGenerating figure...")
    fig = make_figure(no_cbam, cbam, region_labels, FOCUS_REGIONS, mode=mode)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"emissions_simple_validation_{mode}_{timestamp}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
