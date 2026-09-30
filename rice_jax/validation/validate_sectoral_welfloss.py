"""validate_sectoral_welfloss.py

Trains PPO agents under four conditions (2×2 grid):
  CBAM ∈ {off (τ=0), on (τ=0.15)} × sectoral_welfloss ∈ {False, True}

All runs use max trade momentum (ρ=1.0), 7 regions.

Figure layout — one column per non-EU region, two rows:
  Row 0  EU export share of DIRTY sectors (top-50% emissions intensity)
  Row 1  EU export share of CLEAN sectors (bottom-50% emissions intensity)

  Each panel has 4 lines:
    colour  = CBAM state (blue=no CBAM, red=CBAM on)
    style   = welfloss mode (dashed=aggregate, solid=sectoral)

Key question: does sectoral_welfloss cause selective diversion — a larger
drop in dirty-sector EU share under CBAM-on arms vs the baselines?

Output: plots/sectoral_welfloss_validation_<TIMESTAMP>.png

Usage (from rice_jax/ directory):
    /path/to/rice-jax/bin/python validate_sectoral_welfloss.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))



import os
from dataclasses import replace
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxnasium.algorithms import PPO

import jaxnasium as jym

from _experiment_util import run_single_episode
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

# ── Configuration ─────────────────────────────────────────────────────────────

NUM_REGIONS: int = 7
MRIO_DATA_ROOT: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset")
EU_REGION_IDX: int = 0
DEST_ALLOC_PERSISTENCE: float = 1.0  # max momentum

TOTAL_TIMESTEPS: int = 100_000_000
NUM_ENVS: int = 4
NUM_STEPS: int = 100

NUM_EVAL_EPISODES: int = 5
SEED: int = 42

OUTPUT_DIR: str = "plots"
DPI: int = 150

# 4 conditions: (cbam_tariff_rate, sectoral_welfloss)
CONDITIONS: dict[str, dict] = {
    "no-CBAM\naggregate": {"cbam_tariff_rate": 0.0,  "sectoral_welfloss": False},
    "no-CBAM\nsectoral":  {"cbam_tariff_rate": 0.0,  "sectoral_welfloss": True},
    "CBAM\naggregate":    {"cbam_tariff_rate": 0.15, "sectoral_welfloss": False},
    "CBAM\nsectoral":     {"cbam_tariff_rate": 0.15, "sectoral_welfloss": True},
}

# colour encodes CBAM state; linestyle encodes welfloss mode
COND_COLORS: dict[str, str] = {
    "no-CBAM\naggregate": "#5577cc",
    "no-CBAM\nsectoral":  "#5577cc",
    "CBAM\naggregate":    "#e05c2a",
    "CBAM\nsectoral":     "#e05c2a",
}
COND_LS: dict[str, str] = {
    "no-CBAM\naggregate": "--",
    "no-CBAM\nsectoral":  "-",
    "CBAM\naggregate":    "--",
    "CBAM\nsectoral":     "-",
}

# ── Environment builder ────────────────────────────────────────────────────────


def _build_env(cbam_tariff_rate: float, sectoral_welfloss: bool, region_params) -> jym.LogWrapper:
    env = RiceMRIO(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        mrio_data_root=MRIO_DATA_ROOT,
        mrio_trade=True,
        dest_alloc_persistence=DEST_ALLOC_PERSISTENCE,
        cbam_tariff_rate=cbam_tariff_rate,
        eu_region_idx=EU_REGION_IDX,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        sectoral_welfloss=sectoral_welfloss,
        fixed_savings_rate=True,
        no_mitigation=True,
        sector_granularity="simple",
    )
    return jym.LogWrapper(env)


# ── Train + evaluate ───────────────────────────────────────────────────────────


def train_and_collect(
    cbam_tariff_rate: float,
    sectoral_welfloss: bool,
    region_params,
    seed: jax.Array,
) -> np.ndarray:
    """
    Train PPO and roll out NUM_EVAL_EPISODES.

    Returns
    -------
    trade_flows : (E, T, NR, NR, NS)
    """
    wrapped_env = _build_env(cbam_tariff_rate, sectoral_welfloss, region_params)
    cbam_lbl = f"tau={cbam_tariff_rate}" if cbam_tariff_rate > 0 else "no-CBAM"
    sw_lbl   = "sectoral" if sectoral_welfloss else "aggregate"

    agent = PPO(
        total_timesteps=TOTAL_TIMESTEPS,
        learning_rate=2.5e-4,
        num_steps=NUM_STEPS,
        num_envs=NUM_ENVS,
        num_minibatches=4,
        num_epochs=4,
        ent_coef=2.0,
        anneal_ent_coef=0.05,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        clip_coef=0.2,
        clip_coef_vf=0.5,
        vf_coef=0.5,
        normalize_observations=True,
        normalize_rewards=False,
        log_function="tqdm",
    )

    print(f"  [{cbam_lbl}, {sw_lbl}] Training {TOTAL_TIMESTEPS:,} steps...")
    agent = agent.train(seed, wrapped_env)

    eval_env = replace(wrapped_env._env, log_info_fn=full_state_info_log_fn)

    all_flows = []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 10_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)
        all_flows.append(np.array(logs["trade_flows"]))  # (T, NR, NR, NS)

    return np.stack(all_flows, axis=0)  # (E, T, NR, NR, NS)


# ── Metric helpers ─────────────────────────────────────────────────────────────


def sector_eu_share_by_quintile(
    trade_flows: np.ndarray,
    eu_idx: int,
    emissions_intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split sectors into dirty (top-50% intensity) and clean (bottom-50%).

    Parameters
    ----------
    trade_flows         : (E, T, NR, NR, NS)
    emissions_intensity : (NR, NS)

    Returns
    -------
    dirty_share : (E, T, NR)  EU share of dirty-sector exports
    clean_share : (E, T, NR)  EU share of clean-sector exports
    """
    mean_intensity = emissions_intensity.mean(axis=0)  # (NS,)
    median         = np.median(mean_intensity)
    dirty_mask     = mean_intensity >= median           # (NS,) bool

    eu_dirty  = trade_flows[:, :, :, eu_idx, :][:, :, :, dirty_mask].sum(axis=-1)
    eu_clean  = trade_flows[:, :, :, eu_idx, :][:, :, :, ~dirty_mask].sum(axis=-1)
    tot_dirty = trade_flows[:, :, :, :, dirty_mask].sum(axis=(3, 4))
    tot_clean = trade_flows[:, :, :, :, ~dirty_mask].sum(axis=(3, 4))

    with np.errstate(divide="ignore", invalid="ignore"):
        dirty_share = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
        clean_share = np.where(tot_clean > 1e-12, eu_clean / tot_clean, np.nan)
    return dirty_share, clean_share


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_figure(
    results: dict[str, np.ndarray],
    region_labels: list[str],
    emissions_intensity: np.ndarray,
) -> plt.Figure:
    """
    results : {condition_label: trade_flows (E, T, NR, NR, NS)}

    Layout:
      Row 0 — dirty-sector EU export share; 4 lines per panel
      Row 1 — clean-sector EU export share; 4 lines per panel
    """
    non_eu = [r for r in range(NUM_REGIONS) if r != EU_REGION_IDX]
    n      = len(non_eu)
    T      = next(iter(results.values())).shape[1]
    ts     = np.arange(T)

    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n, 8),
        constrained_layout=True,
    )
    if n == 1:
        axes = axes.reshape(2, 1)

    # Pre-compute dirty/clean shares for every condition
    shares: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lbl, flows in results.items():
        shares[lbl] = sector_eu_share_by_quintile(flows, EU_REGION_IDX, emissions_intensity)

    for col_i, r_idx in enumerate(non_eu):
        rlbl    = region_labels[r_idx] if r_idx < len(region_labels) else f"Region {r_idx}"
        ax_dirt = axes[0, col_i]
        ax_cln  = axes[1, col_i]

        for lbl in CONDITIONS:
            dirty_sh, clean_sh = shares[lbl]
            c       = COND_COLORS[lbl]
            ls      = COND_LS[lbl]
            display = lbl.replace("\n", " ")

            # dirty row
            d       = dirty_sh[:, :, r_idx]
            md, sd  = np.nanmean(d, axis=0), np.nanstd(d, axis=0)
            ax_dirt.plot(ts, md, color=c, ls=ls, lw=1.8, label=display)
            ax_dirt.fill_between(ts, md - sd, md + sd, alpha=0.12, color=c)

            # clean row
            cl      = clean_sh[:, :, r_idx]
            mc, sc  = np.nanmean(cl, axis=0), np.nanstd(cl, axis=0)
            ax_cln.plot(ts, mc, color=c, ls=ls, lw=1.8, label=display)
            ax_cln.fill_between(ts, mc - sc, mc + sc, alpha=0.12, color=c)

        ax_dirt.set_title(rlbl, fontsize=11, fontweight="bold")
        ax_dirt.set_ylabel("EU share (dirty sectors)", fontsize=8)
        ax_dirt.set_ylim(bottom=0)
        ax_dirt.legend(fontsize=7, loc="upper right")

        ax_cln.set_ylabel("EU share (clean sectors)", fontsize=8)
        ax_cln.set_ylim(bottom=0)
        ax_cln.legend(fontsize=7, loc="upper right")
        ax_cln.set_xlabel("Episode step", fontsize=8)

    # Row labels on left margin
    for ri, lbl in enumerate(["Dirty-sector\nEU export share", "Clean-sector\nEU export share"]):
        axes[ri, 0].annotate(
            lbl, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=8, rotation=90, va="center", ha="right", color="dimgray",
        )

    note = (
        "colour: blue = no CBAM  |  red = CBAM (tau=0.15)\n"
        "style:  solid = sectoral welfloss  |  dashed = aggregate welfloss"
    )
    fig.suptitle(
        f"Selective diversion: dirty vs clean EU export share across 4 conditions\n"
        f"n_regions={NUM_REGIONS}  |  rho={DEST_ALLOC_PERSISTENCE} (max momentum)  |  "
        f"fixed_savings=0.2  |  no_mitigation  |  "
        f"{TOTAL_TIMESTEPS:,} PPO steps  |  {NUM_EVAL_EPISODES} eval eps  |  seed={SEED}\n"
        f"{note}",
        fontsize=10, y=1.03,
    )
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    seed          = jax.random.PRNGKey(SEED)
    region_params = load_region_yamls(NUM_REGIONS)

    print("=== Sectoral Welfloss Validation (4-condition 2x2) ===")
    print(f"Regions: {NUM_REGIONS}  |  rho={DEST_ALLOC_PERSISTENCE}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print()

    results: dict[str, np.ndarray] = {}
    for cond_i, (lbl, cfg) in enumerate(CONDITIONS.items()):
        cond_key = jax.random.fold_in(seed, cond_i)
        results[lbl] = train_and_collect(
            cbam_tariff_rate=cfg["cbam_tariff_rate"],
            sectoral_welfloss=cfg["sectoral_welfloss"],
            region_params=region_params,
            seed=cond_key,
        )
        print(f"  [{lbl.strip()}] Done. trade_flows: {results[lbl].shape}")

    # Emissions intensity for dirty/clean sector split
    try:
        sample_env          = _build_env(0.0, False, region_params)._env
        region_labels       = list(sample_env.mrio_region_labels)
        emissions_intensity = np.array(sample_env.emissions_intensity)  # (NR, NS)
    except Exception:
        region_labels       = [f"Region {i}" for i in range(NUM_REGIONS)]
        emissions_intensity = np.ones((NUM_REGIONS, 1))

    print(f"\nRegion labels: {region_labels}")
    print(f"Emissions intensity shape: {emissions_intensity.shape}")

    print("\nGenerating figure...")
    fig = make_figure(results, region_labels, emissions_intensity)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"sectoral_welfloss_validation_{timestamp}.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
