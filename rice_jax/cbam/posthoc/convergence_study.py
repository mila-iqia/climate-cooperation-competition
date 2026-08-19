"""convergence_study.py

Train RiceMRIO and track convergence with CBAM on or off:
  - episode return (raw) and reward variance over training
  - per-action-type mean and std over training

Environment: RiceMRIO, canonical 9-region vulnerability setup, mrio_trade=True,
             sector_granularity="emissions-simple",
             eu_region_idx=3.

Outputs:
  training_logs/convergence_<LABEL>_<TIMESTAMP>.csv
  plots/convergence_<LABEL>_<TIMESTAMP>.png

Usage:
    # No CBAM (control; default)
    python cbam/posthoc/convergence_study.py

    # With differential CBAM
    python cbam/posthoc/convergence_study.py --cbam-mode differential

    # Three canonical seeds, with a shorter smoke-test budget
    python cbam/posthoc/convergence_study.py --timesteps 100000 --seeds 0 1 2

    # Regenerate plot from existing CSV (never initialises JAX)
    python cbam/posthoc/convergence_study.py --plot-only training_logs/convergence_no_cbam_XYZ.csv
"""

import os
import sys
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

import argparse
from datetime import datetime

import matplotlib
import jaxnasium as jym

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

jym.enable_compilation_cache()

# JAX / jaxnasium / rice_jax are imported lazily inside main() so that
# --plot-only mode never touches JAX.  JAX's initialization pollutes the
# matplotlib Agg renderer on macOS, causing savefig to produce blank PNGs.

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_REGIONS: int = 9
EU_REGION_IDX: int = 3
NUM_ENVS: int = 8
NUM_STEPS: int = 100
NUM_EVAL_EPISODES: int = 8
EVAL_LAST_T: int = 5
REGION_LABELS = [
    "RoW", "Rus+Eur.", "MENA", "Europe(EU)", "SSA-Mining",
    "Americas", "SE Asia", "China", "India",
]
LEGACY_REGION_LABELS = [
    "SSA", "S.Asia", "N.America", "MENA", "LatAm", "Europe(EU)", "E.Asia",
]
CANONICAL_SEEDS = (0, 1, 2)

# The canonical factory and PPO kwargs are imported inside training functions;
# plot-only mode must remain free of JAX initialization on macOS.

# ── Action layout for emissions-simple ───────────────────────────────────────
_N_SECTORS = 2             # emissions-simple: dirty, clean
_SECTOR_NAMES = ["dirty(CBAM)", "clean"]


def _action_layout(num_regions: int | None = None):
    """Return (export_dims_per_agent, mitigation_base, savings_base)."""
    nr = NUM_REGIONS if num_regions is None else num_regions
    export_per_agent = _N_SECTORS * nr        # 2*NR destinations-by-sector
    mitig_base = nr * export_per_agent        # end of the export block
    savings_base = mitig_base + nr
    return export_per_agent, mitig_base, savings_base

# ── Argument parsing ──────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--timesteps", type=int, default=None,
                   help="PPO steps per training run "
                        "(default: CANONICAL_TRAIN_KWARGS['total_timesteps'])")
    p.add_argument("--seed", type=int, default=None,
                   help="Legacy single-seed alias; superseded by --seeds")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Independent training seeds (default: 0 1 2)")
    p.add_argument("--cbam-rate", type=float, default=0.0,
                   help="Flat tariff rate; used when --cbam-mode=flat")
    p.add_argument("--cbam-mode", choices=("none", "flat", "differential"),
                   default=None,
                   help="CBAM model: none, flat, or differential (default: legacy rate resolution)")
    p.add_argument("--log-interval", type=int, default=None,
                   help="Log every N training iterations (default: ~100 rows over "
                        "the run). CANONICAL_TRAIN_KWARGS uses 50, which yields "
                        "only ~13 rows at the 500k canonical budget -- too sparse "
                        "to read convergence from.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip plot generation after training")
    p.add_argument("--plot-only", metavar="CSV", nargs="+",
                   help="Skip training; regenerate plot from one or more CSVs")
    p.add_argument("--num-regions", type=int, default=9,
                   help="Number of regions (canonical default: 9)")
    p.add_argument("--eu-idx", type=int, default=None,
                   help="EU region index (0-based; canonical 9-region default: 3)")
    p.add_argument("--yaml-dir", type=str, default=None,
                   help="Canonical 9-region YAML directory override/check")
    p.add_argument("--flat-tariff", action="store_true",
                   help="Use legacy flat tariff instead of canonical differential CBAM")
    return p.parse_args()


# ── Environment builder ───────────────────────────────────────────────────────


def build_env(cbam_rate: float = 0.0, log_info_fn=None, wrap: bool = True,
              yaml_dir: str | None = None, flat_tariff: bool = False,
              cbam_mode: str | None = None):
    from cbam.config.canonical_config import CANONICAL_YAML_DIR, make_canonical_env

    if NUM_REGIONS != 9 or EU_REGION_IDX != 3:
        raise ValueError(
            "convergence_study.py now targets the canonical 9-region setup; "
            "use cbam_convergence_multiseed.py for other region aggregations."
        )
    if cbam_mode is not None:
        tariff_mode = "flat" if cbam_mode == "none" else cbam_mode
        tariff_rate = 0.0 if cbam_mode == "none" else cbam_rate
    else:
        # Backward-compatible resolution for old commands using --cbam-rate.
        tariff_mode = "flat" if flat_tariff or cbam_rate <= 0.0 else "differential"
        tariff_rate = cbam_rate
    overrides = {
        "cbam_tariff_rate": tariff_rate,
        "cbam_tariff_mode": tariff_mode,
    }
    if yaml_dir is not None and os.path.abspath(yaml_dir) != os.path.abspath(CANONICAL_YAML_DIR):
        raise ValueError("--yaml-dir must point to the canonical 9-region YAML directory")
    env = make_canonical_env(for_training=wrap, **overrides)
    if log_info_fn is not None:
        from _experiment_util import with_log_info_fn

        env = with_log_info_fn(env, log_info_fn)
    return env


# ── PPO builder ───────────────────────────────────────────────────────────────


def build_ppo(total_timesteps: int, csv_path: str, log_interval: int | None = None):
    from cbam.config.canonical_config import CANONICAL_TRAIN_KWARGS
    from rice_jax.training import (
        RCPOMonitoredPPO,
        make_combined_log_fn,
        make_csv_log_fn,
        make_print_log_fn,
    )

    num_iters = (
        total_timesteps
        // CANONICAL_TRAIN_KWARGS["num_steps"]
        // CANONICAL_TRAIN_KWARGS["num_envs"]
    )
    log_fn = make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )
    # A convergence study needs a dense CSV: the canonical log_interval of 50
    # gives ~13 rows over 625 iterations.  Default to ~100 rows instead.
    if log_interval is None:
        log_interval = max(1, num_iters // 100)

    ppo_kwargs = dict(CANONICAL_TRAIN_KWARGS)
    ppo_kwargs.update(
        total_timesteps=total_timesteps,
        log_function=log_fn,
        log_interval=log_interval,
    )
    print(f"  logging every {log_interval} of {num_iters} iterations "
          f"(~{num_iters // max(1, log_interval)} CSV rows)")
    return RCPOMonitoredPPO(**ppo_kwargs)


# ── Custom log_info_fn capturing trade_flows only ────────────────────────────


def _trade_flows_log_fn(state: dict, actions: dict, **kwargs) -> dict:
    """Minimal log_info_fn returning only trade_flows to avoid memory bloat."""
    tf = state.get("trade_flows")   # (NR, NR, NS)  [from_r, to_r, sector]
    if tf is None:
        return {}
    return {"trade_flows": tf}


# ── EU share eval ─────────────────────────────────────────────────────────────


def eval_eu_shares(agent, cbam_rate: float, key, *, flat_tariff: bool = False,
                   cbam_mode: str | None = None,
                   yaml_dir: str | None = None) -> dict:
    """Run NUM_EVAL_EPISODES with the trained policy and return EU dirty/clean
    export share per agent per episode timestep, averaged across episodes.

    EU share for agent a, sector s at step t:
        share[t, a, s] = trade_flows[t, a, EU_IDX, s] / sum_d(trade_flows[t, a, d, s])

    Returns dict:
        "dirty_eu_share": np.ndarray  (T, NR)
        "clean_eu_share": np.ndarray  (T, NR)
    """
    import jax
    import numpy as np

    from _experiment_util import run_single_episode

    eval_env = build_env(cbam_rate=cbam_rate, log_info_fn=_trade_flows_log_fn,
                         wrap=False, yaml_dir=yaml_dir, flat_tariff=flat_tariff,
                         cbam_mode=cbam_mode)

    dirty_eps, clean_eps = [], []
    for ep in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, ep)
        logs = run_single_episode(ep_key, eval_env, agent)
        tf = np.array(logs["trade_flows"])   # (T, NR, NR, NS)  [t, from, to, sector]
        eu_flow    = tf[:, :, EU_REGION_IDX, :]      # (T, NR, NS)
        total_flow = tf.sum(axis=2) + 1e-9           # (T, NR, NS)
        share = eu_flow / total_flow                  # (T, NR, NS)
        dirty_eps.append(share[:, :, 0])              # (T, NR)
        clean_eps.append(share[:, :, 1])              # (T, NR)

    return {
        "dirty_eu_share": np.mean(dirty_eps, axis=0),   # (T, NR)
        "clean_eu_share": np.mean(clean_eps, axis=0),   # (T, NR)
        "dirty_eu_share_last5": np.mean(
            [episode[-EVAL_LAST_T:].mean(axis=0) for episode in dirty_eps], axis=0
        ),
        "clean_eu_share_last5": np.mean(
            [episode[-EVAL_LAST_T:].mean(axis=0) for episode in clean_eps], axis=0
        ),
    }


# ── EU share plot ─────────────────────────────────────────────────────────────


def make_eu_share_plot(eu_shares: dict, plot_path: str, cbam_rate: float = 0.0,
                       cbam_mode: str | None = None) -> None:
    """2-panel figure: EU dirty share + EU clean share per agent over episode timesteps."""
    dirty = eu_shares["dirty_eu_share"]   # (T, NR)
    clean = eu_shares["clean_eu_share"]   # (T, NR)
    T = dirty.shape[0]
    timesteps = np.arange(T)
    N = dirty.shape[1]
    region_labels = REGION_LABELS[:N]
    cmap = plt.cm.tab10

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    tariff_label = cbam_mode or f"flat tau={cbam_rate}"
    fig.suptitle(
        f"EU export share per agent  ({tariff_label},  avg {NUM_EVAL_EPISODES} eval episodes)",
        fontsize=12,
    )

    for ax, data, sector in zip(axes, [dirty, clean], ["dirty / CBAM", "clean"]):
        for a in range(N):
            lbl = region_labels[a] if a < len(region_labels) else f"r{a}"
            ax.plot(timesteps, data[:, a], color=cmap(a / N), lw=1.5, label=lbl)
        ax.set_ylabel("Share of exports going to EU")
        ax.set_title(f"{sector} sector — EU-destined fraction per exporting agent")
        ax.legend(fontsize=8, ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)

    axes[-1].set_xlabel("Episode timestep")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"EU share plot saved: {plot_path}")


# ── Training convergence plot ─────────────────────────────────────────────────


def make_convergence_plot(csv_path: str, plot_path: str, cbam_rate: float = 0.0,
                          *, region_labels: list[str] | None = None,
                          title_label: str | None = None,
                          cbam_mode: str | None = None) -> None:
    df = pd.read_csv(csv_path)

    action_mean_cols = sorted(
        [c for c in df.columns if c.startswith("action_mean_")],
        key=lambda c: int(c.split("_")[-1]),
    )
    action_var_cols = sorted(
        [c for c in df.columns if c.startswith("action_var_")],
        key=lambda c: int(c.split("_")[-1]),
    )

    iters = df["iteration"].values
    total_dims = len(action_mean_cols)

    act_means = df[action_mean_cols].apply(pd.to_numeric, errors="coerce").values  # (T, D)
    act_vars  = df[action_var_cols].apply(pd.to_numeric, errors="coerce").values.clip(0)
    act_stds  = np.sqrt(act_vars)

    inferred_dims = total_dims
    inferred_n = next(
        (n for n in range(1, 100) if inferred_dims == n * (_N_SECTORS * n + 2)),
        NUM_REGIONS,
    )
    N = inferred_n
    EXPORT_PER_AGENT, MITIG_BASE, SAVINGS_BASE = _action_layout(N)
    max_dim = total_dims - 1

    # ── Per-agent export means (avg over destinations within each sector) ─────
    # Action-key-major: agent a, sector s -> [a*EXPORT_PER_AGENT + s*N .. +N-1]
    def _agent_sector_mean(a, s, arr):
        base = a * EXPORT_PER_AGENT + s * N
        cols = [base + d for d in range(N) if base + d <= max_dim]
        return arr[:, cols].mean(axis=1) if cols else np.full(len(iters), np.nan)

    # ── Cross-agent aggregates for mitigation/savings ─────────────────────────
    # Each lives in its own contiguous block, one entry per agent.
    def _block_indices(base):
        return [base + a for a in range(N) if base + a <= max_dim]

    def _mean_agg(indices, arr):
        return arr[:, indices].mean(axis=1) if indices else np.full(len(iters), np.nan)

    mitig_mean   = _mean_agg(_block_indices(MITIG_BASE),   act_means)
    savings_mean = _mean_agg(_block_indices(SAVINGS_BASE), act_means)
    mitig_std    = _mean_agg(_block_indices(MITIG_BASE),   act_stds)
    savings_std  = _mean_agg(_block_indices(SAVINGS_BASE), act_stds)

    # ── Episode return ────────────────────────────────────────────────────────
    ep_ret = np.full(len(iters), np.nan)
    ep_ret_std = np.full(len(iters), np.nan)
    if "ep_return_mean" in df.columns:
        ep_ret = pd.to_numeric(df["ep_return_mean"], errors="coerce").values
    if "ep_return_std" in df.columns:
        ep_ret_std = pd.to_numeric(df["ep_return_std"], errors="coerce").values
    valid = ~np.isnan(ep_ret)

    # ── Region labels (7-region standard ordering) ───────────────────────────
    region_labels = region_labels or (
        REGION_LABELS[:N] if N == 9 else LEGACY_REGION_LABELS[:N]
    )
    agent_cmap = plt.cm.tab10

    # ── 6-panel figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(13, 22), sharex=False)
    tariff_label = cbam_mode or f"tau={cbam_rate}"
    fig.suptitle(
        f"Convergence  (RiceMRIO, {tariff_label},  {N} regions)"
        + (f" — {title_label}" if title_label else ""),
        fontsize=13,
    )

    # Panel 0 — episode return (full training)
    ax = axes[0]
    if valid.any():
        ax.plot(iters[valid], ep_ret[valid], color="darkorchid", lw=1.8,
                label="ep return mean (raw)")
        ax.fill_between(iters[valid],
                        ep_ret[valid] - ep_ret_std[valid],
                        ep_ret[valid] + ep_ret_std[valid],
                        alpha=0.2, color="darkorchid", label="±1 std")
    ax.set_ylabel("Episode return")
    ax.set_xlabel("Iteration")
    ax.set_title("Episode return — full training  "
                 "(raw env reward; PPO normalises internally for its updates)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 1 — early zoom (first 10%)
    ax = axes[1]
    cutoff = max(1, int(iters.max() * 0.10))
    early = valid & (iters <= cutoff)
    if early.sum() > 2:
        ax.plot(iters[early], ep_ret[early], color="darkorchid", lw=1.8)
        ax.fill_between(iters[early],
                        ep_ret[early] - ep_ret_std[early],
                        ep_ret[early] + ep_ret_std[early],
                        alpha=0.2, color="darkorchid")
    ax.set_ylabel("Episode return")
    ax.set_xlabel("Iteration")
    ax.set_title(f"Early learning phase (first 10% of iterations, ≤{cutoff})")
    ax.grid(True, alpha=0.3)

    # Panel 2 — per-agent dirty (CBAM) sector export mean
    ax = axes[2]
    for a in range(N):
        vals = _agent_sector_mean(a, 0, act_means)
        ax.plot(iters, vals, color=agent_cmap(a / N), lw=1.3,
                label=region_labels[a] if a < len(region_labels) else f"r{a}")
    ax.axhline(5, color="k", lw=0.8, ls="--", alpha=0.4, label="midpoint (5)")
    ax.set_ylabel("Action mean (0–9)")
    ax.set_xlabel("Iteration")
    ax.set_title(f"Export allocation — dirty/CBAM sector  (per agent, avg over {N} destinations)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Panel 3 — per-agent clean sector export mean
    ax = axes[3]
    for a in range(N):
        vals = _agent_sector_mean(a, 1, act_means)
        ax.plot(iters, vals, color=agent_cmap(a / N), lw=1.3,
                label=region_labels[a] if a < len(region_labels) else f"r{a}")
    ax.axhline(5, color="k", lw=0.8, ls="--", alpha=0.4, label="midpoint (5)")
    ax.set_ylabel("Action mean (0–9)")
    ax.set_xlabel("Iteration")
    ax.set_title(f"Export allocation — clean sector  (per agent, avg over {N} destinations)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)

    # Panel 4 — mitigation + savings means (avg across agents)
    ax = axes[4]
    ax.plot(iters, mitig_mean,   color="tab:blue",   lw=1.5, label="mitigation_rate (avg)")
    ax.plot(iters, savings_mean, color="tab:orange",  lw=1.5, label="savings_rate (avg)")
    ax.axhline(5, color="k", lw=0.8, ls="--", alpha=0.4, label="midpoint (5)")
    ax.set_ylabel("Action mean (0–9)")
    ax.set_xlabel("Iteration")
    ax.set_title(f"Mitigation & savings rate means (avg across {N} agents)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5 — action stds (avg across agents per type)
    ax = axes[5]
    for a in range(N):
        dirty_std = _agent_sector_mean(a, 0, act_stds)
        ax.plot(iters, dirty_std, color=agent_cmap(a / N), lw=1.0, alpha=0.7,
                label=region_labels[a] if a < len(region_labels) else f"r{a}")
    ax.plot(iters, mitig_std,   color="tab:blue",   lw=1.5, ls="--", label="mitigation (avg)")
    ax.plot(iters, savings_std, color="tab:orange",  lw=1.5, ls="--", label="savings (avg)")
    ax.set_ylabel("Action std  (↓ = deterministic)")
    ax.set_xlabel("Iteration")
    ax.set_title("Action std — dirty export per agent + mitigation/savings (avg)")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else ".", exist_ok=True)
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {plot_path}")


# ── Multi-seed aggregation ────────────────────────────────────────────────────


def _align_csv_metric(csv_paths: list[str], column: str) -> tuple[np.ndarray, np.ndarray]:
    """Align one numeric CSV metric to a common iteration grid."""
    series = []
    grids = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        frame = pd.DataFrame({"iteration": frame["iteration"], column: values})
        frame = frame.dropna(subset=[column]).set_index("iteration")[column]
        if not frame.empty:
            series.append(frame)
            grids.append(set(frame.index.tolist()))
    if not series:
        return np.array([]), np.empty((0, 0))
    iterations = np.array(sorted(set.union(*grids)))
    values = np.full((len(series), len(iterations)), np.nan)
    for row, current in enumerate(series):
        for col, iteration in enumerate(iterations):
            if iteration in current.index:
                values[row, col] = float(current.loc[iteration])
        valid = np.flatnonzero(~np.isnan(values[row]))
        if len(valid):
            values[row] = np.interp(
                np.arange(len(iterations)), valid, values[row, valid],
                left=values[row, valid[0]], right=values[row, valid[-1]],
            )
    return iterations, values


def make_multiseed_plot(csv_paths: list[str], plot_path: str,
                        cbam_rate: float, seeds: list[int],
                        cbam_mode: str | None = None) -> None:
    """Plot mean ± seed spread for the core convergence signals."""
    metrics = [
        ("ep_return_mean", "Raw episode return", "tab:purple"),
        ("ep_return_std", "Within-batch episode-return std", "tab:blue"),
        ("reward_var", "Per-step reward variance", "tab:orange"),
        ("cbam_lambda", "RCPO lambda", "tab:green"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, (column, ylabel, color) in zip(axes.flat, metrics):
        iterations, values = _align_csv_metric(csv_paths, column)
        if values.size:
            mean = np.nanmean(values, axis=0)
            spread = np.nanstd(values, axis=0)
            ax.plot(iterations, mean, color=color, lw=1.8, label="mean")
            ax.fill_between(iterations, mean - spread, mean + spread,
                            color=color, alpha=0.18, label="±1 seed std")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, f"{column} not logged", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    tariff_label = cbam_mode or f"flat tau={cbam_rate}"
    fig.suptitle(
        f"Multi-seed convergence (9-region {tariff_label})\n"
        f"seeds={seeds}", fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Multi-seed plot saved: {plot_path}")


def _mean_eu_shares(seed_shares: list[dict]) -> dict:
    """Average per-seed EU-share evaluation arrays."""
    return {
        key: np.mean([shares[key] for shares in seed_shares], axis=0)
        for key in seed_shares[0]
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    global NUM_REGIONS, EU_REGION_IDX, REGION_LABELS
    args = _parse_args()

    NUM_REGIONS = args.num_regions
    EU_REGION_IDX = args.eu_idx if args.eu_idx is not None else 3
    if NUM_REGIONS != 9 or EU_REGION_IDX != 3:
        raise SystemExit(
            "This convergence study targets the canonical 9-region setup "
            "(use --num-regions 9 --eu-idx 3)."
        )
    REGION_LABELS = REGION_LABELS[:NUM_REGIONS]

    if args.seed is not None and args.seeds is not None:
        raise SystemExit("Use either --seed or --seeds, not both.")
    seeds = list(args.seeds if args.seeds is not None else (
        [args.seed] if args.seed is not None else CANONICAL_SEEDS
    ))
    if not seeds:
        raise SystemExit("At least one training seed is required.")

    if args.cbam_mode is None:
        cbam_mode = "flat" if args.flat_tariff or args.cbam_rate <= 0.0 else "differential"
    else:
        cbam_mode = args.cbam_mode
    if args.flat_tariff and args.cbam_mode == "differential":
        raise SystemExit("--flat-tariff conflicts with --cbam-mode differential.")
    if cbam_mode == "differential" and args.cbam_rate != 0.0:
        print("  Note: --cbam-rate is ignored in differential mode; MACs set tau_eff.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "training_logs"
    plot_dir = "plots"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    if args.plot_only:
        label = ("differential" if cbam_mode == "differential"
                 else "flat" + f"{int(args.cbam_rate * 100):03d}"
                 if cbam_mode == "flat" and args.cbam_rate > 0 else "no_cbam")
        plot_path = os.path.join(plot_dir, f"convergence_{label}_plot_only.png")
        if len(args.plot_only) == 1:
            make_convergence_plot(args.plot_only[0], plot_path,
                                                                    cbam_rate=args.cbam_rate, cbam_mode=cbam_mode)
        else:
            make_multiseed_plot(args.plot_only, plot_path,
                                                                args.cbam_rate, seeds, cbam_mode=cbam_mode)
        return

    label = (
        "differential" if cbam_mode == "differential"
        else f"flat{int(args.cbam_rate * 100):03d}" if cbam_mode == "flat" and args.cbam_rate > 0
        else "no_cbam"
    )
    # Resolved here (not at argparse time) so --plot-only stays JAX-free.
    from cbam.config.canonical_config import canonical_train_kwargs

    timesteps = (
        args.timesteps if args.timesteps is not None
        else canonical_train_kwargs()["total_timesteps"]
    )

    csv_paths = []
    trained = []
    print("=== Convergence study ===")
    print(f"  cbam_mode   : {cbam_mode}")
    print(f"  cbam_rate   : {args.cbam_rate if cbam_mode == 'flat' else 'n/a'}")
    print(f"  num_regions : {NUM_REGIONS}")
    print(f"  eu_idx      : {EU_REGION_IDX}")
    print(f"  timesteps   : {timesteps:,}")
    print(f"  seeds       : {seeds}")

    import jax
    for seed in seeds:
        csv_path = os.path.join(log_dir, f"convergence_{label}_s{seed}_{ts}.csv")
        print(f"\nTraining seed {seed} → {csv_path}")
        env = build_env(cbam_rate=args.cbam_rate, yaml_dir=args.yaml_dir,
                flat_tariff=args.flat_tariff, cbam_mode=cbam_mode)
        ppo = build_ppo(timesteps, csv_path, log_interval=args.log_interval)
        agent, _metrics = jym.precompile(ppo.train, jax.random.PRNGKey(seed), env)()
        csv_paths.append(csv_path)
        trained.append((agent, seed))

    if args.no_plot:
        print(f"\nTraining complete. CSVs: {csv_paths}  (--no-plot set)")
        return

    plot_path = os.path.join(plot_dir, f"convergence_{label}_multiseed_{ts}.png")
    make_multiseed_plot(csv_paths, plot_path, args.cbam_rate, seeds,
                        cbam_mode=cbam_mode)
    for csv_path in csv_paths:
        make_convergence_plot(
            csv_path,
            csv_path.replace(log_dir, plot_dir).replace(".csv", ".png"),
            cbam_rate=args.cbam_rate,
            cbam_mode=cbam_mode,
        )

    print(f"Running {NUM_EVAL_EPISODES} eval episodes per seed for EU export shares...")
    seed_shares = [
        eval_eu_shares(
            agent, args.cbam_rate, jax.random.PRNGKey(seed),
            yaml_dir=args.yaml_dir, flat_tariff=args.flat_tariff,
            cbam_mode=cbam_mode,
        )
        for agent, seed in trained
    ]
    eu_plot_path = os.path.join(plot_dir, f"convergence_{label}_eu_shares_{ts}.png")
    make_eu_share_plot(_mean_eu_shares(seed_shares), eu_plot_path,
                       cbam_rate=args.cbam_rate, cbam_mode=cbam_mode)


if __name__ == "__main__":
    main()
