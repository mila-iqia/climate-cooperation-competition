"""cbam_convergence_multiseed.py

Multi-seed convergence study for the two conditions that carry the core
argument in the differential CBAM paper:

  ctrl  — no CBAM   (flat τ=0, "burden" baseline)
  cbam  — differential CBAM (self-incentivising tariff, "effort" best case)

Runs N_SEEDS independent training runs per condition and plots three panels:

  Panel 1  ep_return_mean ± σ shaded band across seeds
           Raw (un-normalised) cumulative episode return logged by LogWrapper.
           With diff_reward_mode=True this equals Σ_t Δ-utility[t], i.e. the
           total utility gain over the episode.  This is the absolute metric to
           watch for a plateau — it is NOT the batch-normalised training reward.

  Panel 2  ep_return_std (within-batch spread of episode returns, mean across
           seeds).  Decreasing → episodes becoming more similar → convergence.

  Panel 3  reward_var (per-step reward variance within batch, mean across
           seeds).  A secondary variance-reduction index that is independent of
           episode boundaries.

Note: normalize_rewards=True in PPO normalises the trajectory rewards before
computing GAE/value targets, but LogWrapper accumulates rewards *before* this
step, so ep_return_mean / ep_return_std in the CSV are always in raw units.

Usage (from rice_jax/, rice-jax conda env):
    python validation/cbam_convergence_multiseed.py
    python validation/cbam_convergence_multiseed.py --timesteps 500000 --seeds 5
    python validation/cbam_convergence_multiseed.py --replot <pickle.pkl>
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import pickle
import time
from dataclasses import replace
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import jaxnasium as jym
from training_monitor import (
    MonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
)
from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls


def _cbam_and_utility_log_info_fn(state: dict, actions: dict) -> dict:
    """Combined log_info_fn: RCPO cost + mean utility + RICE economic decomposition.

    Scalar (mean) keys are kept for backward-compat with make_csv_log_fn.
    Per-region arrays (shape NR,) are added for disaggregated CSV logging and
    per-region plot panels; make_csv_log_fn writes them as utility_r0..rN, etc.
    """
    return {
        # RCPO Lagrange update signal
        "cbam_cost_all_regions":       state["cbam_cost_all_regions"],               # (NR,)
        # Aggregate scalars (backward-compat with base CSV columns)
        "mean_utility_step":           state["utility_all_regions"].mean(),           # scalar
        "mean_gross_output_step":      state["gross_output_all_regions"].mean(),      # scalar
        "mean_consumption_step":       state["aggregate_consumption"].mean(),         # scalar
        "mean_abatement_cost_step":    state["abatement_cost_all_regions"].mean(),    # scalar
        "mean_mitigation_step":        state["mitigation_rates_all_regions"].mean(),  # scalar
        # Per-region arrays — written as <name>_r0 .. _rN by make_csv_log_fn
        "utility_per_region":          state["utility_all_regions"],                  # (NR,)
        "gross_output_per_region":     state["gross_output_all_regions"],             # (NR,)
        "consumption_per_region":      state["aggregate_consumption"],                # (NR,)
        "abatement_cost_per_region":   state["abatement_cost_all_regions"],          # (NR,)
        "mitigation_per_region":       state["mitigation_rates_all_regions"],         # (NR,)
    }


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
NON_EU = [r for r in range(NUM_REGIONS) if r != EU_IDX]

TOTAL_TIMESTEPS     = 500_000
NUM_ENVS            = 8
NUM_STEPS           = 100
CBAM_LAMBDA_INIT    = 1.0
WELFARE_LOSS_WEIGHT = 5.0
TRANSFER_MODE       = "abatement"
TRANSFER_ALLOC      = "effort"
REVENUE_SHARE       = 1.0

# Default seeds — 5 yields reasonable error estimates; use --seeds 3 for speed
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Condition definitions.  Both conditions share the same base env; they differ
# only in cbam_tariff_mode and cbam_tariff_rate.
CONDITIONS = {
    "ctrl": dict(
        cbam_tariff_mode = "flat",
        cbam_tariff_rate = 0.0,   # no CBAM cost
        label            = "ctrl (no CBAM)",
        color            = "#1f77b4",   # blue
        linestyle        = "-",
    ),
    "cbam": dict(
        cbam_tariff_mode = "differential",
        cbam_tariff_rate = 0.0,   # placeholder; rate computed per-region at runtime
        label            = "cbam (differential)",
        color            = "#2ca02c",   # green
        linestyle        = "--",
    ),
}

OUTPUT_DIR = "plots"
LOG_DIR    = "training_logs"
LOG_PREFIX = "multiseed_"

_BASE_ENV_KWARGS = dict(
    num_regions                  = NUM_REGIONS,
    mrio_data_root               = MRIO_ROOT,
    mrio_trade                   = True,
    eu_region_idx                = EU_IDX,
    dest_alloc_persistence       = 0.55,
    dest_alloc_baseline_decay    = 0.0,
    diff_reward_mode             = True,
    num_discrete_action_levels   = 10,
    sector_granularity           = "emissions-simple",
    sectoral_welfloss            = True,
    welfare_loss_per_unit_tariff = WELFARE_LOSS_WEIGHT,
    revenue_share                = REVENUE_SHARE,
    transfer_mode                = TRANSFER_MODE,
    transfer_allocation          = TRANSFER_ALLOC,
    reward_mode                  = "additive_cbam",
    cbam_lambda_init             = CBAM_LAMBDA_INIT,
    fixed_savings_rate           = True,   # savings fixed at Nordhaus 0.2; focus on mitigation/export
)

_PPO_KWARGS = dict(
    num_steps              = NUM_STEPS,
    num_envs               = NUM_ENVS,
    learning_rate          = 3e-4,
    num_minibatches        = 4,
    num_epochs             = 8,
    ent_coef               = 0.01,
    anneal_ent_coef        = 0.0,
    gamma                  = 0.99,
    gae_lambda             = 0.95,
    max_grad_norm          = 1.0,
    clip_coef              = 0.2,
    clip_coef_vf           = 0.5,
    vf_coef                = 0.5,
    normalize_observations = True,
    normalize_rewards      = True,
    log_interval           = 50,
)

# ── Training helpers ──────────────────────────────────────────────────────────

def _build_env(cond_name: str) -> jym.LogWrapper:
    cond = CONDITIONS[cond_name]
    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR)
    env = RiceMRIO(
        region_params    = region_params,
        log_info_fn      = _cbam_and_utility_log_info_fn,
        cbam_tariff_mode = cond["cbam_tariff_mode"],
        cbam_tariff_rate = cond["cbam_tariff_rate"],
        **_BASE_ENV_KWARGS,
    )
    return jym.LogWrapper(env)


def _csv_path(cond_name: str, seed: int, run_id: str) -> str:
    _os.makedirs(LOG_DIR, exist_ok=True)
    return _os.path.join(LOG_DIR, f"{LOG_PREFIX}{run_id}_{cond_name}_s{seed}.csv")


def _train_one(cond_name: str, seed: int, total_timesteps: int, run_id: str) -> str:
    """Train one (condition, seed) run and return the CSV path."""
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    key       = jax.random.PRNGKey(seed)
    csv_p     = _csv_path(cond_name, seed, run_id)
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _DROP = {"action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _DROP}, iteration)

    log_fn    = make_combined_log_fn(
        _compact,
        make_csv_log_fn(csv_p),
    )
    env = _build_env(cond_name)
    ppo = MonitoredPPO(
        total_timesteps = total_timesteps,
        log_function    = log_fn,
        **_PPO_KWARGS,
    )
    cond_label = CONDITIONS[cond_name]["label"]
    print(f"\n{'━'*60}")
    print(f"  Training: {cond_label}  seed={seed}")
    print(f"{'━'*60}")
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    print(f"  Done in {time.perf_counter() - t0:.0f}s  →  {csv_p}")
    return csv_p


# ── CSV alignment ─────────────────────────────────────────────────────────────

def _load_and_align(csv_paths: list[str], col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load `col` from each CSV, align to a common iteration grid (union of all
    iteration values), forward-fill NaN gaps, and return:
      - iterations : 1-D array of common iteration indices
      - matrix     : (n_seeds, n_iters) array of values
    NaN rows (no completed episode in that batch) are forward-filled to avoid
    distorting the mean/std.
    """
    series_list = []
    iter_sets   = []
    for p in csv_paths:
        df = pd.read_csv(p)
        # ep_return_mean / ep_return_std are empty-string when no episode done
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("iteration")[col]
        series_list.append(df)
        iter_sets.append(set(df.index.tolist()))

    common_iters = sorted(set.union(*iter_sets))
    aligned = np.full((len(csv_paths), len(common_iters)), np.nan)
    for i, s in enumerate(series_list):
        for j, it in enumerate(common_iters):
            if it in s.index:
                aligned[i, j] = float(s.loc[it])
    # Forward-fill NaN within each seed row
    for i in range(len(csv_paths)):
        mask = np.isnan(aligned[i])
        if mask.any() and not mask.all():
            idx = np.where(~mask)[0]
            aligned[i] = np.interp(
                np.arange(len(common_iters)),
                idx, aligned[i, idx],
                left=aligned[i, idx[0]],
                right=aligned[i, idx[-1]],
            )
    return np.array(common_iters), aligned


# ── Plotting ──────────────────────────────────────────────────────────────────

_ROLL = 10   # rolling-window width (iterations)


def _smooth(arr: np.ndarray) -> np.ndarray:
    """Rolling mean along last axis (per-seed), NaN-safe via pandas."""
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        s = pd.Series(arr[i]).rolling(_ROLL, min_periods=1).mean()
        out[i] = s.values
    return out


def _plot(run_data: dict, timestamp: str) -> str:
    """
    Produces a figure with two sections saved to plots/.

    SECTION A — Aggregate (mean across regions), rows 0-1:
      Row 0: [utility]  [ep_return_std]  [action_var]
      Row 1: [gross output]  [consumption]  [abatement cost + mitigation twinx]

    SECTION B — Per-region disaggregation (if per-region columns present), rows 2-3:
      Row 2: [per-region utility]  [per-region gross output]  [per-region consumption]
      Row 3: [per-region abatement cost]  [per-region mitigation]  [legend panel]
    Each per-region panel shows one line per region (coloured by region), mean across
    seeds, for each condition (linestyle varies by condition).  No σ bands to keep
    the 9-region × 2-condition panels readable.
    """
    # Detect whether per-region columns are present
    _has_per_region = False
    for _info in run_data.values():
        if isinstance(_info, dict) and "csv_paths" in _info and _info["csv_paths"]:
            _probe = pd.read_csv(_info["csv_paths"][0])
            _has_per_region = any(c.startswith("mitigation_r") for c in _probe.columns)
            break

    _nrows = 4 if _has_per_region else 2
    fig = plt.figure(figsize=(18, 5 * _nrows))
    n_steps_k = run_data.get("_meta", {}).get("timesteps", TOTAL_TIMESTEPS) // 1000
    fig.suptitle(
        "Multi-seed convergence — differential CBAM (9-region vuln)\n"
        f"{n_steps_k}K steps · {_ROLL}-iter rolling mean",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(_nrows, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Section A axes ────────────────────────────────────────────────────────
    ax_util = fig.add_subplot(gs[0, 0])
    ax_estd = fig.add_subplot(gs[0, 1])
    ax_avar = fig.add_subplot(gs[0, 2])
    ax_go   = fig.add_subplot(gs[1, 0])
    ax_cons = fig.add_subplot(gs[1, 1])
    ax_abat = fig.add_subplot(gs[1, 2])

    # ── Section B axes (only if per-region data exists) ───────────────────────
    if _has_per_region:
        ax_pr_util  = fig.add_subplot(gs[2, 0])
        ax_pr_go    = fig.add_subplot(gs[2, 1])
        ax_pr_cons  = fig.add_subplot(gs[2, 2])
        ax_pr_abat  = fig.add_subplot(gs[3, 0])
        ax_pr_mit   = fig.add_subplot(gs[3, 1])
        ax_legend   = fig.add_subplot(gs[3, 2])
        ax_legend.axis("off")   # used only for a shared region legend

    # Per-region colours: tab10 palette, fixed mapping region_idx → colour
    _TAB10 = [plt.cm.tab10(i / 10) for i in range(10)]
    _REGION_COLORS = {r: _TAB10[r % 10] for r in range(NUM_REGIONS)}

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _plot_band(ax, paths, col, *, color, ls, label, show_n=False):
        """Plot mean ± σ band across seeds for a scalar column."""
        iters_, mat_ = _load_and_align(paths, col)
        mat_s_ = _smooth(mat_)
        mu_    = np.nanmean(mat_s_, axis=0)
        sig_   = np.nanstd(mat_s_,  axis=0)
        lbl_   = f"{label} (n={len(paths)})" if show_n else label
        ax.plot(iters_, mu_, color=color, linestyle=ls, linewidth=1.8, label=lbl_)
        ax.fill_between(iters_, mu_ - sig_, mu_ + sig_, color=color, alpha=0.15)

    def _plot_per_region(ax, paths, prefix, *, ls, nr=NUM_REGIONS):
        """Plot one line per region (mean across seeds) for column <prefix>_r<i>."""
        for r in range(nr):
            col = f"{prefix}_r{r}"
            try:
                iters_, mat_ = _load_and_align(paths, col)
            except Exception:
                continue
            mat_s_ = _smooth(mat_)
            mu_    = np.nanmean(mat_s_, axis=0)
            ax.plot(iters_, mu_,
                    color=_REGION_COLORS[r], linestyle=ls,
                    linewidth=1.2, alpha=0.85,
                    label=REGION_NAMES.get(r, f"r{r}"))

    # ── Section A: aggregate plots ────────────────────────────────────────────
    for cond_name, info in run_data.items():
        if cond_name == "_meta":
            continue
        cond  = CONDITIONS[cond_name]
        color = cond["color"]
        ls    = cond["linestyle"]
        label = cond["label"]
        paths = info["csv_paths"]

        # Row 0 — convergence signals
        _plot_band(ax_util, paths, "mean_utility",   color=color, ls=ls, label=label, show_n=True)
        _plot_band(ax_estd, paths, "ep_return_std",  color=color, ls=ls, label=label)

        first_df = pd.read_csv(paths[0])
        av_cols  = sorted(
            (c for c in first_df.columns if c.startswith("action_var_")),
            key=lambda c: int(c.split("_")[-1]),
        )
        if av_cols:
            # Compute per-group index slices from the CSV column count.
            # JAX flattens the action pytree alphabetically:
            #   fixed_savings_rate=True  → per agent: export_reallocation[NS*NR], mitigation_rate[1]
            #   fixed_savings_rate=False → per agent: export_reallocation[NS*NR], mitigation_rate[1], savings_rate[1]
            total_dims     = len(av_cols)
            dims_per_agent = total_dims // NUM_REGIONS      # 19 (fixed savings) or 20
            n_export       = dims_per_agent - 1 if dims_per_agent % 2 != 0 else dims_per_agent - 2
            # Heuristic: if total dims / NR == 19, savings is fixed (no savings col)
            has_savings    = (dims_per_agent == 20)
            n_export       = dims_per_agent - (2 if has_savings else 1)
            export_idx = [i * dims_per_agent + j
                          for i in range(NUM_REGIONS) for j in range(n_export)]
            mit_idx    = [i * dims_per_agent + n_export     for i in range(NUM_REGIONS)]
            sav_idx    = [i * dims_per_agent + n_export + 1 for i in range(NUM_REGIONS)] if has_savings else []

            _ACTION_GROUPS = [
                ("export_realloc", export_idx, ":"),
                ("mitigation",     mit_idx,    "--"),
            ]
            if sav_idx:
                _ACTION_GROUPS.append(("savings", sav_idx, "-"))

            for grp_name, grp_idx, grp_ls in _ACTION_GROUPS:
                tmp_col = f"_avar_{grp_name}"
                tmp_paths = []
                for p in paths:
                    df_av = pd.read_csv(p)
                    df_av[tmp_col] = df_av[[av_cols[i] for i in grp_idx]].mean(axis=1)
                    tmp = _os.path.splitext(p)[0] + f"_{grp_name}_tmp.csv"
                    df_av[["iteration", tmp_col]].to_csv(tmp, index=False)
                    tmp_paths.append(tmp)
                iters_av, mat_av = _load_and_align(tmp_paths, tmp_col)
                for tmp in tmp_paths:
                    try: _os.remove(tmp)
                    except OSError: pass
                mat_av_s = _smooth(mat_av)
                mu_av    = np.nanmean(mat_av_s, axis=0)
                sig_av   = np.nanstd(mat_av_s,  axis=0)
                lbl_av   = f"{label} [{grp_name}]"
                ax_avar.plot(iters_av, mu_av, color=color, linestyle=grp_ls,
                             linewidth=1.6, label=lbl_av)
                ax_avar.fill_between(iters_av, mu_av - sig_av, mu_av + sig_av,
                                     color=color, alpha=0.10)

        # Row 1 — economic decomposition
        _plot_band(ax_go,   paths, "mean_gross_output",   color=color, ls=ls, label=label)
        _plot_band(ax_cons, paths, "mean_consumption",    color=color, ls=ls, label=label)
        _plot_band(ax_abat, paths, "mean_abatement_cost", color=color, ls=ls, label=label)
        # Overlay mitigation rate on abatement axis (twinx)
        iters_mu, mat_mu = _load_and_align(paths, "mean_mitigation")
        mat_mu_s = _smooth(mat_mu)
        mu_mu    = np.nanmean(mat_mu_s, axis=0)
        sig_mu   = np.nanstd(mat_mu_s,  axis=0)
        ax2 = ax_abat.twinx()
        ax2.plot(iters_mu, mu_mu, color=color, linestyle=":",
                 linewidth=1.4, alpha=0.8, label=f"{label} μ")
        ax2.fill_between(iters_mu, mu_mu - sig_mu, mu_mu + sig_mu,
                         color=color, alpha=0.07)
        ax2.set_ylabel("Mean mitigation rate μ", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)

        # ── Section B: per-region panels ──────────────────────────────────
        if _has_per_region:
            _plot_per_region(ax_pr_util,  paths, "utility",        ls=ls)
            _plot_per_region(ax_pr_go,    paths, "gross_output",   ls=ls)
            _plot_per_region(ax_pr_cons,  paths, "consumption",    ls=ls)
            _plot_per_region(ax_pr_abat,  paths, "abatement_cost", ls=ls)
            _plot_per_region(ax_pr_mit,   paths, "mitigation",     ls=ls)

    # ── Section A decorations ─────────────────────────────────────────────────
    def _decorate(ax, xlabel, ylabel, title, note, note_y=0.05):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.annotate(note, xy=(0.97, note_y), xycoords="axes fraction",
                    ha="right", va="bottom" if note_y < 0.5 else "top",
                    fontsize=7, color="gray", style="italic")

    _decorate(ax_util, "PPO iteration",
              "Mean utility (regions × steps × envs)",
              "Absolute utility level\nmean ± σ across seeds",
              "Rise then plateau = convergence")
    _decorate(ax_estd, "PPO iteration",
              "Within-batch ep-return std",
              "Episode-return spread\n(within batch, mean ± σ)",
              "Decrease = episodes stabilising", note_y=0.95)
    _decorate(ax_avar, "PPO iteration",
              "Action variance (discrete 0–9 units²)",
              "Action variance by group\n(mean ± σ, linestyle = group)",
              "export≈8.25 max; savings converges fastest", note_y=0.95)
    _decorate(ax_go, "PPO iteration",
              "Mean gross output",
              "Gross output\n(mean ± σ across seeds)",
              "↑ output drives ↑ utility")
    _decorate(ax_cons, "PPO iteration",
              "Mean consumption",
              "Consumption\n(mean ± σ across seeds)",
              "↑ consumption drives ↑ utility")
    ax_abat.set_xlabel("PPO iteration")
    ax_abat.set_ylabel("Mean abatement cost")
    ax_abat.set_title("Abatement cost + mitigation rate\nmean ± σ across seeds", fontsize=9)
    ax_abat.legend(fontsize=7, loc="upper left")
    ax_abat.grid(True, alpha=0.3)

    # ── Section B decorations ─────────────────────────────────────────────────
    if _has_per_region:
        _cond_labels = {k: CONDITIONS[k]["label"]
                        for k in run_data if k != "_meta"}
        _ls_note = "  |  ".join(
            f"{v} = '{CONDITIONS[k]['linestyle']}'"
            for k, v in _cond_labels.items()
        )

        def _decorate_pr(ax, ylabel, title):
            ax.set_xlabel("PPO iteration", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            # Deduplicate legend entries (same region shows once per cond; keep first)
            handles, labels_ = ax.get_legend_handles_labels()
            seen = {}
            dedup_h, dedup_l = [], []
            for h, l in zip(handles, labels_):
                if l not in seen:
                    seen[l] = True
                    dedup_h.append(h)
                    dedup_l.append(l)
            ax.legend(dedup_h, dedup_l, fontsize=6, ncol=2,
                      loc="best", framealpha=0.7)

        _decorate_pr(ax_pr_util,  "Utility",        "Per-region utility\n(mean across seeds)")
        _decorate_pr(ax_pr_go,    "Gross output",   "Per-region gross output\n(mean across seeds)")
        _decorate_pr(ax_pr_cons,  "Consumption",    "Per-region consumption\n(mean across seeds)")
        _decorate_pr(ax_pr_abat,  "Abatement cost", "Per-region abatement cost\n(mean across seeds)")
        _decorate_pr(ax_pr_mit,   "Mitigation rate μ", "Per-region mitigation rate\n(mean across seeds)")

        # Shared legend panel: region colours + condition linestyles
        _region_handles = [
            plt.Line2D([0], [0], color=_REGION_COLORS[r], linewidth=2,
                       label=REGION_NAMES.get(r, f"r{r}"))
            for r in range(NUM_REGIONS)
        ]
        _cond_handles = [
            plt.Line2D([0], [0], color="gray", linestyle=CONDITIONS[k]["linestyle"],
                       linewidth=2, label=v)
            for k, v in _cond_labels.items()
        ]
        ax_legend.legend(
            handles=_region_handles + _cond_handles,
            labels=[REGION_NAMES.get(r, f"r{r}") for r in range(NUM_REGIONS)]
                   + list(_cond_labels.values()),
            fontsize=8, loc="center", framealpha=0.8,
            title="Regions  +  conditions", title_fontsize=9,
        )

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_convergence_multiseed_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed convergence study (ctrl vs differential CBAM)."
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total env steps per run (default {TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--seeds", type=int, default=len(DEFAULT_SEEDS),
        help=f"Number of seeds to use (taken from DEFAULT_SEEDS, default {len(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--replot", type=str, default=None,
        metavar="PICKLE",
        help="Path to an existing .pkl artifact — regenerate plot without retraining",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=list(CONDITIONS.keys()),
        choices=list(CONDITIONS.keys()),
        help="Which conditions to train (default: all)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_timesteps = args.timesteps

    # ── Replot mode ───────────────────────────────────────────────────────────
    if args.replot:
        print(f"Replot mode — loading {args.replot}")
        with open(args.replot, "rb") as f:
            run_data = pickle.load(f)
        _plot(run_data, timestamp)
        return

    # ── Training mode ─────────────────────────────────────────────────────────
    seeds = DEFAULT_SEEDS[: args.seeds]
    active_conds = args.conditions

    print(f"\n{'═'*60}")
    print(f"  cbam_convergence_multiseed.py")
    print(f"  conditions : {active_conds}")
    print(f"  seeds      : {seeds}")
    print(f"  timesteps  : {args.timesteps:,}")
    print(f"  total runs : {len(active_conds) * len(seeds)}")
    print(f"{'═'*60}")

    run_data: dict[str, dict] = {"_meta": {"timesteps": total_timesteps}}

    for cond_name in active_conds:
        csv_paths = []
        for seed in seeds:
            csv_p = _train_one(cond_name, seed, total_timesteps, timestamp)
            csv_paths.append(csv_p)
        run_data[cond_name] = {
            "seeds":     seeds,
            "csv_paths": csv_paths,
        }

    # Save artifact
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_convergence_multiseed_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(run_data, f)
    print(f"\nArtifact saved → {pkl_path}")

    _plot(run_data, timestamp)


if __name__ == "__main__":
    main()
