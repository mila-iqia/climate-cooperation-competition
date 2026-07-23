"""cbam_experiment_2e_clubs.py

Phase 2E — Endogenous CBAM-club design comparison.

Trains one PPO policy per club scenario and compares them on the critical
KPIs.  The three designs (see ``rice_jax/mrio/scenarios.py``):

  B1  MRIOClubCBAM     — open-accession single CBAM club anchored on the EU.
  B2  MRIOSectoralClub — single club whose anchor also negotiates which
                         sectors the border levy covers.
  B3  MRIOMultiClub    — competing clubs, each anchored on a fixed core;
                         an exporter pays every club it is not a member of.

All three share the differential (MAC-gap) CBAM tariff, the additive RCPO
reward channel (λ fixed at cbam_lambda_init), and the 3-stage
propose→evaluate→climate negotiation cycle (negotiation_on=True).

Critical KPIs (per scenario, averaged over evaluation episodes):
  - temp_rise        : end-of-horizon atmospheric temperature anomaly (°C;
                       lower = less global warming) — the headline climate KPI
  - club_size        : mean number of club members  (coalition breadth)
  - club_ambition    : mean binding minimum mitigation rate of club members
                       (the club's ambition / stringency)
  - mu_nonEU         : mean non-EU mitigation rate μ (abatement / free-riding)
  - eu_dirty_share   : mean EU-bound dirty export share (trade diversion;
                       lower = less diversion)
  - cbam_cost        : mean total CBAM cost levied per step (border friction)
  - welfare_nonEU    : mean non-EU utility (distributional cost of the club)
  - ep_return        : final training episode return (rolling-10)

Research question: which club architecture best converts CBAM pressure into
broad membership + high abatement at low diversion and welfare cost?

Usage (from rice_jax/, rice-jax conda env):
    python cbam/drivers/cbam_experiment_2e_clubs.py [--timesteps 1000000]
    python cbam/drivers/cbam_experiment_2e_clubs.py --scenarios B1 B3
    python cbam/drivers/cbam_experiment_2e_clubs.py --replot <pickle.pkl>
"""

import sys
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

import matplotlib
matplotlib.use("Agg")  # BEFORE any JAX import (macOS Agg backend pollution)

import argparse
import os as _os
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
from rice_jax.training import (
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
    rcpo_cbam_log_info_fn,
)
from rice_jax import MRIOClubCBAM, MRIOSectoralClub, MRIOMultiClub
from rice_jax.utils import full_state_info_log_fn, load_region_yamls
from _experiment_util import (
    FixedActionAgent,
    get_output_dir,
    get_log_dir,
    run_single_episode,
)


# ── Config ────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT  = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", "..", ".."))

NUM_REGIONS = 7
EU_IDX      = 5
MRIO_ROOT   = _os.path.join(_REPO_ROOT, "csv_asset")

# Second competing-club core for B3 (North America in the 7-region ordering).
B3_SECOND_CORE = 2

REGION_NAMES = {
    0: "SSA", 1: "S.Asia", 2: "N.America", 3: "MENA",
    4: "LatAm", 5: "EU", 6: "E.Asia",
}
NON_EU = [r for r in range(NUM_REGIONS) if r != EU_IDX]

TOTAL_TIMESTEPS   = 1_000_000
NUM_ENVS          = 8
NUM_STEPS         = 100
NUM_EVAL_EPISODES = 8
SEED              = 42
# Paris-aligned reference used only to annotate the residual warming gap that
# the CBAM clubs cannot close (it is NOT a model target — see _plot_results).
TARGET_TEMP       = 2.0
CBAM_RATE         = 0.80   # flat-mode fallback only; differential ignores this
CBAM_LAMBDA_INIT  = 1.0
WELFARE_LOSS_WEIGHT = 5.0

# RCPO Lagrange-multiplier schedule (Tessler et al. 2019): λ climbs while mean
# CBAM cost exceeds the target, sharpening the abatement-vs-diversion trade-off.
RCPO_ETA_LAMBDA   = 5e-7
RCPO_ALPHA_TARGET = 0.01

# Credible EU anchor — net-zero ramp (EU Climate Law 2021/1119).  Pins the EU
# core's club offer so the differential reference MAC cannot self-collapse to a
# low-ambition equilibrium.  Applied via _MRIOClubBase.step_propose.
EU_MITIGATION_SCHEDULE = (
    0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.80, 0.90, 1.00,
    1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
)

# Scenario registry: label → (class, extra-kwargs, colour, long name)
SCENARIOS = {
    "B1": (MRIOClubCBAM,    {},                                  "#1f77b4",
           "Single club"),
    "B2": (MRIOSectoralClub, {},                                 "#2ca02c",
           "Sectoral club"),
    "B3": (MRIOMultiClub,    {"club_cores": (EU_IDX, B3_SECOND_CORE)}, "#d62728",
           "Competing clubs"),
}

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR    = get_log_dir("training_logs")
LOG_PREFIX = "cbam2e_clubs_"

_BASE_ENV = dict(
    num_regions                  = NUM_REGIONS,
    mrio_data_root               = MRIO_ROOT,
    mrio_trade                   = True,
    eu_region_idx                = EU_IDX,
    cbam_tariff_rate             = CBAM_RATE,
    cbam_randomize               = False,
    dest_alloc_persistence       = 0.55,
    dest_alloc_baseline_decay    = 0.0,        # MUST be 0.0 — keep MRIO anchor
    diff_reward_mode             = True,
    num_discrete_action_levels   = 10,
    sector_granularity           = "emissions-simple",
    sectoral_welfloss            = True,
    welfare_loss_per_unit_tariff = WELFARE_LOSS_WEIGHT,
    cbam_lambda_init             = CBAM_LAMBDA_INIT,
    eu_mitigation_schedule       = EU_MITIGATION_SCHEDULE,
    # cbam_tariff_mode / reward_mode default to differential / additive_cbam
    # inside the club base class — do not override here.
)

_PPO_KWARGS = dict(
    num_steps              = NUM_STEPS,
    num_envs               = NUM_ENVS,
    learning_rate_start    = 3e-4,
    learning_rate_end      = None,
    num_minibatches        = 4,
    num_epochs             = 8,
    ent_coef_start         = 0.01,
    ent_coef_end           = None,
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


# ── Build / train helpers ─────────────────────────────────────────────────────

def _build_env(label, for_training=True):
    cls, extra, _, _ = SCENARIOS[label]
    env = cls(
        region_params = load_region_yamls(NUM_REGIONS),
        reward_mode   = "additive_cbam",
        log_info_fn   = rcpo_cbam_log_info_fn,
        **_BASE_ENV,
        **extra,
    )
    return jym.LogWrapper(env) if for_training else env


def _make_log_fn(label, num_iters):
    _os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = _os.path.join(LOG_DIR, f"{LOG_PREFIX}{label}.csv")
    _print_fn = make_print_log_fn(num_iterations=num_iters)
    _DROP = {"action_mean", "action_var"}

    def _compact(data, iteration):
        _print_fn({k: v for k, v in data.items() if k not in _DROP}, iteration)

    return make_combined_log_fn(_compact, make_csv_log_fn(csv_path)), csv_path


def _train(label, key, total_timesteps):
    num_iters = total_timesteps // (NUM_ENVS * NUM_STEPS)
    env       = _build_env(label, for_training=True)
    log_fn, csv_path = _make_log_fn(label, num_iters)
    ppo = RCPOMonitoredPPO(
        total_timesteps   = total_timesteps,
        log_function      = log_fn,
        rcpo_eta_lambda   = RCPO_ETA_LAMBDA,
        rcpo_alpha_target = RCPO_ALPHA_TARGET,
        **_PPO_KWARGS,
    )
    _, _, _, long_name = SCENARIOS[label]
    print(f"\n{'━'*60}")
    print(f"  Training {label} — {long_name}  (RCPO λ auto-tune)")
    print(f"{'━'*60}")
    t0  = time.perf_counter()
    ppo = ppo.train(key, env)            # CRITICAL: keep the returned object
    print(f"  Done in {time.perf_counter() - t0:.0f}s")
    return ppo, csv_path


# ── Evaluation ────────────────────────────────────────────────────────────────

def _per_region_arr(d):
    """full_state_info_log_fn splits *_all_regions keys into {region: (T,)}."""
    return np.stack([np.array(d[i]) for i in range(NUM_REGIONS)], axis=-1)  # (T, NR)


def _eval(label, key, raw_env, agent):
    """Return a dict of critical KPIs averaged over evaluation episodes."""
    eval_env = replace(raw_env, log_info_fn=full_state_info_log_fn)

    club_sizes, club_ambitions, mu_all, dirty_all, cost_all, welfare_all, temp_all = (
        [], [], [], [], [], [], []
    )
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 70_000 + ep_id)
        logs   = run_single_episode(ep_key, eval_env, agent)

        # Club membership — (T, NR) bool; size = members per step.
        membership = np.array(logs["club_membership"])              # (T, NR)
        n_members  = membership.sum(axis=1)                         # (T,)
        club_sizes.append(n_members.mean())

        # Club ambition — mean binding minimum mitigation rate over members.
        # minimum_mitigation_rate_all_regions = club rate for members, 0 else;
        # averaging over the membership mask gives the club's stringency and
        # works uniformly for single (B1/B2) and competing (B3) clubs.
        mmr = _per_region_arr(logs["minimum_mitigation_rate_all_regions"])  # (T, NR)
        bound = (mmr * membership).sum(axis=1) / np.maximum(n_members, 1.0)  # (T,)
        club_ambitions.append(bound.mean())

        # Mitigation rate μ — non-EU mean over the episode.
        mu = _per_region_arr(logs["mitigation_rates_all_regions"])  # (T, NR)
        mu_all.append(mu[:, NON_EU].mean())

        # EU-bound dirty export share, averaged over non-EU exporters.
        tf    = np.array(logs["trade_flows"])                       # (T, NR, NR, NS)
        dirty_to_eu = tf[:, :, EU_IDX, 0]                           # (T, NR)
        dirty_tot   = tf[:, :, :, 0].sum(axis=2)                    # (T, NR)
        share = dirty_to_eu / (dirty_tot + 1e-10)                   # (T, NR)
        dirty_all.append(share[:, NON_EU].mean())

        # Total CBAM cost levied per step (border friction).
        cost = _per_region_arr(logs["cbam_cost_all_regions"])       # (T, NR)
        cost_all.append(cost.sum(axis=1).mean())

        # Non-EU welfare (utility).
        util = _per_region_arr(logs["utility_all_regions"])         # (T, NR)
        welfare_all.append(util[:, NON_EU].mean())

        # Global atmospheric temperature anomaly — end-of-horizon (peak) value.
        # full_state_info_log_fn splits global_temperature into a dict with
        # "atmosphere" / "lower_ocean" components, each shape (T,).
        atm = np.array(logs["global_temperature"]["atmosphere"])    # (T,)
        temp_all.append(atm[-1])

    return {
        "temp_rise":     float(np.mean(temp_all)),
        "club_size":     float(np.mean(club_sizes)),
        "club_ambition": float(np.mean(club_ambitions)),
        "mu_nonEU":      float(np.mean(mu_all)),
        "eu_dirty_share": float(np.mean(dirty_all)),
        "cbam_cost":     float(np.mean(cost_all)),
        "welfare_nonEU": float(np.mean(welfare_all)),
    }


def _bau_temp(key):
    """No-policy counterfactual warming (°C, end-of-horizon).

    Free-rider baseline: every region takes μ=0 and nobody joins the club, so
    the differential CBAM levy collapses to ~0.  This is the upper-bound
    warming against which the trained clubs' (small) temperature reduction is
    measured — it makes the limited reach of the border tariff visible.
    Uses the B1 env class with the EU schedule disabled so the EU is not
    exogenously forced to abate.
    """
    env = MRIOClubCBAM(
        region_params = load_region_yamls(NUM_REGIONS),
        reward_mode   = "additive_cbam",
        log_info_fn   = full_state_info_log_fn,
        **{**_BASE_ENV, "eu_mitigation_schedule": None},
    )
    agent = FixedActionAgent(env)
    temps = []
    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(key, 90_000 + ep_id)
        logs   = run_single_episode(ep_key, env, agent)
        temps.append(np.array(logs["global_temperature"]["atmosphere"])[-1])
    return float(np.mean(temps))


# ── Plotting ──────────────────────────────────────────────────────────────────

_KPI_PANELS = [
    ("temp_rise",      "Global temp rise (°C, final)",   "lower = less warming"),
    ("club_size",      "Club size (mean members)",       "higher = broader coalition"),
    ("club_ambition",  "Club ambition (min μ, members)", "higher = more stringent club"),
    ("mu_nonEU",       "Non-EU mitigation μ",            "higher = more abatement"),
    ("eu_dirty_share", "EU dirty export share",          "lower = less diversion"),
    ("cbam_cost",      "Total CBAM cost / step",         "lower = less border friction"),
    ("welfare_nonEU",  "Non-EU welfare (utility)",       "higher = better for exporters"),
]


def _plot_results(results, timestamp, bau_temp=None):
    labels = [r["label"] for r in results]
    colors = [SCENARIOS[l][2] for l in labels]
    x      = np.arange(len(labels))

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(
        f"Phase 2E — Endogenous CBAM-Club Design Comparison\n"
        f"differential (MAC-gap) tariff, additive RCPO reward (λ={CBAM_LAMBDA_INIT}), "
        f"{NUM_REGIONS}-region, {TOTAL_TIMESTEPS//1_000_000:.0f}M steps",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.60, wspace=0.30,
                           height_ratios=[1, 1, 1, 0.7])

    # ── Training convergence ──────────────────────────────────────────────────
    ax_conv = fig.add_subplot(gs[0, 0])
    for res, col in zip(results, colors):
        try:
            df = pd.read_csv(res["csv_path"])
            if "ep_return_mean" in df.columns:
                ax_conv.plot(df["ep_return_mean"].rolling(10).mean(),
                             color=col, linewidth=1.5, label=res["label"])
        except Exception:
            pass
    ax_conv.set_xlabel("PPO iteration")
    ax_conv.set_ylabel("ep_return_mean (rolling-10)")
    ax_conv.set_title("Training convergence")
    ax_conv.legend(fontsize=9)

    # ── KPI bar panels (temp_rise first — headline climate KPI) ──────────────
    panel_axes = [
        fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    for ax, (key, title, hint) in zip(panel_axes, _KPI_PANELS):
        vals = [r["kpis"][key] for r in results]
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(f"{title}\n({hint})", fontsize=10)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.3g}", (xi, v), ha="center", va="bottom", fontsize=8)

        # Temp panel: overlay the no-policy BAU ceiling and the 2°C reference
        # so the *limited* warming reduction achievable via CBAM is explicit.
        if key == "temp_rise":
            best = min(vals)
            top  = max([v for v in vals] + ([bau_temp] if bau_temp else []))
            ax.set_ylim(0, top * 1.18)
            if bau_temp is not None:
                ax.axhline(bau_temp, ls="--", lw=1.4, color="0.35")
                ax.annotate(f"no-policy BAU  {bau_temp:.2f}°C",
                            (len(labels) - 0.5, bau_temp), ha="right", va="bottom",
                            fontsize=8, color="0.35")
            ax.axhline(TARGET_TEMP, ls=":", lw=1.6, color="green")
            ax.annotate("2°C reference", (-0.45, TARGET_TEMP), ha="left",
                        va="bottom", fontsize=8, color="green")
            cbam_cut = (bau_temp - best) if bau_temp else None
            cut_txt  = (f"CBAM cuts only {cbam_cut:.2f}°C off BAU\n"
                        if cbam_cut is not None else "")
            ax.annotate(
                f"{cut_txt}residual gap to 2°C:\n+{best - TARGET_TEMP:.2f}°C (best club)",
                xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top",
                fontsize=8, color="firebrick",
                bbox=dict(boxstyle="round", fc="white", ec="firebrick", alpha=0.85),
            )

    # ── Summary table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[3, :])
    ax_tbl.axis("off")
    kpi_keys  = [k for k, _, _ in _KPI_PANELS]
    col_heads = ["Scenario"] + [t for _, t, _ in _KPI_PANELS]
    table_data = []
    for res in results:
        long = SCENARIOS[res["label"]][3]
        row  = [f"{res['label']} — {long}"] + [f"{res['kpis'][k]:.4g}" for k in kpi_keys]
        table_data.append(row)
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_heads,
                       loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    ax_tbl.set_title("Critical-KPI summary", fontsize=10, pad=6)

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_2e_clubs_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved → {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS),
                        help="Club scenarios to train (default: B1 B2 B3)")
    parser.add_argument("--replot", type=str, default=None,
                        help="Path to existing .pkl — regenerate plot without retraining")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.replot:
        with open(args.replot, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict):
            results  = payload["results"]
            bau_temp = payload.get("bau_temp")
        else:                                   # backward-compat: old list pkls
            results  = payload
            bau_temp = None
        print(f"Loaded {len(results)} models from {args.replot}")
        _plot_results(results, timestamp, bau_temp=bau_temp)
        return

    root_key = jax.random.PRNGKey(args.seed)
    results  = []

    for i, label in enumerate(args.scenarios):
        train_key = jax.random.fold_in(root_key, i)
        ppo, csv_path = _train(label, train_key, args.timesteps)

        eval_key = jax.random.fold_in(root_key, 300 + i)
        raw_env  = _build_env(label, for_training=False)
        kpis     = _eval(label, eval_key, raw_env, ppo)

        results.append({"label": label, "kpis": kpis, "csv_path": csv_path})
        print(f"\n  {label}: " + "  ".join(f"{k}={v:.4g}" for k, v in kpis.items()))

    # ── No-policy BAU counterfactual (upper-bound warming) ────────────────────
    bau_temp = _bau_temp(jax.random.fold_in(root_key, 999))
    print(f"\n  BAU (no-policy) warming: {bau_temp:.3f}°C")

    # ── Save artefact ─────────────────────────────────────────────────────────
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_2e_clubs_{timestamp}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {"results": results, "bau_temp": bau_temp, "target_temp": TARGET_TEMP},
            f,
        )
    print(f"\nPickle saved → {pkl_path}")

    _plot_results(results, timestamp, bau_temp=bau_temp)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("  PHASE 2E — CLUB DESIGN COMPARISON (critical KPIs)")
    print("─" * 78)
    header = f"  {'Scenario':18s}" + "".join(
        f"{t.split('(')[0].strip()[:14]:>15s}" for _, t, _ in _KPI_PANELS
    )
    print(header)
    for res in results:
        long = SCENARIOS[res["label"]][3]
        name = f"{res['label']} {long}"
        row  = f"  {name:18s}" + "".join(
            f"{res['kpis'][k]:>15.4g}" for k, _, _ in _KPI_PANELS
        )
        print(row)
    print("═" * 78)


if __name__ == "__main__":
    main()
