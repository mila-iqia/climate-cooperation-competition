"""cbam_posthoc_C_amplifier.py — Diagnostic post-hoc for Experiment C (amplifier sweep).

Eight failure-mode hypotheses are tested against the pkl output of
cbam_experiment_C_amplifier.py. No retraining — reads pkl + training CSVs.

Failure modes tested
--------------------
FD1  μ_pinned confound     — Does attenuation arise because μ_pinned *increases*
                              with M (general abatement subsidy effect) rather than
                              μ_open increasing (diversion specifically suppressed)?
                              Pass: Δμ_open dominates Δμ_pinned at M*.

FD2  EU-share saturation   — Is the OPEN arm EU dirty share already near zero at
                              M=1, leaving no diversion headroom to suppress?
                              Pass: share_open(M=1) > 5% (some diversion is present).

FD3  Abatement cap binding  — Mode=abatement clamps subsidy=min(transfer, abatement_cost).
                              At high M, transfer >> abatement_cost so the excess is
                              forfeited; effective multiplier saturates and μ_open
                              plateaus despite increasing M.
                              Diagnosed by: μ_open plateau gradient vs M.

FD4  Effort-allocation      — At high M where μ is near-uniform across regions,
     degeneracy               effort allocation ≈ equal (marginal incentive
                              ∂T_r/∂μ_r → 0); the performance incentive collapses.
                              Pass: std(μ_open) across CBAM regions does not collapse
                              to near-zero at M*.

FD5  λ saturation / collapse — RCPO λ drives to 0 because amplified abatement subsidies
                               make ΔU persistently large → cbam constraint never binds
                               → CBAM signal vanishes from reward.
                               Pass: terminal λ at M* is comparable to λ at M=1.
                               (Panel optional — requires cbam_lambda CSV column.)

FD6  Training instability   — High M → reward explosion → non-convergent ep_return
                              curves; attenuation at M≥20 is unreliable.
                              Pass: ep_return_mean at M=50 plateaus within training run.

FD7  Divert-and-harvest     — Agents maintain some EU exports deliberately to preserve
                              the endogenous pool × M transfer income, creating a perverse
                              "dirty equilibrium at scale."  EU dirty share does not fall
                              monotonically with M.
                              Diagnosis: EU dirty export proxy vs M in OPEN arm.

FD8  Region heterogeneity   — Aggregate M* is driven by one large region while CBAM-
                              vulnerable small regions (SSA, India) never attenuate.
                              Diagnosis: per-region attenuation heatmap.

Output
------
  Saves <out_prefix>_posthoc.png (8-panel diagnostic figure)
  Prints a text scorecard with pass/fail for each failure mode.

Usage
-----
    python cbam/posthoc/cbam_posthoc_C_amplifier.py \\
        --pkl plots/cbam_C_amplifier_<timestamp>.pkl

    # Optional: direct output directory
    python cbam/posthoc/cbam_posthoc_C_amplifier.py \\
        --pkl ... --out-dir plots/
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))


import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from _experiment_util import get_output_dir


# ── Region metadata ─────────────────────────────────────────────────────────

NUM_REGIONS = 9
EU_IDX      = 3

REGION_NAMES = {
    0: "RoW", 1: "Russia+Eurasia", 2: "MENA", 3: "EU",
    4: "SSA Mining", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
REGION_SHORT = {
    0: "RoW", 1: "Rus+Eur", 2: "MENA", 3: "EU",
    4: "SSA", 5: "Amer.", 6: "SE Asia", 7: "China", 8: "India",
}
NON_EU            = [r for r in range(NUM_REGIONS) if r != EU_IDX]
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]  # drop RoW

EVAL_LAST_T = 5   # match canonical_config


# ── Array helpers ────────────────────────────────────────────────────────────

def _per_region_mu(mitigation_raw: np.ndarray, last_t: int = EVAL_LAST_T) -> np.ndarray:
    """(n_ep, T, NR) → (NR,) mean mitigation over last_t steps."""
    return mitigation_raw[:, -last_t:, :].mean(axis=(0, 1))


def _per_region_eu_dirty_share(trade_flows_raw: np.ndarray,
                                last_t: int = EVAL_LAST_T) -> np.ndarray:
    """(n_ep, T, NR, NR, NS) → (NR,) fraction of each region's dirty output going to EU."""
    tf = trade_flows_raw[:, -last_t:]           # (n_ep, t, NR, NR, NS)
    dirty_to_eu  = tf[:, :, :, EU_IDX, 0]       # (n_ep, t, NR)
    dirty_total  = tf[:, :, :, :, 0].sum(-1)    # (n_ep, t, NR)
    return (dirty_to_eu / (dirty_total + 1e-10)).mean(axis=(0, 1))


def _mean_mu_std(mitigation_raw: np.ndarray, region_idxs: list[int],
                 last_t: int = EVAL_LAST_T) -> tuple[float, float]:
    """Mean and std of mitigation across the given regions (population variance)."""
    mu = _per_region_mu(mitigation_raw, last_t)
    vals = np.array([mu[r] for r in region_idxs])
    return float(vals.mean()), float(vals.std())


def _read_csv(csv_path: str) -> pd.DataFrame:
    """Read a training CSV tolerating relative paths from rice_jax/."""
    if not csv_path or not isinstance(csv_path, str):
        return pd.DataFrame()
    full = (
        os.path.join(str(_RICE_JAX_ROOT), csv_path)
        if not os.path.isabs(csv_path)
        else csv_path
    )
    if not os.path.exists(full):
        return pd.DataFrame()
    return pd.read_csv(full)


# ── Build diagnostic tables ──────────────────────────────────────────────────

def _build_tables(cells: list) -> dict:
    """Compute all per-cell metrics used by the diagnostic panels."""
    rows = []
    for c in cells:
        mu_r    = _per_region_mu(c["mitigation_raw"])
        eu_sh_r = _per_region_eu_dirty_share(c["trade_flows_raw"])
        mu_std  = float(np.array([mu_r[r] for r in CBAM_PLOT_REGIONS]).std())
        for r in CBAM_PLOT_REGIONS:
            rows.append({
                "seed":           c["seed"],
                "multiplier":     c["multiplier"],
                "pinned":         c["pinned"],
                "region":         r,
                "region_short":   REGION_SHORT.get(r, f"R{r}"),
                "mu":             float(mu_r[r]),
                "eu_dirty_share": float(eu_sh_r[r]),
                "mu_std_across_regions": mu_std,
            })

    df = pd.DataFrame(rows)

    # Pivot: μ_pinned, μ_open per (seed, multiplier, region)
    piv = df.pivot_table(
        index=["seed", "multiplier", "region", "region_short"],
        columns="pinned", values=["mu", "eu_dirty_share"],
    )
    piv.columns = ["_".join(str(x) for x in col) for col in piv.columns]
    piv = piv.rename(columns={
        "mu_True": "mu_pinned", "mu_False": "mu_open",
        "eu_dirty_share_True": "eu_sh_pinned", "eu_dirty_share_False": "eu_sh_open",
    }).reset_index()

    piv["crowd_out_gap"] = piv["mu_pinned"] - piv["mu_open"]

    # Attenuation relative to M=1 baseline
    gap_m1 = (piv[piv["multiplier"] == 1.0]
              .set_index(["seed", "region"])["crowd_out_gap"]
              .rename("gap_m1"))
    piv = piv.join(gap_m1, on=["seed", "region"])
    piv["attenuation"]   = piv["gap_m1"] - piv["crowd_out_gap"]
    piv["delta_mu_open"]   = piv["mu_open"]   - piv.join(
        piv[piv["multiplier"] == 1.0].set_index(["seed","region"])["mu_open"].rename("mu_open_m1"),
        on=["seed","region"],
    )["mu_open_m1"]
    piv["delta_mu_pinned"] = piv["mu_pinned"] - piv.join(
        piv[piv["multiplier"] == 1.0].set_index(["seed","region"])["mu_pinned"].rename("mu_pinned_m1"),
        on=["seed","region"],
    )["mu_pinned_m1"]

    # Aggregate μ_std across regions: one value per (seed, multiplier, arm)
    mu_std_df = df.groupby(["seed", "multiplier", "pinned"])["mu_std_across_regions"].mean().reset_index()

    # EU dirty share aggregated over CBAM regions (OPEN arm only)
    eu_agg = (df[~df["pinned"]]
              .groupby(["seed", "multiplier"])["eu_dirty_share"]
              .mean().reset_index()
              .rename(columns={"eu_dirty_share": "eu_sh_mean"}))

    return {
        "df_raw":   df,
        "pivot":    piv,
        "mu_std":   mu_std_df,
        "eu_agg":   eu_agg,
    }


def _load_training_curves(cells: list) -> dict:
    """Return {label: df_csv} for all cells that have a readable CSV."""
    curves = {}
    for c in cells:
        path = c.get("csv_path", "")
        df = _read_csv(path)
        if not df.empty:
            curves[c["label"]] = (df, c["multiplier"], c["pinned"], c["seed"])
    return curves


# ── Scorecard ─────────────────────────────────────────────────────────────────

def _scorecard(tables: dict, curves: dict, m_levels: list) -> dict:
    """Evaluate each failure mode.  Returns {FD_id: {"pass": bool, "detail": str}}."""
    piv    = tables["pivot"]
    mu_std = tables["mu_std"]
    eu_agg = tables["eu_agg"]
    scores = {}

    # ── FD1: μ_pinned confound ─────────────────────────────────────────────
    # At M* (first M where aggregate attenuation > 0), check if
    # Δμ_open > Δμ_pinned (diversion suppression dominates subsidy effect)
    agg_attn = (piv.groupby("multiplier")[["attenuation", "delta_mu_open", "delta_mu_pinned"]]
                .mean().reset_index())
    threshold_rows = agg_attn[agg_attn["attenuation"] > 0]
    if threshold_rows.empty:
        scores["FD1"] = {"pass": None, "detail": "No M* found — FD1 not evaluable"}
    else:
        m_star = threshold_rows["multiplier"].min()
        row = agg_attn[agg_attn["multiplier"] == m_star].iloc[0]
        d_open  = row["delta_mu_open"]
        d_pin   = row["delta_mu_pinned"]
        passes  = d_open > d_pin
        scores["FD1"] = {
            "pass": passes,
            "detail": (f"At M*={m_star:.0f}×: Δμ_open={d_open:+.4f}  "
                       f"Δμ_pinned={d_pin:+.4f} → "
                       f"{'diversion-suppression dominant ✅' if passes else 'pinned-arm confound ❌'}"),
        }

    # ── FD2: EU-share saturation ───────────────────────────────────────────
    eu_m1 = eu_agg[eu_agg["multiplier"] == 1.0]["eu_sh_mean"].mean()
    passes = eu_m1 > 0.05
    scores["FD2"] = {
        "pass": passes,
        "detail": (f"EU dirty share (OPEN, M=1): {eu_m1:.4f}  "
                   f"→ {'enough headroom ✅' if passes else 'near-zero at baseline ❌'}"),
    }

    # ── FD3: abatement cap binding (μ_open plateau) ────────────────────────
    # Compute d(μ_open)/d(M) in the upper half of the sweep.  If gradient ≈ 0
    # above some M, the cap has bound.
    mu_open_agg = (piv.groupby("multiplier")["mu_open"].mean().reset_index()
                   .sort_values("multiplier"))
    m_arr  = mu_open_agg["multiplier"].values
    mu_arr = mu_open_agg["mu_open"].values
    if len(m_arr) >= 3:
        upper_half = m_arr >= np.median(m_arr)
        grad_upper = np.diff(mu_arr[upper_half]) / np.diff(np.log1p(m_arr[upper_half]))
        plateau = bool(np.all(np.abs(grad_upper) < 0.02))
    else:
        plateau = None
    scores["FD3"] = {
        "pass": not plateau if plateau is not None else None,
        "detail": (f"μ_open gradient in upper half of sweep: "
                   f"max|d_mu/d_logM| = {float(np.abs(grad_upper).max()):.4f}  "
                   f"→ {'plateau (cap binding) ❌' if plateau else 'still increasing ✅'}"),
    }

    # ── FD4: effort-allocation degeneracy ─────────────────────────────────
    # std(μ_open) across CBAM regions at M=1 vs M_max
    std_m1   = mu_std[(mu_std["multiplier"]==1.0) & (~mu_std["pinned"])]["mu_std_across_regions"].mean()
    m_max    = max(m_levels)
    std_mmax = mu_std[(mu_std["multiplier"]==m_max) & (~mu_std["pinned"])]["mu_std_across_regions"].mean()
    passes = std_mmax > 0.05
    scores["FD4"] = {
        "pass": passes,
        "detail": (f"std(μ_open) across regions: M=1→{std_m1:.4f}, M={m_max:.0f}→{std_mmax:.4f}  "
                   f"→ {'diversity maintained ✅' if passes else 'allocation uniform — effort incentive collapsed ❌'}"),
    }

    # ── FD5: λ saturation / collapse ──────────────────────────────────────
    # Look for cbam_lambda column in CSVs
    lambda_by_m = {}
    for label, (df_csv, mult, pinned, seed) in curves.items():
        if "cbam_lambda" in df_csv.columns and not pinned:
            val = df_csv["cbam_lambda"].dropna()
            if not val.empty:
                lambda_by_m.setdefault(mult, []).append(float(val.iloc[-1]))

    if not lambda_by_m:
        scores["FD5"] = {"pass": None, "detail": "cbam_lambda not in CSVs (MonitoredPPO used) — not evaluable"}
    else:
        lam_m1   = np.mean(lambda_by_m.get(1.0, [np.nan]))
        lam_mmax = np.mean(lambda_by_m.get(max(m_levels), [np.nan]))
        if np.isnan(lam_m1) or np.isnan(lam_mmax):
            scores["FD5"] = {"pass": None, "detail": "Incomplete λ data"}
        else:
            passes = lam_mmax > 0.1 * lam_m1  # at least 10% of M=1 lambda
            scores["FD5"] = {
                "pass": passes,
                "detail": (f"Terminal λ: M=1→{lam_m1:.4f}, M={max(m_levels):.0f}→{lam_mmax:.4f}  "
                           f"→ {'signal maintained ✅' if passes else 'λ collapsed ❌'}"),
            }

    # ── FD6: training instability ──────────────────────────────────────────
    # Check if ep_return_mean in final 20% of training has higher std than
    # in middle 20% — a divergence signature.
    instability_flags = []
    for label, (df_csv, mult, pinned, seed) in curves.items():
        if "ep_return_mean" not in df_csv.columns:
            continue
        series = df_csv["ep_return_mean"].dropna().values
        if len(series) < 20:
            continue
        n = len(series)
        mid_std  = series[int(0.4*n):int(0.6*n)].std()
        tail_std = series[int(0.8*n):].std()
        if tail_std > 3 * mid_std and tail_std > 0.01:
            instability_flags.append(label)

    passes = len(instability_flags) == 0
    scores["FD6"] = {
        "pass": passes,
        "detail": (f"Unstable cells (tail_std > 3× mid_std): "
                   f"{instability_flags if instability_flags else 'none ✅'}"),
    }

    # ── FD7: divert-and-harvest (non-monotone EU share) ──────────────────
    # EU dirty share in OPEN arm should be non-increasing with M if transfers
    # genuinely suppress diversion.  A U-shaped curve is the failure signature.
    eu_mean_by_m = eu_agg.groupby("multiplier")["eu_sh_mean"].mean().sort_index()
    if len(eu_mean_by_m) >= 3:
        m_vals = eu_mean_by_m.index.values
        sh_vals = eu_mean_by_m.values
        # Check for uptick in upper half
        upper_idx = m_vals >= np.median(m_vals)
        m_up  = m_vals[upper_idx]
        sh_up = sh_vals[upper_idx]
        has_uptick = any(sh_up[i+1] > sh_up[i] + 0.01 for i in range(len(sh_up)-1))
    else:
        has_uptick = False
    scores["FD7"] = {
        "pass": not has_uptick,
        "detail": (f"EU dirty share (OPEN): {dict(zip(eu_mean_by_m.index.astype(int), eu_mean_by_m.round(4).values))}  "
                   f"→ {'non-monotone uptick detected ❌' if has_uptick else 'monotonically non-increasing ✅'}"),
    }

    # ── FD8: region heterogeneity ──────────────────────────────────────────
    # At M*, how many CBAM regions show positive worst-seed attenuation?
    if threshold_rows.empty:
        scores["FD8"] = {"pass": None, "detail": "No M* found — FD8 not evaluable"}
    else:
        m_star = threshold_rows["multiplier"].min()
        attn_at_mstar = piv[piv["multiplier"] == m_star]
        per_region = attn_at_mstar.groupby("region")["attenuation"].min()
        n_passing = (per_region > 0).sum()
        passes = n_passing >= len(CBAM_PLOT_REGIONS) // 2  # majority pass
        neg_regions = [REGION_SHORT.get(r, f"R{r}") for r in per_region[per_region <= 0].index]
        scores["FD8"] = {
            "pass": passes,
            "detail": (f"At M*={m_star:.0f}×: {n_passing}/{len(CBAM_PLOT_REGIONS)} regions "
                       f"have positive worst-seed attenuation  "
                       f"{'→ majority pass ✅' if passes else f'→ failing regions: {neg_regions} ❌'}"),
        }

    return scores


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(tables: dict, curves: dict, scores: dict, m_levels: list, out_path: str):
    piv    = tables["pivot"]
    mu_std = tables["mu_std"]
    eu_agg = tables["eu_agg"]

    sorted_m = sorted(m_levels)
    n_m      = len(sorted_m)
    seed_colors = {s: c for s, c in zip([0, 1, 2], ["#1f77b4","#ff7f0e","#2ca02c"])}

    fig = plt.figure(figsize=(20, 22))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.35)

    ax_decomp  = fig.add_subplot(gs[0, :2])   # FD1: μ_pinned vs μ_open vs M
    ax_eusat   = fig.add_subplot(gs[0, 2:])   # FD2: EU dirty share vs M
    ax_cap     = fig.add_subplot(gs[1, :2])   # FD3: μ_open gradient (cap binding)
    ax_deg     = fig.add_subplot(gs[1, 2:])   # FD4: std(μ_open) across regions
    ax_lambda  = fig.add_subplot(gs[2, :2])   # FD5: λ trajectory
    ax_conv    = fig.add_subplot(gs[2, 2:])   # FD6: training convergence (ep_return)
    ax_harvest = fig.add_subplot(gs[3, :2])   # FD7: EU share non-monotone check
    ax_heat    = fig.add_subplot(gs[3, 2:])   # FD8: per-region attenuation heatmap

    # ── FD1: decomposition ─────────────────────────────────────────────────
    agg = piv.groupby("multiplier")[["mu_pinned","mu_open","delta_mu_pinned","delta_mu_open"]].mean()
    agg_std = piv.groupby("multiplier")[["mu_pinned","mu_open"]].std()
    ax_decomp.fill_between(sorted_m,
        [agg["mu_pinned"][m] - agg_std["mu_pinned"].get(m,0) for m in sorted_m],
        [agg["mu_pinned"][m] + agg_std["mu_pinned"].get(m,0) for m in sorted_m],
        alpha=0.20, color="steelblue")
    ax_decomp.fill_between(sorted_m,
        [agg["mu_open"][m] - agg_std["mu_open"].get(m,0) for m in sorted_m],
        [agg["mu_open"][m] + agg_std["mu_open"].get(m,0) for m in sorted_m],
        alpha=0.20, color="darkorange")
    ax_decomp.plot(sorted_m, [agg["mu_pinned"][m] for m in sorted_m],
                   "o-", color="steelblue", lw=2, label="μ_pinned (δ_max=0)")
    ax_decomp.plot(sorted_m, [agg["mu_open"][m] for m in sorted_m],
                   "s-", color="darkorange", lw=2, label="μ_open (δ_max=3)")
    # Δ traces on secondary axis
    ax2 = ax_decomp.twinx()
    ax2.plot(sorted_m, [agg["delta_mu_pinned"][m] for m in sorted_m],
             "^--", color="steelblue", alpha=0.6, lw=1.5, label="Δμ_pinned vs M=1")
    ax2.plot(sorted_m, [agg["delta_mu_open"][m] for m in sorted_m],
             "v--", color="darkorange", alpha=0.6, lw=1.5, label="Δμ_open vs M=1")
    ax2.axhline(0, color="k", lw=0.8, ls=":")
    ax2.set_ylabel("Δμ vs M=1 (right axis)", fontsize=8)
    ax_decomp.set_xscale("log")
    ax_decomp.set_xlabel("Transfer multiplier")
    ax_decomp.set_ylabel("Mean μ (CBAM regions)")
    fd1_tag = "✅" if scores.get("FD1",{}).get("pass") else ("❌" if scores.get("FD1",{}).get("pass") is False else "—")
    ax_decomp.set_title(f"FD1: μ_pinned vs μ_open decomposition {fd1_tag}")
    lines1, labels1 = ax_decomp.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_decomp.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="upper left")

    # ── FD2: EU dirty share vs M (OPEN arm) ────────────────────────────────
    eu_means = eu_agg.groupby("multiplier")["eu_sh_mean"]
    eu_m = eu_means.mean()
    eu_s = eu_means.std().fillna(0)
    ax_eusat.fill_between(sorted_m,
        [eu_m[m] - eu_s.get(m,0) for m in sorted_m],
        [eu_m[m] + eu_s.get(m,0) for m in sorted_m],
        alpha=0.25, color="firebrick")
    ax_eusat.plot(sorted_m, [eu_m[m] for m in sorted_m],
                  "D-", color="firebrick", lw=2)
    ax_eusat.axhline(0.05, color="k", lw=1.0, ls="--", label="5% saturation threshold")
    ax_eusat.set_xscale("log")
    ax_eusat.set_xlabel("Transfer multiplier")
    ax_eusat.set_ylabel("Mean EU dirty share (OPEN arm)")
    fd2_tag = "✅" if scores.get("FD2",{}).get("pass") else "❌"
    ax_eusat.set_title(f"FD2: EU dirty-share saturation {fd2_tag}")
    ax_eusat.legend(fontsize=8)

    # ── FD3: μ_open gradient (cap binding) ────────────────────────────────
    mu_open_m = piv.groupby("multiplier")["mu_open"].mean()
    ax_cap.plot(sorted_m, [mu_open_m[m] for m in sorted_m],
                "s-", color="mediumseagreen", lw=2, label="μ_open mean")
    # Gradient bars
    m_log = np.log1p(np.array(sorted_m))
    mu_vals = np.array([mu_open_m[m] for m in sorted_m])
    grads = np.gradient(mu_vals, m_log)
    ax_cap_r = ax_cap.twinx()
    ax_cap_r.bar(range(n_m), grads, color="mediumseagreen", alpha=0.35,
                 label="dμ_open/d_logM")
    ax_cap_r.axhline(0.02, color="k", ls=":", lw=0.8, label="plateau threshold=0.02")
    ax_cap_r.axhline(-0.02, color="k", ls=":", lw=0.8)
    ax_cap_r.set_ylabel("Gradient (right axis)", fontsize=8)
    ax_cap.set_xticks(range(n_m)); ax_cap.set_xticklabels([f"{m:.0f}×" for m in sorted_m])
    ax_cap.set_xlabel("Transfer multiplier")
    ax_cap.set_ylabel("μ_open (mean)")
    fd3_tag = "✅" if scores.get("FD3",{}).get("pass") else ("❌" if scores.get("FD3",{}).get("pass") is False else "—")
    ax_cap.set_title(f"FD3: Abatement cap binding (μ_open plateau?) {fd3_tag}")

    # ── FD4: std(μ_open) across regions ────────────────────────────────────
    std_open = mu_std[~mu_std["pinned"]].groupby("multiplier")["mu_std_across_regions"].agg(["mean","std"])
    ax_deg.fill_between(sorted_m,
        [std_open["mean"][m] - std_open["std"].get(m,0) for m in sorted_m],
        [std_open["mean"][m] + std_open["std"].get(m,0) for m in sorted_m],
        alpha=0.25, color="purple")
    ax_deg.plot(sorted_m, [std_open["mean"][m] for m in sorted_m],
                "o-", color="purple", lw=2, label="std(μ_open) across regions")
    ax_deg.axhline(0.05, color="k", ls="--", lw=1.0, label="degeneracy threshold=0.05")
    ax_deg.set_xscale("log")
    ax_deg.set_xlabel("Transfer multiplier")
    ax_deg.set_ylabel("std(μ) across CBAM regions")
    fd4_tag = "✅" if scores.get("FD4",{}).get("pass") else "❌"
    ax_deg.set_title(f"FD4: Effort-allocation degeneracy {fd4_tag}")
    ax_deg.legend(fontsize=8)

    # ── FD5: λ trajectory ──────────────────────────────────────────────────
    lambda_data = {}
    for label, (df_csv, mult, pinned, seed) in curves.items():
        if "cbam_lambda" in df_csv.columns and not pinned:
            series = df_csv["cbam_lambda"].dropna().values
            if len(series):
                lambda_data.setdefault(mult, []).append(series)

    if lambda_data:
        for m in sorted_m:
            if m not in lambda_data:
                continue
            arr = lambda_data[m]
            mean_traj = np.mean([s[:min(len(s),len(arr[0]))] for s in arr], axis=0)
            ax_lambda.plot(mean_traj, label=f"{m:.0f}×", lw=1.5)
        ax_lambda.set_xlabel("Training iteration")
        ax_lambda.set_ylabel("RCPO λ")
        fd5_tag = "✅" if scores.get("FD5",{}).get("pass") else ("❌" if scores.get("FD5",{}).get("pass") is False else "—")
        ax_lambda.set_title(f"FD5: λ trajectory (RCPO signal) {fd5_tag}")
        ax_lambda.legend(fontsize=7, ncol=2)
    else:
        ax_lambda.text(0.5, 0.5, "cbam_lambda not logged\n(MonitoredPPO used — not evaluable)",
                       ha="center", va="center", transform=ax_lambda.transAxes, fontsize=10)
        ax_lambda.set_title("FD5: λ trajectory — N/A")

    # ── FD6: training convergence ───────────────────────────────────────────
    max_iters = 0
    for label, (df_csv, mult, pinned, seed) in curves.items():
        if "ep_return_mean" not in df_csv.columns:
            continue
        series = df_csv["ep_return_mean"].dropna().values
        if not len(series):
            continue
        max_iters = max(max_iters, len(series))
        color = plt.cm.plasma(mult / (max(m_levels) + 1e-8))
        ax_conv.plot(series, color=color, alpha=0.6, lw=1.0)
    if max_iters:
        sm = plt.cm.ScalarMappable(cmap="plasma",
                                   norm=plt.Normalize(vmin=1, vmax=max(m_levels)))
        sm.set_array([])
        plt.colorbar(sm, ax=ax_conv, label="multiplier ×", fraction=0.04)
    fd6_tag = "✅" if scores.get("FD6",{}).get("pass") else "❌"
    ax_conv.set_xlabel("Training iteration")
    ax_conv.set_ylabel("ep_return_mean")
    ax_conv.set_title(f"FD6: Training convergence stability {fd6_tag}")

    # ── FD7: divert-and-harvest (EU share non-monotone check) ──────────────
    for seed, grp in eu_agg.groupby("seed"):
        grp_sorted = grp.sort_values("multiplier")
        ax_harvest.plot(grp_sorted["multiplier"], grp_sorted["eu_sh_mean"],
                        marker="o", alpha=0.7, color=seed_colors.get(seed, "gray"),
                        label=f"seed {seed}")
    ax_harvest.set_xscale("log")
    ax_harvest.set_xlabel("Transfer multiplier")
    ax_harvest.set_ylabel("EU dirty share (OPEN arm)")
    fd7_tag = "✅" if scores.get("FD7",{}).get("pass") else "❌"
    ax_harvest.set_title(f"FD7: Divert-and-harvest (non-monotone EU share?) {fd7_tag}")
    ax_harvest.legend(fontsize=8)

    # ── FD8: per-region attenuation heatmap ────────────────────────────────
    reg_labels = [REGION_SHORT.get(r, f"R{r}") for r in CBAM_PLOT_REGIONS]
    heat = np.full((n_m, len(CBAM_PLOT_REGIONS)), np.nan)
    for mi, m in enumerate(sorted_m):
        for ri, r in enumerate(CBAM_PLOT_REGIONS):
            sub = piv[(piv["multiplier"] == m) & (piv["region"] == r)]
            if not sub.empty:
                heat[mi, ri] = sub["attenuation"].min()  # worst seed

    vmax = max(0.2, float(np.nanmax(np.abs(heat))))
    im = ax_heat.imshow(heat, aspect="auto", cmap="RdYlGn",
                        vmin=-vmax, vmax=vmax)
    ax_heat.set_xticks(range(len(CBAM_PLOT_REGIONS)))
    ax_heat.set_xticklabels(reg_labels, rotation=30, ha="right", fontsize=8)
    ax_heat.set_yticks(range(n_m))
    ax_heat.set_yticklabels([f"{m:.0f}×" for m in sorted_m], fontsize=8)
    ax_heat.set_xlabel("Region")
    ax_heat.set_ylabel("Transfer multiplier")
    fd8_tag = "✅" if scores.get("FD8",{}).get("pass") else ("❌" if scores.get("FD8",{}).get("pass") is False else "—")
    ax_heat.set_title(f"FD8: Per-region attenuation (worst seed) {fd8_tag}")
    plt.colorbar(im, ax=ax_heat, fraction=0.04, label="attenuation")

    # ── Suptitle ──────────────────────────────────────────────────────────
    n_pass = sum(1 for v in scores.values() if v.get("pass") is True)
    n_fail = sum(1 for v in scores.values() if v.get("pass") is False)
    n_na   = sum(1 for v in scores.values() if v.get("pass") is None)
    fig.suptitle(
        "Experiment C — Transfer Amplifier: Post-hoc Failure-Mode Diagnostics\n"
        f"FD scorecard:  {n_pass} PASS  |  {n_fail} FAIL  |  {n_na} N/A",
        fontsize=12, y=1.01,
    )

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Diagnostic figure saved → {out_path}")


# ── Text report ───────────────────────────────────────────────────────────────

def _print_report(scores: dict, m_levels: list):
    print("\n" + "═" * 72)
    print("  EXPERIMENT C POST-HOC — FAILURE MODE SCORECARD")
    print("═" * 72)
    for fd_id in sorted(scores):
        s = scores[fd_id]
        if s["pass"] is True:
            status = "✅ PASS"
        elif s["pass"] is False:
            status = "❌ FAIL"
        else:
            status = "── N/A "
        print(f"  {fd_id}  {status}  {s['detail']}")
    print("═" * 72)

    failing = [k for k, v in scores.items() if v.get("pass") is False]
    na      = [k for k, v in scores.items() if v.get("pass") is None]
    if not failing:
        print("\n  All evaluable failure modes PASS.")
        print("  The attenuation result is structurally clean.")
    else:
        print(f"\n  Active failure modes: {failing}")
        interps = {
            "FD1": "Attenuation driven by abatement subsidy in PINNED arm — "
                   "not genuine diversion suppression. "
                   "Consider adding a no-transfer control arm.",
            "FD2": "EU dirty share near zero at M=1 — diversion already saturated. "
                   "The gap can't shrink further; any attenuation is spurious.",
            "FD3": "μ_open plateaus at high M — abatement cap is binding. "
                   "Mode=abatement forfeits transfers exceeding abatement spending. "
                   "Consider mode=consumption at high M to test without cap.",
            "FD4": "Effort allocation degeneracy — μ converges uniformly across regions "
                   "at high M so the performance incentive collapses. "
                   "Compare vs alloc=equal at same M to isolate.",
            "FD5": "RCPO λ collapsed — CBAM signal vanished from reward. "
                   "Result is driven by subsidised abatement, not the CBAM mechanism.",
            "FD6": "Training instability at high M — reward curves did not converge. "
                   "Results at those M levels are unreliable. Extend timesteps.",
            "FD7": "Divert-and-harvest equilibrium detected — agents maintain dirty EU "
                   "exports to preserve pool income. The transfer is funding diversion.",
            "FD8": "Region heterogeneity — M* aggregate is driven by a subset of regions. "
                   "Report per-region M* rather than aggregate.",
        }
        for fd in failing:
            if fd in interps:
                print(f"\n  [{fd}] {interps[fd]}")
    if na:
        print(f"\n  N/A (missing data): {na}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", required=True,
                    help="Path to the pkl file from cbam_experiment_C_amplifier.py")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="Output directory for the diagnostic figure (default: same as pkl).")
    args = ap.parse_args()

    print(f"  Loading {args.pkl} …")
    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)

    cells     = data["cells"]
    m_levels  = data.get("multiplier_levels", sorted({c["multiplier"] for c in cells}))

    print(f"  {len(cells)} cells loaded  "
          f"({len([c for c in cells if not c['pinned']])} OPEN, "
          f"{len([c for c in cells if c['pinned']])} PINNED)")

    tables = _build_tables(cells)
    curves = _load_training_curves(cells)
    print(f"  {len(curves)} training CSVs loaded")

    scores = _scorecard(tables, curves, list(m_levels))
    _print_report(scores, list(m_levels))

    outdir = args.out_dir or os.path.dirname(os.path.abspath(args.pkl))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(outdir, f"cbam_C_amp_posthoc_{timestamp}.png")
    _plot(tables, curves, scores, list(m_levels), out_path)


if __name__ == "__main__":
    main()
