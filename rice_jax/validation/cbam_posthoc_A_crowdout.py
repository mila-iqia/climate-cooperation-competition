"""cbam_posthoc_A_crowdout.py — Diagnostic post-hoc for Experiment A (crowd-out attenuation).

Designed to answer: WHY doesn't redistribution attenuate crowd-out?

Hypotheses tested
-----------------
H1. Transfer too small: CBAM revenue pool is negligible vs. diversion benefit.
H2. Moral hazard: rs=1 reduces μ_pinned (less incentive to mitigate if income
    is guaranteed) — so crowd-out gap doesn't shrink, it shifts down uniformly.
H3. Region heterogeneity: attenuation works for some regions but not others;
    the aggregate hides it.
H4. Diversion-dominance: μ_open is similar across rs conditions — agents
    divert regardless of transfer because the cost-avoidance incentive dominates.
H5. Training insufficiency: training curves haven't converged.

Panels
------
1. Per-region crowd-out gap: bar chart for each region under rs=0 vs rs=1.
2. Pinned / open decomposition: does redistribution move μ_pinned, μ_open, or both?
3. Transfer-vs-diversion budget: estimated transfer magnitude per region vs.
   CBAM cost avoided by diversion (from trade_flows_raw).
4. Training convergence: ep_return_mean curves per cell to check for divergence.
5. Moral hazard scatter: μ_pinned(rs=1) vs μ_pinned(rs=0) per region.

Usage
-----
    python validation/cbam_posthoc_A_crowdout.py \\
        --pkl experiments/cbam_experiment_A_crowdout_*/plots/cbam_A_crowdout_*.pkl

    # Or from run_experiment.py (once registered):
    python run_experiment.py --posthoc-only experiments/cbam_experiment_A_crowdout_*
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _experiment_util import get_output_dir


# ── Region metadata (9-region vuln setup) ──────────────────────────────────

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
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]  # drop RoW for figures


# ── Helpers ─────────────────────────────────────────────────────────────────

EVAL_LAST_T = 5  # Same as canonical_config

def _region_short(r):
    return REGION_SHORT.get(r, f"R{r}")


def _per_region_mu(mitigation_raw, last_t=EVAL_LAST_T):
    """Mean mitigation per region over last_t steps.  (n_ep, T, NR) → (NR,)"""
    return mitigation_raw[:, -last_t:, :].mean(axis=(0, 1))


def _eu_dirty_share_per_region(trade_flows_raw, last_t=EVAL_LAST_T):
    """Per-region dirty-export share going to EU.  (n_ep, T, NR, NR, NS) → (NR,)"""
    tf = trade_flows_raw[:, -last_t:]              # (n_ep, t, NR, NR, NS)
    dirty_to_eu = tf[:, :, :, EU_IDX, 0]           # (n_ep, t, NR)
    dirty_total = tf[:, :, :, :, 0].sum(-1)        # (n_ep, t, NR)
    share = dirty_to_eu / (dirty_total + 1e-10)    # (n_ep, t, NR)
    return share.mean(axis=(0, 1))                 # (NR,)


def _estimated_cbam_cost_per_region(trade_flows_raw, last_t=EVAL_LAST_T):
    """Rough proxy for CBAM cost: dirty exports TO EU × assumed effective rate.

    This is an approximation since we don't have the exact tariff rates in the
    pkl arrays, but the relative magnitudes across regions are informative.
    """
    tf = trade_flows_raw[:, -last_t:]
    dirty_to_eu = tf[:, :, :, EU_IDX, 0]           # (n_ep, t, NR)
    return dirty_to_eu.mean(axis=(0, 1))            # (NR,) — proportional to cost


def _diversion_benefit(trade_flows_pinned, trade_flows_open, last_t=EVAL_LAST_T):
    """Estimate diversion benefit = reduction in dirty-EU exports when diversion
    channel opens.  Higher means larger incentive to divert.
    """
    tf_p = trade_flows_pinned[:, -last_t:]
    tf_o = trade_flows_open[:, -last_t:]
    eu_dirty_pinned = tf_p[:, :, :, EU_IDX, 0].mean(axis=(0, 1))  # (NR,)
    eu_dirty_open   = tf_o[:, :, :, EU_IDX, 0].mean(axis=(0, 1))  # (NR,)
    return eu_dirty_pinned - eu_dirty_open  # (NR,) — positive = avoided EU exposure


def _read_csv(csv_path):
    """Read a training CSV, resolving relative paths from rice_jax/."""
    if not csv_path:
        return pd.DataFrame()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rice_jax_dir = os.path.dirname(script_dir)
    full = os.path.join(rice_jax_dir, csv_path) if not os.path.isabs(csv_path) else csv_path
    if not os.path.exists(full):
        return pd.DataFrame()
    return pd.read_csv(full)


# ── Core diagnostics ───────────────────────────────────────────────────────

def _build_diagnostic_tables(cells):
    """Build per-region diagnostic arrays from the pkl cells.

    Returns a dict of DataFrames keyed by analysis.
    """
    # Build per-cell per-region metrics
    rows = []
    for c in cells:
        mu_per_region = _per_region_mu(c["mitigation_raw"])
        eu_share      = _eu_dirty_share_per_region(c["trade_flows_raw"])
        cbam_proxy    = _estimated_cbam_cost_per_region(c["trade_flows_raw"])
        for r in CBAM_PLOT_REGIONS:
            rows.append({
                "seed":           c["seed"],
                "revenue_share":  c["revenue_share"],
                "transfer_mode":  c["transfer_mode"],
                "pinned":         c["pinned"],
                "region":         r,
                "region_name":    _region_short(r),
                "mu":             mu_per_region[r],
                "eu_dirty_share": eu_share[r],
                "cbam_proxy":     cbam_proxy[r],
            })
    df = pd.DataFrame(rows)

    # Pivot to get μ_pinned, μ_open per (seed, rs, mode, region)
    pivot = df.pivot_table(
        index=["seed", "revenue_share", "transfer_mode", "region", "region_name"],
        columns="pinned", values="mu",
    ).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={True: "mu_pinned", False: "mu_open"})
    pivot["crowd_out_gap"] = pivot["mu_pinned"] - pivot["mu_open"]

    # Attenuation per (seed, mode, region)
    g0 = pivot[pivot["revenue_share"] == 0.0].set_index(
        ["seed", "transfer_mode", "region"])["crowd_out_gap"]
    g1 = pivot[pivot["revenue_share"] == 1.0].set_index(
        ["seed", "transfer_mode", "region"])["crowd_out_gap"]
    attn = (g0 - g1).rename("attenuation").reset_index()

    # Moral hazard: compare μ_pinned between rs=0 and rs=1
    mu_p0 = pivot[pivot["revenue_share"] == 0.0].set_index(
        ["seed", "transfer_mode", "region"])["mu_pinned"]
    mu_p1 = pivot[pivot["revenue_share"] == 1.0].set_index(
        ["seed", "transfer_mode", "region"])["mu_pinned"]
    moral_hazard = (mu_p1 - mu_p0).rename("delta_mu_pinned").reset_index()

    # μ_open comparison
    mu_o0 = pivot[pivot["revenue_share"] == 0.0].set_index(
        ["seed", "transfer_mode", "region"])["mu_open"]
    mu_o1 = pivot[pivot["revenue_share"] == 1.0].set_index(
        ["seed", "transfer_mode", "region"])["mu_open"]
    delta_open = (mu_o1 - mu_o0).rename("delta_mu_open").reset_index()

    # Diversion benefit: compare pinned vs open trade flows in rs=0 condition
    diversion_rows = []
    for c in cells:
        if c["pinned"] or c["revenue_share"] != 0.0:
            continue
        # Find matching pinned cell
        match = [
            x for x in cells
            if x["seed"] == c["seed"]
            and x["revenue_share"] == c["revenue_share"]
            and x["transfer_mode"] == c["transfer_mode"]
            and x["pinned"]
        ]
        if not match:
            continue
        div_benefit = _diversion_benefit(
            match[0]["trade_flows_raw"], c["trade_flows_raw"])
        for r in CBAM_PLOT_REGIONS:
            diversion_rows.append({
                "seed": c["seed"],
                "transfer_mode": c["transfer_mode"],
                "region": r,
                "region_name": _region_short(r),
                "diversion_benefit": div_benefit[r],
            })
    df_diversion = pd.DataFrame(diversion_rows) if diversion_rows else pd.DataFrame()

    return {
        "cells_per_region": df,
        "pivot": pivot,
        "attenuation": attn,
        "moral_hazard": moral_hazard,
        "delta_open": delta_open,
        "diversion": df_diversion,
    }


def _print_summary(tables):
    """Print a text summary of the diagnostic findings."""
    attn = tables["attenuation"]
    mh   = tables["moral_hazard"]
    do   = tables["delta_open"]

    print("\n" + "═" * 70)
    print("  POST-HOC DIAGNOSTIC: Experiment A — Why doesn't redistribution help?")
    print("═" * 70)

    # H2: Moral hazard — does rs=1 reduce μ_pinned?
    print("\n─── H2: Moral hazard (Δ μ_pinned = μ_pinned(rs=1) − μ_pinned(rs=0)) ───")
    print("  Negative → redistribution reduces incentive to mitigate even when pinned")
    mh_by_mode = mh.groupby("transfer_mode")["delta_mu_pinned"].agg(["mean", "min", "max"])
    print(mh_by_mode.to_string(float_format=lambda x: f"{x:+.4f}"))
    mh_by_region = mh.groupby(["region", "transfer_mode"])["delta_mu_pinned"].mean().unstack()
    print("\n  Per-region breakdown:")
    mh_by_region.index = [_region_short(r) for r in mh_by_region.index]
    print(mh_by_region.to_string(float_format=lambda x: f"{x:+.4f}"))

    # H4: Diversion-dominance — does μ_open change with rs?
    print("\n─── H4: Diversion dominance (Δ μ_open = μ_open(rs=1) − μ_open(rs=0)) ───")
    print("  If ≈0, diversion strength is unchanged by transfers")
    do_by_mode = do.groupby("transfer_mode")["delta_mu_open"].agg(["mean", "min", "max"])
    print(do_by_mode.to_string(float_format=lambda x: f"{x:+.4f}"))
    do_by_region = do.groupby(["region", "transfer_mode"])["delta_mu_open"].mean().unstack()
    print("\n  Per-region breakdown:")
    do_by_region.index = [_region_short(r) for r in do_by_region.index]
    print(do_by_region.to_string(float_format=lambda x: f"{x:+.4f}"))

    # H3: Heterogeneity — attenuation per region
    print("\n─── H3: Per-region attenuation (gap(rs=0) − gap(rs=1)) ───")
    print("  Positive → redistribution helps that region; Negative → hurts")
    attn_by_region = attn.groupby(["region", "transfer_mode"])["attenuation"].mean().unstack()
    attn_by_region.index = [_region_short(r) for r in attn_by_region.index]
    print(attn_by_region.to_string(float_format=lambda x: f"{x:+.4f}"))

    # H1: Transfer budget vs diversion benefit
    if not tables["diversion"].empty:
        print("\n─── H1: Diversion benefit magnitude (dirty-EU exports saved by diversion) ───")
        print("  Larger = stronger incentive to divert regardless of transfer")
        div_by_region = tables["diversion"].groupby("region_name")["diversion_benefit"].mean()
        print(div_by_region.to_string(float_format=lambda x: f"{x:.4f}"))

    # Net diagnosis
    print("\n─── NET DIAGNOSIS ───")
    mean_delta_pinned = mh["delta_mu_pinned"].mean()
    mean_delta_open   = do["delta_mu_open"].mean()
    print(f"  Mean Δ μ_pinned (rs effect on baseline): {mean_delta_pinned:+.4f}")
    print(f"  Mean Δ μ_open   (rs effect on diversion): {mean_delta_open:+.4f}")
    if mean_delta_pinned < -0.02:
        print("  → MORAL HAZARD dominant: redistribution lowers effort even absent diversion")
    elif abs(mean_delta_open) < 0.02 and abs(mean_delta_pinned) < 0.02:
        print("  → TRANSFER TOO SMALL: redistribution barely moves either margin")
    elif mean_delta_open < mean_delta_pinned:
        print("  → PERVERSE INTERACTION: redistribution enables more diversion (lowers μ_open more)")
    else:
        print("  → MIXED: see per-region breakdown for heterogeneous effects")
    print("═" * 70)


# ── Plotting ───────────────────────────────────────────────────────────────

def _plot_diagnostic(tables, cells, out_dir, timestamp):
    """4-panel diagnostic figure."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Post-hoc Diagnostic: Experiment A — Crowd-out under redistribution\n"
        "Why doesn't redistribution attenuate crowd-out?",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    pivot  = tables["pivot"]
    attn   = tables["attenuation"]
    mh     = tables["moral_hazard"]
    do_tbl = tables["delta_open"]
    regions = CBAM_PLOT_REGIONS
    region_labels = [_region_short(r) for r in regions]
    n_regions = len(regions)

    # ── Panel 1: Per-region crowd-out gap, rs=0 vs rs=1 ──
    ax = fig.add_subplot(gs[0, 0])
    for i_mode, mode in enumerate(["consumption", "abatement"]):
        gap_rs0 = pivot[(pivot["revenue_share"] == 0.0) &
                        (pivot["transfer_mode"] == mode)].groupby("region")["crowd_out_gap"].mean()
        gap_rs1 = pivot[(pivot["revenue_share"] == 1.0) &
                        (pivot["transfer_mode"] == mode)].groupby("region")["crowd_out_gap"].mean()
        x = np.arange(n_regions)
        w = 0.18
        offset = (i_mode - 0.5) * 2  # -1 or +1
        ax.bar(x + offset * w - w/2, [gap_rs0.get(r, 0) for r in regions],
               width=w, alpha=0.7, label=f"rs=0 | {mode[:4]}",
               color="tab:blue" if mode == "consumption" else "tab:purple")
        ax.bar(x + offset * w + w/2, [gap_rs1.get(r, 0) for r in regions],
               width=w, alpha=0.5, label=f"rs=1 | {mode[:4]}",
               hatch="///",
               color="tab:blue" if mode == "consumption" else "tab:purple")
    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(region_labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Crowd-out gap (μ_pinned − μ_open)")
    ax.set_title("H3: Per-region crowd-out gap\n(solid=rs=0, hatched=rs=1; lower with rs=1 → attenuation)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 2: Moral hazard — Δμ_pinned per region ──
    ax = fig.add_subplot(gs[0, 1])
    for i_mode, mode in enumerate(["consumption", "abatement"]):
        vals = mh[mh["transfer_mode"] == mode].groupby("region")["delta_mu_pinned"].mean()
        x = np.arange(n_regions)
        w = 0.35
        offset = (i_mode - 0.5) * w
        color = "tab:orange" if mode == "consumption" else "tab:green"
        ax.bar(x + offset, [vals.get(r, 0) for r in regions],
               width=w, alpha=0.7, label=mode, color=color)
    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(region_labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ μ_pinned = μ_pinned(rs=1) − μ_pinned(rs=0)")
    ax.set_title("H2: Moral hazard\n(negative → rs=1 REDUCES baseline mitigation)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 3: Δμ_open per region (diversion dominance) ──
    ax = fig.add_subplot(gs[1, 0])
    for i_mode, mode in enumerate(["consumption", "abatement"]):
        vals = do_tbl[do_tbl["transfer_mode"] == mode].groupby("region")["delta_mu_open"].mean()
        x = np.arange(n_regions)
        w = 0.35
        offset = (i_mode - 0.5) * w
        color = "tab:orange" if mode == "consumption" else "tab:green"
        ax.bar(x + offset, [vals.get(r, 0) for r in regions],
               width=w, alpha=0.7, label=mode, color=color)
    ax.set_xticks(np.arange(n_regions))
    ax.set_xticklabels(region_labels, fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Δ μ_open = μ_open(rs=1) − μ_open(rs=0)")
    ax.set_title("H4: Diversion dominance\n(≈0 → transfers don't change diversion incentive)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # ── Panel 4: Summary text + per-region attenuation heatmap ──
    ax = fig.add_subplot(gs[1, 1])
    # Compute attenuation matrix: region × mode
    attn_matrix = np.zeros((n_regions, 2))
    for i_mode, mode in enumerate(["consumption", "abatement"]):
        vals = attn[attn["transfer_mode"] == mode].groupby("region")["attenuation"].mean()
        for i_r, r in enumerate(regions):
            attn_matrix[i_r, i_mode] = vals.get(r, 0)

    im = ax.imshow(attn_matrix, aspect="auto", cmap="RdYlGn", vmin=-0.25, vmax=0.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["consumption", "abatement"], fontsize=9)
    ax.set_yticks(range(n_regions))
    ax.set_yticklabels(region_labels, fontsize=8)
    # Annotate cells
    for i in range(n_regions):
        for j in range(2):
            v = attn_matrix[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.12 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Per-region attenuation\n(green = transfers help, red = transfers hurt)")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"posthoc_A_diagnostic_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDiagnostic figure → {out_path}")
    return out_path


def _plot_convergence(cells, out_dir, timestamp):
    """Training curve overlay from CSVs (if available)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Training convergence per condition (ep_return_mean)", fontsize=11)

    conditions = [
        ("rs=0, pinned",  0.0, True),
        ("rs=0, open",    0.0, False),
        ("rs=1, pinned",  1.0, True),
        ("rs=1, open",    1.0, False),
    ]
    found_any = False
    for ax, (label, rs, pinned) in zip(axes.flat, conditions):
        ax.set_title(label, fontsize=9)
        matching = [c for c in cells
                    if c["revenue_share"] == rs and c["pinned"] == pinned]
        for c in matching:
            csv_path = c.get("csv_path", "")
            df = _read_csv(csv_path)
            if df.empty or "ep_return_mean" not in df.columns:
                continue
            found_any = True
            lbl = f"s{c['seed']}_{c['transfer_mode'][:4]}"
            ax.plot(df["ep_return_mean"].values, label=lbl, alpha=0.7)
        ax.legend(fontsize=7)
        ax.set_xlabel("log_interval step")
        ax.grid(alpha=0.3)

    if not found_any:
        plt.close(fig)
        print("  [convergence] No CSVs found, skipping convergence plot.")
        return None

    out_path = os.path.join(out_dir, f"posthoc_A_convergence_{timestamp}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Convergence plot → {out_path}")
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc diagnostic for Experiment A (crowd-out attenuation)")
    parser.add_argument("--pkl", required=True,
                        help="Path to the Experiment A pkl bundle")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same dir as pkl)")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        bundle = pickle.load(f)

    cells = bundle["cells"]
    print(f"Loaded {len(cells)} cells from {args.pkl}")
    print(f"  Seeds: {bundle.get('seeds', '?')}")
    print(f"  Pass: {bundle.get('pass_info', {}).get('passed', '?')}")

    out_dir = args.out_dir or os.path.dirname(args.pkl)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build diagnostic tables
    tables = _build_diagnostic_tables(cells)

    # Print summary
    _print_summary(tables)

    # Plots
    _plot_diagnostic(tables, cells, out_dir, timestamp)
    _plot_convergence(cells, out_dir, timestamp)

    print(f"\n  Post-hoc complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
