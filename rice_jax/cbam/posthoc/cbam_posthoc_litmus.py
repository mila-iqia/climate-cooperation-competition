"""cbam_posthoc_litmus.py — Build-Measure-Learn post-hoc analysis.

Layer 1 "scorecard" (no JAX):
  1. Pass table (matches LITMUS_TEST_SPEC.md format)
  2. Per-region decomposition bar charts
  3. Cross-suite consistency checks
  4. Training curve overlays from CSVs
  5. Optionally writes report-ready markdown

Layer 2 "introspection" (requires JAX, needs --save-agents pkls):
  6. Jacobian heatmap: which obs dims drive each agent's policy
  7. First-layer weight norms: coarse connectivity check
  8. Counterfactual obs perturbation: zero out CBAM obs dims, compare actions
  9. Per-test introspection summary guided by Layer 1 triage

Usage (from rice_jax/, any Python 3.11+ env — no JAX needed for Layer 1):
    # Layer 1 only:
    python cbam/drivers/cbam_posthoc_litmus.py \\
        --mech-pkl plots/litmus_mech_1M_freesav_*.pkl \\
        --cond-pkl plots/litmus_cond_1M_freesav_*.pkl

    # Full analysis (Layer 1 + Layer 2, requires JAX and --save-agents pkls):
    python cbam/drivers/cbam_posthoc_litmus.py \\
        --mech-pkl plots/litmus_mech_1M_freesav_*.pkl \\
        --cond-pkl plots/litmus_cond_1M_freesav_*.pkl \\
        --introspect

    # Write report to file:
    python cbam/drivers/cbam_posthoc_litmus.py \\
        --mech-pkl plots/litmus_mech_*.pkl --cond-pkl plots/litmus_cond_*.pkl \\
        --out-report auto --introspect
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import os
import pickle
import sys
from datetime import datetime

from _experiment_util import get_output_dir

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Region metadata (9-region vuln setup) ──────────────────────────────────

NUM_REGIONS = 9
EU_IDX      = 3

REGION_NAMES = {
    0: "RoW", 1: "Russia+Eurasia", 2: "MENA", 3: "EU",
    4: "SSA Mining", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
REGION_SHORT = {
    0: "RoW", 1: "Rus+Eur", 2: "MENA", 3: "EU",
    4: "SSA", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
NON_EU            = [r for r in range(NUM_REGIONS) if r != EU_IDX]
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]   # drop RoW for plots


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fmt(v, precision=3):
    """Format a scalar for display."""
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _load_cbam_exposure():
    """Load baseline (2016 MRIO) CBAM exposure per region.

    Returns dict with keys:
      dirty_eu_share[r]  — share of region r's dirty exports going to EU
      intensity[r]       — normalised emissions intensity (dirty sector)
      exposure[r]        — composite: share × tef × dest_share × intensity
    or None if data can't be loaded.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        rice_jax_dir = os.path.dirname(script_dir)
        project_root = os.path.dirname(rice_jax_dir)
        sys.path.insert(0, rice_jax_dir)
        from rice_jax.mrio.loaders import (
            load_sector_shares,
            load_bilateral_trade_shares,
            load_emissions_intensity,
            _build_sector_groups,
            _aggregate_sector_arrays,
        )
        mrio_dir = os.path.join(project_root, "csv_asset", "mrio",
                                "aggregated", "eora_agg_9")
        cc_csv = os.path.join(project_root, "cbam_yamls",
                              "CountryClass_cbam_vuln_9.csv")
        if not os.path.isdir(mrio_dir) or not os.path.isfile(cc_csv):
            return None

        shares, sector_names, rice_to_mrio = load_sector_shares(
            mrio_dir, NUM_REGIONS, cc_csv)
        dest_alloc_26, tef_26 = load_bilateral_trade_shares(
            mrio_dir, rice_to_mrio)
        intensity_26 = load_emissions_intensity(mrio_dir, rice_to_mrio)

        # Normalise intensity (mirrors RiceMRIO.__post_init__)
        imean = intensity_26.mean()
        if imean > 0:
            intensity_26 = intensity_26 / imean * 0.1

        groups = _build_sector_groups(sector_names, "emissions-simple")
        _, shares_2, tef_2, dest_alloc_2, intensity_2 = \
            _aggregate_sector_arrays(
                groups, shares, tef_26, dest_alloc_26, intensity_26)

        S_DIRTY = 0
        dirty_eu_share = np.array([
            dest_alloc_2[r, S_DIRTY, EU_IDX] for r in range(NUM_REGIONS)])
        intensity = np.array([
            intensity_2[r, S_DIRTY] for r in range(NUM_REGIONS)])
        exposure = np.array([
            shares_2[r, S_DIRTY] * tef_2[r, S_DIRTY]
            * dest_alloc_2[r, S_DIRTY, EU_IDX] * intensity_2[r, S_DIRTY]
            for r in range(NUM_REGIONS)])

        return dict(dirty_eu_share=dirty_eu_share,
                    intensity=intensity, exposure=exposure)
    except Exception as e:
        print(f"  [WARN] Could not load CBAM exposure data: {e}")
        return None


def _read_csv(csv_path):
    """Read a training CSV, resolving relative paths from rice_jax/."""
    if not csv_path:
        return pd.DataFrame()
    # csv_path is stored relative to rice_jax/ working dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rice_jax_dir = os.path.dirname(script_dir)
    full = os.path.join(rice_jax_dir, csv_path) if not os.path.isabs(csv_path) else csv_path
    if not os.path.exists(full):
        return pd.DataFrame()
    return pd.read_csv(full)


def _dict_to_array(d, regions=None):
    """Convert a {region_idx: value} dict to a list aligned with `regions`."""
    if regions is None:
        regions = CBAM_PLOT_REGIONS
    return [d.get(r, float("nan")) for r in regions]


# ── Pass Table ──────────────────────────────────────────────────────────────

def build_pass_table(mech, cond):
    """Build the pass table as a list of row dicts."""
    rows = []

    if mech:
        m1 = mech["m1"]
        rows.append(dict(
            test="M1", family="Fixed diff",
            question="Does diversion exist?",
            metric=f"EU dirty share: {_fmt(m1['data']['share_ctrl'])} → {_fmt(m1['data']['share_diff'])}",
            value=f"{m1['data']['rel_drop']*100:.1f}% rel drop",
            passed=_fmt(m1["passed"]),
            strength="Strong",
        ))

        m2 = mech["m2"]
        rows.append(dict(
            test="M2", family="Fixed diff",
            question="Costless mitigation when pinned?",
            metric=f"mean μ (non-EU) = {_fmt(m2['data']['mean_mu'])}",
            value=f"μ = {_fmt(m2['data']['mean_mu'])}",
            passed=_fmt(m2["passed"]),
            strength="Strong",
        ))

        m3 = mech["m3"]
        gap_m3_m2 = m3["data"]["mean_mu"] - m2["data"]["mean_mu"]
        rows.append(dict(
            test="M3", family="Fixed diff",
            question="Costly mitigation survives when pinned?",
            metric=f"mean μ = {_fmt(m3['data']['mean_mu'])} (gap from M2: {gap_m3_m2:+.3f})",
            value=f"μ = {_fmt(m3['data']['mean_mu'])}",
            passed=_fmt(m3["passed"]),
            strength="Moderate",
        ))

        m4 = mech["m4"]
        gap_m4_m3 = m4["data"]["mean_mu"] - m3["data"]["mean_mu"]
        rows.append(dict(
            test="M4", family="Fixed diff",
            question="Diversion crowds out mitigation?",
            metric=f"μ_M4={_fmt(m4['data']['mean_mu'])} vs μ_M3={_fmt(m3['data']['mean_mu'])} (Δ={gap_m4_m3:+.3f})",
            value=f"μ_M4 < μ_M3 by {abs(gap_m4_m3):.3f}",
            passed=_fmt(m4["passed"]),
            strength="Strong",
        ))

        n1 = mech["n1"]
        rows.append(dict(
            test="N1", family="Null",
            question="Mechanical tariff-relief?",
            metric=f"cost_low={_fmt(n1['data']['cost_low'])}, cost_high={_fmt(n1['data']['cost_high'])}",
            value=f"cost_high < cost_low",
            passed=_fmt(n1["passed"]),
            strength="Mechanical",
        ))

    if cond:
        c1 = cond["c1"]
        rows.append(dict(
            test="C1", family="Randomised diff",
            question="Policy conditions exports on CBAM?",
            metric=f"share_on={_fmt(c1['data']['share_on'])}, share_off={_fmt(c1['data']['share_off'])}",
            value=f"{c1['data']['cond_gap']*100:.1f}% rel gap ({c1['data']['cond_gap_pp']:.1f}pp)",
            passed=_fmt(c1["passed"]),
            strength="Strong",
        ))

        c2 = cond["c2"]
        rows.append(dict(
            test="C2a", family="Randomised diff",
            question="Costless mitigation conditioning?",
            metric=f"μ_on={_fmt(c2['data']['mu_a_on'])}, μ_off={_fmt(c2['data']['mu_a_off'])}",
            value=f"gap={c2['data']['gap_a']:.3f} ({c2['data']['grade_a']})",
            passed=c2["data"]["grade_a"],
            strength="Graded",
        ))
        rows.append(dict(
            test="C2b", family="Randomised diff",
            question="Costly mitigation conditioning?",
            metric=f"μ_on={_fmt(c2['data']['mu_b_on'])}, μ_off={_fmt(c2['data']['mu_b_off'])}",
            value=f"gap={c2['data']['gap_b']:.3f} ({c2['data']['grade_b']})",
            passed=c2["data"]["grade_b"],
            strength="Graded",
        ))

        c3 = cond["c3"]
        c2b_mu_on = c2["data"]["mu_b_on"]
        crowd_out = c2b_mu_on - c3["data"]["mu_on"]
        rows.append(dict(
            test="C3", family="Randomised diff",
            question="Conditioned crowd-out?",
            metric=f"μ_both={_fmt(c3['data']['mu_on'])} vs μ_pinned={_fmt(c2b_mu_on)}",
            value=f"crowd-out = {crowd_out:.3f}",
            passed=_fmt(c3["passed"]),
            strength="Strong",
        ))

    return rows


def print_pass_table(rows):
    """Print the pass table to stdout."""
    print("\n" + "=" * 100)
    print("  LITMUS PASS TABLE")
    print("=" * 100)
    header = f"{'Test':<6} {'Family':<18} {'Question':<40} {'Value':<30} {'Pass?':<8} {'Strength'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['test']:<6} {r['family']:<18} {r['question']:<40} {r['value']:<30} {r['passed']:<8} {r['strength']}")
    print()


# ── Cross-Suite Consistency ─────────────────────────────────────────────────

def cross_suite_consistency(mech, cond):
    """Check consistency between mechanism and conditioning results."""
    checks = []

    if not (mech and cond):
        return checks

    m1_rel  = mech["m1"]["data"]["rel_drop"]
    c1_gap  = cond["c1"]["data"]["cond_gap"]

    # 1) M1 diversion vs C1 conditioning — both should show strong diversion
    agree = (m1_rel > 0.15) == (c1_gap > 0.15)
    checks.append(dict(
        name="Diversion agreement (M1 vs C1)",
        m_val=f"M1 rel_drop = {m1_rel:.3f}",
        c_val=f"C1 cond_gap = {c1_gap:.3f}",
        agree=agree,
        note="Both show strong diversion" if agree else "Disagreement on diversion strength",
    ))

    # 2) M4 crowd-out vs C3 crowd-out — same direction?
    m4_mu = mech["m4"]["data"]["mean_mu"]
    m3_mu = mech["m3"]["data"]["mean_mu"]
    m_crowd = m3_mu - m4_mu   # positive = crowd-out present

    c3_mu   = cond["c3"]["data"]["mu_on"]
    c2b_mu  = cond["c2"]["data"]["mu_b_on"]
    c_crowd = c2b_mu - c3_mu  # positive = crowd-out present

    same_sign = (m_crowd > 0) == (c_crowd > 0)
    checks.append(dict(
        name="Crowd-out agreement (M4 vs C3)",
        m_val=f"M3→M4 Δμ = {-m_crowd:+.3f}",
        c_val=f"C2b→C3 Δμ = {-c_crowd:+.3f}",
        agree=same_sign,
        note=f"Both {'show' if same_sign else 'DISAGREE on'} crowd-out"
             f" (mech={m_crowd:.3f}, cond={c_crowd:.3f})",
    ))

    # 3) M3 μ level vs C2b μ_on level — mitigation survives at similar levels?
    m3_level = mech["m3"]["data"]["mean_mu"]
    c2b_level = cond["c2"]["data"]["mu_b_on"]
    ratio = m3_level / c2b_level if c2b_level > 0 else float("nan")
    close = 0.5 < ratio < 2.0
    checks.append(dict(
        name="Mitigation level agreement (M3 vs C2b)",
        m_val=f"M3 μ = {m3_level:.3f}",
        c_val=f"C2b μ_on = {c2b_level:.3f}",
        agree=close,
        note=f"Ratio = {ratio:.2f} ({'comparable' if close else 'divergent'})",
    ))

    return checks


def print_consistency(checks):
    """Print cross-suite consistency checks."""
    if not checks:
        return
    print("\n" + "=" * 80)
    print("  CROSS-SUITE CONSISTENCY")
    print("=" * 80)
    for c in checks:
        tag = "OK" if c["agree"] else "!!"
        print(f"  [{tag}] {c['name']}")
        print(f"       {c['m_val']}")
        print(f"       {c['c_val']}")
        print(f"       → {c['note']}")
    print()


# ── Per-Region Decomposition ───────────────────────────────────────────────

def plot_per_region(mech, cond, out_path):
    """Per-region decomposition bar charts."""
    n_panels = 0
    if mech:
        n_panels += 3  # M1 dirty share, M2/M3/M4 μ, M4 dirty share
    if cond:
        n_panels += 2  # C1 share gap, C2b μ gap

    # Try to load baseline CBAM exposure
    cbam_exp = _load_cbam_exposure()
    if cbam_exp is not None:
        n_panels += 1  # leading panel

    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0
    labels = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    x = np.arange(len(CBAM_PLOT_REGIONS))
    bar_w = 0.35

    # Panel 0: Baseline CBAM exposure (2016 MRIO)
    if cbam_exp is not None:
        ax = axes[ax_idx]; ax_idx += 1
        eu_share = [cbam_exp["dirty_eu_share"][r] for r in CBAM_PLOT_REGIONS]
        intensity = [cbam_exp["intensity"][r] for r in CBAM_PLOT_REGIONS]
        ax.bar(x - bar_w/2, eu_share, bar_w,
               label="Dirty→EU share", color="#1f77b4", alpha=0.8)
        ax2 = ax.twinx()
        ax2.bar(x + bar_w/2, intensity, bar_w,
                label="Intensity (dirty)", color="#ff7f0e", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Dest. share → EU", fontsize=7, color="#1f77b4")
        ax2.set_ylabel("Emissions intensity", fontsize=7, color="#ff7f0e")
        ax.set_title("Baseline CBAM Exposure (2016 MRIO)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right")

    if mech:
        # M1: per-region EU dirty share (ctrl vs diff)
        ax = axes[ax_idx]; ax_idx += 1
        ctrl = _dict_to_array(mech["m1"]["data"]["pr_ctrl"])
        diff = _dict_to_array(mech["m1"]["data"]["pr_diff"])
        ax.bar(x - bar_w/2, ctrl, bar_w, label="Control (no CBAM)", color="#1f77b4", alpha=0.8)
        ax.bar(x + bar_w/2, diff, bar_w, label="Differential CBAM", color="#d62728", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("EU dirty export share")
        ax.set_title("M1: Diversion per region", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # M2/M3/M4: per-region μ
        ax = axes[ax_idx]; ax_idx += 1
        w = 0.25
        mu2 = _dict_to_array(mech["m2"]["data"]["pr_mu"])
        mu3 = _dict_to_array(mech["m3"]["data"]["pr_mu"])
        mu4 = _dict_to_array(mech["m4"]["data"]["pr_mu"])
        ax.bar(x - w, mu2, w, label=f"M2 costless (μ̄={mech['m2']['data']['mean_mu']:.3f})", color="#2ca02c", alpha=0.8)
        ax.bar(x,     mu3, w, label=f"M3 costly (μ̄={mech['m3']['data']['mean_mu']:.3f})", color="#ff7f0e", alpha=0.8)
        ax.bar(x + w, mu4, w, label=f"M4 both (μ̄={mech['m4']['data']['mean_mu']:.3f})", color="#9467bd", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Mitigation rate μ")
        ax.set_title("M2–M4: Mitigation per region", fontsize=9)
        ax.legend(fontsize=6, loc="upper right")
        ax.tick_params(labelsize=7)

        # M4: per-region EU dirty share
        ax = axes[ax_idx]; ax_idx += 1
        share4 = _dict_to_array(mech["m4"]["data"]["pr_share"])
        ax.bar(x, share4, 0.5, color="#d62728", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("EU dirty export share")
        ax.set_title("M4: Dirty share (both open)", fontsize=9)
        ax.tick_params(labelsize=7)

    if cond:
        # C1: per-region share gap
        ax = axes[ax_idx]; ax_idx += 1
        on  = _dict_to_array(cond["c1"]["data"]["pr_on"])
        off = _dict_to_array(cond["c1"]["data"]["pr_off"])
        ax.bar(x - bar_w/2, off, bar_w, label="CBAM=off", color="#1f77b4", alpha=0.8)
        ax.bar(x + bar_w/2, on,  bar_w, label="CBAM=on",  color="#d62728", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("EU dirty export share")
        ax.set_title("C1: Export conditioning per region", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        # C2: per-region μ gap (costly, C2b)
        ax = axes[ax_idx]; ax_idx += 1
        mu_on  = _dict_to_array(cond["c2"]["data"]["pr_mu_b_on"])
        mu_off = _dict_to_array(cond["c2"]["data"]["pr_mu_b_off"])
        ax.bar(x - bar_w/2, mu_off, bar_w, label="CBAM=off", color="#1f77b4", alpha=0.8)
        ax.bar(x + bar_w/2, mu_on,  bar_w, label="CBAM=on",  color="#d62728", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Mitigation rate μ")
        ax.set_title(f"C2b: Mitigation conditioning ({cond['c2']['data']['grade_b']})", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-region decomposition → {out_path}")


# ── Training Curve Overlays ────────────────────────────────────────────────

def plot_training_curves(mech, cond, out_path):
    """Overlay ep_return_mean from all training CSVs."""
    entries = []

    if mech:
        entries.append(("M1 ctrl", mech["m1"]["data"].get("csv_ctrl"), "#1f77b4"))
        entries.append(("M1 diff", mech["m1"]["data"].get("csv_diff"), "#d62728"))
        entries.append(("M2 costless μ", mech["m2"]["data"].get("csv_path"), "#2ca02c"))
        entries.append(("M3 costly μ", mech["m3"]["data"].get("csv_path"), "#ff7f0e"))
        entries.append(("M4 both", mech["m4"]["data"].get("csv_path"), "#9467bd"))

    if cond:
        entries.append(("C1 export", cond["c1"]["data"].get("csv_path"), "#8c564b"))
        entries.append(("C2a costless", cond["c2"]["data"].get("csv_a"), "#e377c2"))
        entries.append(("C2b costly", cond["c2"]["data"].get("csv_b"), "#7f7f7f"))
        entries.append(("C3 both", cond["c3"]["data"].get("csv_path"), "#bcbd22"))

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    converged_flags = []
    for label, csv_path, color in entries:
        df = _read_csv(csv_path)
        if df.empty or "ep_return_mean" not in df.columns:
            converged_flags.append((label, None))
            continue
        y = df["ep_return_mean"].values
        x = df["timestep"].values if "timestep" in df.columns else np.arange(len(y))
        ax.plot(x, y, label=label, color=color, linewidth=1.2, alpha=0.8)

        # Simple convergence check: last 5 values std < 5% of mean
        if len(y) >= 5:
            tail = y[-5:]
            tail_mean = np.mean(tail)
            tail_std  = np.std(tail)
            stable = tail_std < 0.05 * abs(tail_mean) if tail_mean != 0 else tail_std < 0.01
            converged_flags.append((label, stable))
        else:
            converged_flags.append((label, None))

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode return (mean)")
    ax.set_title("Training Curves — All Litmus Tests")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {out_path}")

    # Print convergence warnings
    warnings = [(label, ok) for label, ok in converged_flags if ok is False]
    if warnings:
        print("\n  ⚠ CONVERGENCE WARNINGS:")
        for label, _ in warnings:
            print(f"    {label}: ep_return tail not stabilised")
    else:
        print("  All training curves appear converged.")


# ── Anomaly Flags ──────────────────────────────────────────────────────────

def flag_anomalies(mech, cond):
    """Flag tests that barely pass or have suspicious patterns."""
    flags = []

    if mech:
        # M1: barely passes?
        m1_drop = mech["m1"]["data"]["rel_drop"]
        if mech["m1"]["passed"] and m1_drop < 0.20:
            flags.append(f"M1 barely passes: {m1_drop*100:.1f}% rel drop (threshold 15%)")

        # M3: very low μ?
        m3_mu = mech["m3"]["data"]["mean_mu"]
        if mech["m3"]["passed"] and m3_mu < 0.05:
            flags.append(f"M3 barely passes: μ={m3_mu:.3f} (threshold 0.01)")

        # M4 crowd-out very small?
        m4_mu = mech["m4"]["data"]["mean_mu"]
        m3_mu = mech["m3"]["data"]["mean_mu"]
        if mech["m4"]["passed"] and (m3_mu - m4_mu) < 0.01:
            flags.append(f"M4 crowd-out marginal: Δμ = {m3_mu - m4_mu:.3f}")

        # M4 crowd-out is in wrong direction despite pass?
        if mech["m4"]["passed"] and m4_mu >= m3_mu:
            flags.append(f"M4 marked pass but μ_M4={m4_mu:.3f} >= μ_M3={m3_mu:.3f}")

        # Per-region sign flips in M1
        pr_ctrl = mech["m1"]["data"]["pr_ctrl"]
        pr_diff = mech["m1"]["data"]["pr_diff"]
        inversions = [r for r in CBAM_PLOT_REGIONS
                      if pr_diff.get(r, 0) > pr_ctrl.get(r, 0)]
        if inversions:
            names = [REGION_NAMES[r] for r in inversions]
            flags.append(f"M1 per-region inversion (diff > ctrl): {', '.join(names)}")

    if cond:
        # C1 barely passes?
        c1_gap = cond["c1"]["data"]["cond_gap"]
        if cond["c1"]["passed"] and c1_gap < 0.20:
            flags.append(f"C1 barely passes: {c1_gap*100:.1f}% gap (threshold 15%)")

        # C2 mixed grades?
        c2 = cond["c2"]["data"]
        if c2["grade_a"] != c2["grade_b"]:
            flags.append(f"C2 mixed grades: C2a={c2['grade_a']}, C2b={c2['grade_b']}")

    return flags


def print_anomalies(flags):
    if not flags:
        print("\n  No anomalies detected.")
        return
    print("\n" + "=" * 60)
    print("  ANOMALY FLAGS")
    print("=" * 60)
    for f in flags:
        print(f"  ⚠ {f}")
    print()


# ── Markdown Output ─────────────────────────────────────────────────────────

def generate_markdown(rows, checks, flags, mech, cond, introspection=None):
    """Generate report-ready markdown."""
    lines = []
    lines.append("## Litmus Results Summary (auto-generated)\n")

    # Pass table
    lines.append("### Pass Table\n")
    lines.append("| Test | Family | Question | Value | Pass? | Strength |")
    lines.append("|------|--------|----------|-------|-------|----------|")
    for r in rows:
        lines.append(f"| {r['test']} | {r['family']} | {r['question']} | {r['value']} | {r['passed']} | {r['strength']} |")
    lines.append("")

    # Key numbers
    if mech and cond:
        lines.append("### Key Numbers\n")
        lines.append(f"- **Diversion (M1):** EU dirty share drops from "
                     f"{mech['m1']['data']['share_ctrl']:.3f} to "
                     f"{mech['m1']['data']['share_diff']:.3f} "
                     f"({mech['m1']['data']['rel_drop']*100:.1f}% relative)")
        lines.append(f"- **Costless mitigation (M2):** μ = {mech['m2']['data']['mean_mu']:.3f}")
        lines.append(f"- **Costly mitigation (M3):** μ = {mech['m3']['data']['mean_mu']:.3f}")
        lines.append(f"- **Crowd-out (M4):** μ drops to {mech['m4']['data']['mean_mu']:.3f} "
                     f"(Δ = {mech['m4']['data']['mean_mu'] - mech['m3']['data']['mean_mu']:+.3f} from M3)")
        lines.append(f"- **Export conditioning (C1):** {cond['c1']['data']['cond_gap']*100:.1f}% "
                     f"within-policy gap ({cond['c1']['data']['cond_gap_pp']:.1f}pp)")
        lines.append(f"- **Mitigation conditioning (C2b):** "
                     f"gap = {cond['c2']['data']['gap_b']:.3f} ({cond['c2']['data']['grade_b']})")
        lines.append(f"- **Conditioned crowd-out (C3):** μ_both = {cond['c3']['data']['mu_on']:.3f} "
                     f"vs μ_pinned = {cond['c2']['data']['mu_b_on']:.3f}")
        lines.append("")

    # Per-region detail tables (use region names)
    if mech:
        lines.append("### Per-Region Detail\n")
        # M1: diversion per region
        lines.append("#### M1: EU Dirty Share (Control vs Differential)\n")
        header = "| Region | " + " | ".join(["Control", "Differential", "Δ"]) + " |"
        lines.append(header)
        lines.append("|--------|---------|--------------|-----|")
        pr_ctrl = mech["m1"]["data"]["pr_ctrl"]
        pr_diff = mech["m1"]["data"]["pr_diff"]
        for r in CBAM_PLOT_REGIONS:
            c = pr_ctrl.get(r, 0)
            d = pr_diff.get(r, 0)
            delta = d - c
            lines.append(f"| {REGION_NAMES[r]} | {c:.3f} | {d:.3f} | {delta:+.3f} |")
        lines.append("")

        # M2-M4: mitigation per region
        lines.append("#### M2–M4: Mitigation Rate μ per Region\n")
        header = "| Region | M2 (costless) | M3 (costly) | M4 (both) | M4−M3 |"
        lines.append(header)
        lines.append("|--------|---------------|-------------|-----------|-------|")
        for r in CBAM_PLOT_REGIONS:
            m2 = mech["m2"]["data"]["pr_mu"].get(r, 0)
            m3 = mech["m3"]["data"]["pr_mu"].get(r, 0)
            m4 = mech["m4"]["data"]["pr_mu"].get(r, 0)
            lines.append(f"| {REGION_NAMES[r]} | {m2:.3f} | {m3:.3f} | {m4:.3f} | {m4-m3:+.3f} |")
        lines.append("")

    if cond:
        # C1: export conditioning per region
        lines.append("#### C1: Export Conditioning per Region\n")
        header = "| Region | CBAM=off | CBAM=on | Δ |"
        lines.append(header)
        lines.append("|--------|----------|---------|-----|")
        pr_on = cond["c1"]["data"]["pr_on"]
        pr_off = cond["c1"]["data"]["pr_off"]
        for r in CBAM_PLOT_REGIONS:
            on = pr_on.get(r, 0)
            off = pr_off.get(r, 0)
            lines.append(f"| {REGION_NAMES[r]} | {off:.3f} | {on:.3f} | {on-off:+.3f} |")
        lines.append("")

        # C2b: mitigation conditioning per region
        lines.append("#### C2b: Mitigation Conditioning per Region\n")
        header = "| Region | μ (CBAM=off) | μ (CBAM=on) | Δ |"
        lines.append(header)
        lines.append("|--------|--------------|-------------|-----|")
        pr_mu_on = cond["c2"]["data"]["pr_mu_b_on"]
        pr_mu_off = cond["c2"]["data"]["pr_mu_b_off"]
        for r in CBAM_PLOT_REGIONS:
            on = pr_mu_on.get(r, 0)
            off = pr_mu_off.get(r, 0)
            lines.append(f"| {REGION_NAMES[r]} | {off:.3f} | {on:.3f} | {on-off:+.3f} |")
        lines.append("")

    # Cross-suite
    if checks:
        lines.append("### Cross-Suite Consistency\n")
        for c in checks:
            tag = "✓" if c["agree"] else "✗"
            lines.append(f"- {tag} **{c['name']}**: {c['note']}")
        lines.append("")

    # Anomalies
    if flags:
        lines.append("### Anomaly Flags\n")
        for f in flags:
            lines.append(f"- ⚠ {f}")
        lines.append("")

    # One-sentence summaries (from LITMUS_TEST_SPEC.md)
    lines.append("### One-Sentence Summaries\n")
    if mech:
        lines.append("> **Fixed-differential:** The fixed-differential litmus suite shows that "
                     "both diversion and mitigation are viable responses to CBAM, but when both "
                     "margins are open, exporters shift toward diversion and away from costly mitigation.\n")
    if cond:
        lines.append("> **Randomised-differential:** The randomised-differential litmus suite "
                     "shows that a single trained policy clearly conditions export reallocation "
                     "on the CBAM signal, while mitigation conditioning is weaker and less robust, "
                     "and crowd-out remains present when both margins are available.\n")

    # Introspection (Layer 2)
    if introspection:
        lines.append("### Layer 2: Network Introspection\n")
        for tag, res in sorted(introspection.items()):
            lines.append(f"**{tag}**\n")
            # Top Jacobian groups
            jac = res.get("jacobian", {})
            if jac:
                sorted_jac = sorted(jac.items(), key=lambda x: -x[1])[:3]
                jac_str = ", ".join(f"{n} ({v:.3f})" for n, v in sorted_jac)
                lines.append(f"- Top Jacobian groups: {jac_str}")
            # Counterfactual
            cf = res.get("counterfactual_delta", float("nan"))
            if not np.isnan(cf):
                lines.append(f"- Counterfactual (zero CBAM obs): L1 logit shift = {cf:.4f}")
            lines.append("")

    return "\n".join(lines)


# ── Triage List (Layer 1 → Layer 2 handoff) ─────────────────────────────────

def build_triage(rows, mech, cond):
    """Build a triage list for Layer 2 introspection."""
    triage = []

    if mech:
        # M4 always interesting for introspection
        triage.append(dict(
            test="M4",
            priority="high",
            question="Which obs group drives the mitigation reduction vs M3? "
                     "Is it cbam_cost or dest_alloc dominance?",
        ))

        # M1 per-region inversions?
        pr_ctrl = mech["m1"]["data"]["pr_ctrl"]
        pr_diff = mech["m1"]["data"]["pr_diff"]
        inversions = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS
                      if pr_diff.get(r, 0) > pr_ctrl.get(r, 0)]
        if inversions:
            triage.append(dict(
                test="M1",
                priority="high",
                question=f"Per-region inversion in {', '.join(inversions)} — "
                         "is this noise or a structural feature?",
            ))

    if cond:
        c2 = cond["c2"]["data"]
        if c2["overall_grade"] in ("weak", "none/mixed"):
            triage.append(dict(
                test="C2",
                priority="high",
                question="Is cbam_tariff_rate obs dim (idx 12) low-Jacobian? "
                         "Mitigation conditioning is weak — confirm network isn't reading CBAM signal.",
            ))
        else:
            triage.append(dict(
                test="C2",
                priority="medium",
                question="Mitigation conditioning is strong — verify Jacobian "
                         "shows cbam_tariff_rate (idx 12) is actually high-weight.",
            ))

    return triage


def print_triage(triage):
    if not triage:
        return
    print("\n" + "=" * 70)
    print("  TRIAGE → Layer 2 Introspection")
    print("=" * 70)
    for t in triage:
        print(f"  [{t['priority'].upper():>6}] {t['test']}: {t['question']}")
    print()


# ── Layer 2: Introspection (JAX required) ──────────────────────────────────

# Obs dimension label groups (alphabetical flatten order, 53 dims total)
# Must match cbam_region_analysis.py
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
OBS_DIM = 53


def _has_agents(data):
    """Check if a pkl data dict contains saved agents."""
    return any(k.startswith("_agent") for k in data.keys())


def _get_agents(data):
    """Extract {label: state_dict} from a data dict.

    Values are multi-agent state dicts: {"region-00": PPOState, ...}.
    (Saved as agent.state from PPO, not the full PPO object.)
    """
    return {k.lstrip("_"): v for k, v in data.items() if k.startswith("_agent")}


def _import_jax():
    """Lazy-import JAX (only for Layer 2)."""
    import jax
    import jax.numpy as jnp
    return jax, jnp


def _compute_jacobian(ppo_state, obs_vec):
    """Gradient of summed action logits w.r.t. obs (discrete action proxy).
    Ported from cbam_region_analysis.py.
    """
    jax, jnp = _import_jax()

    def _forward(o):
        dist = ppo_state.actor(o)
        flat_logits, _ = jax.flatten_util.ravel_pytree(dist.logits)
        return jnp.sum(flat_logits)

    norm_obs = ppo_state.normalizer.normalize_obs(obs_vec)
    grad = jax.grad(_forward)(norm_obs)
    return np.array(grad)


def _first_layer_weight_norms(ppo_state):
    """Column norms of first MLP weight matrix.
    Ported from cbam_region_analysis.py.
    """
    actor = ppo_state.actor
    mlp = actor.mlp
    layers = mlp.layers if hasattr(mlp, "layers") else [mlp]
    w0 = None
    for layer in layers:
        if hasattr(layer, "weight"):
            w0 = np.array(layer.weight)
            break
    if w0 is None:
        for attr in ["layers", "net", "linear"]:
            sub = getattr(actor.obs_processor, attr, None)
            if sub is not None:
                for l in (sub if hasattr(sub, "__iter__") else [sub]):
                    if hasattr(l, "weight"):
                        w0 = np.array(l.weight)
                        break
    if w0 is None:
        return np.zeros(OBS_DIM)
    return np.linalg.norm(w0, axis=0)


def _group_agg(vec):
    """Aggregate a per-obs-dim vector to group-level means."""
    return {name: float(np.mean([vec[i] for i in idxs]))
            for name, idxs in OBS_GROUPS}


def _collect_obs_from_agent(agent, env_builder_fn):
    """Do a quick rollout to collect obs vectors for Jacobian analysis.
    Returns array of shape (T, obs_dim).
    """
    jax, jnp = _import_jax()
    from dataclasses import replace as dc_replace
    from rice_jax.utils import full_state_info_log_fn

    raw_env = env_builder_fn(for_training=False)
    eval_env = dc_replace(raw_env, log_info_fn=full_state_info_log_fn)

    key = jax.random.PRNGKey(999)
    obs_dict, env_state = eval_env.reset(key)

    obs_list = []
    for t in range(20):  # RICE episode length
        # Collect obs for one CBAM-relevant region
        for r in CBAM_PLOT_REGIONS:
            agent_key = f"region-{r:02d}"
            if agent_key in obs_dict:
                obs_list.append(np.array(obs_dict[agent_key].observation))

        act_key, step_key = jax.random.split(key)
        key = step_key
        actions = agent.get_action(act_key, agent.state, obs_dict, deterministic=True)
        (obs_dict, *_), env_state = eval_env.step(step_key, env_state, actions)

    return np.stack(obs_list, axis=0) if obs_list else np.zeros((1, OBS_DIM))


def _counterfactual_perturbation(ppo_state, obs_vec):
    """Zero out CBAM obs dims, compare action distribution change.
    Returns the L1 distance between original and perturbed action logits.
    """
    jax, jnp = _import_jax()

    cbam_dims = []
    for name, idxs in OBS_GROUPS:
        if name in ("cbam_cost", "cbam_lambda", "cbam_tariff_rate", "cbam_revenue"):
            cbam_dims.extend(idxs)

    def _get_logits(o):
        norm = ppo_state.normalizer.normalize_obs(o)
        dist = ppo_state.actor(norm)
        flat, _ = jax.flatten_util.ravel_pytree(dist.logits)
        return flat

    logits_orig = np.array(_get_logits(obs_vec))
    obs_pert = obs_vec.copy()
    obs_pert[cbam_dims] = 0.0
    logits_pert = np.array(_get_logits(obs_pert))

    return float(np.abs(logits_orig - logits_pert).mean())


def run_introspection(mech, cond, triage, out_dir, ts):
    """Layer 2: run Jacobian/weight/counterfactual analysis on saved agents."""
    print("\n" + "=" * 70)
    print("  LAYER 2: INTROSPECTION")
    print("=" * 70)

    # Check which tests have agents
    agents_found = {}
    for label, suite in [("mech", mech), ("cond", cond)]:
        if not suite:
            continue
        for test_id, res in suite.items():
            d = res.get("data", {})
            if _has_agents(d):
                agents_found[test_id] = _get_agents(d)

    if not agents_found:
        print("  No saved agents found in pkls. Re-run training with --save-agents.")
        return {}

    print(f"  Agents available for: {', '.join(sorted(agents_found.keys()))}")

    # For each agent state dict, pick a representative non-EU region
    # and compute introspection metrics on its PPOState.

    introspection_results = {}

    for test_id, agent_dict in sorted(agents_found.items()):
        for agent_label, state_dict in agent_dict.items():
            # state_dict is {"region-XX": PPOState, ...} (multi-agent)
            if not isinstance(state_dict, dict):
                print(f"    Skipping {test_id}/{agent_label}: unexpected type {type(state_dict)}")
                continue

            # Analyse each non-EU region
            for region_key, ppo_state in sorted(state_dict.items()):
                region_idx = int(region_key.split("-")[1])
                if region_idx == EU_IDX or region_idx == 0:  # skip EU and RoW
                    continue

                tag = f"{test_id}/{agent_label}/r{region_idx}"
                print(f"\n  ── {tag} ({REGION_NAMES.get(region_idx, region_key)}) ──")

                # Weight norms
                wnorms = _first_layer_weight_norms(ppo_state)
                wnorms_grouped = _group_agg(wnorms)
                print(f"    Weight norms (top-3 groups):")
                sorted_groups = sorted(wnorms_grouped.items(), key=lambda x: -x[1])
                for name, val in sorted_groups[:3]:
                    print(f"      {name:<20s} {val:.4f}")

                # Use normalizer running mean as representative obs
                obs_mean = None
                if hasattr(ppo_state, 'normalizer') and ppo_state.normalizer is not None:
                    norm_obs = getattr(ppo_state.normalizer, 'obs', None)
                    if norm_obs is not None and hasattr(norm_obs, 'mean'):
                        obs_mean = np.array(norm_obs.mean)

                if obs_mean is not None and obs_mean.shape == (OBS_DIM,):
                    # Jacobian at mean obs
                    jac = _compute_jacobian(ppo_state, obs_mean)
                    jac_grouped = _group_agg(np.abs(jac))
                    print(f"    Jacobian |∂logits/∂obs| (top-3 groups):")
                    sorted_jac = sorted(jac_grouped.items(), key=lambda x: -x[1])
                    for name, val in sorted_jac[:3]:
                        print(f"      {name:<20s} {val:.4f}")

                    # CBAM-specific: which CBAM obs dims matter?
                    cbam_groups = [g for g in sorted_jac
                                   if g[0] in ("cbam_cost", "cbam_lambda",
                                               "cbam_tariff_rate", "cbam_revenue")]
                    if cbam_groups:
                        print(f"    CBAM obs sensitivity:")
                        for name, val in cbam_groups:
                            print(f"      {name:<20s} {val:.4f}")

                    # Counterfactual
                    cf_delta = _counterfactual_perturbation(ppo_state, obs_mean)
                    print(f"    Counterfactual (zero CBAM obs): L1 logit shift = {cf_delta:.4f}")
                else:
                    jac_grouped = {}
                    cf_delta = float("nan")
                    shape_str = obs_mean.shape if obs_mean is not None else "None"
                    print(f"    Skipping Jacobian: obs_mean shape={shape_str}, expected ({OBS_DIM},)")

                introspection_results[tag] = {
                    "weight_norms": wnorms_grouped,
                    "jacobian": jac_grouped,
                    "counterfactual_delta": cf_delta,
                }

    # Plot: weight norms + Jacobian heatmap for all agents
    _plot_introspection(introspection_results, out_dir, ts)

    return introspection_results


def _plot_introspection(results, out_dir, ts):
    """Plot Layer 2 introspection results."""
    if not results:
        return

    group_names = [name for name, _ in OBS_GROUPS]
    n_agents = len(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.5 * n_agents)))

    # Weight norms heatmap
    ax = axes[0]
    wn_matrix = np.zeros((n_agents, len(group_names)))
    agent_labels = sorted(results.keys())
    for i, tag in enumerate(agent_labels):
        wn = results[tag]["weight_norms"]
        for j, gn in enumerate(group_names):
            wn_matrix[i, j] = wn.get(gn, 0)
    if wn_matrix.max() > 0:
        wn_matrix = wn_matrix / wn_matrix.max()
    im = ax.imshow(wn_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agent_labels, fontsize=7)
    ax.set_title("First-Layer Weight Norms (normalised)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Jacobian heatmap
    ax = axes[1]
    jac_matrix = np.zeros((n_agents, len(group_names)))
    for i, tag in enumerate(agent_labels):
        jac = results[tag]["jacobian"]
        for j, gn in enumerate(group_names):
            jac_matrix[i, j] = jac.get(gn, 0)
    if jac_matrix.max() > 0:
        jac_matrix = jac_matrix / jac_matrix.max()
    im = ax.imshow(jac_matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n_agents))
    ax.set_yticklabels(agent_labels, fontsize=7)
    ax.set_title("|Jacobian| at Obs Mean (normalised)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"posthoc_introspection_{ts}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Introspection heatmaps → {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Layer 1 post-hoc analysis for CBAM litmus pkls")
    parser.add_argument("--mech-pkl", help="Path to mechanism pkl")
    parser.add_argument("--cond-pkl", help="Path to conditioning pkl")
    parser.add_argument("--out-dir", default=get_output_dir("plots"),
                        help="Output directory for plots (default: plots or CBAM_EXPERIMENT_DIR/posthoc)")
    parser.add_argument("--markdown", action="store_true",
                        help="Print report-ready markdown to stdout")
    parser.add_argument("--out-report", default=None,
                        help="Write scorecard markdown to this file")
    parser.add_argument("--introspect", action="store_true",
                        help="Run Layer 2 introspection (requires JAX and --save-agents pkls)")
    args = parser.parse_args()

    if not args.mech_pkl and not args.cond_pkl:
        parser.error("Provide at least one of --mech-pkl or --cond-pkl")

    mech = None
    cond = None

    if args.mech_pkl:
        with open(args.mech_pkl, "rb") as f:
            mech = pickle.load(f)
        print(f"  Loaded mechanism pkl: {args.mech_pkl}")

    if args.cond_pkl:
        with open(args.cond_pkl, "rb") as f:
            cond = pickle.load(f)
        print(f"  Loaded conditioning pkl: {args.cond_pkl}")

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Pass table
    rows = build_pass_table(mech, cond)
    print_pass_table(rows)

    # 2. Cross-suite consistency
    checks = cross_suite_consistency(mech, cond)
    print_consistency(checks)

    # 3. Anomaly flags
    flags = flag_anomalies(mech, cond)
    print_anomalies(flags)

    # 4. Per-region decomposition
    plot_per_region(mech, cond, os.path.join(args.out_dir, f"posthoc_regions_{ts}.png"))

    # 5. Training curves
    plot_training_curves(mech, cond, os.path.join(args.out_dir, f"posthoc_curves_{ts}.png"))

    # 6. Triage
    triage = build_triage(rows, mech, cond)
    print_triage(triage)

    # 6b. Layer 2 introspection (optional)
    introspection = {}
    if args.introspect:
        introspection = run_introspection(mech, cond, triage, args.out_dir, ts)

    # 7. Markdown
    md = generate_markdown(rows, checks, flags, mech, cond, introspection)

    if args.out_report:
        report_path = args.out_report
        if report_path == "auto":
            report_path = os.path.join(args.out_dir, f"posthoc_scorecard_{ts}.md")
        with open(report_path, "w") as f:
            f.write(md)
        print(f"  Scorecard written → {report_path}")

    if args.markdown:
        print("\n" + "=" * 60)
        print("  MARKDOWN OUTPUT")
        print("=" * 60)
        print(md)

    print("\n  Done.")


if __name__ == "__main__":
    main()
