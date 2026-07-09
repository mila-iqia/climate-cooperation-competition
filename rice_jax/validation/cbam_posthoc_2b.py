"""cbam_posthoc_2b.py — Post-hoc scorecard for Phase 2B experiments.

Reads tier1 and/or alloc pkl(s), produces:
  1. Per-region bar charts with region names (dirty share, μ, transfer)
  2. Condition-comparison deltas (what changed between conditions?)
  3. Convergence overlay from training CSVs
  4. Report-ready markdown scorecard
  5. Cross-condition ranking tables

Works on:
  - cbam_experiment_2b_tier1.py pkls (list of dicts, keyed by revenue_share)
  - cbam_experiment_2b_alloc.py pkls (list of dicts, keyed by allocation)

Usage (from rice_jax/):
    python validation/cbam_posthoc_2b.py --tier1-pkl plots/cbam_2b_tier1_*.pkl
    python validation/cbam_posthoc_2b.py --alloc-pkl plots/cbam_2b_alloc_*.pkl
    python validation/cbam_posthoc_2b.py --tier1-pkl <t1.pkl> --alloc-pkl <a.pkl> --out-report auto
"""

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
    4: "SSA", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
NON_EU            = [r for r in range(NUM_REGIONS) if r != EU_IDX]
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]  # drop RoW


# ── Helpers ─────────────────────────────────────────────────────────────────

def _fmt(v, precision=3):
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _read_csv(csv_path):
    if not csv_path:
        return pd.DataFrame()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rice_jax_dir = os.path.dirname(script_dir)
    full = os.path.join(rice_jax_dir, csv_path) if not os.path.isabs(csv_path) else csv_path
    if not os.path.exists(full):
        return pd.DataFrame()
    return pd.read_csv(full)


def _region_label(r):
    return REGION_NAMES.get(r, f"Region {r}")


def _region_short(r):
    return REGION_SHORT.get(r, f"R{r}")


# ── Tier1 Analysis ──────────────────────────────────────────────────────────

def _detect_condition_key(results):
    """Detect whether this is a tier1 pkl or alloc pkl."""
    if not results:
        return None
    r0 = results[0]
    if "revenue_share" in r0:
        return "tier1"
    if "allocation" in r0:
        return "alloc"
    return None


def _condition_label(res, kind):
    """Human-readable condition label."""
    if kind == "tier1":
        rs = res["revenue_share"]
        tm = res.get("transfer_mode", "consumption")
        return f"rs={rs:.2f} [{tm[:4]}]"
    elif kind == "alloc":
        return res["allocation"]
    return res.get("label", "?")


def _condition_sort_key(res, kind):
    if kind == "tier1":
        return (res.get("transfer_mode", ""), res["revenue_share"])
    elif kind == "alloc":
        order = {"burden": 0, "effort": 1, "vulnerability": 2, "equal": 3}
        return order.get(res.get("allocation", ""), 99)
    return 0


# ── Pass Table ──────────────────────────────────────────────────────────────

def build_pass_table_tier1(results):
    """Build a tier1 scorecard table."""
    rows = []
    # Sort by revenue_share
    results = sorted(results, key=lambda r: _condition_sort_key(r, "tier1"))
    baseline = next((r for r in results if r["revenue_share"] == 0.0), results[0])

    for res in results:
        if res is baseline:
            continue
        rs = res["revenue_share"]
        tm = res.get("transfer_mode", "consumption")
        # Compute per-region deltas vs baseline
        deltas = {r: res["dirty_share"][r] - baseline["dirty_share"][r]
                  for r in CBAM_PLOT_REGIONS}
        mean_delta = np.mean(list(deltas.values()))
        n_improved = sum(1 for d in deltas.values() if d > 0)  # higher = less diversion

        # Mitigation delta
        mu_deltas = {r: res["mit_rate"][r] - baseline["mit_rate"][r]
                     for r in CBAM_PLOT_REGIONS}
        mean_mu_delta = np.mean(list(mu_deltas.values()))

        passed = n_improved > 0
        rows.append(dict(
            condition=f"rs={rs:.2f} [{tm}]",
            mean_dirty_delta=mean_delta,
            n_improved=n_improved,
            mean_mu_delta=mean_mu_delta,
            passed=passed,
            details={r: deltas[r] for r in CBAM_PLOT_REGIONS},
        ))
    return rows


def build_pass_table_alloc(results):
    """Build an alloc scorecard table."""
    rows = []
    results = sorted(results, key=lambda r: _condition_sort_key(r, "alloc"))
    # Use burden as the reference
    burden_res = next((r for r in results if r["allocation"] == "burden"), results[0])

    for res in results:
        rule = res["allocation"]
        deltas = {r: res["dirty_share"][r] - burden_res["dirty_share"][r]
                  for r in CBAM_PLOT_REGIONS}
        mu_deltas = {r: res["mit_rate"][r] - burden_res["mit_rate"][r]
                     for r in CBAM_PLOT_REGIONS}
        mean_delta = np.mean(list(deltas.values()))
        mean_mu_delta = np.mean(list(mu_deltas.values()))

        rows.append(dict(
            condition=rule,
            mean_dirty_delta=mean_delta,
            mean_mu_delta=mean_mu_delta,
            n_improved_vs_burden=sum(1 for d in deltas.values() if d > 0),
            details={r: (deltas[r], mu_deltas[r]) for r in CBAM_PLOT_REGIONS},
        ))
    return rows


def print_pass_table(rows, kind):
    print(f"\n{'='*90}")
    title = "TIER 1 REVENUE-SHARE SCORECARD" if kind == "tier1" else "ALLOCATION RULE SCORECARD"
    print(f"  {title}")
    print(f"{'='*90}")

    if kind == "tier1":
        header = f"{'Condition':<24} {'Δ dirty (mean)':<16} {'#improved':<12} {'Δ μ (mean)':<14} {'Pass?'}"
        print(header)
        print("-" * len(header))
        for r in rows:
            p = "PASS" if r["passed"] else "FAIL"
            print(f"{r['condition']:<24} {r['mean_dirty_delta']:>+.4f}{'':8} "
                  f"{r['n_improved']:>4}/7{'':6} {r['mean_mu_delta']:>+.4f}{'':8} {p}")
    else:
        header = f"{'Rule':<18} {'Δ dirty vs burden':<20} {'Δ μ vs burden':<16} {'# better'}"
        print(header)
        print("-" * len(header))
        for r in rows:
            print(f"{r['condition']:<18} {r['mean_dirty_delta']:>+.4f}{'':14} "
                  f"{r['mean_mu_delta']:>+.4f}{'':10} {r['n_improved_vs_burden']}/7")
    print()


# ── Per-Region Decomposition ───────────────────────────────────────────────

def plot_per_region(results, kind, out_path):
    """Per-region grouped bar charts with full region names."""
    results = sorted(results, key=lambda r: _condition_sort_key(r, kind))
    n_conds = len(results)
    labels = [_region_short(r) for r in CBAM_PLOT_REGIONS]
    x = np.arange(len(CBAM_PLOT_REGIONS))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    width = 0.8 / n_conds
    colors = plt.cm.tab10(np.linspace(0, 1, n_conds))

    # Panel 1: EU dirty export share
    ax = axes[0]
    for i, res in enumerate(results):
        vals = [res["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        lbl = _condition_label(res, kind)
        ax.bar(x + (i - n_conds/2 + 0.5) * width, vals, width,
               color=colors[i], alpha=0.85, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("EU dirty export share")
    ax.set_title("EU Dirty Export Share by Region", fontsize=10)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    # Panel 2: Mitigation rate
    ax = axes[1]
    for i, res in enumerate(results):
        vals = [res["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        lbl = _condition_label(res, kind)
        ax.bar(x + (i - n_conds/2 + 0.5) * width, vals, width,
               color=colors[i], alpha=0.85, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean mitigation rate μ")
    ax.set_title("Mitigation Rate by Region", fontsize=10)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    # Panel 3: Transfer received
    ax = axes[2]
    for i, res in enumerate(results):
        vals = [res["transfer_mean"][r] for r in CBAM_PLOT_REGIONS]
        lbl = _condition_label(res, kind)
        ax.bar(x + (i - n_conds/2 + 0.5) * width, vals, width,
               color=colors[i], alpha=0.85, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Transfer received (per step)")
    ax.set_title("Transfer Received by Region", fontsize=10)
    ax.legend(fontsize=6, loc="upper right")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-region decomposition → {out_path}")


# ── Delta Plot ──────────────────────────────────────────────────────────────

def plot_deltas(results, kind, out_path):
    """Bar chart showing deltas vs baseline per region."""
    results = sorted(results, key=lambda r: _condition_sort_key(r, kind))
    labels = [_region_short(r) for r in CBAM_PLOT_REGIONS]
    x = np.arange(len(CBAM_PLOT_REGIONS))

    # Choose baseline
    if kind == "tier1":
        baseline = next((r for r in results if r["revenue_share"] == 0.0), results[0])
        comparisons = [r for r in results if r is not baseline]
        title_suffix = "vs no-transfer baseline"
    else:
        baseline = next((r for r in results if r.get("allocation") == "burden"), results[0])
        comparisons = [r for r in results if r is not baseline]
        title_suffix = "vs burden allocation"

    if not comparisons:
        return

    n_conds = len(comparisons)
    width = 0.8 / n_conds
    colors = plt.cm.tab10(np.linspace(0, 1, n_conds))

    fig, (ax_dirty, ax_mu) = plt.subplots(1, 2, figsize=(12, 4.5))

    for i, res in enumerate(comparisons):
        lbl = _condition_label(res, kind)
        # Dirty share delta
        d_dirty = [res["dirty_share"][r] - baseline["dirty_share"][r] for r in CBAM_PLOT_REGIONS]
        ax_dirty.bar(x + (i - n_conds/2 + 0.5) * width, d_dirty, width,
                     color=colors[i], alpha=0.85, label=lbl)
        # μ delta
        d_mu = [res["mit_rate"][r] - baseline["mit_rate"][r] for r in CBAM_PLOT_REGIONS]
        ax_mu.bar(x + (i - n_conds/2 + 0.5) * width, d_mu, width,
                  color=colors[i], alpha=0.85, label=lbl)

    for ax, metric in [(ax_dirty, "Δ EU dirty share"), (ax_mu, "Δ mitigation rate μ")]:
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} {title_suffix}", fontsize=10)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Delta plot → {out_path}")


# ── Training Curves ─────────────────────────────────────────────────────────

def plot_training_curves(results, kind, out_path):
    """Overlay ep_return_mean from all training CSVs."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for i, res in enumerate(sorted(results, key=lambda r: _condition_sort_key(r, kind))):
        csv_path = res.get("csv_path")
        df = _read_csv(csv_path)
        if df.empty or "ep_return_mean" not in df.columns:
            continue
        y = df["ep_return_mean"].rolling(10).mean().values
        lbl = _condition_label(res, kind)
        ax.plot(y, color=colors[i], linewidth=1.2, alpha=0.85, label=lbl)

    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("Episode return (rolling-10 mean)")
    kind_title = "Tier 1 Revenue-Share" if kind == "tier1" else "Allocation Rule"
    ax.set_title(f"Training Convergence — {kind_title} Comparison")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves → {out_path}")


# ── Ranking Table ───────────────────────────────────────────────────────────

def print_ranking(results, kind):
    """Print per-region ranking of conditions by mitigation and diversion."""
    results = sorted(results, key=lambda r: _condition_sort_key(r, kind))

    print(f"\n{'='*80}")
    print("  PER-REGION RANKING")
    print(f"{'='*80}")

    # Rank by μ (descending = best)
    print("\n  By mitigation rate μ (highest first):")
    for r in CBAM_PLOT_REGIONS:
        ranked = sorted(results, key=lambda res: -res["mit_rate"][r])
        ranking_str = " > ".join(
            f"{_condition_label(res, kind)}({res['mit_rate'][r]:.3f})"
            for res in ranked
        )
        print(f"    {_region_label(r):18s}: {ranking_str}")

    # Rank by dirty share (descending = less diversion)
    print("\n  By EU dirty share (highest = least diversion):")
    for r in CBAM_PLOT_REGIONS:
        ranked = sorted(results, key=lambda res: -res["dirty_share"][r])
        ranking_str = " > ".join(
            f"{_condition_label(res, kind)}({res['dirty_share'][r]:.3f})"
            for res in ranked
        )
        print(f"    {_region_label(r):18s}: {ranking_str}")
    print()


# ── Markdown Report ─────────────────────────────────────────────────────────

def generate_markdown(tier1_results, alloc_results, tier1_rows, alloc_rows):
    """Generate combined scorecard markdown."""
    lines = []
    lines.append("## Phase 2B Post-Hoc Scorecard (auto-generated)\n")

    if tier1_results:
        lines.append("### Tier 1: Revenue-Share Ablation\n")
        lines.append("| Condition | Δ dirty share (mean) | # regions improved | Δ μ (mean) | Pass? |")
        lines.append("|-----------|---------------------|-------------------|-----------|-------|")
        for r in tier1_rows:
            p = "PASS" if r["passed"] else "FAIL"
            lines.append(f"| {r['condition']} | {r['mean_dirty_delta']:+.4f} | "
                         f"{r['n_improved']}/7 | {r['mean_mu_delta']:+.4f} | {p} |")
        lines.append("")

        # Per-region detail table
        lines.append("#### Per-Region Detail (EU dirty share)\n")
        header = "| Condition | " + " | ".join(_region_short(r) for r in CBAM_PLOT_REGIONS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(CBAM_PLOT_REGIONS) + 1))
        for res in sorted(tier1_results, key=lambda r: _condition_sort_key(r, "tier1")):
            lbl = _condition_label(res, "tier1")
            vals = " | ".join(f"{res['dirty_share'][r]:.3f}" for r in CBAM_PLOT_REGIONS)
            lines.append(f"| {lbl} | {vals} |")
        lines.append("")

        lines.append("#### Per-Region Detail (mitigation μ)\n")
        header = "| Condition | " + " | ".join(_region_short(r) for r in CBAM_PLOT_REGIONS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(CBAM_PLOT_REGIONS) + 1))
        for res in sorted(tier1_results, key=lambda r: _condition_sort_key(r, "tier1")):
            lbl = _condition_label(res, "tier1")
            vals = " | ".join(f"{res['mit_rate'][r]:.3f}" for r in CBAM_PLOT_REGIONS)
            lines.append(f"| {lbl} | {vals} |")
        lines.append("")

    if alloc_results:
        lines.append("### Allocation Rule Comparison\n")
        lines.append("| Rule | Δ dirty vs burden | Δ μ vs burden | # better |")
        lines.append("|------|-------------------|---------------|----------|")
        for r in alloc_rows:
            lines.append(f"| {r['condition']} | {r['mean_dirty_delta']:+.4f} | "
                         f"{r['mean_mu_delta']:+.4f} | {r['n_improved_vs_burden']}/7 |")
        lines.append("")

        # Per-region detail
        lines.append("#### Per-Region Detail (mitigation μ)\n")
        header = "| Rule | " + " | ".join(_region_short(r) for r in CBAM_PLOT_REGIONS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(CBAM_PLOT_REGIONS) + 1))
        for res in sorted(alloc_results, key=lambda r: _condition_sort_key(r, "alloc")):
            lbl = res["allocation"]
            vals = " | ".join(f"{res['mit_rate'][r]:.3f}" for r in CBAM_PLOT_REGIONS)
            lines.append(f"| {lbl} | {vals} |")
        lines.append("")

    # Interpretation
    lines.append("### Interpretation\n")
    if tier1_results:
        baseline = next((r for r in tier1_results if r["revenue_share"] == 0.0), tier1_results[0])
        best = max((r for r in tier1_results if r["revenue_share"] > 0),
                   key=lambda r: np.mean([r["mit_rate"][i] for i in CBAM_PLOT_REGIONS]),
                   default=None)
        if best:
            mu_gain = np.mean([best["mit_rate"][r] - baseline["mit_rate"][r]
                               for r in CBAM_PLOT_REGIONS])
            lines.append(f"- **Best tier1 for mitigation**: rs={best['revenue_share']:.2f} "
                         f"[{best.get('transfer_mode','consumption')}] "
                         f"(+{mu_gain:.3f} mean μ vs baseline)")

    if alloc_results:
        best_alloc = max(alloc_results,
                         key=lambda r: np.mean([r["mit_rate"][i] for i in CBAM_PLOT_REGIONS]))
        lines.append(f"- **Best allocation for mitigation**: {best_alloc['allocation']} "
                     f"(mean μ = {np.mean([best_alloc['mit_rate'][r] for r in CBAM_PLOT_REGIONS]):.3f})")
    lines.append("")

    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc scorecard for Phase 2B experiments (tier1 + alloc)")
    parser.add_argument("--tier1-pkl", help="Path to tier1 pkl")
    parser.add_argument("--alloc-pkl", help="Path to alloc pkl")
    parser.add_argument("--out-dir", default=get_output_dir("plots"),
                        help="Output directory for plots")
    parser.add_argument("--out-report", default=None,
                        help="Write scorecard markdown to this file ('auto' for timestamped)")
    parser.add_argument("--markdown", action="store_true",
                        help="Print markdown to stdout")
    args = parser.parse_args()

    if not args.tier1_pkl and not args.alloc_pkl:
        parser.error("Provide at least one of --tier1-pkl or --alloc-pkl")

    tier1_results = None
    alloc_results = None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.tier1_pkl:
        with open(args.tier1_pkl, "rb") as f:
            tier1_results = pickle.load(f)
        print(f"  Loaded tier1 pkl: {args.tier1_pkl} ({len(tier1_results)} conditions)")

    if args.alloc_pkl:
        with open(args.alloc_pkl, "rb") as f:
            alloc_results = pickle.load(f)
        print(f"  Loaded alloc pkl: {args.alloc_pkl} ({len(alloc_results)} conditions)")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Tier 1 analysis ──────────────────────────────────────────────────────
    tier1_rows = []
    if tier1_results:
        tier1_rows = build_pass_table_tier1(tier1_results)
        print_pass_table(tier1_rows, "tier1")
        print_ranking(tier1_results, "tier1")
        plot_per_region(tier1_results, "tier1",
                        os.path.join(args.out_dir, f"posthoc_2b_tier1_regions_{ts}.png"))
        plot_deltas(tier1_results, "tier1",
                    os.path.join(args.out_dir, f"posthoc_2b_tier1_deltas_{ts}.png"))
        plot_training_curves(tier1_results, "tier1",
                             os.path.join(args.out_dir, f"posthoc_2b_tier1_curves_{ts}.png"))

    # ── Alloc analysis ───────────────────────────────────────────────────────
    alloc_rows = []
    if alloc_results:
        alloc_rows = build_pass_table_alloc(alloc_results)
        print_pass_table(alloc_rows, "alloc")
        print_ranking(alloc_results, "alloc")
        plot_per_region(alloc_results, "alloc",
                        os.path.join(args.out_dir, f"posthoc_2b_alloc_regions_{ts}.png"))
        plot_deltas(alloc_results, "alloc",
                    os.path.join(args.out_dir, f"posthoc_2b_alloc_deltas_{ts}.png"))
        plot_training_curves(alloc_results, "alloc",
                             os.path.join(args.out_dir, f"posthoc_2b_alloc_curves_{ts}.png"))

    # ── Markdown ─────────────────────────────────────────────────────────────
    md = generate_markdown(tier1_results, alloc_results, tier1_rows, alloc_rows)

    if args.out_report:
        report_path = args.out_report
        if report_path == "auto":
            report_path = os.path.join(args.out_dir, f"posthoc_2b_scorecard_{ts}.md")
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
