"""
mrio_trade_report.py
--------------------
Generates a visual report of initial trade structure from aggregated EORA26 MRIO
data, for any of the 3/7/20 region setups.

Plots produced (saved to csv_asset/mrio/reports/{N}_regions/):
  1. export_ratio.png        — each region's total exports / total output
  2. import_shares.png       — heatmap: % of each region's imports sourced from each partner
  3. domestic_foreign_ratio.png — domestic vs. foreign share of final consumption per region
  4. trade_flow_chord.png    — region-to-region trade flow magnitudes (log scale)

Run:
    python csv_asset/mrio_trade_report.py --regions 20
    python csv_asset/mrio_trade_report.py --regions 7
    python csv_asset/mrio_trade_report.py --regions 3
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_ASSET_DIR = "csv_asset"


def get_paths(n: int):
    agg_dir = os.path.join(CSV_ASSET_DIR, "mrio", "aggregated", f"eora_agg_{n}")
    cc_csv = os.path.join(CSV_ASSET_DIR, f"CountryClass_{n}.csv")
    out_dir = os.path.join(CSV_ASSET_DIR, "mrio", "reports", f"{n}_regions")
    os.makedirs(out_dir, exist_ok=True)
    return agg_dir, cc_csv, out_dir


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mrio(agg_dir: str):
    """
    Returns
    -------
    Z : pd.DataFrame (N_reg*N_sec, N_reg*N_sec)  intermediate flow matrix
    Y : pd.DataFrame (N_reg*N_sec, N_reg*N_fd)   final demand matrix
    x : pd.Series   (N_reg*N_sec,)               total output vector
    regions : list[str]                           ordered region names from MRIO
    """
    Z = pd.read_parquet(os.path.join(agg_dir, "Z.parquet"))
    Y = pd.read_parquet(os.path.join(agg_dir, "Y.parquet"))
    x_raw = pd.read_csv(
        os.path.join(agg_dir, "x.txt"), sep="\t", index_col=[0, 1]
    )
    x_raw.index.names = ["region", "sector"]
    x_raw.columns = ["output"]
    x = x_raw["output"]

    regions = Z.index.get_level_values("region").unique().tolist()
    return Z, Y, x, regions


def build_rice_labels(n: int, cc_csv: str, mrio_regions: list[str]) -> dict[str, str]:
    """
    Map MRIO region label → short RICE-style label (RIG integer).
    Returns {mrio_label: "Region N (short_name)"}.
    """
    df = pd.read_csv(cc_csv)
    df["RI"] = df["RI"].str.strip()
    rig_label = (
        df.dropna(subset=["RIG"])
        .groupby("RIG")["RI"]
        .first()
        .sort_index()
        .to_dict()
    )
    rig_label = {int(k): v for k, v in rig_label.items()}

    label_map = {}
    for rig_idx in range(1, n + 1):
        full_label = rig_label.get(rig_idx, "Rest of World")
        if full_label in mrio_regions:
            label_map[full_label] = f"R{rig_idx}: {full_label}"
        else:
            label_map["Rest of World"] = f"R{rig_idx}: Rest of World"

    return label_map


# ---------------------------------------------------------------------------
# Trade aggregation helpers
# ---------------------------------------------------------------------------

def region_to_region_flows(Z: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate all sector-level flows into a region × region flow matrix.
    Entry (r_from, r_to) = sum of intermediate + final demand flows from r_from to r_to.
    """
    regions_row = Z.index.get_level_values("region").unique().tolist()
    regions_col = Z.columns.get_level_values("region").unique().tolist()
    all_regions = sorted(set(regions_row) | set(regions_col))

    n = len(all_regions)
    flow_matrix = pd.DataFrame(0.0, index=all_regions, columns=all_regions)

    # Intermediate flows Z
    for r_from in all_regions:
        if r_from not in regions_row:
            continue
        z_from = Z.loc[r_from]  # (n_sec, n_reg*n_sec)
        for r_to in all_regions:
            if r_to not in regions_col:
                continue
            flow_matrix.loc[r_from, r_to] += z_from[r_to].values.sum()

    # Final demand flows Y
    y_regions_col = Y.columns.get_level_values("region").unique().tolist()
    for r_from in all_regions:
        if r_from not in regions_row:
            continue
        y_from = Y.loc[r_from]  # (n_sec, n_reg*n_fd_cat)
        for r_to in all_regions:
            if r_to not in y_regions_col:
                continue
            flow_matrix.loc[r_from, r_to] += y_from[r_to].values.sum()

    return flow_matrix


def total_output_by_region(x: pd.Series) -> pd.Series:
    """Sum total output over all sectors for each region."""
    return x.groupby(level="region").sum()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

LABEL_WRAP = 22  # max chars before wrapping long region names

def wrap_label(s: str, maxlen: int = LABEL_WRAP) -> str:
    if len(s) <= maxlen:
        return s
    # break at space nearest to middle
    mid = len(s) // 2
    left = s.rfind(" ", 0, mid)
    right = s.find(" ", mid)
    if left == -1 and right == -1:
        return s
    split = left if (right == -1 or (left != -1 and mid - left <= right - mid)) else right
    return s[:split] + "\n" + s[split + 1:]


def short_labels(regions: list[str]) -> list[str]:
    return [wrap_label(r) for r in regions]


# 1 ─ Export ratio ─────────────────────────────────────────────────────────

def plot_export_ratio(flow_matrix: pd.DataFrame, output_by_region: pd.Series,
                      regions: list[str], out_dir: str) -> None:
    exports = []
    for r in regions:
        total_out = float(output_by_region.get(r, np.nan))
        # exports = total outflow to all regions except self
        outflow_to_others = flow_matrix.loc[r].drop(labels=[r], errors="ignore").sum()
        exports.append(outflow_to_others / total_out if total_out > 0 else 0.0)

    labels = short_labels(regions)
    x_pos = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(max(10, len(regions) * 0.65), 5))
    bars = ax.bar(x_pos, exports, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Exports / Total Output")
    ax.set_title("Export Ratio by Region (EORA26 2016)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, min(1.0, max(exports) * 1.25 + 0.05))
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, exports):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.1%}",
            ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    path = os.path.join(out_dir, "export_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 2 ─ Import shares heatmap ────────────────────────────────────────────────

def plot_import_shares(flow_matrix: pd.DataFrame, regions: list[str],
                       out_dir: str) -> None:
    """
    Heatmap rows = importing region, cols = exporting region.
    Cell value = % of importer's total imports sourced from exporter.
    Diagonal (self-trade) excluded.
    """
    n = len(regions)
    import_share = np.zeros((n, n))
    for i, r_import in enumerate(regions):
        imports_from = np.array([flow_matrix.loc[r_export, r_import] for r_export in regions])
        imports_from[i] = 0.0  # zero out self
        total_imports = imports_from.sum()
        if total_imports > 0:
            import_share[i] = imports_from / total_imports

    labels = short_labels(regions)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.55 + 2), max(7, n * 0.55 + 1.5)))
    im = ax.imshow(import_share * 100, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Exporting Region")
    ax.set_ylabel("Importing Region")
    ax.set_title("Import Source Share (% of total imports, diagonal = self-trade excl.)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("% of imports")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = import_share[i, j] * 100
            if val > 1.0:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=6, color="black" if val < 50 else "white")
    fig.tight_layout()
    path = os.path.join(out_dir, "import_shares.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 3 ─ Domestic vs Foreign consumption ratio ───────────────────────────────

def plot_domestic_foreign(Y: pd.DataFrame, regions: list[str], out_dir: str) -> None:
    """
    For each region r:
      domestic consumption = sum of Y[producers=r, consumers=r]
      foreign consumption  = sum of Y[producers=r, consumers≠r]
    Bar chart: stacked domestic / foreign, normalised to 100%.
    """
    dom_share = []
    for_share = []

    y_col_regions = Y.columns.get_level_values("region").unique().tolist()

    for r in regions:
        if r not in Y.index.get_level_values("region"):
            dom_share.append(np.nan)
            for_share.append(np.nan)
            continue
        y_r = Y.loc[r]  # rows = sectors of r; cols = (consumer_region, fd_cat)
        total_fd = 0.0
        dom_fd = 0.0
        for r_consumer in y_col_regions:
            if r_consumer not in y_r.columns.get_level_values("region"):
                continue
            val = float(y_r[r_consumer].values.sum())
            total_fd += val
            if r_consumer == r:
                dom_fd += val

        if total_fd > 0:
            dom_share.append(dom_fd / total_fd)
            for_share.append((total_fd - dom_fd) / total_fd)
        else:
            dom_share.append(0.0)
            for_share.append(0.0)

    labels = short_labels(regions)
    x_pos = np.arange(len(regions))

    fig, ax = plt.subplots(figsize=(max(10, len(regions) * 0.65), 5))
    ax.bar(x_pos, dom_share, label="Domestic", color="#2166ac", edgecolor="white")
    ax.bar(x_pos, for_share, bottom=dom_share, label="Foreign", color="#d6604d", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Share of Final Demand")
    ax.set_title("Domestic vs. Foreign Final Consumption Share by Region")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    for i, (d, f) in enumerate(zip(dom_share, for_share)):
        if not np.isnan(d):
            ax.text(i, d / 2, f"{d:.0%}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
            if f > 0.04:
                ax.text(i, d + f / 2, f"{f:.0%}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "domestic_foreign_ratio.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 4 ─ Region-to-region trade flow heatmap (log scale) ─────────────────────

def plot_trade_flows(flow_matrix: pd.DataFrame, regions: list[str],
                     out_dir: str) -> None:
    n = len(regions)
    values = np.array(
        [[flow_matrix.loc[r_from, r_to] for r_to in regions] for r_from in regions],
        dtype=float,
    )
    # Log scale; zero → NaN for display
    log_values = np.where(values > 0, np.log10(values + 1), np.nan)

    labels = short_labels(regions)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.55 + 2), max(7, n * 0.55 + 1.5)))
    im = ax.imshow(log_values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Destination Region")
    ax.set_ylabel("Source Region")
    ax.set_title("Trade Flows: Source → Destination (log₁₀ scale, USD million)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("log₁₀(flow + 1)")
    fig.tight_layout()
    path = os.path.join(out_dir, "trade_flow_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# 5 ─ Summary CSV ──────────────────────────────────────────────────────────

def write_summary_csv(flow_matrix: pd.DataFrame, output_by_region: pd.Series,
                      Y: pd.DataFrame, regions: list[str], out_dir: str) -> None:
    rows = []
    y_col_regions = Y.columns.get_level_values("region").unique().tolist()

    for r in regions:
        total_out = float(output_by_region.get(r, 0))
        outflow = flow_matrix.loc[r].drop(labels=[r], errors="ignore").sum()
        export_ratio = outflow / total_out if total_out > 0 else 0.0

        total_imports = sum(
            flow_matrix.loc[r2, r] for r2 in regions if r2 != r
        )

        dom_fd, total_fd = 0.0, 0.0
        if r in Y.index.get_level_values("region"):
            y_r = Y.loc[r]
            for r_con in y_col_regions:
                if r_con not in y_r.columns.get_level_values("region"):
                    continue
                val = float(y_r[r_con].values.sum())
                total_fd += val
                if r_con == r:
                    dom_fd += val
        dom_share = dom_fd / total_fd if total_fd > 0 else 0.0

        row = {
            "region": r,
            "total_output_USD_mn": round(total_out / 1e6, 2),
            "total_exports_USD_mn": round(outflow / 1e6, 2),
            "export_ratio": round(export_ratio, 4),
            "total_imports_USD_mn": round(total_imports / 1e6, 2),
            "domestic_fd_share": round(dom_share, 4),
            "foreign_fd_share": round(1 - dom_share, 4),
        }
        # Import shares from each partner
        for r2 in regions:
            inflow = flow_matrix.loc[r2, r] if r2 != r else 0.0
            share = inflow / total_imports if total_imports > 0 else 0.0
            row[f"import_from_{r2}_pct"] = round(share * 100, 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "trade_summary.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n: int) -> None:
    print(f"\n{'='*60}")
    print(f"  Generating MRIO trade report for {n}-region setup")
    print(f"{'='*60}")

    agg_dir, cc_csv, out_dir = get_paths(n)
    print(f"  MRIO data : {agg_dir}")
    print(f"  Output    : {out_dir}")

    Z, Y, x, mrio_regions = load_mrio(agg_dir)
    print(f"  Loaded {len(mrio_regions)} regions, {len(Z)} sector-rows")

    print("  Building region-to-region flow matrix …")
    flow_matrix = region_to_region_flows(Z, Y)
    output_by_region = total_output_by_region(x)

    # Only plot regions that actually exist in the MRIO aggregation
    regions = [r for r in mrio_regions if r in flow_matrix.index]
    print(f"  Plotting {len(regions)} regions: {regions[:3]} …")

    print("  1/5 Export ratio …")
    plot_export_ratio(flow_matrix, output_by_region, regions, out_dir)

    print("  2/5 Import shares heatmap …")
    plot_import_shares(flow_matrix, regions, out_dir)

    print("  3/5 Domestic vs foreign consumption …")
    plot_domestic_foreign(Y, regions, out_dir)

    print("  4/5 Trade flow heatmap …")
    plot_trade_flows(flow_matrix, regions, out_dir)

    print("  5/5 Summary CSV …")
    write_summary_csv(flow_matrix, output_by_region, Y, regions, out_dir)

    print(f"\nDone. All outputs in: {out_dir}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MRIO trade structure report.")
    parser.add_argument(
        "--regions", type=int, choices=[3, 7, 20], default=20,
        help="Number of regions (3, 7, or 20). Default: 20.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate reports for all three region setups (3, 7, 20).",
    )
    args = parser.parse_args()

    if args.all:
        for n in [3, 7, 20]:
            main(n)
    else:
        main(args.regions)
