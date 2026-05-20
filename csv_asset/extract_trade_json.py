#!/usr/bin/env python3
"""Extract bilateral trade JSONs from aggregated MRIO parquet files.

Generates N_import_2016.json and N_export_2016.json in the same format as
the existing {3,7,20}_import_2016.json files, from the aggregated EORA26
parquet files produced by aggregate_local_mrio.py.

Usage:
    python extract_trade_json.py --n 9
    python extract_trade_json.py --n 9 --mrio-dir csv_asset/mrio/aggregated/eora_agg_9
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def load_x(mrio_dir: str) -> pd.Series:
    """Load total-output vector (x) from pymrio output."""
    x_path = os.path.join(mrio_dir, "x.txt")
    x = pd.read_csv(x_path, sep="\t", index_col=[0, 1])
    return x.iloc[:, 0]


def extract_trade(mrio_dir: str):
    """
    Compute bilateral trade shares and total export fractions
    from MRIO Z and Y matrices.

    Returns
    -------
    import_dict : dict[str, dict[str, float]]
        {importer_region_id: {exporter_region_id: import_intensity}}
        import_intensity = (flow from exporter to importer) / (output of importer)
    export_dict : dict[str, float]
        {region_id: total_export_fraction}
        total_export_fraction = sum(exports to all partners) / total_output
    regions : list[str]
        Ordered region labels from the MRIO data.
    """
    Z = pd.read_parquet(os.path.join(mrio_dir, "Z.parquet"))
    Y = pd.read_parquet(os.path.join(mrio_dir, "Y.parquet"))
    x = load_x(mrio_dir)

    regions = Z.index.get_level_values("region").unique().tolist()
    num_regions = len(regions)

    # Total bilateral trade: sum Z columns by destination region,
    # then add Y columns summed by destination region
    Z_to_r = Z.T.groupby(level="region").sum().T
    Y_to_r = Y.T.groupby(level=0).sum().T
    T = Z_to_r.add(Y_to_r, fill_value=0.0)  # (from_r × sector) → to_r

    # Aggregate over sectors: total bilateral trade from r to r'
    T_agg = T.groupby(level="region").sum()  # (from_r) → (to_r)

    # Total output per region (sum over sectors)
    x_by_region = x.groupby(level="region").sum()

    # Import dict: for each importer r, the intensity of imports from each r'
    # ximport[r'][r] = T_agg[r', r] / x_by_region[r]
    # (What fraction of r's output is received from r')
    import_dict = {}
    export_dict = {}

    for i, r in enumerate(regions):
        region_id = str(i + 1)  # 1-indexed region IDs
        own_output = x_by_region.get(r, 1e-10)

        # Import shares: flow from partner to this region / own output
        im = {}
        for j, r2 in enumerate(regions):
            partner_id = str(j + 1)
            if i == j:
                im[partner_id] = 0.0
            else:
                flow = T_agg.loc[r2, r] if r2 in T_agg.index and r in T_agg.columns else 0.0
                im[partner_id] = float(flow / own_output)
        import_dict[region_id] = im

        # Export fraction: total exports / own output
        total_exports = 0.0
        for j, r2 in enumerate(regions):
            if i == j:
                continue
            flow = T_agg.loc[r, r2] if r in T_agg.index and r2 in T_agg.columns else 0.0
            total_exports += flow
        export_dict[region_id] = float(total_exports / own_output)

    return import_dict, export_dict, regions


def main():
    parser = argparse.ArgumentParser(
        description="Extract bilateral trade JSONs from MRIO parquets."
    )
    parser.add_argument("--n", type=int, required=True,
                        help="Number of regions")
    parser.add_argument("--mrio-dir", type=str, default=None,
                        help="Path to aggregated MRIO dir. "
                             "Default: csv_asset/mrio/aggregated/eora_agg_N")
    parser.add_argument("--output-dir", type=str, default="csv_asset",
                        help="Directory for output JSON files")
    args = parser.parse_args()

    mrio_dir = args.mrio_dir or os.path.join(
        "csv_asset", "mrio", "aggregated", f"eora_agg_{args.n}"
    )

    print(f"Extracting trade data from {mrio_dir}...")
    import_dict, export_dict, regions = extract_trade(mrio_dir)

    print(f"\nRegion ordering ({len(regions)}):")
    for i, r in enumerate(regions):
        ex = export_dict[str(i + 1)]
        print(f"  [{i}] {r:<35}  export_frac={ex:.4f}")

    import_path = os.path.join(args.output_dir, f"{args.n}_import_2016.json")
    export_path = os.path.join(args.output_dir, f"{args.n}_export_2016.json")

    with open(import_path, "w") as f:
        json.dump(import_dict, f, indent=2)
    with open(export_path, "w") as f:
        json.dump(export_dict, f, indent=2)

    print(f"\nWritten: {import_path}")
    print(f"Written: {export_path}")


if __name__ == "__main__":
    main()
