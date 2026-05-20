import os
import copy
import pymrio
import pandas as pd

YEAR = 2015
csv_asset_dir = 'csv_asset'
eora_dir = os.path.join(csv_asset_dir, 'mrio')
unzipped_dir = os.path.join(eora_dir, 'unzipped')
eora_path = os.path.join(unzipped_dir, f'Eora26_{YEAR}_bp')

country_class_files = {
    3:  'CountryClass_3.csv',
    7:  'CountryClass_7.csv',
    20: 'CountryClass_20.csv',
}


def build_region_agg(eora_regions, group_csv_path):
    """
    Build a region aggregation vector for pymrio from a CountryClass CSV.

    The CSV has columns: No, Economy, Code, RI, RIG
      - Code: ISO-3 country code matching EORA regions
      - RIG:  numeric region index (1..N) — this is the actual grouping
      - RI:   human-readable region name within each RIG group

    Countries in EORA not found in the CSV are assigned to 'Rest of World'.
    """
    group_df = pd.read_csv(group_csv_path)
    group_df['Code'] = group_df['Code'].str.strip()
    group_df['RI'] = group_df['RI'].str.strip()

    # Build label: first RI value per RIG group (stable, human-readable)
    rig_label = (
        group_df.dropna(subset=['RIG'])
        .groupby('RIG')['RI']
        .first()
        .to_dict()
    )

    # Map code → group label (via RIG)
    code_to_rig = dict(zip(group_df['Code'], group_df['RIG']))

    def label_for(iso):
        rig = code_to_rig.get(iso)
        if rig is None or (isinstance(rig, float) and pd.isna(rig)):
            return 'Rest of World'
        return rig_label.get(rig, f'Region_{int(rig)}')

    agg_vec = [label_for(c) for c in eora_regions]
    return agg_vec


def aggregate_and_save(eora, n_agents, group_csv_path, out_dir):
    """Parse, aggregate by region grouping, save key matrices."""
    print(f"\n--- Aggregating to {n_agents} regions ---")

    eora_regions = list(eora.get_regions())
    region_agg = build_region_agg(eora_regions, group_csv_path)

    # Work on a copy so we can re-use the parsed eora for each aggegation level
    eora_copy = copy.deepcopy(eora)
    eora_copy.aggregate(region_agg=region_agg, inplace=True)
    eora_copy.calc_all()

    save_path = os.path.join(out_dir, f'eora_agg_{n_agents}')
    os.makedirs(save_path, exist_ok=True)
    eora_copy.save_all(save_path)
    print(f"  Saved to {save_path}")

    # Also export key matrices as parquet for easy loading in JAX
    eora_copy.Z.to_parquet(os.path.join(save_path, 'Z.parquet'))
    eora_copy.Y.to_parquet(os.path.join(save_path, 'Y.parquet'))
    for ext_name in eora_copy.get_extensions():
        ext = getattr(eora_copy, ext_name)
        if hasattr(ext, 'S') and ext.S is not None:
            ext.S.to_parquet(os.path.join(save_path, f'{ext_name}_S.parquet'))

    new_regions = list(eora_copy.get_regions())
    new_sectors = list(eora_copy.get_sectors())
    print(f"  Regions ({len(new_regions)}): {new_regions}")
    print(f"  Sectors ({len(new_sectors)}): {new_sectors}")
    return eora_copy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate EORA26 MRIO tables by a CountryClass CSV."
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to a single CountryClass CSV to aggregate. "
             "If omitted, runs all default aggregations (3, 7, 20)."
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of regions (used for output dir name: eora_agg_N). "
             "Required when --csv is given."
    )
    args = parser.parse_args()

    out_dir = os.path.join(eora_dir, 'aggregated')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Parsing EORA26 {YEAR}...")
    eora = pymrio.parse_eora26(eora_path, year=YEAR)
    print("  Parsed OK.")
    print(f"  Regions: {len(list(eora.get_regions()))}")
    print(f"  Sectors: {list(eora.get_sectors())[:3]} ...")

    if args.csv:
        if args.n is None:
            parser.error("--n is required when --csv is given")
        aggregate_and_save(eora, args.n, args.csv, out_dir)
    else:
        for n_agents, csv_file in country_class_files.items():
            group_csv_path = os.path.join(csv_asset_dir, csv_file)
            aggregate_and_save(eora, n_agents, group_csv_path, out_dir)

    print('\nAll aggregations complete.')

