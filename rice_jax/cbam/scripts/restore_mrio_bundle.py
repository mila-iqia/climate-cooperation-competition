"""restore_mrio_bundle.py

Extract a RICE-MRIO 9-region data bundle (created by save_mrio_bundle.py)
into a target csv_asset directory so that RiceMRIO and canonical_config.py
can load the data for training.

The bundle contains files with paths relative to csv_asset/:
  - mrio/aggregated/eora_agg_9/Z.parquet
  - mrio/aggregated/eora_agg_9/Y.parquet
  - mrio/aggregated/eora_agg_9/Q_S.parquet
  - mrio/aggregated/eora_agg_9/x.txt
  - CountryClass_9.csv

After extraction, point RiceMRIO at the directory by setting:
    export CBAM_MRIO_ROOT=/path/to/target-csv-asset
or pass mrio_data_root= directly when constructing RiceMRIO.

Usage (run from project root):
    python cbam/scripts/restore_mrio_bundle.py
    python cbam/scripts/restore_mrio_bundle.py --bundle /path/to/bundle.zip
    python cbam/scripts/restore_mrio_bundle.py --target-csv-asset /data/csv_asset
    python cbam/scripts/restore_mrio_bundle.py --force   # overwrite existing files
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))


def restore_bundle(bundle_path: str, target_csv_asset: str, force: bool) -> None:
    """Extract the bundle into target_csv_asset, preserving the internal path layout."""
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path!r}")

    print(f"Bundle         : {bundle_path}")
    print(f"Target csv_asset: {target_csv_asset}")
    print(f"Force overwrite: {force}")
    print()

    with zipfile.ZipFile(bundle_path, "r") as zf:
        members = zf.infolist()
        for info in members:
            # Normalize to OS path separators
            rel = info.filename.replace("/", os.sep)
            dest = os.path.join(target_csv_asset, rel)

            if os.path.exists(dest) and not force:
                print(f"  skip (exists) {info.filename}")
                continue

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            print(f"  extracted     {dest}")

    print()
    print("Extraction complete.")
    print()
    print("To use this data for training, set:")
    print(f"    export CBAM_MRIO_ROOT={target_csv_asset}")
    print("or pass  mrio_data_root=<path>  when constructing RiceMRIO directly.")


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_bundle = os.path.join(repo_root, "rice_mrio_9region_bundle.zip")
    default_target = os.path.join(repo_root, "csv_asset")

    parser = argparse.ArgumentParser(
        description="Restore RICE-MRIO 9-region data from a bundle zip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bundle",
        default=default_bundle,
        metavar="ZIP",
        help="Path to the bundle zip file produced by save_mrio_bundle.py.",
    )
    parser.add_argument(
        "--target-csv-asset",
        default=default_target,
        metavar="DIR",
        help=(
            "Directory to extract into.  Files will be placed at "
            "<DIR>/mrio/aggregated/eora_agg_9/ and <DIR>/CountryClass_9.csv."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.  Without this flag, existing files are skipped.",
    )
    args = parser.parse_args()

    restore_bundle(
        bundle_path=os.path.abspath(args.bundle),
        target_csv_asset=os.path.abspath(args.target_csv_asset),
        force=args.force,
    )


if __name__ == "__main__":
    main()
