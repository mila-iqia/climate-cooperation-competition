#!/usr/bin/env python3
"""
Generate 9-region CBAM-vulnerability yamls by re-aggregating from 20-region yamls.

This is the FAST PATH following the CBAM notebook pattern
(scripts/data_preparation/WBAPI_get_data_cbam.ipynb).
The 20-region yamls were themselves fitted from fresh WB API time series.

A separate WBAPI_get_data_cbam_vuln.py does the full fresh pipeline but
requires debugging the data processing chain.

Usage:
    python scripts/data_preparation/generate_vuln_yamls_from_20.py
"""
import json
import os
import sys

import numpy as np
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

YAMLS_20_DIR   = os.path.join(REPO_ROOT, "other_yamls", "20_regions")
CBAM_DIR       = os.path.join(REPO_ROOT, "cbam_yamls")
OUTPUT_DIR     = os.path.join(CBAM_DIR, "setup_vuln_9")
IMPORT_JSON_20 = os.path.join(REPO_ROOT, "csv_asset", "20_import_2016.json")
IMPORT_JSON_9  = os.path.join(REPO_ROOT, "csv_asset", "9_import_2016.json")
EXPORT_JSON_9  = os.path.join(REPO_ROOT, "csv_asset", "9_export_2016.json")
DEFAULT_YAML   = os.path.join(REPO_ROOT, "region_yamls", "default.yml")

GAMMA = 0.3

# ── Region definitions ──────────────────────────────────────────────────────
# Ordered to match MRIO output (eora_agg_9).
# For MOZ/COD: they're in 20-RIG 20 (SSA Low Income) but should be in RIG 5.
# Since we merge whole 20-RIGs here, they stay in RIG 1 (RoW).
# The MRIO trade data correctly assigns them to SSA Metals (RIG 5)
# because aggregate_local_mrio.py uses the per-country CountryClass CSV.
# The yaml economic params are approximate (20-RIG 20 stays in RoW).
# For exact per-country fitting, use the full WB API pipeline.
SETUP_VULN_9 = {
    "description": "CBAM-Vulnerability: 9 regions isolating SSA Metals & SE Asia",
    "regions": [
        (1, "Rest of World",                [6, 11, 20]),
        (2, "Russia + Turkey + Eurasia",    [4, 5]),
        (3, "MENA (Gulf + N.Africa)",       [12, 13, 14]),
        (4, "EU & Western Europe",          [3]),
        (5, "SSA Metals & Mining",          [18, 19]),
        (6, "Americas",                     [1, 2, 15, 16, 17]),
        (7, "SE Asia & Pacific developing", [7, 9]),
        (8, "China",                        [8]),
        (9, "India",                        [10]),
    ],
}

NUM_REGIONS = len(SETUP_VULN_9["regions"])


def load_20_yamls():
    params = {}
    for i in range(1, 21):
        path = os.path.join(YAMLS_20_DIR, f"{i}.yml")
        with open(path) as f:
            data = yaml.safe_load(f)
        params[i] = data["_RICE_CONSTANT"]
    return params


def approx_gdp(p):
    return p["xA_0"] * (p["xK_0"] ** GAMMA) * (p["xL_0"] / 1000) ** (1 - GAMMA)


def build_cbam_index_map(regions):
    m = {}
    for cbam_idx, _label, rigs in regions:
        for r in rigs:
            m[r] = cbam_idx
    return m


def merge_regions(rig_list, params20, gdp20, import_data_20, cbam_index_map):
    """Merge 20-region RIGs into one CBAM region (GDP-weighted)."""
    rigs = [r for r in rig_list if r in params20]
    total_gdp = sum(gdp20[r] for r in rigs)
    w = {r: gdp20[r] / total_gdp for r in rigs}

    xL_0 = sum(params20[r]["xL_0"] for r in rigs)
    xL_a = sum(params20[r]["xL_a"] for r in rigs)
    xK_0 = sum(params20[r]["xK_0"] for r in rigs)

    pop_slope_num = sum(
        (params20[r]["xL_a"] - params20[r]["xL_0"]) * params20[r]["xl_g"] for r in rigs
    )
    pop_slope_den = xL_a - xL_0
    xl_g = pop_slope_num / pop_slope_den if abs(pop_slope_den) > 1e-6 else np.mean([params20[r]["xl_g"] for r in rigs])

    xA_0 = total_gdp / ((xK_0 ** GAMMA) * (xL_0 / 1000) ** (1 - GAMMA))

    xg_A          = sum(w[r] * params20[r]["xg_A"]          for r in rigs)
    xdelta_A      = sum(w[r] * params20[r]["xdelta_A"]       for r in rigs)
    xsigma_0      = sum(w[r] * params20[r]["xsigma_0"]       for r in rigs)
    xmitigation_0 = sum(w[r] * params20[r]["xmitigation_0"]  for r in rigs)
    xsaving_0     = sum(params20[r]["xsaving_0"] * gdp20[r]  for r in rigs) / total_gdp
    xexport       = sum(params20[r].get("xexport", 0.0) * gdp20[r] for r in rigs) / total_gdp

    merged_cbam_idx = cbam_index_map[rigs[0]]
    all_cbam = sorted(set(cbam_index_map.values()))
    ximport_cbam = {str(c): 0.0 for c in all_cbam}
    for r in rigs:
        r_str = str(r)
        if r_str not in import_data_20:
            continue
        for src_str, val in import_data_20[r_str].items():
            src_cbam = cbam_index_map.get(int(src_str))
            if src_cbam is None or src_cbam == merged_cbam_idx:
                continue
            ximport_cbam[str(src_cbam)] += val * w[r]

    return {
        "xA_0":          float(xA_0),
        "xK_0":          float(xK_0),
        "xL_0":          float(xL_0),
        "xL_a":          float(xL_a),
        "xa_1":          0,
        "xa_2":          0.00236,
        "xa_3":          2,
        "xdelta_A":      float(xdelta_A),
        "xg_A":          float(xg_A),
        "xgamma":        0.3,
        "xl_g":          float(xl_g),
        "xmitigation_0": float(xmitigation_0),
        "xsaving_0":     float(xsaving_0),
        "xsigma_0":      float(xsigma_0),
        "xtax":          0.0,
        "xexport":       float(xexport),
        "ximport":       ximport_cbam,
    }


def main():
    print("Loading 20-region yamls...")
    params20 = load_20_yamls()
    gdp20 = {i: approx_gdp(params20[i]) for i in params20}

    with open(IMPORT_JSON_20) as f:
        import_data_20 = {str(k): {str(kk): v for kk, v in vv.items()} for k, vv in json.load(f).items()}

    with open(DEFAULT_YAML) as f:
        default_doc = yaml.safe_load(f)
    dice_block = default_doc.get("_DICE_CONSTANT", {})

    # Also load 9-region trade JSONs (from MRIO parquets — more accurate)
    with open(IMPORT_JSON_9) as f:
        import_9 = json.load(f)
    with open(EXPORT_JSON_9) as f:
        export_9 = json.load(f)

    cbam_map = build_cbam_index_map(SETUP_VULN_9["regions"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nGenerating {NUM_REGIONS} yamls → {OUTPUT_DIR}")
    print(f"{'─'*80}")
    print(f"  {'#':>3}  {'Region':<35}  {'GDP (T$)':>9}  {'Pop (M)':>8}  {'σ':>8}")
    print(f"  {'─'*3}  {'─'*35}  {'─'*9}  {'─'*8}  {'─'*8}")

    for cbam_idx, label, rigs in SETUP_VULN_9["regions"]:
        p = merge_regions(rigs, params20, gdp20, import_data_20, cbam_map)

        # Override trade data with MRIO-derived values (more accurate for 9-region split)
        idx_str = str(cbam_idx)
        if idx_str in import_9:
            p["ximport"] = import_9[idx_str]
        if idx_str in export_9:
            p["xexport"] = export_9[idx_str]

        doc = {"_DICE_CONSTANT": dice_block, "_RICE_CONSTANT": p}
        path = os.path.join(OUTPUT_DIR, f"{cbam_idx}.yml")
        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

        print(f"  {cbam_idx:>3}  {label:<35}  {approx_gdp(p):>9.2f}  "
              f"{p['xL_0']:>8.0f}  {p['xsigma_0']:>8.4f}")

    # Sanity check
    print(f"\n{'─'*80}")
    print("Cobb-Douglas sanity check:")
    all_ok = True
    for cbam_idx, label, rigs in SETUP_VULN_9["regions"]:
        path = os.path.join(OUTPUT_DIR, f"{cbam_idx}.yml")
        with open(path) as f:
            doc = yaml.safe_load(f)
        p = doc["_RICE_CONSTANT"]
        gdp_check = approx_gdp(p)
        expected = sum(gdp20[r] for r in rigs if r in gdp20)
        rel_err = abs(gdp_check - expected) / (expected + 1e-9)
        status = "OK" if rel_err < 1e-6 else f"MISMATCH rel_err={rel_err:.2e}"
        if rel_err >= 1e-6:
            all_ok = False
        print(f"  {cbam_idx}. {label:<35}  check={gdp_check:.4f}  expected={expected:.4f}  {status}")

    print()
    if all_ok:
        print("All checks passed.")
    else:
        print("SOME CHECKS FAILED!")

    print(f"\nYamls written to: {OUTPUT_DIR}")
    print(f"EU region index: 3 (0-indexed, yaml 4.yml)")


if __name__ == "__main__":
    main()
