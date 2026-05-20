#!/usr/bin/env python3
"""Generate CountryClass_cbam_vuln_9.csv for CBAM-vulnerability-focused aggregation.

Maps 218 EORA countries to 9 regions designed to isolate CBAM-vulnerable
economies as separate RL agents, rather than burying them in large geographic
buckets.

Region design rationale:
    1  EU & Western Europe         – CBAM imposer
    2  Russia + Turkey + Eurasia   – High σ (0.90–1.35), major CBAM targets
    3  China                       – σ=0.78, largest exporter
    4  India                       – σ=0.87, steel/cement
    5  MENA (Gulf + N.Africa)      – Cement/fertilizer, σ=0.59–1.21
    6  SSA Metals & Mining         – ZAF aluminium/steel, MOZ aluminium,
                                     ZMB copper, NGA oil/gas, COD cobalt
    7  SE Asia & Pacific developing – Vietnam, Indonesia, Thailand
    8  Americas                    – USA, Canada, Brazil, Mexico
    9  Rest of World (residual)    – Japan, Korea, Australia, Bangladesh,
                                     Pakistan, SSA low-income (non-metal)

Usage:
    python generate_countryclass_vuln.py
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CC20_PATH = os.path.join(REPO_ROOT, "csv_asset", "CountryClass_20.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "CountryClass_cbam_vuln_9.csv")

# ── 20-RIG → 9-RIG base mapping ──
# RIG numbers are chosen to match the pymrio MRIO output ordering so that
# yaml K.yml = MRIO region index (K-1).  Verified against eora_agg_9.
#
# MRIO output order: [0]=RoW, [1]=Russia, [2]=MENA, [3]=EU, [4]=SSA,
#                     [5]=Americas, [6]=SE Asia, [7]=China, [8]=India
RIG20_TO_RIG9 = {
    1:  6,   # North America High Income (USA) → Americas
    2:  6,   # North America High Income (Canada, Bermuda) → Americas
    3:  4,   # Europe & Central Asia High Income → EU & Western Europe
    4:  2,   # Europe & Central Asia Upper Middle → Russia + Turkey + Eurasia
    5:  2,   # Europe & Central Asia Lower Middle → Russia + Turkey + Eurasia
    6:  1,   # East Asia & Pacific High Income → Rest of World
    7:  7,   # East Asia & Pacific Upper Middle → SE Asia & Pacific developing
    8:  8,   # East Asia & Pacific Upper Middle (China only) → China
    9:  7,   # East Asia & Pacific Lower Middle → SE Asia & Pacific developing
    10: 9,   # South Asia Lower Middle (India only) → India
    11: 1,   # South Asia (ex India) → Rest of World
    12: 3,   # MENA High Income → MENA
    13: 3,   # MENA Upper Middle Income → MENA
    14: 3,   # MENA Lower/Low Income → MENA
    15: 6,   # Latin America & Caribbean High Income → Americas
    16: 6,   # Latin America & Caribbean Upper Middle → Americas
    17: 6,   # Latin America & Caribbean Lower Middle → Americas
    18: 5,   # Sub-Saharan Africa Upper Middle → SSA Metals & Mining
    19: 5,   # Sub-Saharan Africa Lower Middle → SSA Metals & Mining
    20: 1,   # Sub-Saharan Africa Low Income → Rest of World (with overrides)
}

# ── Country-level overrides (move CBAM-vulnerable countries to RIG 5) ──
# These countries are in 20-RIG 20 (SSA Low Income) but have significant
# CBAM-relevant metal/mining exports that should be modelled separately.
COUNTRY_OVERRIDES = {
    "MOZ": 5,   # Mozambique: Mozal aluminium smelter, one of Africa's largest
    "COD": 5,   # Congo DRC: cobalt (60% of world supply), copper, tin
}

# ── Region labels (ordered to match MRIO output from pymrio) ──
LABEL_MAP = {
    1: "Rest of World",
    2: "Russia + Turkey + Eurasia",
    3: "MENA (Gulf + N.Africa)",
    4: "EU & Western Europe",
    5: "SSA Metals & Mining",
    6: "Americas",
    7: "SE Asia & Pacific developing",
    8: "China",
    9: "India",
}


def main():
    cc20 = pd.read_csv(CC20_PATH)

    new_rigs = []
    new_labels = []

    for _, row in cc20.iterrows():
        code = row["Code"]
        rig20 = row["RIG"]

        # Country-level override first
        if code in COUNTRY_OVERRIDES:
            rig9 = COUNTRY_OVERRIDES[code]
        elif pd.notna(rig20):
            rig9 = RIG20_TO_RIG9.get(int(rig20))
        else:
            rig9 = None

        if rig9 is not None:
            new_rigs.append(int(rig9))
            new_labels.append(LABEL_MAP[int(rig9)])
        else:
            new_rigs.append(np.nan)
            new_labels.append("")

    cc_new = cc20.copy()
    cc_new["RIG"] = new_rigs
    cc_new["RI"] = new_labels

    # Convert RIG to int where not NaN
    cc_new["RIG"] = cc_new["RIG"].apply(
        lambda x: int(x) if pd.notna(x) else ""
    )
    cc_new.to_csv(OUTPUT_PATH, index=False)

    # Summary
    print(f"Written: {OUTPUT_PATH}")
    print(f"Total countries: {len(cc_new)}")
    assigned = cc_new[cc_new["RIG"] != ""]
    for rig in sorted(LABEL_MAP.keys()):
        count = len(assigned[assigned["RIG"] == rig])
        print(f"  RIG {rig}: {LABEL_MAP[rig]:<35} ({count} countries)")
    unassigned = len(cc_new) - len(assigned)
    if unassigned:
        print(f"  Unassigned: {unassigned} countries (no RIG in source)")


if __name__ == "__main__":
    main()
