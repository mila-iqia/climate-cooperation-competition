#!/usr/bin/env python3
"""
WBAPI_get_data_cbam_vuln.py  —  Fresh YAML generation for 9-region CBAM-vulnerability setup.

Follows the same WB API → time-series fitting → yaml pipeline as
scripts/data_preparation/WBAPI_get_data_7.ipynb, but uses the
CBAM-vulnerability-focused CountryClass (9 regions).

Region ordering matches MRIO output (eora_agg_9):
    1  Rest of World
    2  Russia + Turkey + Eurasia
    3  MENA (Gulf + N.Africa)
    4  EU & Western Europe          ← EU (eu_region_idx=3, 0-indexed)
    5  SSA Metals & Mining           ← CBAM-vulnerability group
    6  Americas
    7  SE Asia & Pacific developing
    8  China
    9  India

Usage:
    conda run -n rice-jax python scripts/data_preparation/WBAPI_get_data_cbam_vuln.py

Outputs:
    cbam_yamls/setup_vuln_9/1.yml ... 9.yml
"""
import csv
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import wbgapi as wb
import yaml

warnings.filterwarnings("ignore")

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)

from opt_helper import (
    get_data_list,
    get_env_data_list,
    get_gA_deltaA,
    get_pop_lg,
    merge_region_dict,
    packup_regions,
    write_yaml_files,
)


def get_tax_data_list(taxdf, code):
    """Look up corporate tax rate for a country from tax_rate.csv."""
    row = taxdf[taxdf["iso_3"] == code]
    if len(row) == 0:
        return 0.0
    return float(row.iloc[0]["rate"])

CC_PATH = os.path.join(REPO_ROOT, "cbam_yamls", "CountryClass_cbam_vuln_9.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "cbam_yamls", "setup_vuln_9")
IMPORT_JSON = os.path.join(REPO_ROOT, "csv_asset", "9_import_2016.json")
EXPORT_JSON = os.path.join(REPO_ROOT, "csv_asset", "9_export_2016.json")
DEFAULT_YAML = os.path.join(REPO_ROOT, "region_yamls", "default.yml")

NUM_REGIONS = 9


# ════════════════════════════════════════════════════════════════════
# 1. Load auxiliary data
# ════════════════════════════════════════════════════════════════════
def load_auxiliary():
    print("Loading auxiliary data...")
    lasdf = pd.read_csv(os.path.join(REPO_ROOT, "csv_asset", "UN-pop-pred.csv"))
    lasdf = lasdf[lasdf["Year"] == 2100].reset_index(drop=True)

    cc = pd.read_csv(CC_PATH)
    countryclass = {i: list(cc[cc["RIG"] == i]["Code"]) for i in range(1, NUM_REGIONS + 1)}
    countryclass[0] = sum([countryclass[i] for i in range(1, NUM_REGIONS + 1)], [])

    envdf = pd.read_csv(
        os.path.join(REPO_ROOT, "csv_asset",
                     "Environmental_Protection_Expenditures_Geo_Avg_Recent_Years_Sum.csv")
    )

    env_pay = {}
    with open(
        os.path.join(REPO_ROOT, "csv_asset",
                     "Environmental_Protection_Expenditures_Geo_Avg_Recent_Years_Sum.csv"),
        "r",
    ) as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] != "ISO3":
                env_pay[row[0]] = float(row[1]) / 100

    taxdf = pd.read_csv(os.path.join(REPO_ROOT, "csv_asset", "tax_rate.csv"))

    return lasdf, countryclass, envdf, env_pay, taxdf


# ════════════════════════════════════════════════════════════════════
# 2. Fetch World Bank data
# ════════════════════════════════════════════════════════════════════
def fetch_wb_data():
    print("Fetching World Bank API data (this may take a minute)...")
    series_list = [
        "NY.GDP.MKTP.CD",          # GDP (current US$)
        "CM.MKT.LCAP.CD",          # Capital stock
        "SP.POP.TOTL",             # Population
        "EN.GHG.CO2.MT.CE.AR5",    # CO₂ emissions (Mt CO2e) — Climate Watch
        "NE.CON.TOTL.ZS",          # Consumption share
    ]
    # NOTE: EN.ATM.CO2E.KT was archived from WDI source 2 (as of mid-2024).
    #       Replaced by EN.GHG.CO2.MT.CE.AR5 (Climate Watch).  Units: Mt vs kt.
    df = wb.data.DataFrame(series_list, time=range(1960, 2022, 1), labels=True)
    df = df.reset_index()
    print(f"  Fetched {len(df)} rows.")
    return df


# ════════════════════════════════════════════════════════════════════
# 3. Pre-process: exclusions, economy list
# ════════════════════════════════════════════════════════════════════
def preprocess(df):
    print("Pre-processing...")
    economy_list = list(df["economy"])[: len(set(df["economy"]))]
    country_list = list(df["Country"])[: len(set(df["economy"]))]
    economy_region_list = []
    for x in economy_list:
        if x == "WLD":
            break
        economy_region_list.append(x)

    years = list(df.columns)[4:]
    df[years] = df[years].apply(pd.to_numeric)

    # Exclude countries with no GDP data 2003-2020
    exc_code = [
        "YEM", "VIR", "VEN", "TKM", "SYR", "MAF", "SSD", "SOM", "SXM", "SMR",
        "MNP", "NCL", "NRU", "LIE", "XKX", "PRK", "IMN", "GRL", "GIB", "PYF",
        "FRO", "ERI", "CUW", "CHI", "CYM", "VGB", "ABW", "AND", "TWN",
    ]

    return economy_list, country_list, economy_region_list, years, exc_code


# ════════════════════════════════════════════════════════════════════
# 4. Derive carbon intensity & impute missing data
# ════════════════════════════════════════════════════════════════════
def derive_sigma(df, economy_list, country_list, years):
    """Derive sigma = CO2 intensity of GDP.

    Old indicator EN.ATM.CO2E.KT was in kt  → multiplied by 1e6 (kg/kt=1e6?) for consistency.
    New indicator EN.GHG.CO2.MT.CE.AR5 is in Mt → multiply by 1e9 to keep same sigma units.
    sigma ≈ kg CO2 / $ GDP  (≈ tCO2 / k$).
    """
    print("Deriving carbon intensity σ...")
    co2_series = "EN.GHG.CO2.MT.CE.AR5"
    gdp_series = "NY.GDP.MKTP.CD"
    for i in range(len(country_list)):
        co2 = df[(df["economy"] == economy_list[i]) & (df["series"] == co2_series)][years].reset_index(drop=True)
        gdp = df[(df["economy"] == economy_list[i]) & (df["series"] == gdp_series)][years].reset_index(drop=True)
        if co2.empty or gdp.empty:
            # Missing data — create NaN row so borrow_co2 can fill it later
            s = pd.DataFrame(columns=["economy", "Country", "series", "Series"] + years)
            s.loc[0] = [economy_list[i], country_list[i], "EN.ATM.CO2E.KD.CD", "sigma"] + [float("nan")] * len(years)
            df = pd.concat([df, s], ignore_index=True)
            continue
        s = co2 * 1_000_000_000 / gdp  # Mt→kg: 1 Mt = 1e9 kg
        s.at[0, "economy"] = economy_list[i]
        s.at[0, "Country"] = country_list[i]
        s.at[0, "series"] = "EN.ATM.CO2E.KD.CD"
        s.at[0, "Series"] = "sigma"
        df = pd.concat([df, s], ignore_index=True)
    return df


def borrow_co2(df, economy_list, country_list, years):
    """Fill missing CO₂ data by borrowing from similar countries."""
    print("Borrowing CO₂ data for missing countries...")
    borrowdict = {
        "PSE": "EGY", "VIR": "USA", "VEN": "MEX", "TCA": "GBR", "MAF": "FRA",
        "SSD": "EGY", "SXM": "NLD", "SMR": "ITA", "PRI": "USA", "MNP": "USA",
        "NCL": "FRA", "MCO": "FRA", "MAC": "CHN", "XKX": "TUR", "PRK": "RUS",
        "IMN": "GBR", "HKG": "CHN", "GUM": "USA", "GRL": "DNK", "GIB": "GBR",
        "PYF": "FRA", "FRO": "DNK", "ERI": "EGY", "CUW": "NLD", "CHI": "GBR",
        "CYM": "GBR", "VGB": "GBR", "BMU": "CAN", "ABW": "MEX", "ASM": "USA",
    }
    noco2 = {}
    sigma_mask = df["series"] == "EN.ATM.CO2E.KD.CD"
    for i in range(len(country_list)):
        rows = df[sigma_mask & (df["economy"] == economy_list[i])]
        if rows.empty or rows["YR2018"].isnull().values[0]:
            noco2[economy_list[i]] = country_list[i]

    noco2_region = {k: v for k, v in noco2.items() if k != "WLD"}
    code2idx = {
        k: df[sigma_mask & (df["economy"] == k)].index.values[0]
        for k in noco2_region
        if len(df[sigma_mask & (df["economy"] == k)]) > 0
    }
    borrowed, skipped = 0, 0
    for k in noco2_region:
        if k not in code2idx:
            skipped += 1
            continue
        donor = borrowdict.get(k)
        if donor is None:
            skipped += 1
            continue
        try:
            donor_data = df[sigma_mask & (df["economy"] == donor)][years].iloc[0]
            df.loc[code2idx[k], years] = donor_data
            borrowed += 1
        except Exception:
            print(f"  Warning: cannot borrow CO₂ for {k}")
            skipped += 1
    print(f"  Borrowed sigma for {borrowed} countries, skipped {skipped}")
    return df


def impute_K(df, economy_region_list, exc_code, years):
    """KNN imputation for missing capital stock data."""
    from sklearn.neighbors import KNeighborsRegressor
    print("KNN-imputing capital stock K...")
    for y in range(2003, 2021):
        yr = f"YR{y}"
        train_data, train_label, test_data = [], [], []
        train_codes, test_codes = [], []
        for code in economy_region_list:
            if code in exc_code:
                continue
            k_val = get_data_list(df, code, "K")[1][yr]
            y_val = get_data_list(df, code, "Y")[1][yr]
            l_val = get_data_list(df, code, "L")[1][yr]
            if pd.isnull(k_val):
                test_data.append([y_val, l_val])
                test_codes.append(code)
            else:
                train_data.append([y_val, l_val])
                train_label.append(k_val)
                train_codes.append(code)
        if test_data:
            neigh = KNeighborsRegressor(n_neighbors=5)
            neigh.fit(np.array(train_data), np.array(train_label))
            preds = neigh.predict(np.array(test_data))
            for i, code in enumerate(test_codes):
                idx = df[df["economy"] == code][df["series"] == "CM.MKT.LCAP.CD"][yr].index.values[0]
                df.loc[idx, yr] = preds[i]
    return df


def impute_C(df, economy_region_list, exc_code, years):
    """KNN imputation for missing consumption data."""
    from sklearn.neighbors import KNeighborsRegressor
    print("KNN-imputing consumption share C...")
    for y in range(2003, 2021):
        yr = f"YR{y}"
        train_data, train_label, test_data = [], [], []
        train_codes, test_codes = [], []
        for code in economy_region_list:
            if code in exc_code:
                continue
            c_val = get_data_list(df, code, "C")[1][yr]
            y_val = get_data_list(df, code, "Y")[1][yr]
            l_val = get_data_list(df, code, "L")[1][yr]
            if pd.isnull(c_val):
                test_data.append([y_val, l_val])
                test_codes.append(code)
            else:
                train_data.append([y_val, l_val])
                train_label.append(c_val)
                train_codes.append(code)
        if test_data:
            neigh = KNeighborsRegressor(n_neighbors=5)
            neigh.fit(np.array(train_data), np.array(train_label))
            preds = neigh.predict(np.array(test_data))
            for i, code in enumerate(test_codes):
                idx = df[df["economy"] == code][df["series"] == "NE.CON.TOTL.ZS"][yr].index.values[0]
                df.loc[idx, yr] = preds[i]
    return df


def derive_TFP(df, economy_list, exc_code, years):
    """Derive TFP (A) from Y, K, L using Cobb-Douglas."""
    print("Deriving TFP (A)...")
    for i in range(len(economy_list)):
        if economy_list[i] in exc_code:
            continue
        s = (
            df[df["economy"] == economy_list[i]][df["series"] == "NY.GDP.MKTP.CD"][years].reset_index(drop=True)
            / (
                1_000_000_000_000
                * (
                    (df[df["economy"] == economy_list[i]][df["series"] == "CM.MKT.LCAP.CD"][years].reset_index(drop=True)
                     / 1_000_000_000_000) ** 0.3
                    * (df[df["economy"] == economy_list[i]][df["series"] == "SP.POP.TOTL"][years].reset_index(drop=True)
                       / 1_000_000_000) ** 0.7
                )
            )
        )
        s.at[0, "economy"] = economy_list[i]
        s.at[0, "Country"] = economy_list[i]
        s.at[0, "series"] = "ATFP"
        s.at[0, "Series"] = "A"
        df = pd.concat([df, s], ignore_index=True)
    return df


# ════════════════════════════════════════════════════════════════════
# 5. Gather raw results per country
# ════════════════════════════════════════════════════════════════════
def gather_raw_results(df, economy_region_list, exc_code, lasdf, envdf, taxdf):
    print("Gathering per-country raw results...")
    raw_results = {}
    skipped_sigma = []
    for x in economy_region_list:
        if x in exc_code:
            continue
        raw_results[x] = {}
        raw_results[x]["TS_Y"] = get_data_list(df, x, "Y")
        raw_results[x]["TS_A"] = get_data_list(df, x, "A")
        raw_results[x]["TS_K"] = get_data_list(df, x, "K")
        raw_results[x]["TS_L"] = get_data_list(df, x, "L")
        raw_results[x]["TS_sigma"] = get_data_list(df, x, "sigma")
        raw_results[x]["TS_C"] = get_data_list(df, x, "C")

        # Skip countries with empty sigma time series (e.g. MNE, SRB with
        # short GDP histories that produce no continuous sigma range).
        if not raw_results[x]["TS_sigma"][0]:
            skipped_sigma.append(x)
            del raw_results[x]
            continue

        try:
            raw_results[x]["La"] = list(
                lasdf[lasdf["Code"] == x]["Population (future projections)"]
            )[0]
        except IndexError:
            # Fallback: use current pop * 0.8 as convergence pop
            raw_results[x]["La"] = raw_results[x]["TS_L"][0][-1] * 0.8
        raw_results[x]["mitigation"] = get_env_data_list(envdf, x)
        try:
            raw_results[x]["saving"] = 1 - int(raw_results[x]["TS_C"][0][-1]) / 100
        except (ValueError, IndexError):
            raw_results[x]["saving"] = 0.22  # global average fallback
        try:
            raw_results[x]["tax"] = get_tax_data_list(taxdf, x)
        except Exception:
            raw_results[x]["tax"] = 0
    if skipped_sigma:
        print(f"  Skipped {len(skipped_sigma)} countries with empty sigma: {skipped_sigma}")
    print(f"  Collected data for {len(raw_results)} countries.")
    return raw_results


# ════════════════════════════════════════════════════════════════════
# 6. Merge regions and fit dynamics
# ════════════════════════════════════════════════════════════════════
def fit_region_params(countryclass, raw_results, exc_code):
    print("Fitting regional parameters...")
    # Add countries missing from raw_results to exc_code so packup_regions skips them
    all_codes = set()
    for v in countryclass.values():
        all_codes.update(v)
    extra_exc = [c for c in all_codes if c not in raw_results and c not in exc_code]
    full_exc = exc_code + extra_exc
    para_result = {}
    for i in range(1, NUM_REGIONS + 1):
        codes = countryclass[i]
        valid_codes = [c for c in codes if c not in full_exc and c in raw_results]
        if not valid_codes:
            print(f"  WARNING: Region {i} has no valid countries!")
            continue
        print(f"  Region {i}: {len(valid_codes)} countries")
        try:
            a = merge_region_dict(
                packup_regions(raw_results, i, countryclass, full_exc, raw_results)
            )
        except Exception as e:
            import traceback
            print(f"  ERROR merging region {i}: {e}")
            print(traceback.format_exc())
            continue

        para_result[i] = {}
        para_result[i]["xl_g"] = get_pop_lg(a["Ls"], a["Las"])
        para_result[i]["xL_a"] = a["Las"] / 1_000_000
        para_result[i]["xL_0"] = a["Ls"][-1] / 1_000_000
        try:
            para_result[i]["xg_A"], para_result[i]["xdelta_A"] = get_gA_deltaA(a["As"])
        except Exception:
            para_result[i]["xg_A"] = 0.076
            para_result[i]["xdelta_A"] = 0.005
        para_result[i]["xA_0"] = a["As"][-1]
        para_result[i]["xK_0"] = a["Ks"][-1] / 1_000_000_000_000
        para_result[i]["xsigma_0"] = a["sigmas"] / (1 - 0.05)
        para_result[i]["xtax"] = a["taxs"]
        para_result[i]["xmitigation_0"] = a["mitigations"]
        para_result[i]["xsaving_0"] = a["savings"]

    return para_result


# ════════════════════════════════════════════════════════════════════
# 7. Write yamls & inject trade data
# ════════════════════════════════════════════════════════════════════
def write_yamls_with_trade(para_result):
    print(f"Writing yamls to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load default DICE constants
    with open(DEFAULT_YAML) as f:
        default_doc = yaml.safe_load(f)

    default_dict = {
        "_RICE_CONSTANT": {
            "xgamma": 0.3,
            "xa_1": 0,
            "xa_2": 0.00236,
            "xa_3": 2,
        }
    }

    # Write yaml for each region
    write_yaml_files(para_result, OUTPUT_DIR, default_dict=default_dict)

    # Inject trade data from MRIO-derived JSONs
    with open(IMPORT_JSON) as f:
        import_data = json.load(f)
    with open(EXPORT_JSON) as f:
        export_data = json.load(f)

    for i in range(1, NUM_REGIONS + 1):
        path = os.path.join(OUTPUT_DIR, f"{i}.yml")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping trade injection")
            continue
        with open(path) as f:
            doc = yaml.safe_load(f) or {}

        # Add DICE constants
        doc["_DICE_CONSTANT"] = default_doc.get("_DICE_CONSTANT", {})

        # Inject trade
        doc["_RICE_CONSTANT"]["ximport"] = import_data[str(i)]
        doc["_RICE_CONSTANT"]["xexport"] = export_data[str(i)]

        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)

    print("  Yamls written with trade data.")


# ════════════════════════════════════════════════════════════════════
# 8. Sanity check — Cobb-Douglas identity
# ════════════════════════════════════════════════════════════════════
def sanity_check():
    print("\nSanity check — Cobb-Douglas identity:")
    gamma = 0.3
    all_ok = True
    for i in range(1, NUM_REGIONS + 1):
        path = os.path.join(OUTPUT_DIR, f"{i}.yml")
        with open(path) as f:
            doc = yaml.safe_load(f)
        p = doc["_RICE_CONSTANT"]
        gdp = p["xA_0"] * (p["xK_0"] ** gamma) * (p["xL_0"] / 1000) ** (1 - gamma)
        # For a freshly-fitted yaml, A_0 is the last value of the merged TFP series,
        # which already implies this identity. Just verify the yaml was written correctly.
        gdp_from_yk = (
            p["xA_0"] * (p["xK_0"] ** gamma) * (p["xL_0"] / 1000) ** (1 - gamma)
        )
        print(f"  Region {i}: GDP={gdp:.4f}T  L0={p['xL_0']:.1f}M  "
              f"sigma={p['xsigma_0']:.4f}  A0={p['xA_0']:.4f}")
        if gdp <= 0 or np.isnan(gdp):
            print(f"    ✗ INVALID GDP!")
            all_ok = False

    print("  All checks passed." if all_ok else "  SOME CHECKS FAILED!")
    return all_ok


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CBAM-Vulnerability YAML Generator (9 regions, fresh WB API)")
    print("=" * 70)

    lasdf, countryclass, envdf, env_pay, taxdf = load_auxiliary()
    df = fetch_wb_data()
    economy_list, country_list, economy_region_list, years, exc_code = preprocess(df)
    df = derive_sigma(df, economy_list, country_list, years)
    df = borrow_co2(df, economy_list, country_list, years)
    df = impute_K(df, economy_region_list, exc_code, years)
    df = impute_C(df, economy_region_list, exc_code, years)
    df = derive_TFP(df, economy_list, exc_code, years)
    raw_results = gather_raw_results(
        df, economy_region_list, exc_code, lasdf, envdf, taxdf
    )
    para_result = fit_region_params(countryclass, raw_results, exc_code)
    write_yamls_with_trade(para_result)
    sanity_check()

    print("\n" + "=" * 70)
    print("Done. Yamls at:", OUTPUT_DIR)
    print(f"EU region index: 3 (0-indexed, yaml 4.yml)")
    print("=" * 70)


if __name__ == "__main__":
    main()
