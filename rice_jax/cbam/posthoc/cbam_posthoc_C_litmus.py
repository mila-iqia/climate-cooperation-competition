"""cbam_posthoc_C_litmus.py — Post-hoc diagnostic for Experiment C (multi-seed litmus).

Analyses per-test per-seed metric values from the Experiment C pkl to identify:
1. Which tests are borderline vs clearly failing
2. Per-seed metric values showing the distribution of pass margins
3. Whether failures cluster on specific seeds (training noise) or specific tests
   (structural weakness)
4. Per-REGION decomposition: which regions respond to the CBAM signal, which don't
5. Regional "Jacobian": finite-difference sensitivity of each region's behaviour
   (diversion, mitigation) to the CBAM on/off signal — identifies who's driving
   aggregate pass/fail and whether non-responding regions drag the average down.

Usage (from rice_jax/):
    python cbam/posthoc/cbam_posthoc_C_litmus.py \\
        --pkl cbam/experiment_results/cbam_experiment_C_litmus_*/plots/cbam_C_litmus_*.pkl
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))


import argparse
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from _experiment_util import get_output_dir


# ── Region config (9-region vulnerability) ──────────────────────────────────

NUM_REGIONS = 9
EU_IDX      = 3
REGION_NAMES = {
    0: "RoW", 1: "Russia+Eur.", 2: "MENA", 3: "EU",
    4: "SSA-Mining", 5: "Americas", 6: "SE Asia", 7: "China", 8: "India",
}
NON_EU            = [r for r in range(NUM_REGIONS) if r != EU_IDX]
CBAM_PLOT_REGIONS = [r for r in NON_EU if r != 0]   # exclude RoW (catch-all)


# ── Test metadata ───────────────────────────────────────────────────────────

TEST_INFO = {
    "m1": {
        "name": "M1: Diversion exists",
        "metric_key": "rel_drop",
        "threshold": 0.15,
        "direction": ">",
        "unit": "rel drop",
    },
    "m2": {
        "name": "M2: Costless mitigation",
        "metric_key": "mean_mu",
        "threshold": 0.15,
        "direction": ">",
        "unit": "μ",
    },
    "m3": {
        "name": "M3: Costly mitigation",
        "metric_key": "mean_mu",
        "threshold": 0.01,
        "direction": ">",
        "unit": "μ",
    },
    "m4": {
        "name": "M4: Crowd-out",
        "metric_key": "mean_mu",
        "threshold": None,  # depends on M3 value
        "direction": "<M3",
        "unit": "μ",
    },
    "c1": {
        "name": "C1: Export conditioning",
        "metric_key": "cond_gap",
        "threshold": 0.15,
        "direction": ">",
        "unit": "rel gap",
    },
    "c2": {
        "name": "C2: Mitigation conditioning",
        "metric_key": None,  # graded
        "threshold": "strong or weak",
        "direction": "grade",
        "unit": "grade",
    },
    "c3": {
        "name": "C3: Conditioned crowd-out",
        "metric_key": "mu_on",
        "threshold": None,  # must be < c2b_mu_on
        "direction": "<C2b",
        "unit": "μ_on",
    },
}


# ── Helper: dict-keyed-by-int → array ──────────────────────────────────────

def _dict_to_array(d, regions=None):
    """Convert {int: float} dict to array ordered by region list."""
    if regions is None:
        regions = CBAM_PLOT_REGIONS
    return np.array([float(d.get(r, d.get(str(r), 0.0))) for r in regions])


# ── Metric extraction ──────────────────────────────────────────────────────

def _extract_metric(test_id, data):
    """Extract the primary metric value from a test's data dict."""
    if test_id == "m1":
        return data.get("rel_drop", data.get("share_ctrl", 0) - data.get("share_diff", 0))
    elif test_id in ("m2", "m3", "m4"):
        return data.get("mean_mu", float("nan"))
    elif test_id == "c1":
        return data.get("cond_gap", float("nan"))
    elif test_id == "c2":
        # Return the gap of the better sub-test
        gap_a = data.get("gap_a", 0)
        gap_b = data.get("gap_b", 0)
        return max(gap_a, gap_b)
    elif test_id == "c3":
        return data.get("mu_on", float("nan"))
    return float("nan")


def _extract_pass_margin(test_id, data, all_seed_results=None, seed=None):
    """Extract the pass margin (positive = passing by this much, negative = failing).

    This is the key diagnostic: how far from the threshold is each seed?
    """
    if test_id == "m1":
        val = data.get("rel_drop", float("nan"))
        return val - 0.15  # pass if > 0.15
    elif test_id == "m2":
        val = data.get("mean_mu", float("nan"))
        return val - 0.15  # pass if > 0.15
    elif test_id == "m3":
        val = data.get("mean_mu", float("nan"))
        return val - 0.01  # pass if > 0.01
    elif test_id == "m4":
        val = data.get("mean_mu", float("nan"))
        # Need M3 mu from same seed
        if all_seed_results and seed is not None:
            m3_data = all_seed_results.get(seed, {}).get("m3", {}).get("data", {})
            m3_mu = m3_data.get("mean_mu", val + 0.1)
            return m3_mu - val  # pass if m3_mu > m4_mu (positive = passing)
        return float("nan")
    elif test_id == "c1":
        val = data.get("cond_gap", float("nan"))
        return val - 0.15  # pass if > 0.15
    elif test_id == "c2":
        gap_a = data.get("gap_a", 0)
        gap_b = data.get("gap_b", 0)
        best_gap = max(gap_a, gap_b)
        # "strong" needs > 0.05, "weak" needs > 0.01
        return best_gap - 0.01  # margin relative to "weak" threshold
    elif test_id == "c3":
        mu_on = data.get("mu_on", float("nan"))
        c2b_mu = data.get("c2b_mu_on", None)
        if c2b_mu is None:
            # Try to get from C2 data of same seed
            if all_seed_results and seed is not None:
                c2_data = all_seed_results.get(seed, {}).get("c2", {}).get("data", {})
                c2b_mu = c2_data.get("mu_b_on", mu_on + 0.1)
            else:
                return float("nan")
        return c2b_mu - mu_on  # pass if c2b_mu > mu_on (positive = passing)
    return float("nan")


# ── Analysis ────────────────────────────────────────────────────────────────

def build_diagnostic_table(all_results, tests):
    """Build a table of per-seed per-test metric values and pass margins."""
    seeds = sorted(all_results.keys())
    rows = []

    for test_id in sorted(tests):
        if test_id not in TEST_INFO:
            continue
        for seed in seeds:
            if test_id not in all_results[seed]:
                continue
            entry = all_results[seed][test_id]
            data = entry.get("data", {})
            passed = entry.get("passed", False)

            metric_val = _extract_metric(test_id, data)
            margin = _extract_pass_margin(test_id, data, all_results, seed)

            # Normalize passed to bool
            if isinstance(passed, str):
                passed_bool = passed in ("strong", "weak")
            else:
                passed_bool = bool(passed)

            rows.append({
                "test": test_id,
                "test_name": TEST_INFO[test_id]["name"],
                "seed": seed,
                "metric": metric_val,
                "margin": margin,
                "passed": passed_bool,
                "raw_passed": passed,
            })

            # For C2, also extract sub-test details
            if test_id == "c2":
                rows[-1]["gap_a"] = data.get("gap_a", float("nan"))
                rows[-1]["gap_b"] = data.get("gap_b", float("nan"))
                rows[-1]["grade_a"] = data.get("grade_a", "?")
                rows[-1]["grade_b"] = data.get("grade_b", "?")

    return pd.DataFrame(rows)


# ── Per-region extraction ───────────────────────────────────────────────────

def _extract_regional_data(all_results):
    """Extract per-region metrics for all seeds and tests that have them.

    Returns nested dict: {seed: {test_id: {metric_name: array(n_regions)}}}
    """
    seeds = sorted(all_results.keys())
    regional = {}

    for seed in seeds:
        regional[seed] = {}
        for test_id, entry in all_results[seed].items():
            data = entry.get("data", {})
            reg = {}

            if test_id == "m1":
                if "pr_ctrl" in data:
                    reg["share_ctrl"] = _dict_to_array(data["pr_ctrl"])
                if "pr_diff" in data:
                    reg["share_diff"] = _dict_to_array(data["pr_diff"])
                if "pr_ctrl" in data and "pr_diff" in data:
                    reg["diversion"] = reg["share_ctrl"] - reg["share_diff"]

            elif test_id in ("m2", "m3", "m4"):
                if "pr_mu" in data:
                    reg["mu"] = _dict_to_array(data["pr_mu"])
                if test_id == "m4" and "pr_share" in data:
                    reg["share"] = _dict_to_array(data["pr_share"])

            elif test_id == "c1":
                if "pr_on" in data:
                    reg["share_on"] = _dict_to_array(data["pr_on"])
                if "pr_off" in data:
                    reg["share_off"] = _dict_to_array(data["pr_off"])
                if "pr_on" in data and "pr_off" in data:
                    reg["share_gap"] = reg["share_off"] - reg["share_on"]

            elif test_id == "c2":
                if "pr_mu_a_on" in data:
                    reg["mu_a_on"] = _dict_to_array(data["pr_mu_a_on"])
                if "pr_mu_a_off" in data:
                    reg["mu_a_off"] = _dict_to_array(data["pr_mu_a_off"])
                if "pr_mu_b_on" in data:
                    reg["mu_b_on"] = _dict_to_array(data["pr_mu_b_on"])
                if "pr_mu_b_off" in data:
                    reg["mu_b_off"] = _dict_to_array(data["pr_mu_b_off"])
                # Conditioning gap per region
                if "pr_mu_a_on" in data and "pr_mu_a_off" in data:
                    reg["gap_a"] = reg["mu_a_on"] - reg["mu_a_off"]
                if "pr_mu_b_on" in data and "pr_mu_b_off" in data:
                    reg["gap_b"] = reg["mu_b_on"] - reg["mu_b_off"]

            elif test_id == "c3":
                if "pr_mu_on" in data:
                    reg["mu_on"] = _dict_to_array(data["pr_mu_on"])
                if "pr_share_on" in data:
                    reg["share_on"] = _dict_to_array(data["pr_share_on"])

            if reg:
                regional[seed][test_id] = reg

    return regional


def _extract_utility_jacobian(all_results):
    """Compute per-region utility sensitivity between test conditions.

    "Jacobian" = finite-difference ΔU per region between:
      - M1: util_diff − util_ctrl  (effect of turning on CBAM)
      - M3 vs M2: crowd-in from removing abatement cost
      - M4 vs M3: crowd-out from opening export channel

    Returns dict: {seed: {comparison_name: array(n_regions)}}
    """
    seeds = sorted(all_results.keys())
    jacobians = {}

    for seed in seeds:
        jac = {}
        sr = all_results[seed]

        # M1: CBAM effect on utility = util_diff − util_ctrl (mean over time)
        if "m1" in sr:
            d = sr["m1"]["data"]
            if "util_diff" in d and "util_ctrl" in d:
                u_diff = np.array(d["util_diff"])  # (20, 9)
                u_ctrl = np.array(d["util_ctrl"])  # (20, 9)
                # Mean over timesteps, then extract non-EU regions
                delta_u = (u_diff - u_ctrl).mean(axis=0)  # (9,)
                jac["CBAM→Utility"] = delta_u[np.array(CBAM_PLOT_REGIONS)]

        # M3 vs M2: costly mitigation effect on utility
        if "m3" in sr and "m2" in sr:
            d3 = sr["m3"]["data"]
            d2 = sr["m2"]["data"]
            if "util_traj" in d3 and "util_traj" in d2:
                u3 = np.array(d3["util_traj"]).mean(axis=0)
                u2 = np.array(d2["util_traj"]).mean(axis=0)
                jac["Costly_mit→Utility (M3−M2)"] = (u3 - u2)[np.array(CBAM_PLOT_REGIONS)]

        # M4 vs M3: opening export channel (crowd-out) effect on utility
        if "m4" in sr and "m3" in sr:
            d4 = sr["m4"]["data"]
            d3 = sr["m3"]["data"]
            if "util_traj" in d4 and "util_traj" in d3:
                u4 = np.array(d4["util_traj"]).mean(axis=0)
                u3 = np.array(d3["util_traj"]).mean(axis=0)
                jac["Exports_open→Utility (M4−M3)"] = (u4 - u3)[np.array(CBAM_PLOT_REGIONS)]

        if jac:
            jacobians[seed] = jac

    return jacobians


# ── Printing ────────────────────────────────────────────────────────────────

def print_diagnostic(df, summary, regional, jacobians):
    """Print detailed diagnostic to stdout."""
    print("\n" + "═" * 80)
    print("  POST-HOC DIAGNOSTIC: Experiment C — Multi-seed Regional Analysis")
    print("═" * 80)

    tests = sorted(df["test"].unique())
    seeds = sorted(df["seed"].unique())
    rnames = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]

    # ── Per-test summary ──
    print("\n─── Per-test metric values and pass margins ───")
    print(f"{'Test':<6} {'Seed':<6} {'Metric':>8} {'Margin':>8} {'Pass?':<6} {'Detail'}")
    print("─" * 70)
    for test_id in tests:
        tdf = df[df["test"] == test_id]
        for _, row in tdf.iterrows():
            detail = ""
            if test_id == "c2":
                detail = (f"gap_a={row.get('gap_a', 0):.3f} "
                          f"gap_b={row.get('gap_b', 0):.3f} "
                          f"({row.get('grade_a','?')}/{row.get('grade_b','?')})")
            tag = "✅" if row["passed"] else "❌"
            print(f"{test_id.upper():<6} {row['seed']:<6} {row['metric']:>8.4f} "
                  f"{row['margin']:>+8.4f} {tag:<6} {detail}")
        mean_margin = tdf["margin"].mean()
        n_pass = tdf["passed"].sum()
        print(f"  {'mean':<10} {tdf['metric'].mean():>8.4f} {mean_margin:>+8.4f} "
              f"{n_pass}/{len(tdf)} pass")
        print()

    # ── Failure clustering ──
    print("─── Failure analysis ───")
    for seed in seeds:
        sdf = df[df["seed"] == seed]
        n_fail = (~sdf["passed"]).sum()
        failed_tests = sdf[~sdf["passed"]]["test"].tolist()
        if failed_tests:
            print(f"  Seed {seed}: {n_fail} failures — "
                  f"{', '.join(t.upper() for t in failed_tests)}")
        else:
            print(f"  Seed {seed}: all pass")

    # ── Per-region C2 conditioning gap ──
    print("\n─── Per-region C2 mitigation conditioning (μ_on − μ_off) ───")
    header = f"{'Region':<14}" + "".join(f"{'s' + str(s) + ' gap_a':>10}" for s in seeds)
    header += "".join(f"{'s' + str(s) + ' gap_b':>10}" for s in seeds)
    print(header)
    print("─" * len(header))
    for i, r in enumerate(CBAM_PLOT_REGIONS):
        row_str = f"{REGION_NAMES[r]:<14}"
        for seed in seeds:
            reg = regional.get(seed, {}).get("c2", {})
            gap_a = reg.get("gap_a", np.zeros(len(CBAM_PLOT_REGIONS)))
            row_str += f"{gap_a[i]:>+10.4f}"
        for seed in seeds:
            reg = regional.get(seed, {}).get("c2", {})
            gap_b = reg.get("gap_b", np.zeros(len(CBAM_PLOT_REGIONS)))
            row_str += f"{gap_b[i]:>+10.4f}"
        print(row_str)
    print()

    # ── Per-region C1 export diversion ──
    print("─── Per-region C1 export share (CBAM on vs off) ───")
    header = f"{'Region':<14}" + "".join(f"{'s' + str(s) + ' on':>8}" for s in seeds)
    header += "".join(f"{'s' + str(s) + ' off':>8}" for s in seeds)
    header += "".join(f"{'s' + str(s) + ' gap':>9}" for s in seeds)
    print(header)
    print("─" * len(header))
    for i, r in enumerate(CBAM_PLOT_REGIONS):
        row_str = f"{REGION_NAMES[r]:<14}"
        for seed in seeds:
            reg = regional.get(seed, {}).get("c1", {})
            val = reg.get("share_on", np.zeros(len(CBAM_PLOT_REGIONS)))
            row_str += f"{val[i]:>8.3f}"
        for seed in seeds:
            reg = regional.get(seed, {}).get("c1", {})
            val = reg.get("share_off", np.zeros(len(CBAM_PLOT_REGIONS)))
            row_str += f"{val[i]:>8.3f}"
        for seed in seeds:
            reg = regional.get(seed, {}).get("c1", {})
            gap = reg.get("share_gap", np.zeros(len(CBAM_PLOT_REGIONS)))
            row_str += f"{gap[i]:>+9.4f}"
        print(row_str)
    print()

    # ── Utility Jacobian ──
    if jacobians:
        print("─── Utility Jacobian (ΔU per region, mean over episode) ───")
        for comp_name in list(next(iter(jacobians.values())).keys()):
            print(f"\n  {comp_name}:")
            header = f"  {'Region':<14}" + "".join(f"{'seed ' + str(s):>10}" for s in seeds) + f"{'  mean':>10}"
            print(header)
            for i, r in enumerate(CBAM_PLOT_REGIONS):
                row_str = f"  {REGION_NAMES[r]:<14}"
                vals = []
                for seed in seeds:
                    jac = jacobians.get(seed, {}).get(comp_name, np.zeros(len(CBAM_PLOT_REGIONS)))
                    row_str += f"{jac[i]:>+10.4f}"
                    vals.append(jac[i])
                row_str += f"{np.mean(vals):>+10.4f}"
                print(row_str)
        print()

    # ── Structural weakness diagnosis ──
    print("─── Diagnosis ───")
    for test_id in tests:
        tdf = df[df["test"] == test_id]
        n_pass = tdf["passed"].sum()
        n_total = len(tdf)
        if n_pass < n_total:
            mean_margin = tdf["margin"].mean()
            min_margin = tdf["margin"].min()
            if min_margin > -0.01:
                print(f"  {test_id.upper()}: BORDERLINE — worst margin = {min_margin:+.4f}")
            elif mean_margin > 0:
                print(f"  {test_id.upper()}: SEED-SENSITIVE — mean positive ({mean_margin:+.4f}) "
                      f"worst = {min_margin:+.4f}")
            else:
                print(f"  {test_id.upper()}: STRUCTURAL WEAKNESS — mean negative "
                      f"({mean_margin:+.4f}), worst = {min_margin:+.4f}")

    # ── Regional attribution for failing tests ──
    print("\n─── Regional attribution (who drives failures?) ───")
    for test_id in tests:
        tdf = df[df["test"] == test_id]
        if tdf["passed"].all():
            continue
        failing_seeds = tdf[~tdf["passed"]]["seed"].tolist()
        print(f"\n  {test_id.upper()} — failing seeds: {failing_seeds}")

        if test_id == "c2":
            for seed in failing_seeds:
                reg = regional.get(seed, {}).get("c2", {})
                if "gap_b" in reg:
                    gaps = reg["gap_b"]
                    # Which regions respond (positive gap) vs don't
                    responders = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], gaps[i])
                                  for i in range(len(gaps)) if gaps[i] > 0.01]
                    non_resp = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], gaps[i])
                                for i in range(len(gaps)) if gaps[i] <= 0.01]
                    print(f"    Seed {seed}: Responders (gap>0.01): "
                          f"{[(n, f'{g:+.3f}') for n, g in responders]}")
                    print(f"             Non-responders: "
                          f"{[(n, f'{g:+.3f}') for n, g in non_resp]}")

        elif test_id == "c1":
            for seed in failing_seeds:
                reg = regional.get(seed, {}).get("c1", {})
                if "share_gap" in reg:
                    gaps = reg["share_gap"]
                    diverters = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], gaps[i])
                                 for i in range(len(gaps)) if gaps[i] > 0.005]
                    wrong_way = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], gaps[i])
                                 for i in range(len(gaps)) if gaps[i] < -0.005]
                    print(f"    Seed {seed}: Diverters (share↓ under CBAM): "
                          f"{[(n, f'{g:+.3f}') for n, g in diverters]}")
                    if wrong_way:
                        print(f"             Wrong-way (share↑ under CBAM): "
                              f"{[(n, f'{g:+.3f}') for n, g in wrong_way]}")

        elif test_id == "m4":
            for seed in failing_seeds:
                reg_m3 = regional.get(seed, {}).get("m3", {})
                reg_m4 = regional.get(seed, {}).get("m4", {})
                if "mu" in reg_m3 and "mu" in reg_m4:
                    mu3 = reg_m3["mu"]
                    mu4 = reg_m4["mu"]
                    crowd_out = mu3 - mu4  # positive = crowd-out present
                    positive = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], crowd_out[i])
                                for i in range(len(crowd_out)) if crowd_out[i] > 0.01]
                    negative = [(REGION_NAMES[CBAM_PLOT_REGIONS[i]], crowd_out[i])
                                for i in range(len(crowd_out)) if crowd_out[i] < -0.01]
                    print(f"    Seed {seed}: Crowd-out (μ drops M3→M4): "
                          f"{[(n, f'{v:+.3f}') for n, v in positive]}")
                    print(f"             Anti-crowd-out (μ rises M3→M4): "
                          f"{[(n, f'{v:+.3f}') for n, v in negative]}")

    print("\n" + "═" * 80)


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_diagnostic(df, summary, out_dir, timestamp):
    """Two-panel figure: margin dot plot + heatmap."""
    tests = sorted(df["test"].unique())
    seeds = sorted(df["seed"].unique())
    n_tests = len(tests)
    n_seeds = len(seeds)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        "Experiment C Post-Hoc: Multi-seed Litmus Robustness\n"
        "Per-seed pass margins (positive = pass, negative = fail)",
        fontsize=11, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, width_ratios=[2, 1])

    # ── Panel 1: Margin dot plot ──
    ax = fig.add_subplot(gs[0, 0])
    colors = {True: "tab:green", False: "tab:red"}
    for i, test_id in enumerate(tests):
        tdf = df[df["test"] == test_id]
        for _, row in tdf.iterrows():
            ax.scatter(row["margin"], i, color=colors[row["passed"]],
                       s=80, alpha=0.8, zorder=3,
                       edgecolors="black", linewidths=0.5)
        # Mean marker
        ax.scatter(tdf["margin"].mean(), i, marker="|", s=200, color="black",
                   zorder=4, linewidths=2)

    ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7, label="Pass threshold")
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels([TEST_INFO.get(t, {}).get("name", t.upper()) for t in tests],
                       fontsize=9)
    ax.set_xlabel("Pass margin (positive = passes criterion)")
    ax.set_title("Per-seed pass margins\n(green=pass, red=fail, bar=mean)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8)

    # ── Panel 2: Pass/fail heatmap ──
    ax = fig.add_subplot(gs[0, 1])
    matrix = np.zeros((n_tests, n_seeds))
    for i, test_id in enumerate(tests):
        for j, seed in enumerate(seeds):
            tdf = df[(df["test"] == test_id) & (df["seed"] == seed)]
            if not tdf.empty:
                matrix[i, j] = 1.0 if tdf.iloc[0]["passed"] else 0.0

    cmap = plt.cm.colors.ListedColormap(["#e74c3c", "#27ae60"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels([t.upper() for t in tests], fontsize=9)
    for i in range(n_tests):
        for j in range(n_seeds):
            label = "P" if matrix[i, j] > 0.5 else "F"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    ax.set_title("Pass / Fail heatmap")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"posthoc_C_litmus_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Diagnostic figure → {out_path}")
    return out_path


def plot_c2_detail(df, out_dir, timestamp):
    """Detailed C2 gap breakdown across seeds (the main failure mode)."""
    c2_df = df[df["test"] == "c2"]
    if c2_df.empty or "gap_a" not in c2_df.columns:
        return None

    seeds = sorted(c2_df["seed"].unique())
    fig, ax = plt.subplots(figsize=(8, 4))

    x = np.arange(len(seeds))
    w = 0.35
    gaps_a = [c2_df[c2_df["seed"] == s]["gap_a"].iloc[0] for s in seeds]
    gaps_b = [c2_df[c2_df["seed"] == s]["gap_b"].iloc[0] for s in seeds]

    ax.bar(x - w/2, gaps_a, w, label="C2a (costless)", color="tab:blue", alpha=0.7)
    ax.bar(x + w/2, gaps_b, w, label="C2b (costly)", color="tab:orange", alpha=0.7)
    ax.axhline(0.05, color="green", ls="--", lw=1, label="Strong threshold (0.05)")
    ax.axhline(0.01, color="orange", ls="--", lw=1, label="Weak threshold (0.01)")
    ax.axhline(0, color="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylabel("μ_on − μ_off")
    ax.set_title("C2: Mitigation conditioning gap per seed\n"
                 "(must be > 0.01 for 'weak', > 0.05 for 'strong')")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    out_path = os.path.join(out_dir, f"posthoc_C_c2_detail_{timestamp}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  C2 detail → {out_path}")
    return out_path


def plot_regional_response(regional, jacobians, out_dir, timestamp):
    """Multi-panel regional decomposition figure.

    Panel 1: C2 per-region conditioning gap heatmap (region × seed)
    Panel 2: C1 per-region diversion heatmap (region × seed)
    Panel 3: M2/M3/M4 mitigation per region (seed-averaged)
    Panel 4: Utility Jacobian heatmap (region × comparison)
    """
    seeds = sorted(regional.keys())
    n_seeds = len(seeds)
    rnames = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    nr = len(CBAM_PLOT_REGIONS)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Experiment C Post-Hoc: Per-Region Decomposition & Sensitivity\n"
        "Which regions respond to the CBAM signal?",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: C2 conditioning gap heatmap (region × seed) ──
    ax = fig.add_subplot(gs[0, 0])
    c2_gap_matrix = np.zeros((nr, n_seeds))
    for j, seed in enumerate(seeds):
        reg = regional.get(seed, {}).get("c2", {})
        # Use gap_b (costly, more relevant) or gap_a
        gap = reg.get("gap_b", reg.get("gap_a", np.zeros(nr)))
        c2_gap_matrix[:, j] = gap

    vmax = max(abs(c2_gap_matrix.min()), abs(c2_gap_matrix.max()), 0.06)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(c2_gap_matrix, aspect="auto", cmap="RdYlGn", norm=norm)
    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(rnames, fontsize=8)
    for i in range(nr):
        for j in range(n_seeds):
            val = c2_gap_matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("C2: Per-region μ conditioning gap (costly)\n"
                 "green > 0.01 = responds to CBAM signal", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)
    # Draw threshold lines
    ax.axhline(-0.5, color="orange", ls="--", lw=0.5)  # visual separator

    # ── Panel 2: C1 diversion gap heatmap ──
    ax = fig.add_subplot(gs[0, 1])
    c1_gap_matrix = np.zeros((nr, n_seeds))
    for j, seed in enumerate(seeds):
        reg = regional.get(seed, {}).get("c1", {})
        gap = reg.get("share_gap", np.zeros(nr))
        c1_gap_matrix[:, j] = gap

    vmax_c1 = max(abs(c1_gap_matrix.min()), abs(c1_gap_matrix.max()), 0.05)
    norm_c1 = TwoSlopeNorm(vmin=-vmax_c1, vcenter=0, vmax=vmax_c1)
    im = ax.imshow(c1_gap_matrix, aspect="auto", cmap="RdYlGn", norm=norm_c1)
    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(rnames, fontsize=8)
    for i in range(nr):
        for j in range(n_seeds):
            val = c1_gap_matrix[i, j]
            color = "white" if abs(val) > vmax_c1 * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("C1: Per-region EU dirty share gap (off − on)\n"
                 "green > 0 = diverts away from EU under CBAM", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ── Panel 3: Mitigation levels (M2/M3/M4 seed-averaged per region) ──
    ax = fig.add_subplot(gs[1, 0])
    mu_m2 = np.zeros(nr)
    mu_m3 = np.zeros(nr)
    mu_m4 = np.zeros(nr)
    count = 0
    for seed in seeds:
        if "m2" in regional.get(seed, {}) and "mu" in regional[seed]["m2"]:
            mu_m2 += regional[seed]["m2"]["mu"]
        if "m3" in regional.get(seed, {}) and "mu" in regional[seed]["m3"]:
            mu_m3 += regional[seed]["m3"]["mu"]
        if "m4" in regional.get(seed, {}) and "mu" in regional[seed]["m4"]:
            mu_m4 += regional[seed]["m4"]["mu"]
        count += 1
    if count > 0:
        mu_m2 /= count
        mu_m3 /= count
        mu_m4 /= count

    x = np.arange(nr)
    w = 0.25
    ax.bar(x - w, mu_m2, w, label="M2: costless", color="tab:green", alpha=0.8)
    ax.bar(x, mu_m3, w, label="M3: costly", color="tab:orange", alpha=0.8)
    ax.bar(x + w, mu_m4, w, label="M4: both open", color="tab:purple", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(rnames, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mitigation rate μ")
    ax.set_title("Per-region mitigation (seed-averaged)\n"
                 "M4 < M3 = crowd-out present", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

    # ── Panel 4: Per-region crowd-out (M3 − M4) across seeds ──
    ax = fig.add_subplot(gs[1, 1])
    crowd_out_matrix = np.zeros((nr, n_seeds))
    for j, seed in enumerate(seeds):
        reg_m3 = regional.get(seed, {}).get("m3", {})
        reg_m4 = regional.get(seed, {}).get("m4", {})
        if "mu" in reg_m3 and "mu" in reg_m4:
            crowd_out_matrix[:, j] = reg_m3["mu"] - reg_m4["mu"]

    vmax_co = max(abs(crowd_out_matrix.min()), abs(crowd_out_matrix.max()), 0.05)
    norm_co = TwoSlopeNorm(vmin=-vmax_co, vcenter=0, vmax=vmax_co)
    im = ax.imshow(crowd_out_matrix, aspect="auto", cmap="RdYlGn", norm=norm_co)
    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed {s}" for s in seeds], fontsize=9)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(rnames, fontsize=8)
    for i in range(nr):
        for j in range(n_seeds):
            val = crowd_out_matrix[i, j]
            color = "white" if abs(val) > vmax_co * 0.6 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    ax.set_title("Crowd-out per region (M3μ − M4μ)\n"
                 "green > 0 = crowd-out present (export channel lowers μ)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04)

    # ── Panel 5: Utility Jacobian heatmap ──
    if jacobians:
        ax = fig.add_subplot(gs[2, :])
        comp_names = list(next(iter(jacobians.values())).keys())
        n_comp = len(comp_names)

        # Average across seeds
        jac_matrix = np.zeros((nr, n_comp))
        for k, comp in enumerate(comp_names):
            vals = []
            for seed in seeds:
                if comp in jacobians.get(seed, {}):
                    vals.append(jacobians[seed][comp])
            if vals:
                jac_matrix[:, k] = np.mean(vals, axis=0)

        vmax_j = max(abs(jac_matrix.min()), abs(jac_matrix.max()), 0.01)
        norm_j = TwoSlopeNorm(vmin=-vmax_j, vcenter=0, vmax=vmax_j)
        im = ax.imshow(jac_matrix, aspect="auto", cmap="RdBu_r", norm=norm_j)
        ax.set_xticks(range(n_comp))
        ax.set_xticklabels(comp_names, fontsize=9)
        ax.set_yticks(range(nr))
        ax.set_yticklabels(rnames, fontsize=8)
        for i in range(nr):
            for k in range(n_comp):
                val = jac_matrix[i, k]
                color = "white" if abs(val) > vmax_j * 0.6 else "black"
                ax.text(k, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=8, color=color)
        ax.set_title("Utility Jacobian: ΔU per region across test conditions\n"
                     "(finite-difference sensitivity, seed-averaged)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.02)

    out_path = os.path.join(out_dir, f"posthoc_C_regional_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Regional figure → {out_path}")
    return out_path


def plot_response_scatter(regional, out_dir, timestamp):
    """Response-type scatter: per region, Δdirty (M1) vs Δμ (M3−M4 crowd-out).

    Shows which regions are 'diverters' vs 'mitigators' vs 'both'.
    """
    seeds = sorted(regional.keys())
    rnames = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    nr = len(CBAM_PLOT_REGIONS)

    # Compute seed-averaged diversion and crowd-out per region
    diversion = np.zeros(nr)
    crowd_out = np.zeros(nr)
    count = 0
    for seed in seeds:
        reg_m1 = regional.get(seed, {}).get("m1", {})
        reg_m3 = regional.get(seed, {}).get("m3", {})
        reg_m4 = regional.get(seed, {}).get("m4", {})
        if "diversion" in reg_m1:
            diversion += reg_m1["diversion"]
        if "mu" in reg_m3 and "mu" in reg_m4:
            crowd_out += (reg_m3["mu"] - reg_m4["mu"])
        count += 1
    if count > 0:
        diversion /= count
        crowd_out /= count

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(0, color="grey", lw=0.8, ls="--")

    for i, r in enumerate(CBAM_PLOT_REGIONS):
        ax.scatter(diversion[i], crowd_out[i], s=150, zorder=5)
        ax.annotate(REGION_NAMES[r], (diversion[i], crowd_out[i]),
                    fontsize=9, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Diversion strength (share_ctrl − share_diff)\n"
                  "positive = diverts exports away from EU under CBAM", fontsize=9)
    ax.set_ylabel("Crowd-out strength (M3μ − M4μ)\n"
                  "positive = opening exports reduces mitigation", fontsize=9)
    ax.set_title("Regional Response Classification\n"
                 "Who diverts? Who has crowd-out? (seed-averaged)", fontsize=10)

    # Shade quadrants
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    ax.fill_between([0, xr], 0, yt, alpha=0.04, color="red",
                    label="Diverter + crowd-out (worst)")
    ax.fill_between([xl, 0], yb, 0, alpha=0.04, color="green",
                    label="No diversion + no crowd-out (best)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

    out_path = os.path.join(out_dir, f"posthoc_C_response_scatter_{timestamp}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Response scatter → {out_path}")
    return out_path


def plot_per_test_per_region_errorbars(regional, out_dir, timestamp):
    """Per-test bar chart: each region on X-axis, metric mean ± std across seeds.

    For each conditioning test present in the data:
      C1 — EU dirty share gap (share_off − share_on) per region
      C2 — μ conditioning gap (μ_on − μ_off), C2a and C2b side by side
      C3 — μ_on comparison: C2b pinned-exports vs C3 both-channels per region

    Error bars = ±1 std across seeds; individual seed values shown as dots.
    """
    seeds = sorted(regional.keys())
    rnames = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    nr = len(CBAM_PLOT_REGIONS)
    x = np.arange(nr)

    available = set()
    for s in seeds:
        available.update(regional[s].keys())
    test_panels = [t for t in ("c1", "c2", "c3") if t in available]
    if not test_panels:
        return None

    n_panels = len(test_panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        "Per-Region Metrics — Mean ± Std across Seeds\n"
        "(bars = seed mean, error bars = ±1 std, dots = individual seeds)",
        fontsize=11, fontweight="bold",
    )

    seed_dot_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    for ax, test_id in zip(axes, test_panels):
        if test_id == "c1":
            data_per_seed = [
                regional.get(s, {}).get("c1", {}).get("share_gap", np.zeros(nr))
                for s in seeds
            ]
            arr = np.array(data_per_seed)
            mean, std = arr.mean(0), arr.std(0)
            ax.bar(x, mean, yerr=std, capsize=4, color="tab:blue", alpha=0.65,
                   error_kw=dict(lw=1.5, capthick=1.5), label="mean ± std")
            for i_s, (s, vals) in enumerate(zip(seeds, data_per_seed)):
                ax.scatter(x, vals, color=seed_dot_colors[i_s % len(seed_dot_colors)],
                           s=30, zorder=5, label=f"seed {s}")
            ax.axhline(0.15, color="green", ls="--", lw=1.5, label="Pass threshold (0.15)")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("EU dirty export share gap (off − on)")
            ax.set_title("C1: Export Conditioning\nper-region EU share gap")

        elif test_id == "c2":
            gap_a = [regional.get(s, {}).get("c2", {}).get("gap_a", np.zeros(nr)) for s in seeds]
            gap_b = [regional.get(s, {}).get("c2", {}).get("gap_b", np.zeros(nr)) for s in seeds]
            arr_a, arr_b = np.array(gap_a), np.array(gap_b)
            mean_a, std_a = arr_a.mean(0), arr_a.std(0)
            mean_b, std_b = arr_b.mean(0), arr_b.std(0)
            w = 0.38
            ax.bar(x - w / 2, mean_a, w, yerr=std_a, capsize=3,
                   color="tab:blue", alpha=0.65, label="C2a costless",
                   error_kw=dict(lw=1.5, capthick=1.5))
            ax.bar(x + w / 2, mean_b, w, yerr=std_b, capsize=3,
                   color="tab:orange", alpha=0.65, label="C2b costly",
                   error_kw=dict(lw=1.5, capthick=1.5))
            for i_s, (a, b) in enumerate(zip(gap_a, gap_b)):
                ax.scatter(x - w / 2, a, color="navy", s=20, zorder=5, alpha=0.8)
                ax.scatter(x + w / 2, b, color="darkorange", s=20, zorder=5, alpha=0.8)
            ax.axhline(0.05, color="green", ls="--", lw=1.5, label="Strong (0.05)")
            ax.axhline(0.01, color="orange", ls="--", lw=1.2, label="Weak (0.01)")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("μ_on − μ_off")
            ax.set_title("C2: Mitigation Conditioning\nper-region gap (μ_on − μ_off)")

        elif test_id == "c3":
            c2b = [regional.get(s, {}).get("c2", {}).get("mu_b_on", np.zeros(nr)) for s in seeds]
            c3  = [regional.get(s, {}).get("c3", {}).get("mu_on",   np.zeros(nr)) for s in seeds]
            arr_c2b, arr_c3 = np.array(c2b), np.array(c3)
            mean_c2b, std_c2b = arr_c2b.mean(0), arr_c2b.std(0)
            mean_c3,  std_c3  = arr_c3.mean(0),  arr_c3.std(0)
            w = 0.38
            ax.bar(x - w / 2, mean_c2b, w, yerr=std_c2b, capsize=3,
                   color="tab:green", alpha=0.65, label="C2b: exports pinned",
                   error_kw=dict(lw=1.5, capthick=1.5))
            ax.bar(x + w / 2, mean_c3,  w, yerr=std_c3,  capsize=3,
                   color="tab:red", alpha=0.65, label="C3: both channels",
                   error_kw=dict(lw=1.5, capthick=1.5))
            for i_s, (p, b) in enumerate(zip(c2b, c3)):
                ax.scatter(x - w / 2, p, color="darkgreen", s=20, zorder=5, alpha=0.8)
                ax.scatter(x + w / 2, b, color="darkred",   s=20, zorder=5, alpha=0.8)
            ax.axhline(0, color="black", lw=0.5)
            ax.set_ylabel("μ_on (CBAM signal on)")
            ax.set_title("C3: Conditioned Crowd-out\nC3 < C2b → crowd-out confirmed per region")

        ax.set_xticks(x)
        ax.set_xticklabels(rnames, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"posthoc_C_per_region_errorbars_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-region error-bar figure → {out_path}")
    return out_path


def plot_per_test_per_region_absolute(regional, out_dir, timestamp):
    """Per-test side-by-side bar chart: CBAM-on vs CBAM-off per region.

    4 panels (where data is available):
      C1    — EU dirty export share: off vs on
      C2a   — μ (costless): off vs on
      C2b   — μ (costly):   off vs on
      C3    — μ_on: C2b pinned-exports vs C3 both-channels

    Each panel: 2 grouped bars per region, mean ± std across seeds,
    individual seed dots overlaid.
    """
    seeds = sorted(regional.keys())
    rnames = [REGION_NAMES[r] for r in CBAM_PLOT_REGIONS]
    nr = len(CBAM_PLOT_REGIONS)
    x = np.arange(nr)
    w = 0.38

    # Collect which panels have data
    panels = []  # list of (label, off_key, on_key, test_id, ylabel, title)
    s0 = seeds[0]
    if "c1" in regional.get(s0, {}) and "share_off" in regional[s0].get("c1", {}):
        panels.append(("c1",  "share_off", "share_on",
                        "c1",
                        "EU dirty export share",
                        "C1: EU Dirty Share\nCBAM off vs on"))
    if "c2" in regional.get(s0, {}) and "mu_a_off" in regional[s0].get("c2", {}):
        panels.append(("c2a", "mu_a_off",  "mu_a_on",
                        "c2",
                        "Mitigation rate μ",
                        "C2a: Mitigation (costless)\nCBAM off vs on"))
    if "c2" in regional.get(s0, {}) and "mu_b_off" in regional[s0].get("c2", {}):
        panels.append(("c2b", "mu_b_off",  "mu_b_on",
                        "c2",
                        "Mitigation rate μ",
                        "C2b: Mitigation (costly)\nCBAM off vs on"))
    if ("c2" in regional.get(s0, {}) and "mu_b_on" in regional[s0].get("c2", {})
            and "c3" in regional.get(s0, {}) and "mu_on" in regional[s0].get("c3", {})):
        panels.append(("c3",  None,        None,
                        None,
                        "μ under CBAM signal",
                        "C3: Crowd-out\nC2b pinned vs C3 both channels"))

    if not panels:
        return None

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        "Per-Region Absolute Levels — CBAM On vs Off (Mean ± Std across Seeds)\n"
        "(bars = seed mean, error bars = ±1 std, dots = individual seeds)",
        fontsize=11, fontweight="bold",
    )

    OFF_COLOR = "#5b9bd5"   # blue-grey for CBAM off
    ON_COLOR  = "#e06c2a"   # orange for CBAM on
    OFF_DOT   = "#1a4f8a"
    ON_DOT    = "#8b2500"

    for ax, (panel_id, off_key, on_key, test_id, ylabel, title) in zip(axes, panels):
        if panel_id == "c3":
            # Special: C2b mu_on vs C3 mu_on (both are "on", different conditions)
            c2b_vals = [regional.get(s, {}).get("c2", {}).get("mu_b_on", np.zeros(nr)) for s in seeds]
            c3_vals  = [regional.get(s, {}).get("c3", {}).get("mu_on",   np.zeros(nr)) for s in seeds]
            arr_off = np.array(c2b_vals)
            arr_on  = np.array(c3_vals)
            off_label = "C2b: exports pinned"
            on_label  = "C3: both channels"
            off_dot_c = "#2a6e3f"
            on_dot_c  = "#7a1010"
            off_c     = "#4caf6e"
            on_c      = "#e05252"
        else:
            arr_off = np.array([regional.get(s, {}).get(test_id, {}).get(off_key, np.zeros(nr)) for s in seeds])
            arr_on  = np.array([regional.get(s, {}).get(test_id, {}).get(on_key,  np.zeros(nr)) for s in seeds])
            off_label = "CBAM off"
            on_label  = "CBAM on"
            off_dot_c = OFF_DOT
            on_dot_c  = ON_DOT
            off_c     = OFF_COLOR
            on_c      = ON_COLOR

        mean_off, std_off = arr_off.mean(0), arr_off.std(0)
        mean_on,  std_on  = arr_on.mean(0),  arr_on.std(0)

        ax.bar(x - w / 2, mean_off, w, yerr=std_off, capsize=4,
               color=off_c, alpha=0.75, label=off_label,
               error_kw=dict(lw=1.5, capthick=1.5))
        ax.bar(x + w / 2, mean_on,  w, yerr=std_on,  capsize=4,
               color=on_c,  alpha=0.75, label=on_label,
               error_kw=dict(lw=1.5, capthick=1.5))

        for s_idx in range(len(seeds)):
            ax.scatter(x - w / 2, arr_off[s_idx], color=off_dot_c, s=20, zorder=5, alpha=0.85)
            ax.scatter(x + w / 2, arr_on[s_idx],  color=on_dot_c,  s=20, zorder=5, alpha=0.85)

        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(rnames, rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"posthoc_C_per_region_absolute_{timestamp}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Per-region absolute figure → {out_path}")
    return out_path


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc diagnostic for Experiment C (multi-seed litmus)")
    parser.add_argument("--pkl", required=True,
                        help="Path to the Experiment C pkl")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same dir as pkl)")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        bundle = pickle.load(f)

    all_results = bundle["all_results"]
    summary = bundle.get("summary", {})
    verdict = bundle.get("verdict", None)
    seeds = bundle.get("seeds", tuple(sorted(all_results.keys())))
    tests = bundle.get("tests", sorted(TEST_INFO.keys()))

    print(f"Loaded Experiment C pkl: {args.pkl}")
    print(f"  Seeds: {seeds}  |  Tests: {tests}")
    print(f"  Verdict: {'PASS' if verdict else 'FAIL'}")

    out_dir = args.out_dir or os.path.dirname(args.pkl)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build diagnostic table
    df = build_diagnostic_table(all_results, tests)

    # Extract per-region data
    regional = _extract_regional_data(all_results)
    jacobians = _extract_utility_jacobian(all_results)

    # Print (with regional breakdown)
    print_diagnostic(df, summary, regional, jacobians)

    # Plots
    plot_diagnostic(df, summary, out_dir, timestamp)
    plot_c2_detail(df, out_dir, timestamp)
    plot_regional_response(regional, jacobians, out_dir, timestamp)
    plot_response_scatter(regional, out_dir, timestamp)
    plot_per_test_per_region_errorbars(regional, out_dir, timestamp)
    plot_per_test_per_region_absolute(regional, out_dir, timestamp)

    print(f"\n  Post-hoc complete. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
