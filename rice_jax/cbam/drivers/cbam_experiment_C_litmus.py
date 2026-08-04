"""cbam_experiment_C_litmus.py

Experiment C — Multi-seed litmus freeze (registry entry C_litmus_multiseed).

Runs the full litmus suite (Part I: M1–M4, Part II: C1–C3) across the
canonical seed set and checks whether each test is seed-robust.

Pass criterion (from registry):
  Each test passes its predefined inequality on at least ceil(N_seeds/2) seeds;
  min-seed value also satisfies the inequality for any claim promoted to the
  headline.

Usage (from rice_jax/, rice-jax conda env):
    python cbam/drivers/cbam_experiment_C_litmus.py [--timesteps 2000000]
    python cbam/drivers/cbam_experiment_C_litmus.py --seeds 0,1,2
    python cbam/drivers/cbam_experiment_C_litmus.py --replot <pickle.pkl>
"""

import argparse
import math
import os as _os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import cloudpickle
import jax
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import cbam.drivers.cbam_litmus_conditioning as _cond

# These are the actual test runners from the two litmus scripts.
# They accept a single JAX PRNG key and return (passed, data) tuples.
import cbam.drivers.cbam_litmus_mechanism as _mech
from _experiment_util import get_log_dir, get_output_dir, save_run_config
from cbam.config.canonical_config import (
    CANONICAL_SEEDS,
    CANONICAL_TRAIN_KWARGS,
    SENSITIVITY_PARAMS,
    canonical_train_kwargs,
)
from cbam.config.registry import REGISTRY

matplotlib.use("Agg")
_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

EXPERIMENT_ID = "C_litmus_multiseed"
_REGISTRY = REGISTRY[EXPERIMENT_ID]

OUTPUT_DIR = get_output_dir("plots")
LOG_DIR = get_log_dir("training_logs")

_DEFAULT_TIMESTEPS = canonical_train_kwargs()["total_timesteps"]

# ── Test catalogue ──────────────────────────────────────────────────────────

ALL_MECHANISM_TESTS = ("m1", "m2", "m3", "m4")
ALL_CONDITIONING_TESTS = ("c1", "c2", "c3")
ALL_TESTS = ALL_MECHANISM_TESTS + ALL_CONDITIONING_TESTS
DEFAULT_TESTS = ALL_CONDITIONING_TESTS  # M tests available via --tests m1,m2,m3,m4 but not run by default


# ── Runner ──────────────────────────────────────────────────────────────────


def _run_mechanism_suite(key, tests):
    """Run Part I (mechanism) tests for one seed. Returns {test_id: {passed, data}}."""
    keys = jax.random.split(key, 10)
    results = {}

    if "m1" in tests:
        passed, data = _mech.run_m1(keys[0])
        results["m1"] = {"passed": passed, "data": data}

    if "m2" in tests:
        passed, data = _mech.run_m2(keys[1])
        results["m2"] = {"passed": passed, "data": data}

    m3_mu = 0.05  # fallback
    if "m3" in tests:
        m2_mu = results["m2"]["data"]["mean_mu"] if "m2" in results else None
        passed, data = _mech.run_m3(keys[2], m2_mu=m2_mu)
        results["m3"] = {"passed": passed, "data": data}
        m3_mu = data["mean_mu"]

    if "m4" in tests:
        passed, data = _mech.run_m4(keys[3], m3_mu)
        results["m4"] = {"passed": passed, "data": data}

    return results


def _run_conditioning_suite(key, tests):
    """Run Part II (conditioning) tests for one seed. Returns {test_id: {passed, data}}."""
    keys = jax.random.split(key, 10)
    results = {}

    if "c1" in tests:
        passed, data = _cond.run_c1(keys[0])
        results["c1"] = {"passed": passed, "data": data}

    c2b_mu_on = 0.5  # fallback
    if "c2" in tests:
        grade, data = _cond.run_c2(keys[1])
        results["c2"] = {"passed": grade, "data": data}
        c2b_mu_on = data["mu_b_on"]

    if "c3" in tests:
        passed, data = _cond.run_c3(keys[2], c2b_mu_on)
        results["c3"] = {"passed": passed, "data": data}

    return results


def _test_passed(result_entry):
    """Normalise heterogeneous pass indicators to bool."""
    p = result_entry["passed"]
    if isinstance(p, bool):
        return p
    if isinstance(p, str):
        return p in ("strong", "weak")  # "none" / "mixed" → fail
    return False


def run_all_seeds(seeds, tests, timesteps):
    """Run the requested tests across all seeds. Returns nested dict."""
    # Propagate timesteps to both sub-modules
    _mech.TOTAL_TIMESTEPS = timesteps
    _cond.TOTAL_TIMESTEPS = timesteps

    mech_tests = sorted(set(tests) & set(ALL_MECHANISM_TESTS))
    cond_tests = sorted(set(tests) & set(ALL_CONDITIONING_TESTS))

    all_results = {}  # {seed: {test_id: {passed, data}}}

    for seed in seeds:
        print(f"\n{'═' * 60}")
        print(f"  SEED {seed}")
        print(f"{'═' * 60}")
        key = jax.random.PRNGKey(seed)
        seed_results = {}

        if mech_tests:
            mech_key = jax.random.fold_in(key, 1000)
            seed_results.update(_run_mechanism_suite(mech_key, mech_tests))

        if cond_tests:
            cond_key = jax.random.fold_in(key, 2000)
            seed_results.update(_run_conditioning_suite(cond_key, cond_tests))

        all_results[seed] = seed_results

    return all_results


# ── Summary & criterion checking ───────────────────────────────────────────


def compute_summary(all_results, tests):
    """Aggregate per-seed results into pass rates and seed-robustness verdict."""
    seeds = sorted(all_results.keys())
    n_seeds = len(seeds)
    majority = math.ceil(n_seeds / 2)

    summary = {}
    for test_id in sorted(tests):
        passes = []
        for seed in seeds:
            if test_id in all_results[seed]:
                passes.append(_test_passed(all_results[seed][test_id]))
            else:
                passes.append(False)

        n_pass = sum(passes)
        majority_pass = n_pass >= majority
        all_pass = n_pass == n_seeds

        summary[test_id] = {
            "n_pass": n_pass,
            "n_seeds": n_seeds,
            "seeds_passed": [s for s, p in zip(seeds, passes) if p],
            "seeds_failed": [s for s, p in zip(seeds, passes) if not p],
            "majority_pass": majority_pass,
            "all_pass": all_pass,
        }

    return summary


def overall_verdict(summary):
    """Registry criterion: all tests pass on majority of seeds."""
    return all(s["majority_pass"] for s in summary.values())


# ── Plotting ────────────────────────────────────────────────────────────────


def plot_summary(summary, file_tag):
    """Simple heatmap: tests × seeds → pass/fail."""
    tests = sorted(summary.keys())
    if not tests:
        return None

    first = summary[tests[0]]
    seeds = sorted(first["seeds_passed"] + first["seeds_failed"])
    n_tests = len(tests)
    n_seeds = len(seeds)

    # Build matrix (1=pass, 0=fail)
    matrix = np.zeros((n_tests, n_seeds), dtype=float)
    for i, test_id in enumerate(tests):
        for j, seed in enumerate(seeds):
            if seed in summary[test_id]["seeds_passed"]:
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(figsize=(max(6, n_seeds * 1.2), max(4, n_tests * 0.7)))
    cmap = plt.cm.colors.ListedColormap(["#e74c3c", "#27ae60"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(n_seeds))
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels([t.upper() for t in tests])

    # Annotate cells
    for i in range(n_tests):
        for j in range(n_seeds):
            label = "PASS" if matrix[i, j] > 0.5 else "FAIL"
            color = "white"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    # Title with overall verdict
    verdict = overall_verdict(summary)
    verdict_str = "PASS ✅" if verdict else "FAIL ❌"
    ax.set_title(
        f"Experiment C: Multi-seed Litmus Robustness  [{verdict_str}]\n"
        f"Criterion: each test passes on ≥ ceil(N/2) = {math.ceil(n_seeds / 2)} seeds",
        fontsize=11,
        fontweight="bold",
    )

    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = _os.path.join(OUTPUT_DIR, f"cbam_C_litmus_{file_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")
    return out_path


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment C — Multi-seed litmus freeze (registry C_litmus_multiseed)"
    )
    parser.add_argument("--timesteps", type=int, default=_DEFAULT_TIMESTEPS)
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in CANONICAL_SEEDS),
        help="Comma-separated seed list (default: canonical seeds)",
    )
    parser.add_argument(
        "--tests",
        type=str,
        default=",".join(DEFAULT_TESTS),
        help="Comma-separated test subset (default: c1,c2,c3; M tests available but not default)",
    )
    parser.add_argument(
        "--replot",
        type=str,
        default=None,
        help="Path to existing .pkl — skip training, re-plot only",
    )
    parser.add_argument(
        "--fixed-savings", dest="fixed_savings", action="store_true", default=False
    )
    parser.add_argument("--free-savings", dest="fixed_savings", action="store_false")
    parser.add_argument(
        "--env-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a sensitivity param (from SENSITIVITY_PARAMS). "
        "e.g. --env-override action_window_size=10. "
        "Can be repeated. Logged in config.json for provenance.",
    )
    args = parser.parse_args()

    # Parse env overrides
    env_overrides: dict = {}
    for item in args.env_override:
        if "=" not in item:
            parser.error(f"--env-override requires KEY=VALUE format, got: {item!r}")
        key, val_str = item.split("=", 1)
        key = key.strip()
        if key not in SENSITIVITY_PARAMS:
            parser.error(
                f"--env-override: '{key}' is not a sensitivity param. "
                f"Allowed: {sorted(SENSITIVITY_PARAMS)}"
            )
        # Auto-cast value
        try:
            val = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        env_overrides[key] = val

    if env_overrides:
        print(f"  env-overrides: {env_overrides}")

    seeds = tuple(int(s.strip()) for s in args.seeds.split(","))
    tests = {t.strip().lower() for t in args.tests.split(",")}
    timesteps = args.timesteps

    # Propagate savings preference to sub-modules
    _mech.FIXED_SAVINGS = args.fixed_savings
    _cond.FIXED_SAVINGS = args.fixed_savings

    # Propagate env overrides to sub-modules
    _mech.ENV_OVERRIDES = env_overrides
    _cond.ENV_OVERRIDES = env_overrides

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_tag = f"{len(seeds)}seeds_{timesteps // 1_000_000}M_{timestamp}"

    if args.replot:
        with open(args.replot, "rb") as f:
            data = pickle.load(f)
        all_results = data["all_results"]
        summary = compute_summary(all_results, tests)
        plot_summary(summary, file_tag)
        return

    t0 = time.perf_counter()
    all_results = run_all_seeds(seeds, tests, timesteps)
    elapsed = time.perf_counter() - t0

    # Strip agents from results to keep pkl small
    for seed_results in all_results.values():
        for test_id, res in seed_results.items():
            d = res.get("data", {})
            for k in list(d.keys()):
                if k.startswith("_agent") or k.startswith("csv"):
                    del d[k]

    summary = compute_summary(all_results, tests)
    verdict = overall_verdict(summary)

    # Save pickle
    _os.makedirs(OUTPUT_DIR, exist_ok=True)
    pkl_path = _os.path.join(OUTPUT_DIR, f"cbam_C_litmus_{file_tag}.pkl")
    with open(pkl_path, "wb") as f:
        cloudpickle.dump(
            {
                "all_results": all_results,
                "summary": summary,
                "verdict": verdict,
                "seeds": seeds,
                "tests": sorted(tests),
                "timesteps": timesteps,
                "elapsed_s": elapsed,
                "env_overrides": env_overrides,
            },
            f,
        )
    print(f"\n  Results saved: {pkl_path}")

    save_run_config(
        {
            "script": "cbam/drivers/cbam_experiment_C_litmus.py",
            "experiment_id": EXPERIMENT_ID,
            "timesteps": timesteps,
            "seeds": seeds,
            "tests": sorted(tests),
            "fixed_savings": args.fixed_savings,
            "env_overrides": env_overrides,
            "pkl_path": pkl_path,
            "elapsed_s": elapsed,
        }
    )

    # Plot
    plot_summary(summary, file_tag)

    # Print summary
    print(f"\n{'═' * 60}")
    print(f"  EXPERIMENT C — MULTI-SEED LITMUS SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Seeds: {seeds}  |  Timesteps: {timesteps:,}")
    print(f"  Elapsed: {elapsed / 3600:.1f}h")
    print()

    for test_id, s in summary.items():
        status = "✅" if s["majority_pass"] else "❌"
        print(
            f"  {test_id.upper():4s}  {s['n_pass']}/{s['n_seeds']} seeds pass  "
            f"{'(majority)' if s['majority_pass'] else '(BELOW majority)'}  {status}"
        )
        if s["seeds_failed"]:
            print(f"        failed on seeds: {s['seeds_failed']}")

    print()
    overall = "PASS ✅" if verdict else "FAIL ❌"
    print(f"  Overall criterion (all tests majority-pass): {overall}")
    print()


if __name__ == "__main__":
    main()
