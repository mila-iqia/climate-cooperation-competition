"""cbam_merge_C_litmus.py — Merge per-seed Experiment C runs into one bundle.

A 50-seed sweep is run as 50 independent single-seed jobs (see
``cluster/slurm_C_litmus.sbatch``), each producing its own
``plots/cbam_C_litmus_1seeds_*.pkl``.  This script collects those pkls,
re-derives the registry criterion over the *pooled* seed set, writes a merged
pkl with the **same schema as a single multi-seed run**, and renders one
summary figure that stays legible at 50 seeds.

Because the merged pkl is schema-compatible, the existing per-region post-hoc
works on it unchanged::

    python cbam/posthoc/cbam_posthoc_C_litmus.py --pkl <merged.pkl>

Usage (from rice_jax/):
    # Merge every seed under a sweep root (the usual case)
    python cbam/posthoc/cbam_merge_C_litmus.py --runs-root cbam/experiment_results/C_litmus_sweep

    # Or point at explicit pkls / a glob
    python cbam/posthoc/cbam_merge_C_litmus.py path/to/*.pkl

    # Merge and immediately chain the per-region post-hoc
    python cbam/posthoc/cbam_merge_C_litmus.py --runs-root <root> --posthoc
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cloudpickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Patch

# Single source of truth for the pass criterion — imported, not re-implemented,
# so the merged verdict can never drift from the per-run verdict.
from cbam.drivers.cbam_experiment_C_litmus import compute_summary, overall_verdict
from cbam.posthoc.cbam_posthoc_C_litmus import TEST_INFO, build_diagnostic_table

matplotlib.use("Agg")


_RICE_JAX_ROOT = Path(__file__).resolve().parents[2]
if str(_RICE_JAX_ROOT) not in sys.path:
    sys.path.insert(0, str(_RICE_JAX_ROOT))

# ── Status palette ──────────────────────────────────────────────────────────
# Pass/fail is a *status* encoding, so the two colors are reserved and never
# reused for series identity.  They differ in lightness as well as hue, and
# every mark that uses them also carries a glyph or a number, so identity is
# never conveyed by color alone (CVD / print / forced-colors safe).

PASS_C = "#1a7f5a"
FAIL_C = "#c1441e"
GRID_C = "#b0b0b0"
INK = "#222222"
MUTED = "#666666"


# ── Discovery ───────────────────────────────────────────────────────────────


def discover_pkls(runs_root: str | None, patterns: list[str]) -> list[str]:
    """Collect candidate Experiment-C pkls from a sweep root and/or globs."""
    found: list[str] = []
    if runs_root:
        found += glob.glob(
            os.path.join(runs_root, "**", "cbam_C_litmus_*.pkl"), recursive=True
        )
    for pat in patterns:
        found += glob.glob(pat) if any(c in pat for c in "*?[") else [pat]

    # Drop anything we produced ourselves on an earlier merge.
    found = [p for p in found if "_merged_" not in os.path.basename(p)]
    return sorted(set(found))


def load_runs(pkl_paths: list[str], on_duplicate: str) -> tuple[dict, dict]:
    """Load each pkl and pool its per-seed results.

    Returns ``(all_results, provenance)``.  ``all_results`` is keyed by seed,
    exactly as a single multi-seed run's ``all_results`` is.
    """
    all_results: dict = {}
    seed_source: dict = {}
    prov = {
        "pkls": [],
        "timesteps": set(),
        "tests": set(),
        "env_overrides": set(),
        "elapsed_s": 0.0,
        "skipped": [],
        "duplicates": [],
    }

    for path in pkl_paths:
        try:
            with open(path, "rb") as fh:
                bundle = pickle.load(fh)
        except Exception as exc:  # truncated / still-running / not ours
            prov["skipped"].append((path, f"unreadable: {exc}"))
            continue

        if not isinstance(bundle, dict) or "all_results" not in bundle:
            prov["skipped"].append((path, "not an Experiment-C bundle"))
            continue

        prov["pkls"].append(path)
        if bundle.get("timesteps") is not None:
            prov["timesteps"].add(int(bundle["timesteps"]))
        prov["tests"].update(bundle.get("tests", []) or [])
        prov["env_overrides"].add(
            tuple(sorted((bundle.get("env_overrides") or {}).items()))
        )
        prov["elapsed_s"] += float(bundle.get("elapsed_s") or 0.0)

        for seed, seed_results in bundle["all_results"].items():
            seed = int(seed)
            if seed in all_results:
                prev = seed_source[seed]
                if on_duplicate == "error":
                    raise SystemExit(
                        f"Duplicate results for seed {seed}:\n  {prev}\n  {path}\n"
                        f"Re-run with --on-duplicate newest to keep the newer pkl."
                    )
                keep_new = os.path.getmtime(path) > os.path.getmtime(prev)
                prov["duplicates"].append(
                    (seed, prev, path, "kept newer" if keep_new else "kept existing")
                )
                if not keep_new:
                    continue
            all_results[seed] = seed_results
            seed_source[seed] = path

    return all_results, prov


# ── Figure ──────────────────────────────────────────────────────────────────


def plot_merged(all_results: dict, summary: dict, tests: list, out_path: str) -> str:
    """Three-panel merged summary, designed to stay readable at 50 seeds.

    A  pass rate per test        — magnitude, horizontal bars, majority line
    B  pass-margin distribution  — box + one dot per seed, threshold at 0
    C  pass/fail matrix          — tests x seeds, no per-cell text
    """
    tests = [t for t in sorted(tests) if t in summary]
    seeds = sorted(all_results.keys())
    n_tests, n_seeds = len(tests), len(seeds)
    df = build_diagnostic_table(all_results, tests)

    fig = plt.figure(figsize=(16, max(4.2, 1.05 * n_tests + 2.5)))
    verdict = overall_verdict(summary)
    majority = int(np.ceil(n_seeds / 2))
    fig.suptitle(
        f"Experiment C — Multi-seed Litmus Robustness   "
        f"[{'PASS' if verdict else 'FAIL'}]\n"
        f"{n_seeds} seeds pooled   |   criterion: each test passes on "
        f"≥ ceil(N/2) = {majority} seeds",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=0.98,
    )
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=[1.05, 1.35, 1.0],
        wspace=0.32,
        top=0.78,
        bottom=0.22,
    )
    ypos = np.arange(n_tests)[::-1]  # first test on top
    ylabels = [TEST_INFO.get(t, {}).get("name", t.upper()) for t in tests]

    # ── A: pass rate ────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    rates = np.array(
        [summary[t]["n_pass"] / max(1, summary[t]["n_seeds"]) for t in tests]
    )
    passing = np.array([summary[t]["majority_pass"] for t in tests])

    for y, rate, ok, t in zip(ypos, rates, passing, tests):
        ax.barh(
            y,
            rate * 100,
            height=0.55,
            color=PASS_C if ok else FAIL_C,
            hatch=None if ok else "///",
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        # Direct label carries the count, so the bar is never read by color alone.
        ax.text(
            rate * 100 + 1.5,
            y,
            f"{summary[t]['n_pass']}/{summary[t]['n_seeds']}  {'✓' if ok else '✗'}",
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )

    ax.axvline(50, color=MUTED, lw=1.4, ls="--", zorder=2, label="majority (50%)")
    ax.set_xlim(0, 118)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_xlabel("Seeds passing (%)", fontsize=9, color=INK)
    ax.set_title(
        "A  Pass rate per test", fontsize=10, fontweight="bold", loc="left", pad=10
    )
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    _recede(ax, axis="x")

    # ── B: margin distribution ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)  # fixed jitter → figure is reproducible
    for y, t in zip(ypos, tests):
        vals = df.loc[df["test"] == t, "margin"].to_numpy(dtype=float)
        ok = df.loc[df["test"] == t, "passed"].to_numpy(dtype=bool)
        finite = np.isfinite(vals)
        if not finite.any():
            continue
        _hbox(ax, vals[finite], y, height=0.5)
        jitter = rng.uniform(-0.16, 0.16, size=finite.sum())
        ax.scatter(
            vals[finite],
            y + jitter,
            s=18,
            c=[PASS_C if p else FAIL_C for p in ok[finite]],
            marker="o",
            edgecolors="white",
            linewidths=0.5,
            alpha=0.9,
            zorder=4,
        )

    ax.axvline(0, color=INK, lw=1.5, ls="--", zorder=3)
    ax.text(
        0,
        ypos.max() + 0.62,
        " pass threshold",
        fontsize=8,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    ax.set_yticks(ypos)
    ax.set_yticklabels([t.upper() for t in tests], fontsize=9)
    ax.set_ylim(-0.7, ypos.max() + 0.95)
    ax.set_xlabel(
        "Pass margin  (> 0 = passes criterion)   ·   box = quartiles, dot = one seed",
        fontsize=9,
        color=INK,
    )
    ax.set_title(
        "B  Margin distribution across seeds",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    _recede(ax, axis="x")

    # ── C: pass/fail matrix ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    matrix = np.zeros((n_tests, n_seeds))
    for i, t in enumerate(tests):
        passed = set(summary[t]["seeds_passed"])
        for j, s in enumerate(seeds):
            matrix[i, j] = 1.0 if s in passed else 0.0

    ax.imshow(
        matrix,
        cmap=plt.cm.colors.ListedColormap([FAIL_C, PASS_C]),
        aspect="auto",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    # At 50 seeds a tick per seed is unreadable; show a sparse, honest subset.
    step = max(1, n_seeds // 12)
    xt = list(range(0, n_seeds, step))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(seeds[j]) for j in xt], fontsize=8, color=INK)
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels([t.upper() for t in tests], fontsize=9)
    ax.set_xlabel("seed", fontsize=9, color=INK)
    ax.set_title(
        "C  Per-seed pass / fail", fontsize=10, fontweight="bold", loc="left", pad=10
    )
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # One shared status key for panels B and C — the two colors mean the same
    # thing in both, so a single legend is clearer than two.
    fig.legend(
        handles=[
            Patch(facecolor=PASS_C, edgecolor="white", label="seed passes test"),
            Patch(facecolor=FAIL_C, edgecolor="white", label="seed fails test"),
            Patch(
                facecolor=FAIL_C,
                edgecolor="white",
                hatch="///",
                label="test fails the majority criterion (panel A)",
            ),
        ],
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# Friendly labels for training-run series stored under data["train_metrics"].
_CURVE_LABELS = {
    "c1_export_cond": "C1 export conditioning",
    "c2a_costless": "C2a costless μ",
    "c2b_costly": "C2b costly μ",
    "c3_both": "C3 both channels",
    "m1_ctrl": "M1 control (τ=0)",
    "m1_diff": "M1 differential CBAM",
    "m2_costless_mu": "M2 costless μ",
    "m3_costly_mu": "M3 costly μ",
    "m4_both": "M4 both open",
}

_CURVE_COLORS = [
    "#1f4e79",
    "#c1441e",
    "#1a7f5a",
    "#7a5c00",
    "#5b3d8a",
    "#0b6e6e",
]


def collect_train_curves(all_results: dict, tests: list) -> dict[str, dict[str, list]]:
    """Gather per-seed episode-return curves keyed by test → run label.

    Expects each test's ``data["train_metrics"]`` to be ``{label: 1d array}``,
    as written by the litmus drivers after jaxnasium ``train()``.
    """
    out: dict[str, dict[str, list]] = {}
    for test_id in tests:
        by_label: dict[str, list] = {}
        for seed in sorted(all_results.keys()):
            entry = all_results[seed].get(test_id)
            if not entry:
                continue
            metrics = (entry.get("data") or {}).get("train_metrics") or {}
            if not isinstance(metrics, dict):
                continue
            for label, curve in metrics.items():
                arr = np.asarray(curve, dtype=float).ravel()
                if arr.size == 0 or not np.isfinite(arr).any():
                    continue
                by_label.setdefault(label, []).append(arr)
        if by_label:
            out[test_id] = by_label
    return out


def plot_learning_curves(
    all_results: dict,
    tests: list,
    out_path: str,
    *,
    steps_per_iter: int | None = None,
) -> str | None:
    """Mean ± std learning curves across seeds, one panel per litmus test.

    Uses the same ``train_metrics`` that came from the litmus trainings — no
    extra training runs.  Returns ``None`` if no curves are present (e.g. older
    pkls from before metrics were saved).
    """
    curves = collect_train_curves(all_results, tests)
    if not curves:
        return None

    n_panels = len(curves)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.2 * ncols, 3.6 * nrows),
        squeeze=False,
        sharey=False,
    )
    fig.suptitle(
        "Experiment C — Learning curves (same litmus trainings)\n"
        f"mean ± std episode return across {len(all_results)} seeds",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=1.02,
    )

    for ax, (test_id, by_label) in zip(axes.ravel(), curves.items()):
        for i, (label, seed_curves) in enumerate(sorted(by_label.items())):
            # Align to shortest curve if any seed truncated early.
            n = min(len(c) for c in seed_curves)
            stack = np.stack([c[:n] for c in seed_curves], axis=0)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            x = (
                np.arange(n) * steps_per_iter
                if steps_per_iter
                else np.arange(n)
            )
            color = _CURVE_COLORS[i % len(_CURVE_COLORS)]
            ax.plot(
                x,
                mean,
                color=color,
                lw=1.8,
                label=_CURVE_LABELS.get(label, label),
            )
            ax.fill_between(
                x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0
            )

        ax.set_title(TEST_INFO.get(test_id, {}).get("name", test_id.upper()), fontsize=10)
        ax.set_xlabel("env steps" if steps_per_iter else "train iteration", fontsize=9)
        ax.set_ylabel("mean episode return", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="best")
        _recede(ax, axis="both")
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(GRID_C)

    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def _hbox(ax, vals: np.ndarray, y: float, height: float = 0.5) -> None:
    """Horizontal quartile box with 1.5·IQR whiskers.

    Drawn by hand rather than via ``ax.boxplot`` so the mark stays thin and the
    call is stable across matplotlib's boxplot orientation API change.
    """
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    lo = vals[vals >= q1 - 1.5 * iqr].min()
    hi = vals[vals <= q3 + 1.5 * iqr].max()
    h = height / 2

    ax.plot([lo, hi], [y, y], color=MUTED, lw=1.1, zorder=2, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [y - h / 2, y + h / 2], color=MUTED, lw=1.1, zorder=2)
    ax.add_patch(
        plt.Rectangle(
            (q1, y - h),
            q3 - q1,
            2 * h,
            facecolor="white",
            edgecolor=MUTED,
            lw=1.1,
            zorder=2,
        )
    )
    ax.plot([med, med], [y - h, y + h], color=INK, lw=1.8, zorder=3)


def _recede(ax, axis="x"):
    """Push grid and spines behind the data."""
    ax.grid(axis=axis, color=GRID_C, alpha=0.35, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID_C)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    for lbl in ax.get_yticklabels():
        lbl.set_color(INK)


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-seed Experiment C pkls into one bundle + figure"
    )
    parser.add_argument(
        "pkls",
        nargs="*",
        default=[],
        help="Explicit pkl paths or globs (optional if --runs-root is given)",
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Sweep root; searched recursively for cbam_C_litmus_*.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <runs-root>/merged, else ./merged_C_litmus)",
    )
    parser.add_argument(
        "--tests",
        default=None,
        help="Comma-separated test subset (default: union of tests found)",
    )
    parser.add_argument(
        "--expect-seeds",
        type=int,
        default=None,
        metavar="N",
        help="Warn loudly if fewer than N seeds were merged (e.g. 50)",
    )
    parser.add_argument(
        "--on-duplicate",
        choices=["newest", "error"],
        default="newest",
        help="What to do when two pkls carry the same seed (default: newest)",
    )
    parser.add_argument(
        "--posthoc",
        action="store_true",
        help="Chain cbam_posthoc_C_litmus.py on the merged pkl",
    )
    args = parser.parse_args()

    if not args.runs_root and not args.pkls:
        parser.error("Provide --runs-root and/or explicit pkl paths")

    pkl_paths = discover_pkls(args.runs_root, args.pkls)
    if not pkl_paths:
        raise SystemExit("No cbam_C_litmus_*.pkl found — nothing to merge.")
    print(f"  Candidate pkls: {len(pkl_paths)}")

    all_results, prov = load_runs(pkl_paths, args.on_duplicate)
    if not all_results:
        raise SystemExit("No usable Experiment-C bundles among the candidates.")

    for path, why in prov["skipped"]:
        print(f"  [skip] {path}: {why}")
    for seed, prev, new, action in prov["duplicates"]:
        print(f"  [dup]  seed {seed}: {action}\n           {prev}\n           {new}")

    seeds = sorted(all_results.keys())
    tests = (
        [t.strip().lower() for t in args.tests.split(",")]
        if args.tests
        else sorted(prov["tests"] or {k for s in seeds for k in all_results[s]})
    )

    if len(prov["timesteps"]) > 1:
        print(
            f"  [WARN] pkls disagree on total_timesteps: {sorted(prov['timesteps'])} "
            f"— the merged seed set is not headline-comparable."
        )
    if len(prov["env_overrides"]) > 1:
        print(
            f"  [WARN] pkls disagree on env_overrides: "
            f"{[dict(e) for e in prov['env_overrides']]}"
        )
    if args.expect_seeds and len(seeds) < args.expect_seeds:
        missing = sorted(set(range(args.expect_seeds)) - set(seeds))
        print(
            f"  [WARN] merged {len(seeds)}/{args.expect_seeds} seeds. "
            f"Missing (assuming seeds 0..{args.expect_seeds - 1}): {missing}"
        )

    summary = compute_summary(all_results, tests)
    verdict = overall_verdict(summary)

    out_dir = args.out_dir or (
        os.path.join(args.runs_root, "merged") if args.runs_root else "merged_C_litmus"
    )
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"merged_{len(seeds)}seeds_{stamp}"

    timesteps = sorted(prov["timesteps"])[0] if prov["timesteps"] else None
    pkl_path = os.path.join(out_dir, f"cbam_C_litmus_{tag}.pkl")
    with open(pkl_path, "wb") as fh:
        cloudpickle.dump(
            {
                # Same schema as a single multi-seed run, so the existing
                # per-region post-hoc consumes this unchanged.
                "all_results": all_results,
                "summary": summary,
                "verdict": verdict,
                "seeds": tuple(seeds),
                "tests": sorted(tests),
                "timesteps": timesteps,
                "elapsed_s": prov["elapsed_s"],
                "env_overrides": dict(next(iter(prov["env_overrides"]), ())),
                # Merge provenance
                "merged_from": prov["pkls"],
                "n_runs_merged": len(prov["pkls"]),
                "timesteps_all": sorted(prov["timesteps"]),
            },
            fh,
        )
    print(f"\n  Merged pkl  → {pkl_path}")

    png_path = plot_merged(
        all_results, summary, tests, os.path.join(out_dir, f"cbam_C_litmus_{tag}.png")
    )
    print(f"  Merged plot → {png_path}")

    steps_per_iter = None
    if timesteps:
        # Canonical litmus uses CANONICAL_TRAIN_KWARGS num_envs×num_steps per iter;
        # recover from timesteps / curve length when available.
        sample_curves = collect_train_curves(all_results, tests)
        for by_label in sample_curves.values():
            for seed_curves in by_label.values():
                if seed_curves:
                    n_iter = min(len(c) for c in seed_curves)
                    if n_iter > 0:
                        steps_per_iter = max(1, int(round(timesteps / n_iter)))
                    break
            if steps_per_iter:
                break

    curves_path = plot_learning_curves(
        all_results,
        tests,
        os.path.join(out_dir, f"cbam_C_litmus_{tag}_learning_curves.png"),
        steps_per_iter=steps_per_iter,
    )
    if curves_path:
        print(f"  Learning curves → {curves_path}")
    else:
        print(
            "  Learning curves → skipped (no train_metrics in merged pkls; "
            "re-run seeds after the jaxnasium metrics update)"
        )

    # ── Text summary ────────────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    print("  EXPERIMENT C — MERGED MULTI-SEED SUMMARY")
    print(f"{'=' * 66}")
    print(f"  Runs merged: {len(prov['pkls'])}  |  Seeds: {len(seeds)}")
    print(f"  Timesteps:   {timesteps:,}" if timesteps else "  Timesteps:   unknown")
    print(f"  Total compute: {prov['elapsed_s'] / 3600:.1f}h summed over jobs\n")
    for test_id in sorted(summary):
        s = summary[test_id]
        mark = "PASS" if s["majority_pass"] else "FAIL"
        print(
            f"  {test_id.upper():4s}  {s['n_pass']:>3d}/{s['n_seeds']:<3d} seeds pass  "
            f"({100 * s['n_pass'] / max(1, s['n_seeds']):5.1f}%)  {mark}"
        )
    print()
    print(
        f"  Overall criterion (all tests majority-pass): "
        f"{'PASS' if verdict else 'FAIL'}\n"
    )

    if args.posthoc:
        posthoc = os.path.join(
            _RICE_JAX_ROOT, "cbam", "posthoc", "cbam_posthoc_C_litmus.py"
        )
        subprocess.run(
            [sys.executable, posthoc, "--pkl", pkl_path, "--out-dir", out_dir],
            cwd=str(_RICE_JAX_ROOT),
            check=False,
        )


if __name__ == "__main__":
    main()
