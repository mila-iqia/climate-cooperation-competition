"""run_experiment.py — Experiment runner for CBAM-RICE.

Creates a self-contained experiment folder and runs a validation script at
the requested *depth*, then optionally launches post-hoc analysis.

Folder layout
-------------
experiments/
  <script_stem>_<YYYYMMDD_HHMMSS>/
  ├── config.json          frozen args + metadata
  ├── plots/               all PNGs and PKLs from the run
  │     ├── <stem>_*.png
  │     └── <stem>_*.pkl
  ├── logs/                training CSVs
  │     └── *.csv
  └── posthoc/             scorecard + introspection outputs (depth=full)
        ├── scorecard.md
        ├── per_region.png
        ├── training_curves.png
        └── introspection.png  (only with --save-agents)

Usage (from rice_jax/):
    python run_experiment.py validation/cbam_litmus_mechanism.py \\
        --depth train --timesteps 1000000 --free-savings

    python run_experiment.py validation/cbam_litmus_mechanism.py \\
        --depth visualize --timesteps 1000000 --free-savings

    python run_experiment.py validation/cbam_litmus_mechanism.py \\
        --depth full --timesteps 1000000 --free-savings --save-agents

    # Re-run only posthoc on an existing experiment folder:
    python run_experiment.py --posthoc-only experiments/cbam_litmus_mechanism_20260512_140000

Depth levels
------------
  train     Run the script; training CSVs and PKLs land in the run folder.
            The experiment script already produces its own summary plot, so
            this is usually all you need for a quick check.
  visualize Same as train.  (Reserved for a future --replot-only pass;
            currently identical to train.)
  full      train + run cbam_posthoc_litmus.py (Layer 1 scorecard, per-region
            decomposition, training-curve overlays).  If --save-agents was
            passed, also triggers Layer 2 introspection.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Scripts that emit a litmus-style pkl understood by cbam_posthoc_litmus.py.
# Value is the --<key>-pkl flag name.
_POSTHOC_LITMUS_SCRIPTS = {
    "cbam_litmus_mechanism":    "mech",
    "cbam_litmus_conditioning": "cond",
}
_POSTHOC_LITMUS = os.path.join(_SCRIPT_DIR, "validation", "cbam_posthoc_litmus.py")

# Scripts that emit a 2B-style pkl understood by cbam_posthoc_2b.py.
# Value is the --<key>-pkl flag name.
_POSTHOC_2B_SCRIPTS = {
    "cbam_experiment_2b_tier1": "tier1",
    "cbam_experiment_2b_alloc": "alloc",
}
_POSTHOC_2B = os.path.join(_SCRIPT_DIR, "validation", "cbam_posthoc_2b.py")

# Union of all scripts that have posthoc support
_POSTHOC_SCRIPTS = {**_POSTHOC_LITMUS_SCRIPTS, **_POSTHOC_2B_SCRIPTS}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _script_stem(script_path: str) -> str:
    return os.path.splitext(os.path.basename(script_path))[0]


def _make_run_dir(stem: str, experiments_root: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stem}_{ts}"
    run_dir = os.path.join(experiments_root, name)
    return run_dir, ts


def _setup_dirs(run_dir: str) -> dict[str, str]:
    dirs = {
        "root":    run_dir,
        "plots":   os.path.join(run_dir, "plots"),
        "logs":    os.path.join(run_dir, "logs"),
        "posthoc": os.path.join(run_dir, "posthoc"),
    }
    os.makedirs(dirs["plots"], exist_ok=True)
    os.makedirs(dirs["logs"],  exist_ok=True)
    return dirs


def _write_config(run_dir: str, meta: dict) -> None:
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"  config  → {path}")


def _run_subprocess(cmd: list[str], env: dict, label: str) -> int:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, env=env, cwd=_SCRIPT_DIR)
    return result.returncode


def _find_pkl(plots_dir: str, stem: str) -> str | None:
    """Find the freshest pkl in plots_dir whose name starts with <stem>."""
    pattern = os.path.join(plots_dir, f"{stem}_*.pkl")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _run_posthoc(dirs: dict, stem: str, save_agents: bool) -> int:
    os.makedirs(dirs["posthoc"], exist_ok=True)

    # Find the pkl
    all_pkls = sorted(glob.glob(os.path.join(dirs["plots"], "*.pkl")))
    if not all_pkls:
        print("  [posthoc] No pkl found in plots dir, skipping.")
        return 0
    pkl_path = all_pkls[-1]

    # Route to the right posthoc script
    if stem in _POSTHOC_LITMUS_SCRIPTS:
        pkl_type = _POSTHOC_LITMUS_SCRIPTS[stem]
        cmd = [
            sys.executable, _POSTHOC_LITMUS,
            f"--{pkl_type}-pkl", pkl_path,
            "--out-dir", dirs["posthoc"],
            "--out-report", os.path.join(dirs["posthoc"], "scorecard.md"),
        ]
        if save_agents:
            cmd.append("--introspect")
    elif stem in _POSTHOC_2B_SCRIPTS:
        pkl_type = _POSTHOC_2B_SCRIPTS[stem]
        cmd = [
            sys.executable, _POSTHOC_2B,
            f"--{pkl_type}-pkl", pkl_path,
            "--out-dir", dirs["posthoc"],
            "--out-report", os.path.join(dirs["posthoc"], "scorecard.md"),
        ]
    else:
        print(f"  [posthoc] No posthoc mapping for '{stem}', skipping.")
        return 0

    return _run_subprocess(cmd, os.environ.copy(), "POST-HOC ANALYSIS")


# ── Posthoc-only mode ────────────────────────────────────────────────────────

def run_posthoc_only(run_dir: str, introspect: bool) -> None:
    """Re-run posthoc analysis on an existing experiment folder."""
    config_path = os.path.join(run_dir, "config.json")
    stem = None
    if os.path.isfile(config_path):
        with open(config_path) as fh:
            cfg = json.load(fh)
        stem = cfg.get("script_stem")
    if stem is None:
        stem = os.path.basename(run_dir).rsplit("_", 2)[0]  # best-effort

    dirs = {
        "root":    run_dir,
        "plots":   os.path.join(run_dir, "plots"),
        "logs":    os.path.join(run_dir, "logs"),
        "posthoc": os.path.join(run_dir, "posthoc"),
    }
    rc = _run_posthoc(dirs, stem, save_agents=introspect)
    sys.exit(rc)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBAM-RICE experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "script",
        nargs="?",
        help="Path to the validation script to run (relative to rice_jax/)",
    )
    parser.add_argument(
        "--depth",
        choices=["train", "visualize", "full"],
        default="train",
        help="How far to run: train / visualize (same as train) / full (+ posthoc)",
    )
    parser.add_argument(
        "--experiments-dir",
        default=os.path.join(_SCRIPT_DIR, "experiments"),
        help="Root directory for experiment folders (default: rice_jax/experiments/)",
    )
    parser.add_argument(
        "--posthoc-only",
        metavar="RUN_DIR",
        default=None,
        help="Skip training; re-run posthoc on an existing experiment folder",
    )
    parser.add_argument(
        "--introspect",
        action="store_true",
        default=False,
        help="Enable Layer 2 introspection in posthoc (requires --save-agents)",
    )
    parser.add_argument(
        "--save-agents",
        action="store_true",
        default=False,
        dest="save_agents",
        help="Include trained PPO agents in the pkl (larger files). "
             "Enables Layer 2 introspection and is forwarded to litmus scripts only.",
    )

    # Everything the runner doesn't recognise is forwarded to the experiment script.
    # Use '--' to explicitly separate runner flags from script flags if needed.
    args, passthrough = parser.parse_known_args()

    # ── posthoc-only shortcut ────────────────────────────────────────────────
    if args.posthoc_only:
        run_posthoc_only(args.posthoc_only, introspect=args.introspect)
        return  # unreachable (sys.exit inside)

    if not args.script:
        parser.error("Provide a script path or --posthoc-only <run_dir>")

    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(_SCRIPT_DIR, script_path)
    if not os.path.isfile(script_path):
        parser.error(f"Script not found: {script_path}")

    stem = _script_stem(script_path)
    run_dir, ts = _make_run_dir(stem, args.experiments_dir)
    dirs = _setup_dirs(run_dir)

    save_agents = args.save_agents or args.introspect

    # --save-agents is a runner flag; only forward it to litmus scripts that
    # accept it (those in _POSTHOC_SCRIPTS).  All other script flags are passed
    # through unchanged via `passthrough`.
    script_cmd = [sys.executable, script_path] + passthrough
    if save_agents and stem in _POSTHOC_SCRIPTS:
        script_cmd.append("--save-agents")

    # ── Write frozen config ──────────────────────────────────────────────────
    _write_config(run_dir, {
        "script":           args.script,
        "script_stem":      stem,
        "depth":            args.depth,
        "timestamp":        ts,
        "passthrough_args": passthrough,
        "save_agents":      save_agents,
        "run_dir":          run_dir,
    })

    print(f"\n  Experiment: {run_dir}")

    # ── Set env vars so the script writes to our run_dir ────────────────────
    env = os.environ.copy()
    env["CBAM_EXPERIMENT_DIR"] = run_dir

    # ── Run the experiment script ────────────────────────────────────────────
    rc = _run_subprocess(script_cmd, env, f"EXPERIMENT: {stem}")
    if rc != 0:
        print(f"\n  [ERROR] Script exited with code {rc}. Aborting.")
        sys.exit(rc)

    # ── Post-hoc (depth=full) ────────────────────────────────────────────────
    if args.depth == "full":
        if stem in _POSTHOC_SCRIPTS:
            rc = _run_posthoc(dirs, stem, save_agents=save_agents)
            if rc != 0:
                print(f"\n  [WARN] Posthoc exited with code {rc}.")
        else:
            print(f"\n  [posthoc] '{stem}' has no posthoc mapping — skipping.")

    print(f"\n  Run complete → {run_dir}")
    print(f"  Artifacts:")
    for subdir in ("plots", "logs", "posthoc"):
        d = dirs[subdir]
        if os.path.isdir(d):
            files = os.listdir(d)
            if files:
                print(f"    {subdir}/  ({len(files)} files)")


if __name__ == "__main__":
    main()
