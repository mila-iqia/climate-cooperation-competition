"""
Destination-Allocation Persistence Comparison  —  Perturb-then-Relax Protocol
===========================================================================
Directly calls ``_compute_trade_flows`` in a Python loop to isolate the
pure AR(1) dynamics without a full training loop.

Protocol
--------
  Phase A  t = 0 … T//2-1   "divert from EU" action
             export_reallocation[:, :, EU_IDX] = 0.0   →  δ = -delta_max
             export_reallocation[:, :, d≠EU]   = 1.0   →  δ = +delta_max

  Phase B  t = T//2 … T-1   "relax" (neutral midpoint)
             export_reallocation = 0.5 everywhere       →  δ = 0

Expected behaviour
------------------
  ρ = 0  → reverts immediately to 2016 baseline at t = T//2
  ρ = 1  → never reverts (fully adaptive, carries forward diverted state)
  0 < ρ < 1  → exponential reversion; faster for lower ρ

  The rate of reversion for a neutral action is:
    dest_alloc[t+1] = (1-ρ)*baseline + ρ*dest_alloc[t]
  so the gap to baseline decays as ρ^k after k relax steps.

Run from rice_jax/ with:
    conda run -n rice-jax python validate_persistence.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from __future__ import annotations

import os
import sys
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import jax
import jax.numpy as jnp

MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csv_asset")
NR        = 3
EU_IDX    = 1
CBAM_RATE = 0.5
T         = 30                        # simulation steps (not tied to episode_length)

RHO_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
COLORS     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls

# ── Build envs ─────────────────────────────────────────────────────────────────
params = load_region_yamls(NR)
non_eu = [r for r in range(NR) if r != EU_IDX]

envs: dict[float, RiceMRIO] = {}
for rho in RHO_VALUES:
    envs[rho] = RiceMRIO(
        num_regions=NR,
        region_params=params,
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        cbam_tariff_rate=CBAM_RATE,
        eu_region_idx=EU_IDX,
        diff_reward_mode=True,
        dest_alloc_persistence=rho,
    )
    if envs[rho].dest_alloc_baseline is None:
        print(f"[ABORT] MRIO data not found at {MRIO_ROOT}")
        sys.exit(1)

env0 = envs[0.0]
NS   = env0.num_sectors

# ── Action templates (shape: NR × NS × NR, then reshaped to NR × NS*NR) ───────
# "Divert from EU": push exports AWAY from EU, toward other destinations.
divert_3d   = np.ones((NR, NS, NR), dtype=np.float32)
divert_3d[:, :, EU_IDX] = 0.0                                # low weight → EU
divert_flat = divert_3d.reshape(NR, NS * NR)

# "Relax": neutral action (δ = 0); anchor fully determines allocation.
neutral_flat = np.full((NR, NS * NR), 0.5, dtype=np.float32)

# Simple uniform production for isolating trade-flow math.
production = np.ones((NR, NS), dtype=np.float32)

# ── Simulate T steps per ρ ─────────────────────────────────────────────────────
print("=" * 64)
print(f"Persistence  NR={NR}  NS={NS}  EU_IDX={EU_IDX}  T={T}  ρ={RHO_VALUES}")
print(f"Protocol: divert t=0..{T//2-1}, relax t={T//2}..{T-1}")
print("=" * 64)

results: dict[float, dict] = {}

for rho in RHO_VALUES:
    env = envs[rho]
    baseline = np.array(env.dest_alloc_baseline, dtype=np.float32)  # (NR, NS, NR)

    dest_alloc_over_time = np.zeros((T, NR, NS, NR), dtype=np.float32)
    prev_alloc = baseline.copy()

    for t in range(T):
        action_flat = divert_flat if t < T // 2 else neutral_flat
        trade_flows, dest_alloc_new = env._compute_trade_flows(
            jnp.array(production),
            jnp.array(action_flat),
            jnp.array(prev_alloc),
            jnp.array(t + 1),  # 1-based timestep (matches step_climate_and_economy)
        )
        dest_alloc_np = np.array(dest_alloc_new)
        dest_alloc_over_time[t] = dest_alloc_np
        prev_alloc = dest_alloc_np

    # EU export share per region: dest_alloc[r, :, EU_IDX].mean(over sectors)
    eu_share = dest_alloc_over_time[:, :, :, EU_IDX].mean(axis=2)   # (T, NR)

    # Distance to baseline (L2 over sectors & destinations, averaged over regions)
    diff = dest_alloc_over_time - baseline[np.newaxis]               # (T, NR, NS, NR)
    dist = np.linalg.norm(diff.reshape(T, NR, -1), axis=-1)         # (T, NR)

    results[rho] = {
        "dest_alloc": dest_alloc_over_time,  # (T, NR, NS, NR)
        "eu_share":   eu_share,              # (T, NR)
        "dist":       dist,                  # (T, NR)  distance from 2016 baseline
    }

    # Steady-state check: last 5 relax steps
    ss_share = eu_share[max(T-5, T//2):, non_eu].mean()
    ss_dist  = dist[max(T-5, T//2):, non_eu].mean()
    divert_share = eu_share[:T//2, non_eu].mean()
    print(f"  ρ={rho:.2f}  divert_eu_share={divert_share:.4f}  "
          f"ss_eu_share={ss_share:.4f}  ss_dist_from_baseline={ss_dist:.6f}")

timesteps = np.arange(T)

try:
    region_labels = list(env0.mrio_region_labels)
except Exception:
    region_labels = [f"R{r}" for r in range(NR)]

# ── Checks ─────────────────────────────────────────────────────────────────────
print()
all_pass = True

# PC1: All ρ values should divert away from EU during phase A
for rho in RHO_VALUES:
    baseline_eu_share = float(np.array(env0.dest_alloc_baseline)[:, :, EU_IDX].mean())
    divert_share_rho  = results[rho]["eu_share"][:T//2, non_eu].mean()
    ok = divert_share_rho < baseline_eu_share
    all_pass &= ok
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  PC1 ρ={rho:.2f}: EU share during divert "
          f"({divert_share_rho:.4f}) < baseline ({baseline_eu_share:.4f})")

print()
# PC2: After relax, ρ=0 should fully revert to baseline (within tol)
ss_dist_rho0 = results[0.0]["dist"][T//2:, non_eu].mean()
ok = ss_dist_rho0 < 1e-4
all_pass &= ok
print(f"  {'PASS' if ok else 'FAIL'}  PC2 ρ=0 fully reverts to baseline "
      f"after relax (ss_dist={ss_dist_rho0:.2e})")

# PC3: ρ=1 should NOT revert (retains diverted allocation)
ss_eu_share_rho1 = results[1.0]["eu_share"][max(T-5,T//2):, non_eu].mean()
base_eu_share    = float(np.array(env0.dest_alloc_baseline)[:, :, EU_IDX].mean())
# ρ=1 EU share should stay below baseline (still diverted) or at least differ from ρ=0
ss_eu_rho0 = results[0.0]["eu_share"][max(T-5,T//2):, non_eu].mean()
ok = ss_eu_share_rho1 < ss_eu_rho0 - 1e-5
all_pass &= ok
print(f"  {'PASS' if ok else 'FAIL'}  PC3 ρ=1 retains diversion after relax "
      f"(ss_EU_share_ρ1={ss_eu_share_rho1:.4f} vs ρ0={ss_eu_rho0:.4f})")

# PC4: Monotonic reversion ordering: ρ=0 reverts fastest, ρ=1 slowest.
# Compare mean distance-from-baseline in relax phase.
ss_dists = [results[rho]["dist"][T//2:, non_eu].mean() for rho in RHO_VALUES]
is_mono  = all(ss_dists[i] <= ss_dists[i+1] + 1e-8 for i in range(len(ss_dists)-1))
all_pass &= is_mono
print(f"  {'PASS' if is_mono else 'FAIL'}  PC4 monotonic reversion: "
      f"dist(ρ=0)≤…≤dist(ρ=1)  {[f'{d:.4f}' for d in ss_dists]}")

print()
status = "ALL PASS" if all_pass else "SOME CHECKS FAILED"
print(f"  {status}")
print("=" * 64)

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, len(non_eu), figsize=(7 * len(non_eu), 10),
                          gridspec_kw={"hspace": 0.45, "wspace": 0.35})
if len(non_eu) == 1:
    axes = [[ax] for ax in axes]

for col_i, r in enumerate(non_eu):
    lbl = str(region_labels[r]) if r < len(region_labels) else f"R{r}"

    # Row 0: EU export share (dest_alloc[:, r, :, EU_IDX].mean(sectors))
    ax0 = axes[0][col_i]
    for i, rho in enumerate(RHO_VALUES):
        ax0.plot(timesteps, results[rho]["eu_share"][:, r],
                 color=COLORS[i], linewidth=2, label=f"ρ={rho}", alpha=0.9)
    # Shade divert / relax phases
    ax0.axvspan(0, T//2 - 0.5, alpha=0.07, color="red",  label="_divert phase")
    ax0.axvspan(T//2 - 0.5, T - 1, alpha=0.07, color="blue", label="_relax phase")
    ax0.axvline(T//2 - 0.5, color="grey", linewidth=1.0, linestyle="--")
    # Baseline reference
    base_val = float(np.array(env0.dest_alloc_baseline)[r, :, EU_IDX].mean())
    ax0.axhline(base_val, color="black", linewidth=1, linestyle=":", label="2016 baseline")
    ax0.set_xlabel("Timestep")
    ax0.set_ylabel("EU destination share")
    ax0.set_title(f"EU destination share — {lbl}\n"
                  f"(divert t<{T//2}, relax t≥{T//2})")
    ax0.legend(fontsize=7)
    ax0.grid(True, alpha=0.3)

    # Row 1: Distance from 2016 baseline
    ax1 = axes[1][col_i]
    for i, rho in enumerate(RHO_VALUES):
        ax1.plot(timesteps, results[rho]["dist"][:, r],
                 color=COLORS[i], linewidth=2, label=f"ρ={rho}", alpha=0.9)
    ax1.axvspan(0, T//2 - 0.5, alpha=0.07, color="red")
    ax1.axvspan(T//2 - 0.5, T - 1, alpha=0.07, color="blue")
    ax1.axvline(T//2 - 0.5, color="grey", linewidth=1.0, linestyle="--")
    ax1.axhline(0, color="black", linewidth=1, linestyle=":", label="baseline")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("‖dest_alloc − baseline‖₂")
    ax1.set_title(f"Distance from 2016 baseline — {lbl}\n"
                  f"(↑ = more diverted; ρ=0 snaps back, ρ=1 persists)")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

fig.suptitle(
    "AR(1) Trade-Allocation Persistence  —  Perturb-then-Relax Protocol\n"
    f"ρ ∈ {RHO_VALUES}  |  NR={NR}  EU_IDX={EU_IDX}  "
    f"T={T}  (red=divert, blue=relax)\n"
    f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
    fontsize=12,
)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"plots/persistence_comparison_{ts}.png"
os.makedirs("plots", exist_ok=True)
fig.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
if not all_pass:
    sys.exit(1)
