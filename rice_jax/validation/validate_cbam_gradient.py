"""
CBAM Gradient Validation Script
================================
Checks whether the Phase 1 CBAM gradient design changes (CBAM_GRADIENT_DESIGN.md)
produce the intended incentive signals:

  V1. Welfloss is sensitive to EU export volume (Approach B).
      Manually diverting exports away from EU must increase welfloss for
      non-EU exporters.

  V2. Welfloss is sensitive to sector composition of EU-bound exports (Approach A).
      Shifting EU-bound basket from high-intensity to low-intensity sectors must
      increase welfloss (lower CBAM burden) — only meaningful with heterogeneous σ.

  V3. Effective CBAM rate reflects sector intensity mix, not just uniform σ.
      Two exporters with identical volume but different sector mixes must receive
      different effective rates when intensities are heterogeneous.

  V4. EU itself is never penalised (self-trade = 0).

  V5. CBAM revenue scales with EU import volume (sanity on accounting).

  V6. FixedActionAgent rollout: EU export share is in [0, 1] and cbam_revenue > 0
      for all timesteps (end-to-end smoke test).

Run from rice_jax/ with:
    conda run -n rice-jax python validate_cbam_gradient.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))



from __future__ import annotations
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")
NR   = 3
EU_IDX = 1        # RIG 2 = Europe & Central Asia
SEED = 42
KEY  = jax.random.PRNGKey(SEED)

# ── Imports ────────────────────────────────────────────────────────────────────
from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls, full_state_info_log_fn
from _experiment_util import FixedActionAgent, run_single_episode

params = load_region_yamls(NR)

# ── Helpers ────────────────────────────────────────────────────────────────────
PASSES: list[bool] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    PASSES.append(bool(cond))
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}" + (f"  ({detail})" if detail else ""))


def _make_env(cbam_rate: float = 0.5, intensity: np.ndarray | None = None) -> RiceMRIO:
    env = RiceMRIO(
        num_regions=NR,
        region_params=params,
        mrio_data_root=MRIO_ROOT,
        mrio_trade=True,
        cbam_tariff_rate=cbam_rate,
        eu_region_idx=EU_IDX,
        diff_reward_mode=True,
        log_info_fn=full_state_info_log_fn,
    )
    if intensity is not None:
        object.__setattr__(env, "emissions_intensity", intensity)
    return env


def _one_step(env: RiceMRIO, realloc_actions: np.ndarray | None = None):
    """
    Run one step from reset.  realloc_actions shape: (NR, NS*NR) or None (zeros).
    Returns the post-step state.
    """
    obs, state = env.reset(KEY)
    NS = env.num_sectors

    # Build action dict: savings=0.25, mitigation=0, reallocation=given or midpoint
    mid = env.num_discrete_action_levels // 2
    realloc = (
        np.full((NR, NS * NR), mid, dtype=np.int32)
        if realloc_actions is None
        else realloc_actions.astype(np.int32)
    )
    action = {
        f"agent_{r}": {
            "savings_rate":        np.array(2, dtype=np.int32),
            "mitigation_rate":     np.array(0, dtype=np.int32),
            "export_reallocation": realloc[r],
        }
        for r in range(NR)
    }
    (_, _, _, _, _), state = env.step_env(KEY, state, action)
    return state


def _eu_imports_from_r(state) -> np.ndarray:
    """EU row of gross_imports_mrio → how much EU imports from each region. (NR,)"""
    # In Phase 2A, import_bids_all_regions stores gross_imports_mrio [to, from]
    return np.array(state["import_bids_all_regions"][EU_IDX])  # (NR,)


def _effective_rates(state) -> np.ndarray:
    """EU row of import_tariffs → effective CBAM rate on each exporter. (NR,)"""
    return np.array(state["import_tariffs"][EU_IDX])  # (NR,)


def _welfloss_from_state(env: RiceMRIO, state) -> np.ndarray:
    """Reconstruct welfloss vector from state."""
    gross_outputs   = np.array(state["gross_output_all_regions"])
    eu_imports      = _eu_imports_from_r(state)
    eff_rate        = _effective_rates(state)
    wl = 1.0 - (eu_imports / (gross_outputs + 1e-8)) * eff_rate * 0.4
    return np.clip(wl, 0, 1)


# Derive realistic heterogeneous intensities: 2 sectors with σ_0 ≪ σ_1
def _hetero_intensity(ns: int) -> np.ndarray:
    """Per-sector intensities that vary across sectors (clean=0.02, dirty=0.50)."""
    intensity = np.zeros((NR, ns), dtype=np.float32)
    # Sector 0 = clean (services-like), sector 1 = dirty (industry-like),
    # remainder = medium.
    intensity[:, 0] = 0.02
    for s in range(1, ns):
        intensity[:, s] = 0.02 + (0.48 / max(ns - 1, 1)) * s
    return intensity


# ── Run checks ─────────────────────────────────────────────────────────────────
print("=" * 64)
print("CBAM Gradient Validation — Phase 1")
print(f"  NR={NR}, EU_IDX={EU_IDX}, SEED={SEED}")
print("=" * 64)

env_base = _make_env(cbam_rate=0.5)
if env_base.dest_alloc_baseline is None:
    print("\n[ABORT] MRIO data not found at:", MRIO_ROOT)
    print("  Run aggregate_local_mrio.py first, or point MRIO_ROOT to csv_asset/.")
    sys.exit(1)

NS = env_base.num_sectors
print(f"\n  num_sectors = {NS}")
print(f"  intensity range (after normalisation): "
      f"[{env_base.emissions_intensity.min():.4f}, "
      f"{env_base.emissions_intensity.max():.4f}]")
print(f"  intensity mean: {env_base.emissions_intensity.mean():.4f}")
print()

# ────────────────────────────────────────────────────────────────────────────────
# V1: Welfloss increases when exporter diverts exports from EU (Approach B)
# ────────────────────────────────────────────────────────────────────────────────
print("── V1: Welfloss responds to EU export volume ────────────────────────────")

mid    = env_base.num_discrete_action_levels // 2
D      = env_base.num_discrete_action_levels

# Baseline: midpoint action → MRIO baseline allocation  
realloc_baseline = np.full((NR, NS * NR), mid, dtype=np.int32)

# Diversion: shift exports away from EU by reducing EU-destination logit
# (action = 0 → δ = -delta_max → very negative logit for EU destination)
# We zero out the EU-destination columns of the realloc action.
# realloc is shaped (NR, NS*NR) and is viewed as (NR, NS, NR):
# dim 2 = destination, EU_IDX = EU destination.
realloc_divert = realloc_baseline.copy()
# For each region r (non-EU), for each sector s, set EU destination to 0 (lowest)
realloc_divert_3d = realloc_divert.reshape(NR, NS, NR)
realloc_divert_3d[:, :, EU_IDX] = 0   # minimise EU destination logits
realloc_divert_flat = realloc_divert_3d.reshape(NR, NS * NR)

s_base   = _one_step(env_base, realloc_baseline)
s_divert = _one_step(env_base, realloc_divert_flat)

eu_imp_base   = _eu_imports_from_r(s_base)
eu_imp_divert = _eu_imports_from_r(s_divert)

non_eu = [r for r in range(NR) if r != EU_IDX]
for r in non_eu:
    check(
        f"V1a R{r}: diverting from EU reduces EU import volume",
        eu_imp_divert[r] < eu_imp_base[r] - 1e-6,
        f"base={eu_imp_base[r]:.4f}  diverted={eu_imp_divert[r]:.4f}",
    )

wl_base   = _welfloss_from_state(env_base, s_base)
wl_divert = _welfloss_from_state(env_base, s_divert)

for r in non_eu:
    check(
        f"V1b R{r}: diverting from EU increases welfloss (lower CBAM burden)",
        wl_divert[r] > wl_base[r] - 1e-8,
        f"wl_base={wl_base[r]:.6f}  wl_diverted={wl_divert[r]:.6f}",
    )
print()

# ────────────────────────────────────────────────────────────────────────────────
# V2: Effective rate reflects sector intensity mix (Approach A).
# Bypass the action encoding: call _compute_cbam directly with hand-crafted
# trade_flows where we control exactly which sectors flow to EU.
# This isolates the math from the 26-sector baseline allocation complexity.
# ────────────────────────────────────────────────────────────────────────────────
print("── V2: Effective rate reflects sector intensity mix ─────────────────────")

if NS < 2:
    check("V2 Skipped", True, f"need ≥2 sectors, got NS={NS}")
else:
    hetero    = _hetero_intensity(NS)          # σ_0=0.02 (clean) … σ_NS-1=0.50 (dirty)
    env_hetero = _make_env(cbam_rate=0.5, intensity=hetero)

    # Two hand-crafted trade_flows (NR, NR, NS):
    # Same total volume to EU per region (1.0 unit), but different sector splits.
    # "All-clean": entire EU-bound volume goes through sector 0 (σ=0.02).
    # "All-dirty": entire EU-bound volume goes through sector NS-1 (σ=0.50).
    volume = 1.0
    for r in non_eu:
        tf_clean = np.zeros((NR, NR, NS), dtype=np.float32)
        tf_dirty = np.zeros((NR, NR, NS), dtype=np.float32)
        tf_clean[r, EU_IDX, 0]      = volume   # all clean sector
        tf_dirty[r, EU_IDX, NS - 1] = volume   # all dirty sector

        gi_clean = tf_clean.sum(axis=2).T   # (NR, NR) [to, from]
        gi_dirty = tf_dirty.sum(axis=2).T

        cm_clean, _ = env_hetero._compute_cbam(
            jnp.array(tf_clean), jnp.array(gi_clean)
        )
        cm_dirty, _ = env_hetero._compute_cbam(
            jnp.array(tf_dirty), jnp.array(gi_dirty)
        )

        rate_clean = float(cm_clean[EU_IDX, r])
        rate_dirty = float(cm_dirty[EU_IDX, r])

        check(
            f"V2a R{r}: all-clean sector → lower effective rate than all-dirty",
            rate_clean < rate_dirty - 1e-6,
            f"σ_clean={hetero[r,0]:.3f}  σ_dirty={hetero[r,NS-1]:.3f}  "
            f"rate_clean={rate_clean:.6f}  rate_dirty={rate_dirty:.6f}",
        )

        # welfloss follows rate (same volume, different rate)
        gross_output = 100.0   # fixed dummy
        wl_clean = 1 - (volume / gross_output) * rate_clean * 0.4
        wl_dirty = 1 - (volume / gross_output) * rate_dirty * 0.4
        check(
            f"V2b R{r}: clean sector mix → higher welfloss than dirty mix",
            wl_clean > wl_dirty - 1e-8,
            f"wl_clean={wl_clean:.6f}  wl_dirty={wl_dirty:.6f}",
        )
print()

# ────────────────────────────────────────────────────────────────────────────────
# V3: Two exporters with identical volume but different sector mixes → different rates.
# Also uses direct _compute_cbam to avoid action-encoding confounds.
# ────────────────────────────────────────────────────────────────────────────────
print("── V3: Sector mix determines effective rate (heterogeneous σ) ───────────")

if NS < 2:
    check("V3 Skipped", True, f"need ≥2 sectors, got NS={NS}")
elif len(non_eu) < 2:
    check("V3 Skipped", True, f"need ≥2 non-EU regions")
else:
    hetero  = _hetero_intensity(NS)
    env_v3  = _make_env(cbam_rate=0.5, intensity=hetero)
    r0, r1  = non_eu[0], non_eu[1]

    # Same total volume (1.0) to EU but R0 sends clean sector, R1 sends dirty
    tf_v3 = np.zeros((NR, NR, NS), dtype=np.float32)
    tf_v3[r0, EU_IDX, 0]      = 1.0   # R0: clean
    tf_v3[r1, EU_IDX, NS - 1] = 1.0   # R1: dirty
    gi_v3 = tf_v3.sum(axis=2).T

    cm_v3, _ = env_v3._compute_cbam(jnp.array(tf_v3), jnp.array(gi_v3))
    rate_r0 = float(cm_v3[EU_IDX, r0])
    rate_r1 = float(cm_v3[EU_IDX, r1])

    check(
        f"V3: R{r0} (clean, σ={hetero[r0,0]:.3f}) has lower rate than "
        f"R{r1} (dirty, σ={hetero[r1,NS-1]:.3f})",
        rate_r0 < rate_r1 - 1e-6,
        f"rate_R{r0}={rate_r0:.6f}  rate_R{r1}={rate_r1:.6f}",
    )
print()

# ────────────────────────────────────────────────────────────────────────────────
# V4: EU region is never assigned a CBAM penalty
# ────────────────────────────────────────────────────────────────────────────────
print("── V4: EU incurs no CBAM self-penalty ───────────────────────────────────")

s4 = _one_step(env_base, realloc_baseline)
rate_eu = _effective_rates(s4)[EU_IDX]
wl_eu   = np.array(s4["utility_times_welfloss_all_regions"])[EU_IDX]

check("V4a: EU effective CBAM rate = 0", abs(rate_eu) < 1e-8,
      f"rate={rate_eu:.2e}")
check("V4b: EU welfloss = utility (no penalty)",
      abs(float(s4["utility_all_regions"][EU_IDX]) -
          float(s4["utility_times_welfloss_all_regions"][EU_IDX])) < 1e-5,
      f"utility={float(s4['utility_all_regions'][EU_IDX]):.4f}  "
      f"u*wl={wl_eu:.4f}")
print()

# ────────────────────────────────────────────────────────────────────────────────
# V5: CBAM revenue scales with EU import volume
# ────────────────────────────────────────────────────────────────────────────────
print("── V5: CBAM revenue proportional to EU import volume ────────────────────")

s5_base   = _one_step(env_base, realloc_baseline)
s5_divert = _one_step(env_base, realloc_divert_flat)

rev_base   = float(np.array(s5_base["cbam_revenue"])[EU_IDX])
rev_divert = float(np.array(s5_divert["cbam_revenue"])[EU_IDX])

check("V5: Diverting exports from EU reduces EU CBAM revenue",
      rev_divert < rev_base - 1e-8,
      f"rev_base={rev_base:.4f}  rev_diverted={rev_divert:.4f}")
print()

# ────────────────────────────────────────────────────────────────────────────────
# V6: End-to-end rollout smoke test
# ────────────────────────────────────────────────────────────────────────────────
print("── V6: End-to-end rollout smoke test ────────────────────────────────────")

env_e2e  = _make_env(cbam_rate=0.5)
agent    = FixedActionAgent(env_e2e)
episode  = run_single_episode(KEY, env_e2e, agent)

trade    = np.array(episode["trade_flows"])   # (T, NR, NR, NS)
eu_exp   = trade[:, :, EU_IDX, :].sum(axis=-1)    # (T, NR) total exported to EU
total_exp = trade.sum(axis=(2, 3)) + 1e-8          # (T, NR)
eu_shares = eu_exp / total_exp                     # (T, NR)

rev_eu = np.array(episode["cbam_revenue"])[:, EU_IDX]  # (T,)

check("V6a: EU export shares in [0, 1] every timestep (non-EU regions)",
      np.all((eu_shares[:, non_eu] >= -1e-6) & (eu_shares[:, non_eu] <= 1 + 1e-6)),
      f"min={eu_shares[:, non_eu].min():.4f}  max={eu_shares[:, non_eu].max():.4f}")

check("V6b: CBAM revenue > 0 every timestep",
      np.all(rev_eu > 0),
      f"min={rev_eu.min():.4f}  mean={rev_eu.mean():.4f}")

# Welfloss for non-EU exporters should be < 1 (CBAM does bite)
utw = np.array(episode["utility_times_welfloss_all_regions"])   # (T, NR) after log_fn reshape?
try:
    # full_state_info_log_fn restructures per-region arrays as {region_id: array(T)}
    utw_stacked = np.stack(
        [np.array(episode["utility_times_welfloss_all_regions"][r]) for r in range(NR)],
        axis=1,
    )  # (T, NR)
    ut_stacked = np.stack(
        [np.array(episode["utility_all_regions"][r]) for r in range(NR)],
        axis=1,
    )  # (T, NR)
    wl_inferred = utw_stacked / (ut_stacked + 1e-12)  # (T, NR)
    for r in non_eu:
        check(
            f"V6c R{r}: welfloss < 1.0 across episode (CBAM penalises exporter)",
            float(wl_inferred[:, r].mean()) < 1.0 - 1e-6,
            f"mean_welfloss={wl_inferred[:, r].mean():.6f}",
        )
except Exception as exc:
    check("V6c welfloss < 1 check", False, f"could not infer welfloss: {exc}")
print()

# ────────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────────
print("=" * 64)
n_pass = sum(PASSES)
n_total = len(PASSES)
print(f"  {n_pass}/{n_total} checks passed")
if n_pass == n_total:
    print("  ALL PASS — gradient design is working as intended.")
else:
    fails = [i + 1 for i, p in enumerate(PASSES) if not p]
    print(f"  FAILED checks at positions: {fails}")
print("=" * 64)

sys.exit(0 if n_pass == n_total else 1)
