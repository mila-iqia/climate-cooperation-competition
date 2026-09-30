"""probe_reward_scale.py

Diagnostic: measure the ratio  λ·cbam_cost / |ΔU|  under the canonical config
to verify that the CBAM penalty is commensurable with the utility signal.

Run from rice_jax/:
    conda run -n rice-jax python validation/probe_reward_scale.py

Prints per-region and aggregate statistics for each episode step, then
summarises:
  - mean(|ΔU|)                           utility signal scale
  - mean(cbam_cost)                      raw CBAM cost scale
  - mean(λ·cbam_cost / |ΔU|)            effective penalty ratio
  - cbam_cost / ΔU range across regions  heterogeneity check
  - raw EORA intensity mean (pre-rescale) for reference
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # must precede JAX import to avoid backend conflict

import numpy as np
import jax
import jax.numpy as jnp

from validation.canonical_config import (
    make_canonical_env,
    CANONICAL_SEEDS,
    EU_REGION_IDX,
    REGION_NAMES,
)
from _experiment_util import FixedActionAgent, run_single_episode


# ---------------------------------------------------------------------------
# 1. Raw intensity stats (post-rescaling) and pre-rescale recovery
# ---------------------------------------------------------------------------

env_raw = make_canonical_env(for_training=False)

# The rescaling formula is:  stored = raw / raw_mean * target_mean (=0.1)
# To recover raw_mean, build a throwaway env directly via RiceMRIO (bypassing
# the canonical guard) with cbam_intensity_target_mean=1.0 — a no-op rescale
# that leaves EORA values at their original scale.
from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls
from validation.canonical_config import (
    _CANONICAL_ENV_DEFAULTS,
    CANONICAL_YAML_DIR,
    NUM_REGIONS,
)
from training_monitor import rcpo_cbam_log_info_fn as _log_fn
_kwargs_unscaled = dict(_CANONICAL_ENV_DEFAULTS)
_kwargs_unscaled["cbam_intensity_target_mean"] = 1.0
_rp = load_region_yamls(NUM_REGIONS, yaml_dir=CANONICAL_YAML_DIR)
env_unscaled = RiceMRIO(region_params=_rp, log_info_fn=_log_fn, **_kwargs_unscaled)
raw_mean_eora = float(env_unscaled.emissions_intensity.mean())
# Note: stored mean != exactly target_mean because rescaling happens at full
# 26-sector granularity before aggregation to emissions-simple (2 sectors).
# The post-aggregation mean is therefore slightly less than 0.1.
stored_mean_check = float(env_raw.emissions_intensity.mean())
print(f"  (post-aggregation mean: {stored_mean_check:.5f}, target was {env_raw.cbam_intensity_target_mean})")

print("=" * 60)
print("INTENSITY STATS")
print(f"  raw EORA mean (pre-rescale)   = {raw_mean_eora:.6f}")
print(f"  stored mean (post-rescale)    = {float(env_raw.emissions_intensity.mean()):.6f}  (target=0.1)")
print(f"  stored min / max              = {float(env_raw.emissions_intensity.min()):.6f} / {float(env_raw.emissions_intensity.max()):.6f}")
print(f"  dirty/clean ratio (sector 0 vs 1, mean over regions):")
print(f"    dirty (s=0) mean: {float(env_raw.emissions_intensity[:, 0].mean()):.6f}")
print(f"    clean (s=1) mean: {float(env_raw.emissions_intensity[:, 1].mean()):.6f}")
print(f"    ratio:            {float(env_raw.emissions_intensity[:, 0].mean() / (env_raw.emissions_intensity[:, 1].mean() + 1e-10)):.2f}x")


# ---------------------------------------------------------------------------
# 2. Rollout with canonical config — collect ΔU and cbam_cost per step
# ---------------------------------------------------------------------------

from dataclasses import replace as _replace

def _probe_log_fn(state, actions):
    return {
        "utility_all_regions": state["utility_all_regions"],
        "cbam_cost_all_regions": state["cbam_cost_all_regions"],
    }

env = make_canonical_env(for_training=False)
env = _replace(env, log_info_fn=_probe_log_fn)
# Use a midpoint action (D//2 = 5 for D=10) — MRIO baseline flows, modest μ
D = env.num_discrete_action_levels
mid = D // 2
agent = FixedActionAgent(env)
# Override with midpoint savings (savings_rate=5) and low mitigation (mu=1/10=10%)
# to produce a realistic non-trivial τ_eff from the differential formula.
for astr in agent.default_actions:
    agent.default_actions[astr] = dict(agent.default_actions[astr])
    agent.default_actions[astr]["savings_rate"] = jnp.array(mid, dtype=jnp.float32)
    # Keep mitigation low (=1 = 10% of D range) so τ_eff is large and cbam_cost nonzero
    agent.default_actions[astr]["mitigation_rate"] = jnp.array(1, dtype=jnp.float32)
    # Midpoint export reallocation → MRIO baseline destination shares
    nr, ns = env.num_regions, env.num_sectors
    agent.default_actions[astr]["export_reallocation"] = jnp.full(
        (ns * nr,), mid, dtype=jnp.float32
    )

print("\n" + "=" * 60)
print(f"ROLLOUT SCALE PROBE  (seed={CANONICAL_SEEDS[0]}, mid-action)")
print("=" * 60)

key = jax.random.PRNGKey(CANONICAL_SEEDS[0])
info = run_single_episode(key, env, agent)

# info["utility_all_regions"]      shape (T, NR)
# info["cbam_cost_all_regions"]    shape (T, NR)
# ΔU is already diff_reward_mode output via generate_rewards, but
# utility_all_regions in state is the *undifferenced* value per step.
# We compute ΔU manually across timesteps.

U     = np.array(info["utility_all_regions"])       # (T, NR)
cost  = np.array(info["cbam_cost_all_regions"])     # (T, NR)
lam   = float(env.cbam_lambda_init)

# ΔU[t] = U[t] - U[t-1], skip t=0
dU = np.diff(U, axis=0)          # (T-1, NR)
c  = cost[1:]                    # (T-1, NR) — align with dU

# Non-EU only (EU has cbam_cost=0 by construction)
non_eu = [r for r in range(env.num_regions) if r != EU_REGION_IDX]

dU_ne = dU[:, non_eu]            # (T-1, NR-1)
c_ne  = c[:, non_eu]             # (T-1, NR-1)

abs_dU = np.abs(dU_ne)
penalty = lam * c_ne
ratio   = np.where(abs_dU > 1e-10, penalty / abs_dU, np.nan)

print(f"\nλ = {lam}")
print(f"\nPer-step statistics (non-EU exporters, all timesteps):")
print(f"  mean |ΔU|              = {np.nanmean(abs_dU):.6f}")
print(f"  mean cbam_cost         = {np.nanmean(c_ne):.6f}")
print(f"  mean λ·cbam_cost       = {np.nanmean(penalty):.6f}")
print(f"  mean λ·cost / |ΔU|     = {np.nanmean(ratio):.4f}  ({np.nanmean(ratio)*100:.1f}%)")
print(f"  median λ·cost / |ΔU|   = {np.nanmedian(ratio):.4f}  ({np.nanmedian(ratio)*100:.1f}%)")
print(f"  p10 / p90              = {np.nanpercentile(ratio,10):.4f} / {np.nanpercentile(ratio,90):.4f}")

print(f"\nPer-region breakdown (mean over timesteps):")
print(f"  {'Region':<22}  {'mean|ΔU|':>10}  {'mean_cost':>10}  {'penalty':>10}  {'ratio%':>8}")
print(f"  {'-'*22}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
for i, r in enumerate(non_eu):
    name = REGION_NAMES.get(r, f"r{r}")
    mu_du    = np.nanmean(np.abs(dU[:, r]))
    mu_cost  = np.nanmean(c[:, r])
    mu_pen   = lam * mu_cost
    mu_ratio = mu_pen / mu_du if mu_du > 1e-10 else float("nan")
    print(f"  {name:<22}  {mu_du:>10.6f}  {mu_cost:>10.6f}  {mu_pen:>10.6f}  {mu_ratio*100:>7.1f}%")

print(f"\nInterpretation:")
r_mean = np.nanmean(ratio)
if r_mean < 0.01:
    verdict = "WEAK — penalty is <1% of ΔU; agents will likely ignore CBAM signal."
elif r_mean < 0.10:
    verdict = "MARGINAL — penalty is 1-10% of ΔU; signal detectable but may need larger λ."
elif r_mean < 2.0:
    verdict = "COMMENSURABLE — penalty is 10-200% of ΔU; well-calibrated range."
else:
    verdict = "DOMINANT — penalty >200% of ΔU; CBAM signal drowns welfare; reduce λ."
print(f"  {verdict}")

print("\nDone.")
