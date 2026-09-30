#!/usr/bin/env python3
"""Smoke test for 9-region CBAM-vulnerability RiceMRIO setup."""
import os
import sys

import numpy as np

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
YAML_DIR = os.path.join(REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_ROOT = os.path.join(REPO_ROOT, "csv_asset")

# Test 1: load_region_yamls with yaml_dir
print("=== Test 1: load_region_yamls ===")
from rice_jax.utils import load_region_yamls
rp = load_region_yamls(9, yaml_dir=YAML_DIR)
print(f"  xA_0 shape: {rp.xA_0.shape}")
print(f"  xL_0 shape: {rp.xL_0.shape}")
print(f"  ximport shape: {rp.ximport.shape}")
print(f"  xsigma_0: {rp.xsigma_0}")
assert rp.xA_0.shape == (9,), f"Expected (9,), got {rp.xA_0.shape}"
assert rp.ximport.shape == (9, 9), f"Expected (9,9), got {rp.ximport.shape}"
print("  PASS\n")

# Test 2: RiceMRIO initialization
print("=== Test 2: RiceMRIO init ===")
from rice_jax import RiceMRIO
env = RiceMRIO(
    region_params=rp,
    num_regions=9,
    mrio_data_root=MRIO_ROOT,
    mrio_trade=True,
    cbam_tariff_rate=0.0,
    eu_region_idx=3,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    sectoral_welfloss=True,
    sector_granularity="emissions-simple",
    welfare_loss_per_unit_tariff=5.0,
)
print(f"  num_regions: {env.num_regions}")
print(f"  num_sectors: {env.num_sectors}")
print(f"  eu_region_idx: {env.eu_region_idx}")
print(f"  sector_names: {env.sector_names}")
assert env.num_regions == 9
assert env.num_sectors == 2
assert env.eu_region_idx == 3
print("  PASS\n")

# Test 3: reset
print("=== Test 3: reset ===")
import jax
key = jax.random.PRNGKey(42)
obs, state = env.reset(key)
print(f"  trade_flows shape: {state['trade_flows'].shape}")
print(f"  production_by_sector shape: {state['production_by_sector'].shape}")
assert state["trade_flows"].shape == (9, 9, 2), f"Got {state['trade_flows'].shape}"
assert state["production_by_sector"].shape == (9, 2)
print("  PASS\n")

# Test 4: one step
print("=== Test 4: step ===")
import jax.numpy as jnp
actions = env.sample_action(key)
# Set midpoint discrete actions
D = env.num_discrete_action_levels
mid = D // 2
for agent in actions:
    for k, v in actions[agent].items():
        actions[agent][k] = jnp.full_like(v, mid)
processed = env.process_actions(actions, state)
state2 = env.step_climate_and_economy(state, processed)
print(f"  trade_flows sum: {state2['trade_flows'].sum():.4f}")
print(f"  cbam_revenue (no CBAM, should be ~0): {state2['cbam_revenue']}")
assert state2["trade_flows"].sum() > 0, "Trade flows should be positive"
print("  PASS\n")

# Test 5: CBAM active
print("=== Test 5: CBAM τ=0.80 ===")
env_cbam = RiceMRIO(
    region_params=rp,
    num_regions=9,
    mrio_data_root=MRIO_ROOT,
    mrio_trade=True,
    cbam_tariff_rate=0.80,
    eu_region_idx=3,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    sectoral_welfloss=True,
    sector_granularity="emissions-simple",
    welfare_loss_per_unit_tariff=5.0,
)
_, state_c = env_cbam.reset(key)
proc_c = env_cbam.process_actions(actions, state_c)
state_c2 = env_cbam.step_climate_and_economy(state_c, proc_c)
eu_rev = state_c2["cbam_revenue"][3]
print(f"  EU CBAM revenue (idx=3): {eu_rev:.6f}")
assert eu_rev > 0, f"Expected positive EU CBAM revenue, got {eu_rev}"
print("  PASS\n")

print("=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
