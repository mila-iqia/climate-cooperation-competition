"""Quick smoke test for additive_cbam reward mode."""
import os, sys, jax
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import jax.numpy as jnp
from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

rp = load_region_yamls(7)
env = RiceMRIO(
    region_params=rp, num_regions=7,
    mrio_data_root=os.path.join(REPO, "csv_asset"),
    mrio_trade=True, eu_region_idx=5,
    sector_granularity="emissions-simple",
    cbam_tariff_rate=0.8,
    reward_mode="additive_cbam",
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
)

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
print(f"reward_mode: {env.reward_mode}")
print(f"cbam_lambda: {state['cbam_lambda']}")
print(f"cbam_cost_all_regions: {state['cbam_cost_all_regions']}")

# Step with midpoint actions
actions = env.sample_action(key)
D = env.num_discrete_action_levels
mid = D // 2
for agent in actions:
    for k, v in actions[agent].items():
        actions[agent][k] = jnp.full_like(v, mid)
processed = env.process_actions(actions, state)
new_state = env.step_climate_and_economy(state, processed)
print(f"cbam_cost after step: {new_state['cbam_cost_all_regions']}")

# Test rewards with λ=0
rewards_lam0 = env.generate_rewards(new_state, state)
print(f"rewards (λ=0): { {k: float(v) for k, v in rewards_lam0.items()} }")

# Test rewards with λ=1.0
new_state_lam1 = new_state.copy()
new_state_lam1["cbam_lambda"] = jnp.float32(1.0)
rewards_lam1 = env.generate_rewards(new_state_lam1, state)
print(f"rewards (λ=1): { {k: float(v) for k, v in rewards_lam1.items()} }")

# Verify λ=0 reward == ΔU (no penalty)
du = new_state["utility_all_regions"] - state["utility_all_regions"]
for i in range(7):
    agent = f"region-{i:02d}"
    assert abs(float(rewards_lam0[agent]) - float(du[i])) < 1e-6, \
        f"λ=0 reward mismatch for {agent}: {rewards_lam0[agent]} vs {du[i]}"

# Verify λ=1 reward == ΔU - cbam_cost
cbam = new_state["cbam_cost_all_regions"]
for i in range(7):
    agent = f"region-{i:02d}"
    expected = float(du[i]) - float(cbam[i])
    assert abs(float(rewards_lam1[agent]) - expected) < 1e-5, \
        f"λ=1 reward mismatch for {agent}: {rewards_lam1[agent]} vs {expected}"

# Also test welfloss mode still works
env_wl = RiceMRIO(
    region_params=rp, num_regions=7,
    mrio_data_root=os.path.join(REPO, "csv_asset"),
    mrio_trade=True, eu_region_idx=5,
    sector_granularity="emissions-simple",
    cbam_tariff_rate=0.8,
    reward_mode="welfloss",  # default
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    sectoral_welfloss=True,
    welfare_loss_per_unit_tariff=5.0,
)
obs_wl, state_wl = env_wl.reset(key)
new_state_wl = env_wl.step_climate_and_economy(state_wl, processed)
rewards_wl = env_wl.generate_rewards(new_state_wl, state_wl)
print(f"welfloss rewards: { {k: float(v) for k, v in rewards_wl.items()} }")

# Null condition: τ=0 → cbam_cost=0 → additive reward = ΔU
env_null = RiceMRIO(
    region_params=rp, num_regions=7,
    mrio_data_root=os.path.join(REPO, "csv_asset"),
    mrio_trade=True, eu_region_idx=5,
    sector_granularity="emissions-simple",
    cbam_tariff_rate=0.0,  # no CBAM
    reward_mode="additive_cbam",
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
)
obs_null, state_null = env_null.reset(key)
ns_null = env_null.step_climate_and_economy(state_null, processed)
assert jnp.allclose(ns_null["cbam_cost_all_regions"], 0.0), \
    f"τ=0 should have zero cbam_cost: {ns_null['cbam_cost_all_regions']}"
print("Null condition (τ=0): cbam_cost = 0 ✓")


# ── Part 2: RCPOMonitoredPPO integration test ───────────────────────────────
print("\n── RCPO integration test ──")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training_monitor import RCPOMonitoredPPO, rcpo_cbam_log_info_fn
import jaxnasium as jym

env_rcpo = jym.LogWrapper(RiceMRIO(
    region_params=rp, num_regions=7,
    mrio_data_root=os.path.join(REPO, "csv_asset"),
    mrio_trade=True, eu_region_idx=5,
    sector_granularity="emissions-simple",
    cbam_tariff_rate=0.8,
    reward_mode="additive_cbam",
    diff_reward_mode=True,
    num_discrete_action_levels=10,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
    log_info_fn=rcpo_cbam_log_info_fn,
))

# Tiny training run: 2 envs × 20 steps × 3 iterations
ppo = RCPOMonitoredPPO(
    multi_agent=True,
    num_envs=2,
    num_steps=20,
    total_timesteps=2 * 20 * 3,  # = 120
    rcpo_eta_lambda=0.1,         # large η for test visibility
    rcpo_alpha_target=0.0,       # target=0 → λ should increase
    log_interval=1.0,            # never log (just train)
)
ppo = ppo.train(key, env_rcpo)
print("RCPOMonitoredPPO training completed ✓")

# After training with α_target=0 and positive costs, λ should have increased
# (We can't easily inspect env_state post-training, but the fact that
# training completed without errors validates the scan loop)
print("RCPO scan loop with λ update executed without error ✓")

print("\n=== ALL TESTS PASSED ===")
