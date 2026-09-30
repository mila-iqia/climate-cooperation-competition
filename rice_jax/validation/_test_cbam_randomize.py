"""Quick smoke test for cbam_randomize feature."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls
import jax
import jaxnasium as jym

rp = load_region_yamls(7)

# --- Test 1: conditioned mode (randomized) ---
env = RiceMRIO(
    region_params=rp, num_regions=7, mrio_data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset"),
    mrio_trade=True, cbam_tariff_rate=0.8, cbam_randomize=True,
    cbam_tariff_rates=(0.0, 0.8),
    dest_alloc_persistence=0.55, dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True, num_discrete_action_levels=10,
    sectoral_welfloss=True, fixed_savings_rate=True, no_mitigation=True,
    sector_granularity="emissions-simple", welfare_loss_per_unit_tariff=50.0,
)

key = jax.random.PRNGKey(0)
obs, state = env.reset_env(key)
print("state cbam_tariff_rate:", state["cbam_tariff_rate"])

# The obs dict from generate_observation includes cbam_tariff_rate;
# jaxnasium flattens it into a single array for the policy network.
# Verify it's in the raw generate_observation output:
raw_obs = env.generate_observation(state)
first_agent = list(raw_obs.keys())[0]
print("first agent key:", first_agent)
print("raw obs keys:", sorted(raw_obs[first_agent].keys()))
print("obs cbam_tariff_rate:", raw_obs[first_agent].get("cbam_tariff_rate", "NOT FOUND"))

# Check randomization
rates = []
for i in range(20):
    k = jax.random.PRNGKey(i)
    _, s = env.reset_env(k)
    rates.append(float(s["cbam_tariff_rate"]))
print("sampled rates:", rates)
n_zero = sum(1 for r in rates if r == 0.0)
n_cbam = sum(1 for r in rates if r > 0)
print(f"{n_zero} no-CBAM, {n_cbam} CBAM out of 20")
assert n_zero > 0, "Expected some 0.0 rates"
assert n_cbam > 0, "Expected some 0.8 rates"
print("randomization OK")

# --- Test 2: fixed mode ---
env2 = RiceMRIO(
    region_params=rp, num_regions=7, mrio_data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset"),
    mrio_trade=True, cbam_tariff_rate=0.0, cbam_randomize=False,
    dest_alloc_persistence=0.55, dest_alloc_baseline_decay=1.0,
    diff_reward_mode=True, num_discrete_action_levels=10,
    sectoral_welfloss=True, fixed_savings_rate=True, no_mitigation=True,
    sector_granularity="emissions-simple", welfare_loss_per_unit_tariff=50.0,
)
_, s2 = env2.reset_env(key)
print("fixed mode rate:", float(s2["cbam_tariff_rate"]))
assert float(s2["cbam_tariff_rate"]) == 0.0

# --- Test 3: obs/action space compatibility (can PPO handle it?) ---
wrapped = jym.LogWrapper(env)
wobs, wstate = wrapped.reset_env(key)
print("LogWrapper state cbam_tariff_rate:", wstate["cbam_tariff_rate"])

print("\nAll tests passed!")
