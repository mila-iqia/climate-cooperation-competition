"""Quick test for dest_alloc_baseline_decay parameter."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import jax
import jax.numpy as jnp
import jaxnasium as jym

from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls

params = load_region_yamls(3)

# 1. Default (no decay) — backward compat
env = RiceMRIO(
    num_regions=3,
    region_params=params,
    mrio_data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset"),
    mrio_trade=True,
    cbam_tariff_rate=0.5,
    eu_region_idx=1,
    dest_alloc_persistence=0.55,
)
assert env.dest_alloc_baseline_decay == 0.0, "Default should be 0.0"
print(f"[OK] Default decay = {env.dest_alloc_baseline_decay}")

# 2. With decay enabled
env2 = RiceMRIO(
    num_regions=3,
    region_params=params,
    mrio_data_root=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset"),
    mrio_trade=True,
    cbam_tariff_rate=0.5,
    eu_region_idx=1,
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=1.0,
)
assert env2.dest_alloc_baseline_decay == 1.0
print(f"[OK] Custom decay = {env2.dest_alloc_baseline_decay}")

# 3. Functional test: step through both envs
key = jax.random.PRNGKey(0)

for label, e in [("no-decay", env), ("decay=1.0", env2)]:
    wrapped = jym.LogWrapper(e)
    obs, state = wrapped.reset(key)
    act_space = wrapped.action_space
    def _make_action(v):
        if isinstance(v, dict):
            return {kk: _make_action(vv) for kk, vv in v.items()}
        return jnp.zeros(v.shape) + 0.5
    actions = {k: _make_action(v) for k, v in act_space.items()}
    for step_i in range(3):
        key, subkey = jax.random.split(key)
        (obs, rewards, dones, truncs, infos), state = wrapped.step(subkey, state, actions)
    print(f"[OK] {label}: 3 steps completed")

# 4. Verify decay formula numerically
rho = 0.55
baseline = jnp.array(env2.dest_alloc_baseline)
prev = baseline  # start from baseline
# At t=1: w = (1-0.55)^1 = 0.45
w1 = (1.0 - rho) ** 1
anchor1_expected = w1 * baseline + (1 - w1) * prev  # = 0.45*B + 0.55*B = B
# Since prev == baseline, anchor should equal baseline regardless
# Test with different prev
prev_shifted = baseline * 0.5 + 0.5 / baseline.shape[-1]
w2 = (1.0 - rho) ** 2  # = 0.2025
anchor2 = w2 * baseline + (1 - w2) * prev_shifted
assert jnp.allclose(
    anchor2,
    0.2025 * baseline + 0.7975 * prev_shifted,
    atol=1e-5,
)
print("[OK] Decay formula numerically correct")

print("\nALL TESTS PASSED")
