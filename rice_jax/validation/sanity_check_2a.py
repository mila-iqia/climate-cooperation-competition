import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rice_jax import Rice
from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls

KEY = jax.random.PRNGKey(42)
NR = 3  # 3-region standard setup (eora_agg_3 must exist)

# csv_asset is one level above rice_jax/
MRIO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "csv_asset")


def make_zero_actions(env, key=KEY):
    act = optax.tree.zeros_like(env.sample_action(key))
    for a in act:
        act[a]["savings_rate"] = 2.5
    return act


# ────────────────────────────────────────────────────────────────────────────
# CHECK 1: Phase 1B unchanged (mrio_trade=False bit-identical to Rice)
# ────────────────────────────────────────────────────────────────────────────
params = load_region_yamls(NR)
rice = Rice(num_regions=NR, region_params=params)
mrio_1b = RiceMRIO(num_regions=NR, region_params=params, mrio_data_root=MRIO_ROOT)

_, s_rice = rice.reset(KEY)
_, s_mrio = mrio_1b.reset(KEY)
act_rice = make_zero_actions(rice)
act_mrio_1b = make_zero_actions(mrio_1b)

for _ in range(5):
    (_, _, _, _, _), s_rice = rice.step_env(KEY, s_rice, act_rice)
    (_, _, _, _, _), s_mrio = mrio_1b.step_env(KEY, s_mrio, act_mrio_1b)

diff = float(jnp.max(jnp.abs(s_rice["production_all_regions"] - s_mrio["production_all_regions"])))
assert diff < 1e-5, f"Phase 1B bit-identity FAILED: max diff = {diff:.2e}"
print(f"CHECK 1 Phase 1B bit-identity: PASS  (max diff {diff:.2e})")


# ────────────────────────────────────────────────────────────────────────────
# CHECK 2: Phase 2A init — data loads without error
# ────────────────────────────────────────────────────────────────────────────
mrio_2a = RiceMRIO(
    num_regions=NR,
    region_params=params,
    mrio_data_root=MRIO_ROOT,
    mrio_trade=True,
    cbam_tariff_rate=0.0,
    eu_region_idx=0,
)
assert mrio_2a.dest_alloc_baseline is not None, "dest_alloc_baseline not loaded"
assert mrio_2a.total_export_frac is not None, "total_export_frac not loaded"
assert mrio_2a.emissions_intensity is not None, "emissions_intensity not loaded"
diag = mrio_2a.validate_shares()
assert diag["dest_alloc_no_nan"], "NaN in dest_alloc_baseline"
assert diag["total_export_frac_in_0_1"], "total_export_frac out of [0,1]"
print(f"CHECK 2 Phase 2A data load: PASS")
print(f"  dest_alloc_baseline shape: {mrio_2a.dest_alloc_baseline.shape}")
print(f"  total_export_frac range:   [{mrio_2a.total_export_frac.min():.3f}, {mrio_2a.total_export_frac.max():.3f}]")
print(f"  emissions_intensity range:  [{mrio_2a.emissions_intensity.min():.3e}, {mrio_2a.emissions_intensity.max():.3e}]")


# ────────────────────────────────────────────────────────────────────────────
# CHECK 3: Phase 2A reset — new state keys present
# ────────────────────────────────────────────────────────────────────────────
_, s0 = mrio_2a.reset(KEY)
assert "trade_flows" in s0, "trade_flows missing from initial state"
assert "cbam_revenue" in s0, "cbam_revenue missing from initial state"
assert s0["trade_flows"].shape == (NR, NR, mrio_2a.num_sectors)
assert s0["cbam_revenue"].shape == (NR,)
print(f"CHECK 3 Phase 2A state keys: PASS")


# ────────────────────────────────────────────────────────────────────────────
# CHECK 4: Zero export_reallocation → trade_flows ≈ MRIO baseline
# action=0 → uniform δ=-δ_max → softmax shift-invariant → baseline flows
# ────────────────────────────────────────────────────────────────────────────
act_2a = make_zero_actions(mrio_2a)

# Step once
(_, _, _, _, _), s1 = mrio_2a.step_env(KEY, s0, act_2a)

# Expected baseline trade flows: production * total_export_frac * dest_alloc_baseline
Y = s1["production_all_regions"]
shares = jnp.array(mrio_2a.sector_output_shares)
prod_by_sector = shares * Y[:, None]  # (NR, NS)
tef = jnp.array(mrio_2a.total_export_frac)
dab = jnp.array(mrio_2a.dest_alloc_baseline)  # (NR, NS, NR)
export_vol = prod_by_sector * tef  # (NR, NS)
trade_baseline_rsd = export_vol[:, :, None] * dab  # (NR, NS, NR)
trade_baseline = trade_baseline_rsd.transpose(0, 2, 1)  # (NR, NR, NS) [from,to,s]

max_diff_trade = float(jnp.max(jnp.abs(s1["trade_flows"] - trade_baseline)))
print(f"CHECK 4 Zero-delta → MRIO baseline: max diff = {max_diff_trade:.2e}  ", end="")
assert max_diff_trade < 1e-4, f"FAILED: {max_diff_trade:.2e}"
print("PASS")


# ────────────────────────────────────────────────────────────────────────────
# CHECK 5: Budget conservation — exports ≤ production_by_sector
# ────────────────────────────────────────────────────────────────────────────
# total exports from (r, s) = sum over destinations
total_exports_by_sector = s1["trade_flows"].sum(axis=1)  # (NR, NS) [from, sector]
overshoot = float(jnp.max(jnp.maximum(total_exports_by_sector - s1["production_by_sector"], 0)))
print(f"CHECK 5 Budget conservation: max overshoot = {overshoot:.2e}  ", end="")
assert overshoot < 1e-4, f"FAILED: exports exceed production by {overshoot:.2e}"
print("PASS")


# ────────────────────────────────────────────────────────────────────────────
# CHECK 6: CBAM wedge — non-zero cbam_tariff_rate reduces exporter welfare
# Uses synthetic emissions_intensity (scaled to produce visible wedge) because
# the raw EORA Q_S values have a unit mismatch (CO2 kt / monetary USD) that
# makes them ~3e-12.  The mechanism is correct; calibration is a data-prep task.
# ────────────────────────────────────────────────────────────────────────────
mrio_cbam = RiceMRIO(
    num_regions=NR,
    region_params=params,
    mrio_data_root=MRIO_ROOT,
    mrio_trade=True,
    cbam_tariff_rate=0.5,
    eu_region_idx=0,
)
# Override emissions intensity to 0.3 tCO2/unit for all region-sector pairs
# so CBAM has a visible bite (object.__setattr__ is the equinox-approved mutator)
synthetic_intensity = np.full(
    (NR, mrio_cbam.num_sectors), 0.3, dtype=np.float32
)
object.__setattr__(mrio_cbam, "emissions_intensity", synthetic_intensity)

_, s0_cbam = mrio_cbam.reset(KEY)
act_cbam = make_zero_actions(mrio_cbam)

(_, _, _, _, _), s1_nocbam = mrio_2a.step_env(KEY, s0, act_2a)
(_, _, _, _, _), s1_cbam = mrio_cbam.step_env(KEY, s0_cbam, act_cbam)

# Non-EU exporters should have strictly lower welfare with CBAM
for r in range(1, NR):
    u_no = float(s1_nocbam["utility_times_welfloss_all_regions"][r])
    u_with = float(s1_cbam["utility_times_welfloss_all_regions"][r])
    diff_pct = (u_no - u_with) / (abs(u_no) + 1e-8) * 100
    print(f"CHECK 6 Region {r} welfare: no_cbam={u_no:.6f}  cbam={u_with:.6f}  "
          f"reduction={diff_pct:.3f}%  ", end="")
    assert u_with < u_no - 1e-6, (
        f"CBAM should strictly reduce welfare for exporter {r}; "
        f"diff={u_no - u_with:.2e} (check: emissions_intensity scale and trade to EU)"
    )
    print("PASS")

# EU (region 0) cbam_revenue should be positive
cbam_rev = float(s1_cbam["cbam_revenue"][0])
print(f"CHECK 6 EU cbam_revenue = {cbam_rev:.6f}  ", end="")
assert cbam_rev > 0, "EU should collect positive CBAM revenue"
print("PASS")

# Confirm raw EORA intensities are near-zero (expected; unit calibration needed)
raw_max = float(mrio_2a.emissions_intensity.max())
print(f"  NOTE: raw EORA emissions_intensity max = {raw_max:.2e} "
      f"(unit mismatch CO2-kt vs monetary; calibration required before training)")


print("\nAll checks passed.")
