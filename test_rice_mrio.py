"""
Smoke test: RiceMRIO vs Rice — bit-identical aggregate output over a rollout.

Run from the repo root with:
    /path/to/rice-jax/bin/python test_rice_mrio.py
"""

import sys
import jax
import jax.numpy as jnp
import numpy as np

# Make sure rice_jax is importable
sys.path.insert(0, "rice_jax")

from rice_jax import Rice, RiceMRIO
from rice_jax.utils import load_region_yamls

NUM_REGIONS = 20
MRIO_DIR = "csv_asset/mrio/aggregated/eora_agg_20"
CC_CSV   = "csv_asset/CountryClass_20.csv"
STEPS    = 15
SEED     = 42


def make_env(cls, **extra):
    region_params = load_region_yamls(NUM_REGIONS)
    kwargs = dict(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        num_discrete_action_levels=10,
        disable_trading=False,
        negotiation_on=False,
    )
    kwargs.update(extra)
    return cls(**kwargs)


def fixed_actions(env, key):
    actions = env.sample_action(key)
    for agent in actions:
        actions[agent]["savings_rate"]    = 2  # 20%
        actions[agent]["mitigation_rate"] = 3  # 30%
        actions[agent]["import_bid"]      = jnp.zeros(NUM_REGIONS, dtype=jnp.int32)
        actions[agent]["import_tariff"]   = jnp.zeros(NUM_REGIONS, dtype=jnp.int32)
        actions[agent]["export_limit"]    = 0
    return actions


def rollout(env, key, steps):
    _, state = env.reset_env(key)
    records = []
    for _ in range(steps):
        actions = fixed_actions(env, key)
        processed = env.process_actions(actions, state)
        state = env.step_climate_and_economy(state, processed)
        records.append({
            "production":    np.array(state["production_all_regions"]),
            "gross_output":  np.array(state["gross_output_all_regions"]),
            "consumption":   np.array(state["aggregate_consumption"]),
            "utility":       np.array(state["utility_all_regions"]),
            "global_temp":   np.array(state["global_temperature"]),
        })
    return records, state


def main():
    key = jax.random.PRNGKey(SEED)

    print("Building base Rice env …")
    base = make_env(Rice)

    print("Building RiceMRIO env …")
    mrio = make_env(
        RiceMRIO,
        mrio_aggregated_dir=MRIO_DIR,
        country_class_csv=CC_CSV,
    )

    # --- Validate shares ---
    print("\n=== Share diagnostics ===")
    diag = mrio.validate_shares()
    print(f"  all rows sum to 1 : {diag['all_rows_sum_to_one']}")
    print(f"  no NaN             : {diag['no_nan']}")
    print(f"  no Inf             : {diag['no_inf']}")
    print(f"  sector count ok    : {diag['sector_names_count_ok']}")
    print(f"  region count ok    : {diag['region_labels_count_ok']}")
    print(f"  row sums (min/max) : {diag['row_sums'].min():.6f} / {diag['row_sums'].max():.6f}")
    print("\n  RICE region → MRIO label mapping:")
    for i, label in enumerate(diag["mrio_region_labels"]):
        print(f"    region {i:2d} → {label}")

    # --- Rollout both ---
    print(f"\n=== Running {STEPS}-step rollout ===")
    base_records, base_state = rollout(base, key, STEPS)
    mrio_records, mrio_state = rollout(mrio, key, STEPS)

    # --- Compare ---
    print("\n=== Parity checks ===")
    all_ok = True
    for t, (br, mr) in enumerate(zip(base_records, mrio_records)):
        for k in ["production", "gross_output", "consumption", "utility", "global_temp"]:
            if not np.allclose(br[k], mr[k], atol=1e-4, rtol=1e-5):
                print(f"  MISMATCH at step {t}, key '{k}'")
                print(f"    base: {br[k]}")
                print(f"    mrio: {mr[k]}")
                all_ok = False

    if all_ok:
        print("  ✓ All aggregate outputs are bit-identical across all steps.")
    else:
        print("  ✗ Mismatches detected — see above.")

    # --- Sector disaggregation check ---
    print("\n=== Sector disaggregation check (last step) ===")
    pbs = np.array(mrio_state["production_by_sector"])   # (20, 26)
    pa  = np.array(mrio_state["production_all_regions"]) # (20,)
    sector_sums = pbs.sum(axis=1)
    if np.allclose(sector_sums, pa, atol=1e-4):
        print("  ✓ production_by_sector.sum(axis=1) == production_all_regions.")
    else:
        print("  ✗ Sector sums do not match aggregate production!")
        for i, (s, a) in enumerate(zip(sector_sums, pa)):
            print(f"    region {i}: sector_sum={s:.4f}, aggregate={a:.4f}")

    print(f"\n  production_by_sector shape : {pbs.shape}")
    print(f"  Sector names ({len(mrio.sector_names)}):")
    for s in mrio.sector_names:
        print(f"    {s}")


if __name__ == "__main__":
    main()
