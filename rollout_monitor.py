#!/usr/bin/env python3
"""
Simple rollout script to monitor production and production factors for region 18
in the JAX RICE environment with placeholder actions.
"""

import jax
import jax.numpy as jnp
from rice_jax import Rice
from rice_jax.utils import load_region_yamls

def main():
    # Load region parameters for 20 regions
    region_params = load_region_yamls(20)

    # Create RICE environment
    env = Rice(
        region_params=region_params,
        num_regions=20,
        num_discrete_action_levels=10,
        disable_trading=False,
        negotiation_on=False,
        dmg_function="base",
        temperature_calibration="base",
        carbon_model="base",
        apply_welfloss=True,
        apply_welfgain=True
    )

    # Reset environment
    key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
    obs, state = env.reset_env(key)
    print(obs)

    print("Starting rollout with placeholder actions...")
    print("Timestep | Region 18 Production Factor | Region 18 Production | Global Temp")
    print("-" * 70)

    # Run rollout for full episode
    timestep = 0
    done = False

    while not done and timestep < env.episode_length:
        # Print initial values for region 18
        # prod_factor = state["production_factor_all_regions"][18]
        # production = state["production_all_regions"][18]
        # global_temp = state["global_temperature"][0]

        print("5d")

        # Use placeholder actions: fixed savings rate, zero mitigation
        actions = env.sample_action(key)
        # Override with placeholder values
        for agent in actions:
            actions[agent]["savings_rate"] = 0.25  # 25% savings
            actions[agent]["mitigation_rate"] = 0.0  # No mitigation

        # Convert actions to the format expected by step_climate_and_economy
        actions_array = env.process_actions(actions, state)

        # Debug: check state before step
        print(f"Before step - production_factor shape: {state['production_factor_all_regions'].shape if state['production_factor_all_regions'] is not None else 'None'}")

        # Step environment directly (bypass observation generation)
        state = env.step_climate_and_economy(state, actions_array)
        # prod_factor = state["production_factor_all_regions"][18]
        # production = state["production_all_regions"][18]
        # global_temp = state["global_temperature"][0]

        # Debug: check state after step
        print(f"After step - production_factor shape: {state['production_factor_all_regions'].shape if state['production_factor_all_regions'] is not None else 'None'}")
        print(f"State keys: {list(state.keys())}")

        done = state["current_simulation_year"] >= env.start_year + (env.episode_length - 1) * env.years_per_step
        timestep += 1

        # Stop after reasonable number of steps for monitoring
        if timestep >= 10:  # Just show first 10 steps
            print("... (stopping after 10 steps for brevity)")
            break

    print("\nRollout complete!")
    print(f"Final timestep: {timestep}")
    print(".2f")
    print(".2f")

if __name__ == "__main__":
    main()