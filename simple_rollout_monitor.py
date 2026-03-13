#!/usr/bin/env python3
"""
Simple script to run a rollout of the JAX RICE environment with placeholder actions
and monitor production factor and production values for region 18.
"""

import jax
import jax.numpy as jnp
from rice_jax import Rice
from rice_jax.utils import load_region_yamls


def main():
    # Set up the environment
    num_regions = 20
    region_params = load_region_yamls(num_regions)

    env = Rice(
        num_regions=num_regions,
        region_params=region_params,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        action_window_size=1,
        disable_trading=False,
        negotiation_on=False,
        dmg_function="base",
        temperature_calibration="base",
        carbon_model="base",
        apply_welfloss=True,
        apply_welfgain=True,
        init_capital_multiplier=10.0,
        balance_interest_rate=0.1,
        consumption_substitution_rate=0.5,
        preference_for_domestic=0.5,
        init_gamma=0.99,
    )

    # Wrap with LogWrapper if needed (following main.py)
    from jaxnasium import LogWrapper
    env = LogWrapper(env)

    # Set up placeholder actions (fixed actions for all regions)
    # Using similar structure to FixedActionAgent
    random_action = env.sample_action(jax.random.PRNGKey(0))
    random_action_one_agent = random_action['agent_0']
    zero_action_one_agent = jax.tree.map(lambda x: jnp.zeros_like(x), random_action_one_agent)

    # Set some reasonable default values
    default_action_one_agent = zero_action_one_agent.copy()
    default_action_one_agent["savings_rate"] = 2  # Index for savings rate (out of 10 levels)
    default_action_one_agent["mitigation_rate"] = 0  # Index for mitigation rate
    default_action_one_agent["export_limit"] = 5  # Index for export limit

    # Create actions for all regions (same action for all)
    actions = {
        f'agent_{i}': default_action_one_agent.copy()
        for i in range(num_regions)
    }

    # Reset the environment
    key = jax.random.PRNGKey(42)
    obs, state = env.reset(key)

    print("Starting rollout...")
    print("Region 18 (index 17) monitoring:")
    print("Timestep | Production Factor | Production")
    print("-" * 40)

    # Run the rollout
    for timestep in range(env.episode_length):
        # Step the environment
        (obs, reward, terminated, truncated, info), state = env.step(key, state, actions)

        # Extract values for region 18 (index 17)
        production_factor = float(state["production_factor_all_regions"][17])
        production = float(state["production_all_regions"][17])

        print("5d")

        # Check if episode ended
        if terminated or truncated:
            print(f"\nEpisode ended at timestep {timestep}")
            break

        # Split key for next step
        key, _ = jax.random.split(key)

    print("\nRollout complete.")


if __name__ == "__main__":
    main()