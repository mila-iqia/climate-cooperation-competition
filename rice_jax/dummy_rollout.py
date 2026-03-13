import os
import jax
from dataclasses import replace
import jaxnasium as jym

from rice_jax import Rice
from rice_jax.utils import load_region_yamls, full_state_info_log_fn, log_episode_to_json, create_plots
from _experiment_util import FixedActionAgent, run_single_episode


def main():
    # Configure environment
    num_regions = 20  # Change to 3 or 7 if needed
    region_params = load_region_yamls(num_regions)
    env_settings = {
        "region_params": region_params,
        "num_regions": num_regions,
        "diff_reward_mode": True,
        "num_discrete_action_levels": 10,
        # Add other settings as needed
    }
    env = Rice(**env_settings)
    env = replace(env, log_info_fn=full_state_info_log_fn)

    # Create dummy actions agent
    agent = FixedActionAgent(env)

    # Run episode
    seed = jax.random.PRNGKey(42)
    episode_logs = run_single_episode(seed, env, agent)

    # Log to JSON
    output_dir = "episode_logs"
    os.makedirs(output_dir, exist_ok=True)
    log_filepath = log_episode_to_json(
        episode_logs,
        output_folder=output_dir,
        agent=agent,
        env=env,
        episode_id=0,
        additional_metadata={"seed": int(seed[0])},
    )
    print(f"Logs saved to: {log_filepath}")

    # Optional: Create plots
    plot_files = create_plots(
        json_log_path=log_filepath,
        output_dir="plots",
        parameter_keys=[
            "global_temperature",
            "production_all_regions",
            "utility_all_regions",
            "actions.savings_rate",
            "actions.mitigation_rate",
        ],
        figsize=(12, 8),
        dpi=300,
    )
    print(f"Plots saved to: {plot_files}")


if __name__ == "__main__":
    main()