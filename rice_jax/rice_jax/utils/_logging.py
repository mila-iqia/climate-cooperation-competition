import json
import os
import time
from copy import copy
from typing import Any, Dict, List, Optional

import equinox as eqx
import jax
import matplotlib.pyplot as plt


def empty_info_log_fn(state: dict, actions: dict) -> dict:
    """Simply returns an empty dict. When the training algorithm logs
    all the info dicts per step, this is useful to avoid massive memory requirements.
    Note that env.wrappers could still insert info into the info dict.
    """
    return {}


def full_state_info_log_fn(state: dict, actions: dict) -> dict:
    info = copy(state)

    keys = [key for key in info.keys()]
    per_region_keys = [key for key in keys if key.endswith("_all_regions")]
    per_region_keys += ["aggregate_consumption"]
    trade_states = [
        "import_tariffs",
        "normalized_import_bids_all_regions",
        "import_bids_all_regions",
        "imports_minus_tariffs",
    ]
    per_region_keys = set(per_region_keys) - set(trade_states)
    for key in per_region_keys:
        this_key_region_dict = {
            region_id: info[key][region_id] for region_id in range(info[key].shape[0])
        }
        info[key] = this_key_region_dict
    # for key in trade_states:
    # # NOTE: this causes insane memory requirements in creating eval runs
    #     this_key_region_dict = {
    #         f"from-{region_id}": {
    #             f"to-{region_id_2}": info[key][region_id, region_id_2]
    #             for region_id_2 in range(info[key].shape[1])
    #         }
    #         for region_id in range(info[key].shape[0])
    #     }
    #     info[key] = this_key_region_dict
    info["global_temperature"] = {
        "atmosphere": info["global_temperature"][0],
        "lower_ocean": info["global_temperature"][1],
    }
    info["global_carbon_mass"] = {
        "atmosphere": info["global_carbon_mass"][0],
        "upper_ocean": info["global_carbon_mass"][1],
        "lower_ocean": info["global_carbon_mass"][2],
    }

    # actions
    # Actions are in form {'agent_id': {action_key: action_value, ...}, ...}
    # We want to transpose that to {action_key: {region_id: action_value, ...}, ...}
    all_actions, agent_structure = eqx.tree_flatten_one_level(actions)
    action_structure = jax.tree.structure(all_actions[0])
    actions = jax.tree.transpose(agent_structure, action_structure, actions)
    info["actions"] = actions

    return info


def _validate_full_state_logging(episode_log: Dict[str, Any]) -> None:
    """Validates that episode_log contains keys that indicate full_state_info_log_fn was used.

    Args:
        episode_log: The episode log from run_single_episode
    """
    # Check for key indicators that full_state_info_log_fn was used
    expected_keys = [
        "actions",  # Added by full_state_info_log_fn
        "global_temperature",  # Restructured by full_state_info_log_fn
        "global_carbon_mass",  # Restructured by full_state_info_log_fn
    ]

    for key in expected_keys:
        assert key in episode_log, (
            f"Key '{key}' not found in episode log. This suggests full_state_info_log_fn was not used."
        )


def log_episode_to_json(
    episode_log: Dict[str, Any],
    output_folder: str,
    agent: Any,
    env: Any,
    episode_id: int = 0,
    additional_metadata: Dict[str, Any] = None,
) -> str:
    """Logs episode data to a JSON file with agent and environment parameters.

    Args:
        episode_log: The episode log from run_single_episode
        output_folder: Directory to save the JSON file
        agent: The agent used (PPO, FixedActionAgent, etc.)
        env: The Rice environment instance
        episode_id: Optional episode identifier
        additional_metadata: Additional metadata to include in the log

    Returns:
        str: Path to the created JSON file

    Raises:
        AssertionError: If episode_log doesn't contain full_state_info_log_fn data
    """
    # Validate that full_state_info_log_fn was used
    _validate_full_state_logging(episode_log)

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Generate timestamp and filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"episode_log_{timestamp}_ep{episode_id}.json"
    filepath = os.path.join(output_folder, filename)

    # Extract agent parameters
    agent_params = {}
    if hasattr(agent, "__dict__"):
        # For PPO agents, extract relevant parameters
        agent_params = {
            "agent_type": type(agent).__name__,
            "learning_rate": getattr(agent, "learning_rate", None),
            "batch_size": getattr(agent, "batch_size", None),
            "num_epochs": getattr(agent, "num_epochs", None),
            "clip_epsilon": getattr(agent, "clip_epsilon", None),
            "value_loss_coef": getattr(agent, "value_loss_coef", None),
            "entropy_coef": getattr(agent, "entropy_coef", None),
        }
        # Remove None values
        agent_params = {k: v for k, v in agent_params.items() if v is not None}
    elif hasattr(agent, "default_actions"):
        # For FixedActionAgent
        agent_params = {
            "agent_type": "FixedActionAgent",
            "default_actions": agent.default_actions,
        }

    # Extract environment parameters
    env_params = {
        "num_regions": env.num_regions,
        "diff_reward_mode": env.diff_reward_mode,
        "relative_reward_mode": env.relative_reward_mode,
        "num_discrete_action_levels": env.num_discrete_action_levels,
        "action_window_size": env.action_window_size,
        "disable_trading": env.disable_trading,
        "negotiation_on": env.negotiation_on,
        "dmg_function": env.dmg_function,
        "temperature_calibration": env.temperature_calibration,
        "carbon_model": env.carbon_model,
        "apply_welfloss": env.apply_welfloss,
        "apply_welfgain": env.apply_welfgain,
        "init_capital_multiplier": env.init_capital_multiplier,
        "balance_interest_rate": env.balance_interest_rate,
        "consumption_substitution_rate": env.consumption_substitution_rate,
        "preference_for_domestic": env.preference_for_domestic,
        "init_gamma": env.init_gamma,
        "episode_length": env.episode_length,
        "start_year": env.start_year,
        "years_per_step": env.years_per_step,
    }

    # Prepare the complete log data
    log_data = {
        "metadata": {
            "timestamp": timestamp,
            "episode_id": episode_id,
            "log_info_fn": "full_state_info_log_fn",
            "num_timesteps": len(episode_log),
        },
        "agent_parameters": agent_params,
        "environment_parameters": env_params,
        "episode_data": episode_log,
    }

    # Add additional metadata if provided
    if additional_metadata:
        log_data["additional_metadata"] = additional_metadata

    # Convert JAX arrays to lists for JSON serialization
    def convert_jax_arrays(obj):
        """Recursively convert JAX arrays to Python lists for JSON serialization."""
        if hasattr(obj, "tolist"):  # JAX/NumPy array
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_jax_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_jax_arrays(item) for item in obj]
        else:
            return obj

    # Convert all JAX arrays to serializable format
    log_data = convert_jax_arrays(log_data)

    # Write to JSON file
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2)

    return filepath


def create_plots(
    json_log_path: str,
    parameter_keys: List[str],
    output_dir: str = "plots",
    figsize: tuple = (12, 8),
    dpi: int = 300,
) -> List[str]:
    """Creates matplotlib plots from JSON episode log data.

    Args:
        json_log_path: Path to the JSON log file created by log_episode_to_json
        output_dir: Directory to save the plot files
        parameter_keys: List of parameter keys to plot.
        figsize: Figure size for matplotlib plots
        dpi: DPI for saved plots

    Returns:
        List[str]: Paths to the created plot files

    Raises:
        FileNotFoundError: If JSON log file doesn't exist
        KeyError: If required keys are missing from the log data
    """

    # Load JSON data
    if not os.path.exists(json_log_path):
        raise FileNotFoundError(f"JSON log file not found: {json_log_path}")

    with open(json_log_path, "r") as f:
        log_data = json.load(f)

    # Extract metadata and episode data
    metadata = log_data["metadata"]
    episode_data = log_data["episode_data"]
    env_params = log_data["environment_parameters"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate base filename from metadata
    timestamp = metadata["timestamp"]
    episode_id = metadata["episode_id"]
    base_filename = f"episode_{timestamp}_ep{episode_id}"

    # Calculate time axis (years) - use actual data length instead of metadata
    start_year = env_params["start_year"]
    years_per_step = env_params["years_per_step"]

    # Get actual data length from the first available data array
    actual_timesteps = _get_actual_timesteps(episode_data)
    years = [start_year + i * years_per_step for i in range(actual_timesteps)]

    # Create single combined plot with grid layout
    plot_file = _create_combined_plot(
        episode_data, parameter_keys, years, output_dir, base_filename, figsize, dpi
    )

    return [plot_file] if plot_file else []


def _get_actual_timesteps(episode_data: Dict[str, Any]) -> int:
    """Gets the actual number of timesteps from the episode data."""
    # Try to find a data array to determine the actual length
    for key, data in episode_data.items():
        if isinstance(data, dict):
            # Check if it's region-specific data
            if all(isinstance(v, list) for v in data.values()):
                return len(next(iter(data.values())))
            # Check if it's nested data like global_temperature
            elif isinstance(data, dict) and any(
                isinstance(v, list) for v in data.values()
            ):
                for sub_data in data.values():
                    if isinstance(sub_data, list):
                        return len(sub_data)
        elif isinstance(data, list):
            return len(data)

    # Fallback: return 0 if no data found
    return 0


def _create_combined_plot(
    episode_data: Dict[str, Any],
    parameter_keys: List[str],
    years: List[int],
    output_dir: str,
    base_filename: str,
    figsize: tuple,
    dpi: int,
) -> Optional[str]:
    """Creates a single combined plot with grid layout for all parameters."""

    # Calculate grid dimensions (2 columns)
    num_plots = len(parameter_keys)
    num_rows = (num_plots + 1) // 2  # Round up for odd numbers
    num_cols = 2

    # Create figure with subplots
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(figsize[0] * 2, figsize[1] * num_rows)
    )

    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Line styles for multiple regions
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    plot_count = 0

    for i, param_key in enumerate(parameter_keys):
        if plot_count >= len(axes):
            break

        ax = axes[plot_count]

        try:
            # Special handling for combined temperature plot
            if param_key == "global_temperature":
                temp_data = episode_data["global_temperature"]
                ax.plot(
                    years,
                    temp_data["atmosphere"],
                    label="Atmosphere",
                    linewidth=2,
                    linestyle="-",
                    color="#1f77b4",
                )
                ax.plot(
                    years,
                    temp_data["lower_ocean"],
                    label="Ocean",
                    linewidth=2,
                    linestyle="--",
                    color="#ff7f0e",
                )
                ax.legend()
            else:
                # Handle nested keys (e.g., "global_temperature.atmosphere")
                if "." in param_key:
                    key_parts = param_key.split(".")
                    data = episode_data
                    for part in key_parts:
                        data = data[part]
                else:
                    data = episode_data[param_key]

                # Handle different data structures
                if isinstance(data, dict):
                    # Region-specific data (e.g., production_all_regions)
                    if all(isinstance(v, list) for v in data.values()):
                        # Data is organized by region
                        for j, (region_id, values) in enumerate(data.items()):
                            style = line_styles[j % len(line_styles)]
                            color = colors[j % len(colors)]
                            ax.plot(
                                years,
                                values,
                                label=f"Region {region_id}",
                                linewidth=2,
                                linestyle=style,
                                color=color,
                            )
                        ax.legend()
                    else:
                        # Single value data (e.g., global_temperature components)
                        ax.plot(years, data, linewidth=2)
                elif isinstance(data, list):
                    # Direct list data
                    ax.plot(years, data, linewidth=2)
                else:
                    raise ValueError(
                        f"Unexpected data type for {param_key}: {type(data)}"
                    )

            # Customize plot
            ax.set_xlabel("Year")
            ax.set_ylabel(param_key)
            ax.set_title(param_key)
            ax.grid(True, alpha=0.3)

            # Format x-axis to show years nicely
            ax.tick_params(axis="x", rotation=45)

            plot_count += 1

        except KeyError as e:
            print(f"Warning: Could not plot {param_key}: {e}")
            # Hide this subplot
            ax.set_visible(False)
            plot_count += 1
            continue
        except Exception as e:
            print(f"Error plotting {param_key}: {e}")
            # Hide this subplot
            ax.set_visible(False)
            plot_count += 1
            continue

    # Hide unused subplots
    for i in range(plot_count, len(axes)):
        axes[i].set_visible(False)

    # Save plot
    filename = f"{base_filename}_plot.png"
    filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()

    return filepath
