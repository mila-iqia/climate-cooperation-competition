import json
import os
import time
from copy import copy
from typing import Any

import equinox as eqx
import jax
import matplotlib.pyplot as plt
from matplotlib import gridspec


def empty_info_log_fn(*args, **kwargs) -> dict:
    """Simply returns an empty dict. When the training algorithm logs
    all the info dicts per step, this is useful to avoid massive memory requirements.
    Note that env.wrappers could still insert info into the info dict.
    """
    return {}


def _transpose_actions(actions: dict) -> dict:
    """Transpose agent-keyed actions to action-keyed form.

    Input:  ``{agent_id: {action_key: value, ...}, ...}``
    Output: ``{action_key: {agent_id: value, ...}, ...}``
    """
    all_actions, agent_structure = eqx.tree_flatten_one_level(actions)
    action_structure = jax.tree.structure(all_actions[0])
    return jax.tree.transpose(agent_structure, action_structure, actions)


def actions_rewards_info_log_fn(state: dict, actions: dict, rewards, **kwargs) -> dict:
    """Minimal per-step logger: transposed actions and bare rewards.

    Default ``log_info_fn`` for the base Rice env.
    """
    info = {"actions": _transpose_actions(actions)}
    info["rewards"] = rewards
    return info


def full_state_info_log_fn(state: dict, actions: dict, rewards=None, **kwargs) -> dict:
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

    info.update(actions_rewards_info_log_fn(state, actions, rewards=rewards, **kwargs))
    return info


def _validate_full_state_logging(episode_log: dict[str, Any]) -> None:
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
    episode_log: dict[str, Any],
    output_folder: str,
    agent: Any,
    env: Any,
    episode_id: int = 0,
    additional_metadata: dict[str, Any] = None,
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
            "learning_rate_start": getattr(agent, "learning_rate_start", None),
            "ent_coef_start": getattr(agent, "ent_coef_start", None),
            "batch_size": getattr(agent, "batch_size", None),
            "num_epochs": getattr(agent, "num_epochs", None),
            "clip_coef": getattr(agent, "clip_coef", None),
            "vf_coef": getattr(agent, "vf_coef", None),
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
    parameter_keys: list[str],
    output_dir: str = "plots",
    figsize: tuple = (12, 8),
    dpi: int = 300,
) -> list[str]:
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

    # known number of true timesteps
    N = 20

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

    number_negotiation_steps = actual_timesteps / N

    years = [start_year + i * years_per_step for i in range(N)]

    # Create single combined plot with grid layout
    plot_file = _create_combined_plot(
        episode_data,
        parameter_keys,
        years,
        output_dir,
        base_filename,
        figsize,
        dpi,
        number_negotiation_steps,
    )

    return [plot_file] if plot_file else []


def _get_actual_timesteps(episode_data: dict[str, Any]) -> int:
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
    episode_data: dict[str, Any],
    parameter_keys: list[str],
    years: list[int],
    output_dir: str,
    base_filename: str,
    figsize: tuple,
    dpi: int,
    number_negotiation_steps: int,
) -> str | None:
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

                # use every Nth value depending on number of negotiation steps.
                temp_data_atmosphere = values = [
                    x
                    for i, x in enumerate(temp_data["atmosphere"], 1)
                    if i % number_negotiation_steps == 0
                ]
                ax.plot(
                    years,
                    temp_data_atmosphere,
                    label="Atmosphere",
                    linewidth=2,
                    linestyle="-",
                    color="#1f77b4",
                )

                # use every Nth value depending on number of negotiation steps.
                temp_data_lower_ocean = [
                    x
                    for i, x in enumerate(temp_data["lower_ocean"], 1)
                    if i % number_negotiation_steps == 0
                ]
                ax.plot(
                    years,
                    temp_data_lower_ocean,
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
                            # get every Nth value depending on number of intermediate negotiation steps
                            values = [
                                x
                                for i, x in enumerate(values, 1)
                                if i % number_negotiation_steps == 0
                            ]

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


import numpy as np


def compute_consumption_breakdown(
    json_log_path: str,
    output_dir: str = "plots",
    figsize: tuple = (14, 8),
    dpi: int = 300,
) -> str:
    """Computes and visualizes domestic vs foreign consumption breakdown over time.

    The resulting figure contains two parts:
    1. A set of bar charts (one per region) showing the share of consumption
       coming from domestic production vs imported goods.  Each region also
       overlays a secondary line plot (twin y-axis) displaying the export
       ratio (`export_limit_all_regions`) over time.
    2. A heatmap of average bilateral imports expressed as a percentage of the
       recipient region's consumption.

    Args:
        json_log_path: Path to the JSON log file created by log_episode_to_json
        output_dir: Directory to save the plot file
        figsize: Figure size for matplotlib plots
        dpi: DPI for saved plots

    Returns:
        str: Path to the created plot file
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

    # Get actual data length
    actual_timesteps = _get_actual_timesteps(episode_data)
    N = 20  # True number of timesteps (excluding negotiation steps)
    number_negotiation_steps = actual_timesteps / N

    # Get time axis (years)
    start_year = env_params["start_year"]
    years_per_step = env_params["years_per_step"]
    years = [start_year + i * years_per_step for i in range(N)]

    # Extract relevant data
    aggregate_consumption = episode_data["aggregate_consumption"]
    imports_minus_tariffs = episode_data["imports_minus_tariffs"]

    # imports_minus_tariffs is [timesteps, from_region, to_region]
    # We need to convert it to account for negotiation steps
    imports_array = np.array(imports_minus_tariffs)

    # Get every Nth value depending on number of negotiation steps
    imports_sampled = imports_array[
        [
            (i - 1) % number_negotiation_steps == 0
            for i in range(1, actual_timesteps + 1)
        ]
    ]

    num_regions = imports_sampled.shape[1]

    # We'll combine the bar charts and heatmap in a single figure using gridspec

    # calculate height ratio: 3 for bars, 1 for heatmap
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, num_regions, height_ratios=[3, 1])

    colors = ["#2ca02c", "#d62728"]  # Green for domestic, red for foreign

    # top row: bar charts for each region
    for region_id in range(num_regions):
        ax = fig.add_subplot(gs[0, region_id])

        # Get consumption for this region
        region_key = str(region_id)
        consumption_data = aggregate_consumption[region_key]
        consumption_sampled = [
            x
            for i, x in enumerate(consumption_data, 1)
            if i % number_negotiation_steps == 0
        ]
        consumption_array = np.array(consumption_sampled)

        # Get total imports into this region (sum over all source regions)
        imported_consumption = imports_sampled[:, :, region_id].sum(axis=1)

        # Calculate domestic consumption
        domestic_consumption = consumption_array - imported_consumption

        # Ensure non-negative (handle small numerical errors)
        domestic_consumption = np.maximum(domestic_consumption, 0)
        imported_consumption = np.maximum(imported_consumption, 0)

        # Calculate percentages
        total_consumption = domestic_consumption + imported_consumption
        domestic_pct = (
            domestic_consumption / np.maximum(total_consumption, 1e-8)
        ) * 100
        foreign_pct = (imported_consumption / np.maximum(total_consumption, 1e-8)) * 100

        # Create stacked bar chart
        ax.bar(years, domestic_pct, label="Domestic", color=colors[0], alpha=0.8)
        ax.bar(
            years,
            foreign_pct,
            bottom=domestic_pct,
            label="Foreign (Imported)",
            color=colors[1],
            alpha=0.8,
        )

        # Add export ratio line using secondary y-axis
        if "export_limit_all_regions" in episode_data:
            export_data = episode_data["export_limit_all_regions"][str(region_id)]
            export_sampled = [
                x
                for i, x in enumerate(export_data, 1)
                if i % number_negotiation_steps == 0
            ]
            ax2 = ax.twinx()
            ax2.plot(
                years,
                export_sampled,
                color="#9467bd",
                linestyle="-",
                linewidth=2,
                label="Export ratio",
            )
            ax2.set_ylabel("Export limit")
            if region_id == 0:
                ax2.legend(loc="lower right")

        # Customize plot
        ax.set_xlabel("Year")
        ax.set_ylabel("Consumption (%)")
        ax.set_title(f"Region {region_id}: Domestic vs Foreign")
        ax.set_ylim([0, 100])
        if region_id == 0:
            ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45)

    # bottom row: heatmap showing average imports share per receiving region
    ax_heat = fig.add_subplot(gs[1, :])

    # compute average imports over timesteps
    avg_imports = imports_sampled.mean(axis=0)  # shape (from, to)

    # average consumption per receiving region
    avg_consumption_per_region = np.array(
        [
            np.mean(
                [
                    x
                    for i, x in enumerate(aggregate_consumption[str(r)])
                    if i % number_negotiation_steps == 0
                ]
            )
            for r in range(num_regions)
        ]
    )

    # percentage of receiving region consumption
    heatmap_data = (
        avg_imports / np.maximum(avg_consumption_per_region[np.newaxis, :], 1e-8)
    ) * 100

    im = ax_heat.imshow(heatmap_data, cmap="viridis", aspect="auto")
    ax_heat.set_title("Average imports (% of recipient consumption)")
    ax_heat.set_xlabel("Recipient region")
    ax_heat.set_ylabel("Source region")
    ax_heat.set_xticks(range(num_regions))
    ax_heat.set_yticks(range(num_regions))
    ax_heat.set_xticklabels([f"R{r}" for r in range(num_regions)])
    ax_heat.set_yticklabels([f"R{r}" for r in range(num_regions)])
    plt.colorbar(im, ax=ax_heat, fraction=0.05, pad=0.05)

    # finalize and save
    filename = f"episode_{timestamp}_ep{episode_id}_consumption_breakdown.png"
    filepath = os.path.join(output_dir, filename)

    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()

    return filepath
