# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Evaluation script for the rice environment
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
import json
import numpy as np
import yaml

from pathlib import Path

_path = Path(os.path.abspath(__file__))

from fixed_paths import PUBLIC_REPO_DIR
from gym.spaces import MultiDiscrete

# climate-cooperation-competition
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
print("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

# mila-sfdc-...
_PRIVATE_REPO_DIR = os.path.join(
    _path.parent.parent.parent.absolute(), "private-repo-clone"
)
sys.path.append(os.path.join(_PRIVATE_REPO_DIR, "backend"))
print("Using _PRIVATE_REPO_DIR = {}".format(_PRIVATE_REPO_DIR))


# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_SEED = 1234567890  # seed used for evaluation

_INDEXES_FILENAME = "climate_economic_min_max_indices.txt"

_METRICS_TO_LABEL_DICT = OrderedDict()
# Read the dict values below as
# (label, decimal points used to round off value: 0 becomes an integer)
_METRICS_TO_LABEL_DICT["reward_all_regions"] = ("Episode Reward", 2)
_METRICS_TO_LABEL_DICT["global_temperature"] = ("Temperature Rise", 2)
_METRICS_TO_LABEL_DICT["global_carbon_mass"] = ("Carbon Mass", 0)
_METRICS_TO_LABEL_DICT["capital_all_regions"] = ("Capital", 0)
_METRICS_TO_LABEL_DICT["production_all_regions"] = ("Production", 0)
_METRICS_TO_LABEL_DICT["gross_output_all_regions"] = ("Gross Output", 0)
_METRICS_TO_LABEL_DICT["investment_all_regions"] = ("Investment", 0)
_METRICS_TO_LABEL_DICT["abatement_cost_all_regions"] = ("Abatement Cost", 2)


def get_imports(framework=None):
    """
    Fetch relevant imports.
    """
    assert framework is not None
    if framework == "rllib":
        from train_with_rllib import (
            create_trainer,
            fetch_episode_states,
            load_model_checkpoints,
        )
    elif framework == "warpdrive":
        from train_with_warp_drive import (
            create_trainer,
            fetch_episode_states,
            load_model_checkpoints,
        )
    else:
        raise ValueError(f"Unknown framework {framework}!")
    return create_trainer, load_model_checkpoints, fetch_episode_states


def get_results_dir(zip_path=None):
    """
    Obtain the 'results' directory from the system arguments.
    """
    if zip_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--results_dir",
            "-r",
            type=str,
            default=".",
            help="the directory where all the submission files are saved. Can also be "
            "the zipped file containing all the submission files.",
        )
        args = parser.parse_args()
        if "results_dir" not in args:
            raise ValueError(
                "Please provide a results directory to evaluate with the argument -r"
            )
        if not os.path.exists(args.results_dir):
            raise ValueError(
                "The results directory is missing. Please make sure the correct path "
                "is specified!"
            )
        try:
            results_dir = args.results_dir

            # Also handle a zipped file
            if results_dir.endswith(".zip"):
                unzipped_results_dir = os.path.join("/tmp", str(time.time()))
                shutil.unpack_archive(results_dir, unzipped_results_dir)
                results_dir = unzipped_results_dir
            return results_dir, parser
        except Exception as err:
            raise ValueError("Cannot obtain the results directory") from err
    else:
        assert zip_path.endswith(".zip"), "The zip_path must end with .zip"
        try:
            unzipped_results_dir = os.path.join("/tmp", str(time.time()))
            shutil.unpack_archive(results_dir, unzipped_results_dir)
            results_dir = unzipped_results_dir
            return results_dir, parser
        except Exception as err:
            raise ValueError("Cannot obtain the results directory") from err


def fetch_base_env(base_folder=".tmp/_base"):
    """
    Download the base version of the code from GitHub.
    """
    if not base_folder.startswith('/'):
      base_folder = os.path.join(PUBLIC_REPO_DIR, base_folder)
      #print(f"Using tmp dir {base_folder}")
    if os.path.exists(base_folder):
        shutil.rmtree(base_folder)
    os.makedirs(base_folder, exist_ok=False)

    print(
        "\nDownloading a base version of the code from GitHub"
        " to run consistency checks..."
    )
    prev_dir = os.getcwd()
    os.chdir(base_folder)
    subprocess.call(["wget", "-O", "rice.py", _BASE_RICE_PATH])
    subprocess.call(["wget", "-O", "rice_helpers.py", _BASE_RICE_HELPERS_PATH])
    if "region_yamls" not in os.listdir(base_folder):
        shutil.copytree(
            os.path.join(PUBLIC_REPO_DIR, "region_yamls"),
            os.path.join(base_folder, "region_yamls"),
        )

    base_rice = import_class_from_path("Rice", os.path.join(base_folder, "rice.py"))()

    # Clean up base code
    os.chdir(prev_dir)
    shutil.rmtree(base_folder)
    return base_rice


def validate_dir(results_dir=None):
    """
    Validate that all the required files are present in the 'results' directory.
    """
    assert results_dir is not None
    framework = None

    files = os.listdir(results_dir)
    if ".warpdrive" in files:
        framework = "warpdrive"
        # Warpdrive was used for training
        for file in [
            "rice.py",
            "rice_helpers.py",
            "rice_cuda.py",
            "rice_step.cu",
            "rice_warpdrive.yaml",
        ]:
            if file not in files:
                success = False
                logging.error(
                    "%s is not present in the results directory: %s!", file, results_dir
                )
                comment = f"{file} is not present in the results directory!"
                break
            success = True
            comment = "Valid submission"
    elif ".rllib" in files:
        framework = "rllib"
        # RLlib was used for training
        for file in ["rice.py", "rice_helpers.py", "rice_rllib.yaml"]:
            if file not in files:
                success = False
                logging.error(
                    "%s is not present in the results directory: %s!", file, results_dir
                )
                comment = f"{file} is not present in the results directory!"
                break
            success = True
            comment = "Valid submission"
    else:
        success = False
        logging.error(
            "Missing identifier file! Either the .rllib or the .warpdrive "
            "file must be present in the results directory: %s",
            results_dir,
        )
        comment = "Missing identifier file!"
    print("comment", comment)
    return framework, success, comment


def compute_metrics(
    fetch_episode_states, trainer, framework, num_episodes=1, include_c_e_idx=True
):
    """
    Generate episode rollouts and compute metrics.
    """
    assert trainer is not None
    available_frameworks = ["rllib", "warpdrive"]
    assert (
        framework in available_frameworks
    ), f"Invalid framework {framework}, should be in f{available_frameworks}."

    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())

    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            if fetch_episode_states is not None:
                episode_states[episode_id] = fetch_episode_states(
                    trainer, desired_outputs
                )
            else:
                episode_states[episode_id] = trainer.fetch_episode_global_states(
                    desired_outputs
                )

        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )
        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    return success, comment, eval_metrics


def val_metrics(logged_ts, framework, num_episodes=1, include_c_e_idx=True):
    """
    Generate episode rollouts and compute metrics.
    """
    available_frameworks = ["rllib", "warpdrive"]
    assert (
        framework in available_frameworks
    ), f"Invalid framework {framework}, should be in f{available_frameworks}."

    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())
    episode_states = {}
    eval_metrics = {}
    try:
        for episode_id in range(num_episodes):
            episode_states[episode_id] = logged_ts

        for feature in desired_outputs:
            feature_values = [None for _ in range(num_episodes)]

            if feature == "global_temperature":
                # Get the temp rise for upper strata
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = (
                        episode_states[episode_id][feature][-1, 0]
                        - episode_states[episode_id][feature][0, 0]
                    )

            elif feature == "global_carbon_mass":
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = episode_states[episode_id][feature][
                        -1, 0
                    ]

            else:
                for episode_id in range(num_episodes):
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id][feature]
                    )

            # Compute mean feature value across episodes
            mean_feature_value = np.mean(feature_values)

            # Formatting the values
            metrics_to_label_dict = _METRICS_TO_LABEL_DICT[feature]

            eval_metrics[metrics_to_label_dict[0]] = perform_format(
                mean_feature_value, metrics_to_label_dict[1]
            )
        if include_c_e_idx:
            if not os.path.exists(_INDEXES_FILENAME):
                # Write min, max climate and economic index values to a file
                # for use during evaluation.
                indices_dict = generate_min_max_climate_economic_indices()
                # Write indices to a file
                with open(_INDEXES_FILENAME, "w", encoding="utf-8") as file_ptr:
                    file_ptr.write(json.dumps(indices_dict))
            with open(_INDEXES_FILENAME, "r", encoding="utf-8") as file_ptr:
                index_dict = json.load(file_ptr)
            eval_metrics["climate_index"] = np.round(
                (eval_metrics["Temperature Rise"] - index_dict["min_ci"])
                / (index_dict["max_ci"] - index_dict["min_ci"]),
                2,
            )
            eval_metrics["economic_index"] = np.round(
                (eval_metrics["Gross Output"] - index_dict["min_ei"])
                / (index_dict["max_ei"] - index_dict["min_ei"]),
                2,
            )
        success = True
        comment = "Successful submission"
    except Exception as err:
        logging.error(err)
        success = False
        comment = "Could not obtain an episode rollout!"
        eval_metrics = {}

    return success, comment, eval_metrics


def perform_format(val, num_decimal_places):
    """
    Format value to the number of desired decimal points.
    """
    if np.isnan(val):
        return val
    assert num_decimal_places >= 0
    rounded_val = np.round(val, num_decimal_places)
    if num_decimal_places == 0:
        return int(rounded_val)
    return rounded_val


def perform_evaluation(
    results_directory=None,
    num_episodes=1,
    eval_seed=None,
    skip_tests=True,
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None
    eval_metrics = {}
    assert num_episodes > 0

    framework, success, comment = validate_dir(results_directory)
    if success:
        logging.info("Running unit tests...")
        this_file_dir = os.path.dirname(os.path.abspath(__file__))

        try:
            if skip_tests:
                logging.info("Skipping check_output test")
            else:
                subprocess.check_output(
                    [
                        "python",
                        os.path.join(this_file_dir, "run_unittests.py"),
                        "--results_dir",
                        results_directory,
                    ],
                )
                logging.info("check_output test is done")

            if success:
                (
                    create_trainer,
                    load_model_checkpoints,
                    fetch_episode_states,
                ) = get_imports(framework=framework)

                logging.info("Performing eval...")

                # Load a run configuration
                config_file = os.path.join(results_directory, f"rice_{framework}.yaml")

                if not os.path.exists(config_file):
                    success = False
                    comment = (
                        f"The run configuration is missing in {results_directory}."
                    )

                else:
                    with open(config_file, "r", encoding="utf-8") as file_ptr:
                        run_config = yaml.safe_load(file_ptr)

                    # Create trainer object
                    try:
                        if "region_yamls" not in os.listdir(results_directory):
                          shutil.copytree(
                            os.path.join(PUBLIC_REPO_DIR, "region_yamls"),
                            os.path.join(results_directory, "region_yamls"),
                          )

                        if not os.path.exists(
                            os.path.join(results_directory, "rice_build.cu")
                        ):
                            shutil.copyfile(
                                os.path.join(PUBLIC_REPO_DIR, "rice_build.cu"),
                                os.path.join(results_directory, "rice_build.cu"),
                            )

                        # ==================== debugging ====================
                        shutil.copyfile(
                            os.path.join(PUBLIC_REPO_DIR, "rice_helpers.py"),
                            os.path.join(results_directory, "rice_helpers.py"),
                            follow_symlinks=False,
                        )
                        print("results_directory", results_directory)
                        # ==================== debugging ====================
                        trainer, _ = create_trainer(
                            run_config, source_dir=results_directory, seed=eval_seed
                        )

                        # Load model checkpoints
                        try:
                            load_model_checkpoints(trainer, results_directory)

                            # Compute metrics
                            try:
                                success, comment, eval_metrics = compute_metrics(
                                    fetch_episode_states,
                                    trainer,
                                    framework,
                                    num_episodes=num_episodes,
                                )

                                if framework == "warpdrive":
                                    trainer.graceful_close()
                                logging.info("DONE!")

                            except Exception as err:
                                logging.error(err)
                                success = False
                                comment = "Count not fetch episode and compute metrics."

                        except Exception as err:
                            logging.error(err)
                            success = False
                            comment = "Could not load model checkpoints."

                    except Exception as err:
                        logging.error(err)
                        success = False
                        comment = (
                            "Could not create trainer with the run_config provided."
                        )
        except subprocess.CalledProcessError as err:
            logging.error(err)
            success = False
            comment = "Unit tests were not successful."

    return framework, success, eval_metrics, comment


def get_temp_rise_and_gross_output(env, actions):
    env.reset()
    for _ in range(env.episode_length):
        env.step(actions)
    temperature_array = env.global_state["global_temperature"]["value"]
    temperature_rise = temperature_array[-1, 0] - temperature_array[0, 0]

    total_gross_production = np.sum(
        env.global_state["gross_output_all_regions"]["value"]
    )
    return temperature_rise, total_gross_production


def generate_min_max_climate_economic_indices():
    """
    Generate min and max climate and economic indices for the leaderboard.
    0% savings, 100% mitigation => best climate index, worst economic index
    100% savings, 0% mitigation => worst climate index, best economic index
    """
    env = fetch_base_env()  # base rice env
    assert isinstance(
        env.action_space[0], MultiDiscrete
    ), "Unknown action space for env."
    all_zero_actions = {
        agent_id: np.zeros(
            len(env.action_space[agent_id].nvec),
            dtype=np.int32,
        )
        for agent_id in range(env.num_agents)
    }

    # 0% savings, 100% mitigation
    low_savings_high_mitigation_actions = {}
    savings_action_idx = 0
    mitigation_action_idx = 1
    for agent_id in range(env.num_agents):
        low_savings_high_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        low_savings_high_mitigation_actions[agent_id][
            mitigation_action_idx
        ] = env.num_discrete_action_levels
    # Best climate index, worst economic index
    best_ci, worst_ei = get_temp_rise_and_gross_output(
        env, low_savings_high_mitigation_actions
    )

    high_savings_low_mitigation_actions = {}
    for agent_id in range(env.num_agents):
        high_savings_low_mitigation_actions[agent_id] = all_zero_actions[
            agent_id
        ].copy()
        high_savings_low_mitigation_actions[agent_id][
            savings_action_idx
        ] = env.num_discrete_action_levels
    worst_ci, best_ei = get_temp_rise_and_gross_output(
        env, high_savings_low_mitigation_actions
    )

    index_dict = {
        "min_ci": float(worst_ci),
        "max_ci": float(best_ci),
        "min_ei": float(worst_ei),
        "max_ei": float(best_ei),
    }
    return index_dict


if __name__ == "__main__":
    logging.info("This script performs evaluation of your code.")
    results_dir = get_results_dir()[0]
    print("results_dir:", results_dir)
    framework_used, succeeded, metrics, comments = perform_evaluation(
        results_dir, eval_seed=_SEED
    )
    print(f"Framework used: {framework_used}")
    print(f"Succeeded: {succeeded}")
    print(f"Metrics: {metrics}")
    print(f"Comments: {comments}")
