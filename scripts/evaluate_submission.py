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
from pathlib import Path
import json
import numpy as np
import yaml
import wandb

_path = Path(os.path.abspath(__file__))

from fixed_paths import PUBLIC_REPO_DIR
from run_unittests import fetch_base_env
from gymnasium.spaces import MultiDiscrete

# climate-cooperation-competition
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
logging.info("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

# mila-sfdc-...
_PRIVATE_REPO_DIR = os.path.join(
    _path.parent.parent.parent.absolute(), "private-repo-clone"
)
sys.path.append(os.path.join(_PRIVATE_REPO_DIR, "backend"))
logging.info("Using _PRIVATE_REPO_DIR = {}".format(_PRIVATE_REPO_DIR))


# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_EVAL_SEED = 1234567890  # seed used for evaluation

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
_METRICS_TO_LABEL_DICT["mitigation_rates_all_regions"] = ("Mitigation  Rate", 2)
_METRICS_TO_LABEL_DICT["savings_all_regions"] = ("Savings  Rate", 2)

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


def try_to_unzip_file(path):
    """
    Obtain the 'results' directory from the system arguments.
    """
    try:
        _unzipped_dir = os.path.join("/tmp", str(time.time()))
        shutil.unpack_archive(path, _unzipped_dir)
        return _unzipped_dir
    except Exception as err:
        raise ValueError("Cannot obtain the results directory") from err


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
                    "%s is not present in the results directory: %s!",
                    file,
                    results_dir,
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
                    "%s is not present in the results directory: %s!",
                    file,
                    results_dir,
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
    fetch_episode_states,
    trainer,
    framework,
    num_episodes=1,
    include_c_e_idx=True,
    log_config=None
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
    # Add auxiliary outputs required for processing
    required_outputs = desired_outputs + ["activity_timestep"]

    if log_config and log_config["enabled"]:
        wandb_config = log_config["wandb_config"]
        wandb.login(key=wandb_config["login"])
        wandb.init(project=wandb_config["project"],
            name=f'{wandb_config["run"]}_eval',
            entity=wandb_config["entity"])

    episode_states = {}
    eval_metrics = {}
    for episode_id in range(num_episodes):
        if fetch_episode_states is not None:
            episode_states[episode_id] = fetch_episode_states(
                trainer, required_outputs
            )
        else:
            episode_states[
                episode_id
            ] = trainer.fetch_episode_global_states(required_outputs)

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
                feature_values[episode_id] = episode_states[episode_id][
                    feature
                ][-1, 0]

        elif feature == "gross_output_all_regions":
            for episode_id in range(num_episodes):
                # collect gross output results based on activity timestep
                activity_timestep = episode_states[episode_id][
                    "activity_timestep"
                ]
                activity_index = np.append(
                    1.0, np.diff(activity_timestep.squeeze())
                )
                activity_index = [
                    np.isclose(v, 1.0) for v in activity_index
                ]
                feature_values[episode_id] = np.sum(
                    episode_states[episode_id]["gross_output_all_regions"][
                        activity_index
                    ]
                )

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

        if log_config and log_config["enabled"]:
            #TODO: fix dirty method to remove negotiation steps from results
            interval = (len(episode_states[episode_id][feature]) - 1) // 20
            ys = episode_states[episode_id][feature][0::interval].T


            xs = list(range(len(ys[0])))
            plot_name = feature.replace("_", " ").capitalize()

            if feature == "global_temperature":
                plot = wandb.plot.line_series(
                    xs=xs,
                    ys=ys.tolist(),
                    keys=["Atmosphere", "Ocean"],
                    title=plot_name,
                    xname="step",
                )
                wandb.log({plot_name: plot})
            elif feature == "global_carbon_mass":
                plot = wandb.plot.line_series(
                    xs=xs,
                    ys=ys.tolist(),
                    keys=["Atmosphere", "Upper ocean", "Lower ocean"],
                    title=plot_name,
                    xname="step",
                )
                wandb.log({plot_name: plot})
            elif feature.endswith("_all_regions"):
                value_name = feature[:-12].replace("_", " ")
                plot_name = value_name.capitalize()
                plot_name_mean = f"Mean {value_name}"
                ys_mean = np.mean(ys, axis=0)
                data = [[x, y] for (x, y) in zip(xs, ys_mean.tolist())]
                table = wandb.Table(data=data, columns=["step", value_name])
                plot_mean = wandb.plot.line(
                    table, "step", value_name, title=plot_name_mean
                )
                plot = wandb.plot.line_series(
                    xs=xs,
                    ys=ys.tolist(),
                    keys=[f"Region {x}" for x in range(len(ys))],
                    title=plot_name,
                    xname="step",
                )
                wandb.log({plot_name_mean: plot_mean})
                wandb.log({plot_name: plot})


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
    # except Exception as err:
    #     logging.error(err)
    #     success = False
    #     comment = "Could not obtain an episode rollout!"
    #     eval_metrics = {}

    if log_config and log_config["enabled"]:
        wandb.log({"climate_index": eval_metrics["climate_index"]})
        wandb.log({"economic_index": eval_metrics["economic_index"]})
        # attach submission file as artifact (needs to be named after the nego class)
        # artifact = wandb.Artifact("submission", type="model")
        # artifact.add_file(submission_file)
        # wandb.log_artifact(artifact)
        wandb.finish()

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
                    feature_values[episode_id] = episode_states[episode_id][
                        feature
                    ][-1, 0]

            elif feature == "gross_output_all_regions":
                for episode_id in range(num_episodes):
                    # collect gross output results based on activity timestep
                    activity_timestep = episode_states[episode_id][
                        "activity_timestep"
                    ]
                    activity_index = np.append(
                        1.0, np.diff(activity_timestep.squeeze())
                    )
                    activity_index = [
                        np.isclose(v, 1.0) for v in activity_index
                    ]
                    feature_values[episode_id] = np.sum(
                        episode_states[episode_id]["gross_output_all_regions"][
                            activity_index
                        ]
                    )
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
    results_directory,
    framework,
    num_episodes=1,
    eval_seed=None,
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None
    assert num_episodes > 0

    (
        create_trainer,
        load_model_checkpoints,
        fetch_episode_states,
    ) = get_imports(framework=framework)

    # Load a run configuration
    config_file = os.path.join(results_directory, f"rice_{framework}.yaml")
    

    try:
        assert os.path.exists(config_file)
    except Exception as err:
        logging.error(
            f"The run configuration is missing in {results_directory}."
        )
        raise err

    # Copy the PUBLIC region yamls and rice_build.cu to the results directory.
    if not os.path.exists(os.path.join(results_directory, "region_yamls")):
        shutil.copytree(
            os.path.join(PUBLIC_REPO_DIR, "region_yamls"),
            os.path.join(results_directory, "region_yamls"),
        )
    if not os.path.exists(os.path.join(results_directory, "rice_build.cu")):
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "rice_build.cu"),
            os.path.join(results_directory, "rice_build.cu"),
        )

    # Create Trainer object
    try:
        with open(config_file, "r", encoding="utf-8") as file_ptr:
            run_config = yaml.safe_load(file_ptr)
        if "logging" in run_config.keys():
            log_config = run_config["logging"]
        else:
            log_config = None
        trainer = create_trainer(
            run_config, source_dir=results_directory, seed=eval_seed
        )

    except Exception as err:
        logging.error(f"Could not create Trainer with the run_config provided.")
        raise err

    # Load model checkpoints
    try:
        load_model_checkpoints(trainer, results_directory)
    except Exception as err:
        logging.error(f"Could not load model checkpoints.")
        raise err

    # Compute metrics
    try:
        success, comment, eval_metrics = compute_metrics(
            fetch_episode_states,
            trainer,
            framework,
            num_episodes=num_episodes,
            log_config = log_config
        )

        if framework == "warpdrive":
            trainer.graceful_close()

        return success, eval_metrics, comment

    except Exception as err:
        logging.error(f"Count not fetch episode and compute metrics.")
        raise err


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

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="./Submissions/1680502535.zip",  # an example of a submission file
        help="The directory where all the submission files are saved. Can also be "
        "a zip-file containing all the submission files.",
    )
    args = parser.parse_args()

    # Check the submission zip file or directory.
    if "results_dir" not in args:
        raise ValueError(
            "Please provide a results directory to evaluate with the argument -r"
        )

    if not os.path.exists(args.results_dir):
        raise ValueError(
            "The results directory is missing. Please make sure the correct path "
            "is specified!"
        )

    results_dir = (
        try_to_unzip_file(args.results_dir)
        if args.results_dir.endswith(".zip")
        else args.results_dir
    )

    logging.info(f"Using submission files in {results_dir}")

    # Validate the submission directory
    framework, results_dir_is_valid, comment = validate_dir(results_dir)
    if not results_dir_is_valid:
        raise AssertionError(
            f"{results_dir} is not a valid submission directory."
        )

    # Run unit tests on the simulation files
    skip_unit_tests = True  # = args.skip_unit_tests

    try:
        if skip_unit_tests:
            logging.info("Skipping check_output test")
        else:
            logging.info("Running unit tests...")
            subprocess.check_output(
                [
                    "python",
                    "run_unittests.py",
                    "--results_dir",
                    results_dir,
                ],
            )
            logging.info("run_unittests.py is done")
    except subprocess.CalledProcessError as err:
        logging.error(f"{results_dir}: unit tests were not successful.")
        raise err

    # Run evaluation with submitted simulation and trained agents.
    logging.info("Starting eval...")
    succeeded, metrics, comments = perform_evaluation(
        results_dir, framework, eval_seed=_EVAL_SEED
    )

    # Report results.
    eval_result_str = "\n".join(
        [
            f"Framework used: {framework}",
            f"Succeeded: {succeeded}",
            f"Metrics: {metrics}",
            f"Comments: {comments}",
        ]
    )
    logging.info(eval_result_str)
    print(eval_result_str)
