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
from evaluate_submission import _METRICS_TO_LABEL_DICT, try_to_unzip_file, validate_dir
import numpy as np
import yaml
import ray
from pathlib import Path

from train_with_rllib import (
            create_trainer,
            load_model_checkpoints,
            fetch_episode_states
        )
from experiments import fetch_episode_states_tariff_test, fetch_episode_states_trade_preference, fetch_episode_states_get_imports

EXP = {"tariff":fetch_episode_states_tariff_test,
        "none":fetch_episode_states,
        "dompref":fetch_episode_states_trade_preference,
        "gettrade":fetch_episode_states_get_imports}

_path = Path(os.path.abspath(__file__))

from fixed_paths import PUBLIC_REPO_DIR
sys.path.append(os.path.join(PUBLIC_REPO_DIR, "scripts"))
print("Using PUBLIC_REPO_DIR = {}".format(PUBLIC_REPO_DIR))

_PRIVATE_REPO_DIR = os.path.join(_path.parent.parent.parent.absolute(), "private-repo-clone")
sys.path.append(os.path.join(_PRIVATE_REPO_DIR, "backend"))
print("Using _PRIVATE_REPO_DIR = {}".format(_PRIVATE_REPO_DIR))


# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_SEED = np.random.randint(0,1000) #1234567890  # seed used for evaluation


def get_results_dir():
    """
    Obtain the 'results' directory from the system arguments.
    """
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="./Submissions/1709895189_default.zip",  # an example of a submission file
        help="The directory where all the submission files are saved. Can also be "
        "a zip-file containing all the submission files.",
    )

    parser.add_argument(
        "--exp",
        "-e",
        type=str,
        default="none",
        help="experiment_type",
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
    
    experiment = args.exp
    logging.info(f"performing experiment {experiment}")
    return results_dir, experiment

def perform_evaluation(
    results_directory=None,
    eval_seed=None,
    experiment = "none"
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None

    framework = 'rllib'
    config_file = os.path.join(results_directory, f"rice_{framework}.yaml")
    with open(config_file, "r", encoding="utf-8") as file_ptr:
                        run_config = yaml.safe_load(file_ptr)
    run_config["num_workers"]=8
    run_config["num_gpus"]=0
    trainer = create_trainer(
            run_config, source_dir=results_directory, seed=eval_seed
        )
    load_model_checkpoints(trainer, results_directory)
    # Fetch all the desired outputs to compute various metrics.
    desired_outputs = list(_METRICS_TO_LABEL_DICT.keys())
    # Add auxiliary outputs required for processing
    required_outputs = desired_outputs + ["activity_timestep"]
    logging.info("beginning experiment")
    #run experiment
    EXP[experiment](trainer, required_outputs)
    logging.info("experiment complete")


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True, local_mode = True)
    results_dir, experiment = get_results_dir()

    perform_evaluation(
        results_dir, eval_seed=_SEED, experiment = experiment
    )