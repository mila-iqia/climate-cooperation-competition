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
from tqdm import tqdm
from train_with_rllib import (
            create_trainer,
            load_model_checkpoints,
        )
from experiments import *
from evaluate_submission import get_imports

#more exps will grow
EXP = {"cl":run_carbon_leakage}
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
        default="./Submissions/1718376483_carbonleakage.zip",  # an example of a submission file
        help="The directory where all the submission files are saved. Can also be "
        "a zip-file containing all the submission files.",
    )

    parser.add_argument(
        "--exp",
        "-e",
        type=str,
        default="cl",
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
    framework, results_dir_is_valid, comment, _ = validate_dir(results_dir)
    if not results_dir_is_valid:
        raise AssertionError(
            f"{results_dir} is not a valid submission directory."
        )
    
    experiment = args.exp
    logging.info(f"performing experiment {experiment}")
    return results_dir, experiment

def perform_evaluation(
    results_directory,
    experiment,
    condition=None,
    num_episodes=1,
    framework="rllib",
    discrete = True,
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
        set_num_agents
    ) = get_imports(framework=framework)
    
    # Load a run configuration
    if discrete:
        yaml_path = f"rice_{framework}_discrete.yaml"
    else:
        yaml_path = f"rice_{framework}_cont.yaml"
    config_file = os.path.join(results_directory, yaml_path)

    
    
    try:
        assert os.path.exists(config_file)
    except Exception as err:
        logging.error(
            f"The run configuration is missing in {results_directory}."
        )
        raise err
    
    with open(config_file, "r", encoding="utf-8") as file_ptr:
        run_config = yaml.safe_load(file_ptr)
        #force eval on single worker
        run_config["trainer"]["num_workers"] = 0
        log_config = run_config["logging"]
    #update region yamls
    set_num_agents(run_config)
    

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
    # Fetch all the desired outputs to compute various metrics.
    # Add auxiliary outputs required for processing
    logging.info("beginning experiment")
    #run experiment
    EXP[experiment](trainer,condition=condition)
    logging.info("experiment complete")

if __name__ == "__main__":

    results_dir, experiment = get_results_dir()
    # ray.init(ignore_reinit_error=True, local_mode = True)
        
    # perform_evaluation(
    #     results_dir,condition="control", experiment = experiment
    # )
    for i in tqdm(range(20)):
        ray.init(ignore_reinit_error=True, local_mode = True)
        
        perform_evaluation(
            results_dir,condition="treatment", experiment = experiment
        )
        # perform_evaluation(
        #     results_dir,condition="control", experiment = experiment
        # )