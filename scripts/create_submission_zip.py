# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Script to create the zipped submission file from the results directory
"""
import os
import shutil
import sys
import argparse
from evaluate_submission import validate_dir
import time
import yaml
from fixed_paths import PUBLIC_REPO_DIR
from pathlib import Path


sys.path.append(PUBLIC_REPO_DIR)

BACKWARDS_COMPAT_CONFIG = """
trainer:
    num_envs: 1 # number of environment replicas
    rollout_fragment_length: 100 # divide episodes into fragments of this many steps each during rollouts.
    train_batch_size: 2000 # total batch size used for training per iteration (across all the environments)
    num_episodes: 100 # number of episodes to run the training for
    framework: torch # framework setting.
    # Note: RLlib supports TF as well, but our end-to-end pipeline is built for Pytorch only.
    # === Hardware Settings ===
    num_workers: 1 # number of rollout worker actors to create for parallel sampling.
    # Note: Setting the num_workers to 0 will force rollouts to be done in the trainer actor.
    num_gpus: 0 # number of GPUs to allocate to the trainer process. This can also be fractional (e.g., 0.3 GPUs).
"""


def get_results_dir():
    """
    Obtain the 'results' directory from the system arguments.
    """
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


def prepare_submission_(results_dir=None):
    """
    # Validate all the submission files and compress into a .zip.
    Note: This method is also invoked in the trainer script itself!
    So if you ran the training script, you may not need to re-run this.
    Args results_dir: the directory where all the training files were saved.
    """
    assert results_dir is not None
    submission_filename = results_dir.split("/")[-1]
    submission_file = os.path.join(PUBLIC_REPO_DIR, "Submissions", submission_filename)

    validate_dir(results_dir)

    # Only copy the latest policy model file for submission
    results_dir_copy = os.path.join("/tmp", "_copies_", submission_filename)
    shutil.copytree(results_dir, results_dir_copy)

    policy_models = [
        os.path.join(results_dir, file)
        for file in os.listdir(results_dir)
        if file.endswith(".state_dict")
    ]
    sorted_policy_models = sorted(policy_models, key=os.path.getmtime)
    # Delete all but the last policy model file
    for policy_model in sorted_policy_models[:-1]:
        os.remove(os.path.join(results_dir_copy, policy_model.split("/")[-1]))

    shutil.make_archive(submission_file, "zip", results_dir_copy)
    print("NOTE: The submission file is created at:", submission_file + ".zip")
    shutil.rmtree(results_dir_copy)

def prepare_submission(results_dir: Path) -> Path:
    """
    # Validate all the submission files and compress into a .zip.
    Note: This method is also invoked in the trainer script itself!
    So if you ran the training script, you may not need to re-run this.
    Args results_dir: the directory where all the training files were saved.
    """
    #assert isinstance(results_dir, Path)

    # Validate the results directory
    _, success, comment = validate_dir(results_dir)
    if not success:
        raise FileNotFoundError(comment)

    # Remove all the checkpoint state files from the tmp directory except for the last one
    policy_models = list(results_dir.glob("*.state_dict"))
    policy_models = sorted(policy_models, key=lambda x: x.stat().st_mtime)

    # assemble list of files to copy
    files_to_copy = list(results_dir.glob("*.py"))
    files_to_copy.extend(list(results_dir.glob(".*")))
    files_to_copy.append(results_dir / "rice_rllib.yaml")
    files_to_copy.append(policy_models[-1])

    # Make a temporary copy of the results directory for zipping
    results_dir_copy = results_dir.parent / "tmp_copy"
    results_dir_copy.mkdir(parents=True)

    for file in files_to_copy:
        shutil.copy(file, results_dir_copy / file.name)

    # Create the submission file and delete the temporary copy
    submission_file = Path("submissions") / results_dir.name
    shutil.make_archive(submission_file, "zip", results_dir_copy)
    print("NOTE: The submission file is created at:\t\t\t", submission_file.with_suffix(".zip"))

    # open rice config yaml file in copied directory
    config_path = results_dir_copy / "rice_rllib.yaml"
    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    # modify the rice_config yaml to work with the original code
    backwards_config = yaml.safe_load(BACKWARDS_COMPAT_CONFIG)
    run_config["trainer"] = backwards_config["trainer"]
    del run_config["logging"]

    # write rice_config yaml file to tmp directory
    with open(config_path, "w", encoding="utf8") as fp:
        yaml.dump(run_config, fp, default_flow_style=False)

    # Create the backwards compatible submission file and delete the temporary copy
    submission_file_bc = Path("submissions") / "backwards_compatible" / results_dir.name
    shutil.make_archive(submission_file_bc, "zip", results_dir_copy)
    print("NOTE: The backwards compatible submission file is created at:\t", submission_file_bc.with_suffix(".zip"))
    
    # delete temporary directory
    shutil.rmtree(results_dir_copy)

    return submission_file.with_suffix(".zip")


if __name__ == "__main__":
    results_dir = Path(get_results_dir()[0])
    prepare_submission(results_dir=results_dir)
