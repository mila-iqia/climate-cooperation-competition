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
from fixed_paths import PUBLIC_REPO_DIR

sys.path.append(PUBLIC_REPO_DIR)


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


def prepare_submission(results_dir=None):
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


if __name__ == "__main__":
    prepare_submission(results_dir=get_results_dir()[0])
