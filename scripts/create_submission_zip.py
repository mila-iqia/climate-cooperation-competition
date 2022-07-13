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

from evaluate_submission import get_results_dir, validate_dir

from fixed_paths import PUBLIC_REPO_DIR
sys.path.append(PUBLIC_REPO_DIR)


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
