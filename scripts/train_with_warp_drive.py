# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Training script for the rice environment using WarpDrive
www.github.com/salesforce/warp-drive
"""

import logging
import os
import shutil
import subprocess
import sys
import numpy as np
import yaml
from desired_outputs import desired_outputs
from opt_helper import get_mean_std
from fixed_paths import PUBLIC_REPO_DIR

sys.path.append(PUBLIC_REPO_DIR)

from scripts.run_unittests import import_class_from_path

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)


def perform_other_imports():
    """
    WarpDrive-related imports.
    """
    import torch

    num_gpus_available = torch.cuda.device_count()
    assert num_gpus_available > 0, "This script needs a GPU to run!"

    from warp_drive.env_wrapper import EnvWrapper
    from warp_drive.training.trainer import Trainer
    from warp_drive.utils.env_registrar import EnvironmentRegistrar

    return torch, EnvWrapper, Trainer, EnvironmentRegistrar


try:
    other_imports = perform_other_imports()
except ImportError:
    print("Installing requirements...")
    subprocess.call(["pip", "install", "rl-warp-drive>=1.6.5"])

    other_imports = perform_other_imports()

torch, EnvWrapper, Trainer, EnvironmentRegistrar = other_imports


def create_trainer(run_config=None, source_dir=None, seed=None):
    """
    Create the WarpDrive trainer.
    """
    torch.cuda.FloatTensor(8)  # add this line for successful cuda_init

    assert run_config is not None
    if source_dir is None:
        source_dir = PUBLIC_REPO_DIR
    if seed is not None:
        run_config["trainer"]["seed"] = seed

    # Create a wrapped environment object via the EnvWrapper
    # Ensure that use_cuda is set to True (in order to run on the GPU)

    # Register the environment
    env_registrar = EnvironmentRegistrar()

    rice_cuda_class = import_class_from_path(
        "RiceCuda", os.path.join(source_dir, "rice_cuda.py")
    )

    env_registrar.add_cuda_env_src_path(
        rice_cuda_class.name, os.path.join(source_dir, "rice_build.cu")
    )

    env_wrapper = EnvWrapper(
        rice_cuda_class(**run_config["env"]),
        num_envs=run_config["trainer"]["num_envs"],
        use_cuda=True,
        env_registrar=env_registrar,
    )

    # Policy mapping to agent ids: agents can share models
    # The policy_tag_to_agent_id_map dictionary maps
    # policy model names to agent ids.
    # ----------------------------------------------------
    policy_tag_to_agent_id_map = {
        "regions": list(range(env_wrapper.env.num_agents)),
    }

    # Create the Trainer object
    # -------------------------
    trainer_obj = Trainer(
        env_wrapper=env_wrapper,
        config=run_config,
        policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
    )
    return trainer_obj, trainer_obj.save_dir


def load_model_checkpoints(trainer=None, save_directory=None, ckpt_idx=-1):
    """
    Load trained model checkpoints.
    """
    assert trainer is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to load the checkpoints from."
    )
    files = [file for file in os.listdir(save_directory) if file.endswith("state_dict")]
    assert len(files) >= len(trainer.policies), "Missing policy checkpoints"

    ckpts_dict = {}
    for policy in trainer.policies_to_train:
        policy_models = [
            os.path.join(save_directory, file) for file in files if policy in file
        ]
        # If there are multiple files, then use the ckpt_idx to specify the checkpoint
        assert ckpt_idx < len(policy_models)
        sorted_policy_models = sorted(policy_models, key=os.path.getmtime)
        policy_model_file = sorted_policy_models[ckpt_idx]
        logging.info(f"Loaded model checkpoints {policy_model_file}.")

        ckpts_dict.update({policy: policy_model_file})
    trainer.load_model_checkpoint(ckpts_dict)


def fetch_episode_states(trainer_obj=None, episode_states=None, env_id=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert isinstance(
        episode_states, list
    ), "Please pass the 'episode states' args as a list."
    assert len(episode_states) > 0
    return trainer_obj.fetch_episode_states(episode_states, env_id)


def copy_source_files(trainer):
    """
    Copy source files to the saving directory.
    """
    for file in ["rice.py", "rice_helpers.py", "rice_cuda.py", "rice_step.cu"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(trainer.save_dir, file),
        )

    for file in [
        "rice_warpdrive.yaml",
    ]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "scripts", file),
            os.path.join(trainer.save_dir, file),
        )

    # Add an identifier file
    with open(
        os.path.join(trainer.save_dir, ".warpdrive"), "x", encoding="utf-8"
    ) as file_pointer:
        pass
    file_pointer.close()


def trainer(
    negotiation_on=0,
    num_envs=100,
    train_batch_size=1024,
    num_episodes=30000,
    lr=0.0005,
    model_params_save_freq=5000,
    desired_outputs=desired_outputs,
    output_all_envs=False,
):
    """
    Main function to run the trainer.
    """
    # Load the run_config
    print("Training with WarpDrive...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_warpdrive.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path"
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_configuration = yaml.safe_load(fp)
    run_configuration["env"]["negotiation_on"] = negotiation_on
    run_configuration["trainer"]["num_envs"] = num_envs
    run_configuration["trainer"]["train_batch_size"] = train_batch_size
    run_configuration["trainer"]["num_episodes"] = num_episodes
    run_configuration["policy"]["regions"]["lr"] = lr
    run_configuration["saving"]["model_params_save_freq"] = model_params_save_freq
    # run_configuration trainer
    # --------------
    trainer_object, _ = create_trainer(run_config=run_configuration)

    # Copy the source files into the results directory
    # ------------------------------------------------
    copy_source_files(trainer_object)

    # Perform training!
    # -----------------
    trainer_object.train()

    # Create a (zipped) submission file
    # ---------------------------------
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "create_submission_zip.py"),
            "--results_dir",
            trainer_object.save_dir,
        ]
    )
    outputs_ts = [
        fetch_episode_states(trainer_object, desired_outputs, env_id=i)
        for i in range(num_envs)
    ]
    for i in range(len(outputs_ts)):
        outputs_ts[i]["global_consumption"] = np.sum(
            outputs_ts[i]["consumption_all_regions"], axis=-1
        )
        outputs_ts[i]["global_production"] = np.sum(
            outputs_ts[i]["gross_output_all_regions"], axis=-1
        )
    if not output_all_envs:
        outputs_ts, _ = get_mean_std(outputs_ts)
    # Shut off the trainer gracefully
    # -------------------------------
    trainer_object.graceful_close()
    return trainer_object, outputs_ts


if __name__ == "__main__":
    print("Training with WarpDrive...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_warpdrive.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path"
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_configuration = yaml.safe_load(fp)

    # Create trainer
    # --------------
    trainer_object, _ = create_trainer(run_config=run_configuration)

    # Copy the source files into the results directory
    # ------------------------------------------------
    copy_source_files(trainer_object)

    # Perform training!
    # -----------------
    trainer_object.train()

    # Create a (zipped) submission file
    # ---------------------------------
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "create_submission_zip.py"),
            "--results_dir",
            trainer_object.save_dir,
        ]
    )

    # Shut off the trainer gracefully
    # -------------------------------
    trainer_object.graceful_close()
