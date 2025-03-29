# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Training script for the rice environment using RLlib
https://docs.ray.io/en/latest/rllib-training.html
"""

import logging
import os
import shutil
import json

# import shutil
import subprocess
import sys
import time
import wandb
import numpy as np
import yaml
from desired_outputs import desired_outputs
from fixed_paths import PUBLIC_REPO_DIR
from run_unittests import import_class_from_path
from opt_helper import save
from rice import Rice
from scenarios import (
    OptimalMitigation,
    MinimalMitigation,
    BasicClub,
    ExportAction,
    CarbonLeakage,
    CarbonLeakageFixed,
    BasicClubTariffAmbition,
    MinimalMitigationActionWindow,
    OptimalMitigationActionWindow,
    BasicClubFixed,
    BasicClubAblateMasks,
    BasicClubConvergence,
    ConvergenceMitigationSavings,
)
import argparse
from collections import OrderedDict
from tqdm import tqdm

sys.path.append(PUBLIC_REPO_DIR)
# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.DEBUG)

# scenarios
SCENARIO_MAPPING = {
    "default": Rice,
    "OptimalMitigation": OptimalMitigation,
    "MinimalMitigation": MinimalMitigation,
    "BasicClub": BasicClub,
    "ExportAction": ExportAction,
    "CarbonLeakage": CarbonLeakage,
    "CarbonLeakageFixed": CarbonLeakageFixed,
    "BasicClubTariffAmbition": BasicClubTariffAmbition,
    "MinimalMitigationActionWindow": MinimalMitigationActionWindow,
    "OptimalMitigationActionWindow": OptimalMitigationActionWindow,
    "BasicClubFixed": BasicClubFixed,
    "BasicClubAblateMasks": BasicClubAblateMasks,
    "ConvergenceMitigationSavings": ConvergenceMitigationSavings,
    "BasicClubConvergence": BasicClubConvergence,
}

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from typing import Dict
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.policy.policy import Policy


class Callbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        **kwargs,
    ):
        # Initialize storage for metrics if it doesn't exist
        if not hasattr(worker, "custom_metrics_storage"):
            worker.custom_metrics_storage = []

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        **kwargs,
    ):
        # Check and initialize storage if not already done
        if not hasattr(worker, "custom_metrics_storage"):
            worker.custom_metrics_storage = []

        # Collect metrics at the end of the episode
        episode.custom_metrics["temperature_rise"] = episode._last_infos["__common__"][
            "temp_rise"
        ]
        metrics = episode._last_infos["__common__"]["metrics"]
        num_regions = episode._last_infos["__common__"]["num_regions"]

        for metric in metrics:
            for region in range(num_regions):
                metric_key = f"{metric}_region_{region}"
                value = episode._last_infos[region][metric]
                episode.custom_metrics[metric_key] = value

        # Append episode metrics to worker's storage
        worker.custom_metrics_storage.append(episode.custom_metrics)

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Aggregate metrics from all workers
        all_metrics = algorithm.workers.foreach_worker(
            lambda w: getattr(w, "custom_metrics_storage", [])
        )

        # Flatten the list of lists
        all_metrics_flat = [metric for sublist in all_metrics for metric in sublist]

        # Calculate variance across all collected metrics
        variance_metrics = {}
        if all_metrics_flat:
            for metric_key in all_metrics_flat[0]:
                values = [m[metric_key] for m in all_metrics_flat if metric_key in m]
                if len(values) > 1:
                    mean = sum(values) / len(values)
                    squared_diff_sum = sum((x - mean) ** 2 for x in values)
                    variance = squared_diff_sum / (len(values) - 1)
                    variance_metrics[f"{metric_key}_variance"] = variance

        # Update the result with variance metrics
        result["custom_metrics"].update(variance_metrics)

        # Clear metrics storage for the next iteration
        algorithm.workers.foreach_worker(
            lambda w: setattr(w, "custom_metrics_storage", [])
        )


def get_config_yaml(yaml_path):
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", yaml_path)
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path "
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)
    return run_config


import ray
import torch
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from datetime import datetime
from ray.tune.logger import NoopLogger

logging.info("Finished imports")


_BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        spaces.Dict: A dictionary of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for key, val in obs.items():
        # list of lists are 'listified' np arrays
        _val = val
        if isinstance(val, list):
            _val = np.array(val)
        elif isinstance(val, (int, np.integer, float, np.floating)):
            _val = np.array([val])

        # assign Space
        if isinstance(_val, np.ndarray):
            large_num = float(_BIG_NUMBER)
            box = Box(
                low=-large_num,
                high=large_num,
                shape=_val.shape,
                dtype=_val.dtype,
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num,
                    high=large_num,
                    shape=_val.shape,
                    dtype=_val.dtype,
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[key] = box

        elif isinstance(_val, dict):
            dict_of_spaces[key] = recursive_obs_dict_to_spaces_dict(_val)
        else:
            raise TypeError
    return Dict(dict_of_spaces)


def recursive_list_to_np_array(dictionary):
    """
    Numpy-ify dictionary object to be used with RLlib.
    """
    if isinstance(dictionary, dict):
        new_d = {}
        for key, val in dictionary.items():
            if isinstance(val, list):
                new_d[key] = np.array(val)
            elif isinstance(val, dict):
                new_d[key] = recursive_list_to_np_array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                new_d[key] = np.array([val])
            elif isinstance(val, np.ndarray):
                new_d[key] = val
            else:
                raise AssertionError
        return new_d
    raise AssertionError


class EnvWrapper(MultiAgentEnv):
    """
    The environment wrapper class.
    """

    def __init__(self, env_config=None):
        super().__init__()

        env_config_copy = env_config.copy()
        if env_config_copy is None:
            env_config_copy = {}
        source_dir = env_config_copy.get("source_dir", None)
        if "source_dir" in env_config_copy:
            del env_config_copy["source_dir"]
        if source_dir is None:
            source_dir = PUBLIC_REPO_DIR
        assert isinstance(env_config_copy, dict)
        self.env = SCENARIO_MAPPING[env_config["scenario"]](**env_config_copy)

        self.action_space = self.env.action_space

        obs, info = self.env.reset()

        self.observation_space = recursive_obs_dict_to_spaces_dict(obs)

        # necessary parameters for rllib?
        self.agent_ids = list(range(self.env.num_regions))
        self._agent_ids = self.agent_ids
        self.num_agents = len(self.agent_ids)

    def reset(self, *, seed=None, options=None):
        """Reset the env."""
        obs, info = self.env.reset()
        super().reset(seed=seed)
        return recursive_list_to_np_array(obs), info

    def step(self, actions=None):
        """Step through the env."""
        assert actions is not None
        assert isinstance(actions, dict)
        obs, rew, terminateds, truncateds, info = self.env.step(actions)
        return (
            recursive_list_to_np_array(obs),
            rew,
            terminateds,
            truncateds,
            info,
        )


def get_rllib_config(config_yaml=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert config_yaml is not None
    assert env_class is not None

    env_config = get_env_config(config_yaml)
    assert isinstance(env_config, dict)

    env_object = create_env_object(env_class, env_config)

    multiagent_policies_config = get_multiagent_policies_config(
        config_yaml=config_yaml, env_object=env_object
    )

    trainer_config = get_trainer_config(config_yaml)

    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": config_yaml["env"],
        "framework": trainer_config["framework"],
        "multiagent": multiagent_policies_config,
        "num_workers": trainer_config["num_workers"],
        "num_gpus": trainer_config["num_gpus"],
        "num_cpus_per_worker": trainer_config["num_cpus_per_worker"],
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
        "train_batch_size": trainer_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config


def get_env_config(config_yaml):
    env_config = config_yaml["env"]
    return env_config


def create_env_object(env_class, env_config):
    env_object = env_class(env_config=env_config)
    return env_object


def get_multiagent_policies_config(config_yaml=None, env_object=None):
    assert config_yaml is not None
    assert env_object is not None

    # Define all the policies here
    regions_policy_config = config_yaml["policy"]["regions"]
    multi_model = config_yaml["policy"]["multi_model"]
    env_config = config_yaml["env"]
    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    if multi_model and env_config["clubs_enabled"]:
        policies = {
            "regionsclub": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
            "regionsnonclub": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
        }

        club_members = env_config["club_members"]

        # Function mapping agent ids to policy ids.
        def policy_mapping_fn(agent_id=None, episode=None, worker=None, **kwargs):
            assert agent_id is not None
            if agent_id in club_members:
                return "regionsclub"
            else:
                return "regionsnonclub"

    else:
        policies = {
            "regions": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
        }

        # Function mapping agent ids to policy ids.
        def policy_mapping_fn(agent_id=None, episode=None, worker=None, **kwargs):
            assert agent_id is not None
            return "regions"

    # Optional list of policies to train, or None for all policies.
    policies_to_train = None

    # Settings for Multi-Agent Environments
    multiagent_config = {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fn,
    }

    return multiagent_config


def get_trainer_config(config_yaml):
    trainer_config = config_yaml["trainer"]
    return trainer_config


def save_model_checkpoint(trainer_obj=None, save_directory=None, current_timestep=0):
    """
    Save trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to save the checkpoints."
    )
    model_params = trainer_obj.get_weights()
    for policy in model_params:
        filepath = os.path.join(
            save_directory,
            f"{policy}_{current_timestep}.state_dict",
        )
        logging.info(
            f"Saving the model checkpoints for policy {policy} to {filepath}.",
        )
        torch.save(model_params[policy], filepath)


def load_model_checkpoints(trainer_obj=None, save_directory=None, ckpt_idx=-1):
    """
    Load trained model checkpoints.
    """
    assert trainer_obj is not None
    assert save_directory is not None
    assert os.path.exists(save_directory), (
        "Invalid folder path. "
        "Please specify a valid directory to load the checkpoints from."
    )
    files = [f for f in os.listdir(save_directory) if f.endswith("state_dict")]

    assert len(files) == len(trainer_obj.config["multiagent"]["policies"])

    model_params = trainer_obj.get_weights()
    for policy in model_params:
        policy_models = [
            os.path.join(save_directory, file) for file in files if policy in file
        ]
        # If there are multiple files, then use the ckpt_idx to specify the checkpoint
        assert ckpt_idx < len(policy_models)
        sorted_policy_models = sorted(policy_models, key=os.path.getmtime)
        policy_model_file = sorted_policy_models[ckpt_idx]
        model_params[policy] = torch.load(policy_model_file)
        logging.info(f"Loaded model checkpoints {policy_model_file}.")

    trainer_obj.set_weights(model_params)


def create_trainer(config_yaml=None, source_dir=None, seed=None):
    """
    Create the RLlib trainer.
    """

    # Create the A2C trainer.
    config_yaml["env"]["source_dir"] = source_dir

    if config_yaml["env"]["action_space_type"] == "discrete":
        from scripts.torch_models_discrete import TorchLinear
    elif config_yaml["env"]["action_space_type"] == "continuous":
        if "beta" in config_yaml["policy"]["regions"]["model"]["custom_model"].lower():
            from scripts.torch_models_cont_beta import CustomBetaPolicyModel
            from ray.rllib.models import ModelCatalog
            from beta_action_dist import BetaActionDistribution

            ModelCatalog.register_custom_action_dist(
                "beta_distribution", BetaActionDistribution
            )

            from beta_action_dist import BetaActionDistribution
        elif (
            "cont" in config_yaml["policy"]["regions"]["model"]["custom_model"].lower()
        ):
            from scripts.torch_models_cont import TorchLinear
        elif (
            "discrete"
            in config_yaml["policy"]["regions"]["model"]["custom_model"].lower()
        ):
            from scripts.torch_models_discrete import TorchLinear

    rllib_config = get_rllib_config(
        config_yaml=config_yaml,
        env_class=EnvWrapper,
        seed=seed,
    )

    config = A2CConfig()

    # config.num_agents = rllib_config["num_envs_per_worker"]

    config = config.training(train_batch_size=rllib_config["train_batch_size"])
    config = config.environment(disable_env_checking=True)
    config = config.multi_agent(
        policies=rllib_config["multiagent"]["policies"],
        policy_mapping_fn=rllib_config["multiagent"]["policy_mapping_fn"],
        policies_to_train=rllib_config["multiagent"]["policies_to_train"],
    )

    config = config.resources(num_gpus=rllib_config["num_gpus"])
    config = config.rollouts(
        num_rollout_workers=rllib_config["num_workers"],
        num_envs_per_worker=rllib_config["num_envs_per_worker"],
    )
    config = config.framework(rllib_config["framework"])
    config = config.environment(
        EnvWrapper,
        env_config=rllib_config["env_config"],
    )
    config = config.callbacks(Callbacks)

    config.seed = seed

    rllib_trainer = config.build()

    return rllib_trainer


def create_save_dir_path(exp_run_config, results_dir=None):
    assert exp_run_config is not None
    if results_dir is None:
        # Use the current time as the name for the results directory.
        results_dir = f"{time.time():10.0f}"

    # Directory to save model checkpoints and metrics
    save_config = exp_run_config["saving"]
    results_save_dir = os.path.join(
        save_config["basedir"],
        save_config["name"],
        save_config["tag"],
        "is_perturbation",
        save_config["is_perturbation"],
        results_dir,
    )

    return results_save_dir


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(
                obj,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, type):
                return str(obj)
            # You can add more type-specific conversions here if needed
        except TypeError:
            pass

        # Catch-all: Convert the object to a string
        return str(obj)


def fetch_episode_states(trainer_obj=None, episode_states=None, file_name=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    outputs = {}

    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    obs, _ = env_object.reset()

    env = env_object.env

    for state in episode_states:
        assert state in env.global_state, f"{state} is not in global state!"
        # Initialize the episode states
        array_shape = env.global_state[state]["value"].shape
        outputs[state] = np.nan * np.ones(array_shape)

    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_agents):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    for timestep in range(env.episode_length):
        for state in episode_states:
            outputs[state][timestep] = env.global_state[state]["value"][timestep]

        actions = {}
        # TODO: Consider using the `compute_actions` (instead of `compute_action`)
        # API below for speed-up when there are many agents.
        for region_id in range(env.num_agents):
            if (
                len(agent_states[region_id]) == 0
            ):  # stateless, with a linear model, for example

                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )

            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_actions(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
        obs, rewards, done, truncateds, info = env_object.step(actions)
        if done["__all__"]:
            for state in episode_states:
                outputs[state][timestep + 1] = env.global_state[state]["value"][
                    timestep + 1
                ]
            if file_name:
                # Get the current script's directory
                current_directory = os.path.dirname(__file__)
                # Construct the path to the 'eval' directory
                eval_directory = os.path.join(current_directory, "..", "evals")
                # Ensure the path is absolute
                eval_directory = os.path.abspath(eval_directory)
                formatted_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
                name = f"global_state_{file_name}_{formatted_datetime}.json"
                # Define the file name and construct the full file path
                file_path = os.path.join(eval_directory, name)

                with open(file_path, "w") as f:
                    json.dump(env.global_state, f, cls=NumpyArrayEncoder)
            break

    return outputs


def set_num_agents(config_yaml):
    """
    updates the region_yamls folder with the appropriate number of regions
    as defined by the yaml config
    """
    num_agents = config_yaml["regions"]["num_agents"]

    assert num_agents in [3, 7, 20, 27]

    # Get the directory where the script is located
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Get the parent directory of the script
    parent_directory = os.path.dirname(script_directory)

    # Path to the 'other_yamls' directory
    target_directory = os.path.join(parent_directory, "other_yamls")

    # List everything in the 'other_yamls' directory
    entries = os.listdir(target_directory)

    # Filter out files, only keep directories
    folders = [
        entry
        for entry in entries
        if os.path.isdir(os.path.join(target_directory, entry))
    ]

    # Get target directory containing relevant .yml files
    target_region_yamls = os.path.join(
        target_directory,
        [folder for folder in folders if folder.startswith(str(num_agents))][0],
    )

    # Path to the 'test_regions' directory
    test_regions_directory = os.path.join(parent_directory, "region_yamls")

    # Delete all files in 'test_regions' except 'default.yml'
    for file in os.listdir(test_regions_directory):
        file_path = os.path.join(test_regions_directory, file)
        if os.path.isfile(file_path) and file != "default.yml":
            os.remove(file_path)

    # Copy all .yml files from target_region_yamls to test_regions
    for file in os.listdir(target_region_yamls):
        if file.endswith(".yml"):
            source_file_path = os.path.join(target_region_yamls, file)
            destination_file_path = os.path.join(test_regions_directory, file)
            shutil.copy2(source_file_path, destination_file_path)
    logging.info(f"region yamls updated to {num_agents} regions")


if __name__ == "__main__":
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, default="rice_rllib_discrete.yaml")
    parser.add_argument("--perturbation", "-p", type=float, default=0.0)
    args = parser.parse_args()
    config_yaml = get_config_yaml(yaml_path=args.yaml)
    config_yaml["saving"]["is_perturbation"] = str(args.perturbation)
    config_yaml["env"]["is_perturbation"] = args.perturbation
    print("CURRENT CONFIG: \n\n")
    print(config_yaml)
    ray.init(ignore_reinit_error=True)

    if config_yaml["logging"]["enabled"]:
        wandb_config = config_yaml["logging"]["wandb_config"]
        wandb.login(key=wandb_config["login"])
        wandb.init(
            project=wandb_config["project"],
            name=f'{wandb_config["run"]}_train',
            entity=wandb_config["entity"],
        )

    set_num_agents(config_yaml)
    trainer = create_trainer(config_yaml)
    save_dir = create_save_dir_path(config_yaml)
    os.makedirs(save_dir)

    for file in ["rice.py", "rice_helpers.py", "scenarios.py"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(save_dir, file),
        )
    for file in [args.yaml]:
        file = os.path.basename(file)
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "scripts", file),
            os.path.join(save_dir, file),
        )

    # Add an identifier file
    with open(os.path.join(save_dir, ".rllib"), "x", encoding="utf-8") as fp:
        pass
    fp.close()

    # Perform training
    # ----------------
    trainer_config = config_yaml["trainer"]
    num_episodes = trainer_config["num_episodes"]
    train_batch_size = trainer_config["train_batch_size"]

    # Fetch the env object from the trainer
    if trainer_config["num_workers"] > 0:
        # Fetch the env object from the trainer
        envs = trainer.workers.foreach_worker(lambda worker: worker.env)
        env_obj = envs[1].env
    else:
        env_obj = trainer.workers.local_worker().env.env

    episode_length = env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size
    logs = []
    file_name = (
        save_dir.split("/")[-1]
        + "_"
        + config_yaml["env"]["scenario"]
        + "_"
        + str(config_yaml["regions"]["num_agents"])
    )

    for iteration in tqdm(range(num_iters)):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.train()
        current_logs = result["custom_metrics"]
        keys_to_log = [
            "num_agent_steps_sampled",
            "num_agent_steps_trained",
            "num_env_steps_sampled",
            "num_env_steps_trained",
            "num_env_steps_sampled_this_iter",
            "num_env_steps_trained_this_iter",
            "num_env_steps_sampled_throughput_per_sec",
            "num_env_steps_trained_throughput_per_sec",
            "timesteps_total",
            "num_steps_trained_this_iter",
            "agent_timesteps_total",
            "config",
        ]
        current_logs["learner_stats"] = result["info"]["learner"]["regions"][
            "learner_stats"
        ]

        for key in keys_to_log:
            current_logs[key] = result[key]
        trained_steps = current_logs["num_env_steps_trained_this_iter"]
        print(f"STEPS LOGGED: {trained_steps}")
        if config_yaml["logging"]["enabled"]:
            wandb.log(
                {
                    "episode_reward_min": result["episode_reward_min"],
                    "episode_reward_mean": result["episode_reward_mean"],
                    "episode_reward_max": result["episode_reward_max"],
                },
                step=result["episodes_total"],
            )
            if (
                config_yaml["env"]["clubs_enabled"]
                and config_yaml["policy"]["multi_model"]
            ):
                wandb.log(
                    result["info"]["learner"]["regionsclub"]["learner_stats"],
                    step=result["episodes_total"],
                )
                wandb.log(
                    result["info"]["learner"]["regionsnonclub"]["learner_stats"],
                    step=result["episodes_total"],
                )
            else:
                wandb.log(
                    result["info"]["learner"]["regions"]["learner_stats"],
                    step=result["episodes_total"],
                )

        total_timesteps = result.get("timesteps_total")
        if (
            iteration % config_yaml["saving"]["model_params_save_freq"] == 0
            or iteration == num_iters - 1
        ):
            save_model_checkpoint(trainer, save_dir, total_timesteps)

        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")
        if iteration != 0:
            with open(
                os.path.join(PUBLIC_REPO_DIR, "callback_logs", f"{file_name}.json"), "r"
            ) as f:
                file_logs = json.load(f)
            file_logs.append(current_logs)
            with open(
                os.path.join(PUBLIC_REPO_DIR, "callback_logs", f"{file_name}.json"), "w"
            ) as f:
                json.dump(file_logs, f, cls=NumpyArrayEncoder)
        else:
            with open(
                os.path.join(PUBLIC_REPO_DIR, "callback_logs", f"{file_name}.json"), "w"
            ) as f:
                json.dump([current_logs], f, cls=NumpyArrayEncoder)
    # Create a (zipped) submission file
    # ---------------------------------
    subprocess.call(
        [
            "python",
            os.path.join(PUBLIC_REPO_DIR, "scripts", "create_submission_zip.py"),
            "--results_dir",
            save_dir,
        ]
    )

    # Close Ray gracefully after completion
    ray.shutdown()

    if config_yaml["logging"]["enabled"]:
        wandb.finish()
