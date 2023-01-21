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
import subprocess
import sys
import time

import numpy as np
import yaml
from run_unittests import import_class_from_path
from desired_outputs import desired_outputs
from fixed_paths import PUBLIC_REPO_DIR

import ray
from ray import tune
from ray import air
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

sys.path.append(PUBLIC_REPO_DIR)

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.DEBUG)


def perform_other_imports():
    """
    RLlib-related imports.
    """
    import ray
    from ray import air, tune # SY add
    from ray.rllib.algorithms.ppo import PPO # SY add
    import torch
    from gym.spaces import Box, Dict
    from ray.rllib.agents.a3c import A2CTrainer
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.tune.logger import NoopLogger

    return ray, torch, Box, Dict, MultiAgentEnv, A2CTrainer, NoopLogger


try:
    print("other imports performing")
    other_imports = perform_other_imports()
except ImportError:
    print("Installing requirements...")

    # Install gym
    subprocess.call(["pip", "install", "gym==0.21.0"])
    # Install RLlib v1.0.0
    subprocess.call(["pip", "install", "ray[rllib]==1.0.0"])
    # Install PyTorch
    subprocess.call(["pip", "install", "torch==1.9.0"])

    other_imports = perform_other_imports()

ray, torch, Box, Dict, MultiAgentEnv, A2CTrainer, NoopLogger = other_imports

from torch_models import TorchLinear

_BIG_NUMBER = 1e20


def get_tuner(run_config):
    config1 = run_config['saving']
    config2 = run_config['trainer']
    config3 = run_config['env']
    config4 = run_config['policy']
    
    p1 = config1['metrics_log_freq']
    p2 = config1['model_params_save_freq']
    
    p3 = config2['num_envs']
    p4 = config2['rollout_fragment_length']
    p5 = config2['train_batch_size']
    p6 = config2['num_episodes']
    p7 = config2['num_workers']
    p8 = config2['num_gpus']
    
    p9 = config3['num_discrete_action_levels']
    p10 = config3['negotiation_on']
    
    config5 = config4['regions']
    
    
    tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
      metric="episode_reward_mean",
      mode="max",
      #scheduler=pbt,
      num_samples=1),
      param_space={
        'num_workers': p7,
        'num_gpus':p8
    }
    )
    
    return tuner
    
def get_tuner_simple(run_config):
    
    tuner = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"episode_reward_mean": 150},
    ),
    param_space=config)
    
    return tuner

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
                low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
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
        env_config_copy = env_config.copy()
        if env_config_copy is None:
            env_config_copy = {}
        source_dir = env_config_copy.get("source_dir", None)
        # Remove source_dir key in env_config if it exists
        if "source_dir" in env_config_copy:
            del env_config_copy["source_dir"]
        if source_dir is None:
            source_dir = PUBLIC_REPO_DIR
        assert isinstance(env_config_copy, dict)
        self.env = import_class_from_path("Rice", os.path.join(source_dir, "rice.py"))(
            **env_config_copy
        )

        self.action_space = self.env.action_space

        self.observation_space = recursive_obs_dict_to_spaces_dict(self.env.reset())

    def reset(self):
        """Reset the env."""
        obs = self.env.reset()
        return recursive_list_to_np_array(obs)

    def step(self, actions=None):
        """Step through the env."""
        assert actions is not None
        assert isinstance(actions, dict)
        obs, rew, done, info = self.env.step(actions)
        return recursive_list_to_np_array(obs), rew, done, info


def get_rllib_config(exp_run_config=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert exp_run_config is not None
    assert env_class is not None

    env_config = exp_run_config["env"]
    assert isinstance(env_config, dict)
    env_object = env_class(env_config=env_config)

    # Define all the policies here
    policy_config = exp_run_config["policy"]["regions"]

    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    policies = {
        "regions": (
            None,  # uses default policy
            env_object.observation_space[0],
            env_object.action_space[0],
            policy_config,
        ),
    }

    # Function mapping agent ids to policy ids.
    def policy_mapping_fn(agent_id=None):
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

    train_config = exp_run_config["trainer"]
    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": exp_run_config["env"],
        "framework": train_config["framework"],
        "multiagent": multiagent_config,
        "num_workers": train_config["num_workers"],
        "num_gpus": train_config["num_gpus"],
        "num_envs_per_worker": train_config["num_envs"] // train_config["num_workers"],
        "train_batch_size": train_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config


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
            "Saving the model checkpoints for policy %s to %s.", (policy, filepath)
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


def create_trainer(exp_run_config=None, source_dir=None, results_dir=None, seed=None):
    """
    Create the RLlib trainer.
    """
    logging.warning("creating trainer")
    assert exp_run_config is not None
    if results_dir is None:
        # Use the current time as the name for the results directory.
        results_dir = f"{time.time():10.0f}"

    # Directory to save model checkpoints and metrics
    logging.warning("creating save config")
    save_config = exp_run_config["saving"]
    results_save_dir = os.path.join(
        save_config["basedir"],
        save_config["name"],
        save_config["tag"],
        results_dir,
    )
    run_config1= exp_run_config # input to tuner
    logging.warning("ray init call")
    #if not ray.is_initialized():
    ray.init() # ignore_reinit_error=True)
    #context = ray.init() # do no init?
    logging.warning("ray initiated")
    # Create the A2C trainer.
    exp_run_config["env"]["source_dir"] = source_dir
    logging.warning("exp run config is:", exp_run_config)
    # set config here?
    #my_config = A3CConfig().training(lr=0.01, grad_clip=30.0)
    
    """rllib_trainer = A2CTrainer(
        env=EnvWrapper,
        config=get_rllib_config(
            exp_run_config=exp_run_config, env_class=EnvWrapper, seed=seed
        ),
    )"""
    config1 = PPOConfig().training(lr=tune.grid_search([0.01, 0.001, 0.0001]))
    tuner1= get_tuner_simple(config1)
    
    logging.warning("created simple trainer")
    return tuner1, results_save_dir # return rllib_trainer


def fetch_episode_states(trainer_obj=None, episode_states=None):
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
    obs = env_object.reset()

    env = env_object.env

    for state in episode_states:
        assert state in env.global_state, f"{state} is not in global state!"
        # Initialize the episode states
        array_shape = env.global_state[state]["value"].shape
        outputs[state] = np.nan * np.ones(array_shape)

    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_regions):
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
                actions[region_id] = trainer_obj.compute_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
        obs, _, done, _ = env_object.step(actions)
        if done["__all__"]:
            for state in episode_states:
                outputs[state][timestep + 1] = env.global_state[state]["value"][
                    timestep + 1
                ]
            break

    return outputs


def trainer(
    negotiation_on=0,
    num_envs=100,
    train_batch_size=1024,
    num_episodes=30000,
    lr=0.0005,
    model_params_save_freq=5000,
    desired_outputs=desired_outputs,
    num_workers=4,
):
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_rllib.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path "
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)
    # replace the default setting
    run_config["env"]["negotiation_on"] = negotiation_on
    run_config["trainer"]["num_envs"] = num_envs
    run_config["trainer"]["train_batch_size"] = train_batch_size
    run_config["trainer"]["num_workers"] = num_workers
    run_config["trainer"]["num_episodes"] = num_episodes
    run_config["policy"]["regions"]["lr"] = lr
    run_config["saving"]["model_params_save_freq"] = model_params_save_freq

    # Create trainer
    # --------------
    trainer, save_dir = create_trainer(run_config)

    # Copy the source files into the results directory
    # ------------------------------------------------
    os.makedirs(save_dir)
    # Copy source files to the saving directory
    for file in ["rice.py", "rice_helpers.py"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(save_dir, file),
        )
    for file in ["rice_rllib.yaml"]:
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
    trainer_config = run_config["trainer"]
    # num_episodes = trainer_config["num_episodes"]
    # train_batch_size = trainer_config["train_batch_size"]
    # Fetch the env object from the trainer
    env_obj = trainer.workers.local_worker().env.env
    episode_length = env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        #result = trainer.train() # replace with tuner
        result = trainer.fit()
        total_timesteps = result.get("timesteps_total")
        if (
            iteration % run_config["saving"]["model_params_save_freq"] == 0
            or iteration == num_iters - 1
        ):
            save_model_checkpoint(trainer, save_dir, total_timesteps)
            logging.info(result)
        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")

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
    outputs_ts = fetch_episode_states(trainer, desired_outputs)
    ray.shutdown()
    return trainer, outputs_ts


if __name__ == "__main__":
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    config_path = os.path.join(PUBLIC_REPO_DIR, "scripts", "rice_rllib.yaml")
    if not os.path.exists(config_path):
        raise ValueError(
            "The run configuration is missing. Please make sure the correct path "
            "is specified."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    # Create trainer
    print("run config is:", run_config) # print about run config
    trainer, save_dir = create_trainer(run_config)
    print(" created trainer")
    # Copy the source files into the results directory
    # ------------------------------------------------
    os.makedirs(save_dir)
    # Copy source files to the saving directory
    for file in ["rice.py", "rice_helpers.py"]:
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, file),
            os.path.join(save_dir, file),
        )
    for file in ["rice_rllib.yaml"]:
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
    trainer_config = run_config["trainer"]
    num_episodes = trainer_config["num_episodes"]
    train_batch_size = trainer_config["train_batch_size"]
    # Fetch the env object from the trainer
    #env_obj = trainer.workers.local_worker().env.env # not sure what to replace with
    episode_length = 20 #env_obj.episode_length
    num_iters = (num_episodes * episode_length) // train_batch_size

    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.fit() # trainer.train()
        logging.warning("trainer fitted")
        total_timesteps = result.get("timesteps_total")
        if (
            iteration % run_config["saving"]["model_params_save_freq"] == 0
            or iteration == num_iters - 1
        ):
            save_model_checkpoint(trainer, save_dir, total_timesteps)
            logging.info(result)
        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")

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
