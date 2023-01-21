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
from ray.rllib.algorithms.ppo import PPO

#from sklearn import datasets

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


def get_tuner():
    
    
    tuner = tune.Tuner(
    "PPO",
    tune_config=tune.TuneConfig(
      metric="episode_reward_mean",
      mode="max",
      #scheduler=pbt,
      num_samples=1),
      param_space={
        'num_workers': 1
    }
    )
    
    return tuner


if __name__ == "__main__":
    print("Training with RLlib...")
    
    """
    digits = datasets.load_digits()
    x = digits.data
    y = digits.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    """


    episode_length = 20 #env_obj.episode_length
    num_iters = 5
    
    trainer = get_tuner()
    
    for iteration in range(num_iters):
        print(f"********** Iter : {iteration + 1:5d} / {num_iters:5d} **********")
        result = trainer.fit() # trainer.train()
        logging.warning("trainer fitted")
        total_timesteps = result.get("timesteps_total")
        
        print(f"""episode_reward_mean: {result.get('episode_reward_mean')}""")

    # Close Ray gracefully after completion
    ray.shutdown()
