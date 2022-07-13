# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


# Run this via
# !python run_cpu_gpu_env_consistency_checks.py

import logging
import os
import sys

import torch

from fixed_paths import PUBLIC_REPO_DIR
sys.path.append(PUBLIC_REPO_DIR)

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar

from rice import Rice
from rice_cuda import RiceCuda

logging.getLogger().setLevel(logging.ERROR)

_NUM_GPUS_AVAILABLE = torch.cuda.device_count()
assert _NUM_GPUS_AVAILABLE > 0, "This script needs a GPU to run!"

env_registrar = EnvironmentRegistrar()

env_registrar.add_cuda_env_src_path(Rice.name, os.path.join(PUBLIC_REPO_DIR, "rice_build.cu"))
env_configs = {
    "no_negotiation": {
        "num_discrete_action_levels": 100,
        "negotiation_on": False,
    },
    "with_negotiation": {
        "num_discrete_action_levels": 100,
        "negotiation_on": True,
    },
}
testing_class = EnvironmentCPUvsGPU(
    cpu_env_class=Rice,
    cuda_env_class=RiceCuda,
    env_configs=env_configs,
    num_envs=2,
    num_episodes=2,
    use_gpu_testing_mode=False,
    env_registrar=env_registrar,
)

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)
