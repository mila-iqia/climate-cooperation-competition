# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
CUDA version of the Regional Integrated model of Climate and the Economy (RICE).
This subclasses the python version of the model and also the CUDAEnvironmentContext
for running with WarpDrive (https://github.com/salesforce/warp-drive)
"""

import os
import sys
import numpy as np
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_PUBLIC_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [_PUBLIC_REPO_DIR] + sys.path

from rice import Rice

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


class RiceCuda(Rice, CUDAEnvironmentContext):
    """
    Rice env class that invokes the CUDA step function.
    """

    name = "Rice"

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device.
        """
        data_feed = DataFeed()

        # Add constants
        for key, value in sorted(self.dice_constant.items()):
            data_feed.add_data(name=key, data=value)

        for key, value in sorted(self.rice_constant.items()):
            data_feed.add_data(name=key, data=value)

        # Add all the global states at timestep 0.
        timestep = 0
        for key in sorted(self.global_state.keys()):
            data_feed.add_data(
                name=key,
                data=self.global_state[key]["value"][timestep],
                save_copy_and_apply_at_reset=True,
            )

        for key in sorted(self.global_state.keys()):
            data_feed.add_data(
                name=key + "_norm",
                data=self.global_state[key]["norm"],
            )

        # Env config parameters
        data_feed.add_data(
            name="aux_ms",
            data=np.zeros(self.num_regions, dtype=np.float32),
            save_copy_and_apply_at_reset=True,
        )

        # Env config parameters
        data_feed.add_data(
            name="num_discrete_action_levels",
            data=self.num_discrete_action_levels,
        )

        data_feed.add_data(
            name="balance_interest_rate",
            data=self.balance_interest_rate,
        )

        data_feed.add_data(name="negotiation_on", data=self.negotiation_on)

        # Armington agg. parameters
        data_feed.add_data_list(
            [
                ("sub_rate", self.sub_rate),
                ("dom_pref", self.dom_pref),
                ("for_pref", self.for_pref),
            ]
        )

        # Year parameters
        data_feed.add_data_list(
            [("current_year", self.current_year, True), ("end_year", self.end_year)]
        )

        return data_feed

    @staticmethod
    def get_tensor_dictionary():
        """
        Create a dictionary of pytorch-accessible tensors to push to the device.
        """
        tensor_dict = DataFeed()
        return tensor_dict

    def step(self):
        constants_keys = list(sorted(self.dice_constant.keys())) + list(
            sorted(self.rice_constant.keys())
        )
        args = (
            constants_keys
            + list(sorted(self.global_state.keys()))
            + [key + "_norm" for key in list(sorted(self.global_state.keys()))]
            + [
                "num_discrete_action_levels",
                "balance_interest_rate",
                "negotiation_on",
                "aux_ms",
                "sub_rate",
                "dom_pref",
                "for_pref",
                "current_year",
                "end_year",
                _OBSERVATIONS + "_features",
                _OBSERVATIONS + "_action_mask",
                _ACTIONS,
                _REWARDS,
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
            ]
        )

        self.cuda_step(
            *self.cuda_step_function_feed(args),
            block=self.cuda_function_manager.block,
            grid=self.cuda_function_manager.grid,
        )
