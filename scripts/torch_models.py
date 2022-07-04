# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Custom Pytorch policy models to use with RLlib.
"""

# API reference:
# https://docs.ray.io/en/latest/rllib/rllib-models.html#custom-pytorch-models

import numpy as np
from gym.spaces import Box, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()

_ACTION_MASK = "action_mask"


class TorchLinear(TorchModelV2, nn.Module):
    """
    Fully-connected Pytorch policy model.
    """

    custom_name = "torch_linear"

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, fc_dims=None
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        if fc_dims is None:
            fc_dims = [256, 256]

        # Check Observation spaces

        self.observation_space = obs_space.original_space

        if not isinstance(self.observation_space, Dict):
            if isinstance(self.observation_space, Box):
                raise TypeError(
                    "({name}) Observation space should be a gym Dict. "
                    "Is a Box of shape {self.observation_space.shape}"
                )
            raise TypeError(
                f"({name}) Observation space should be a gym Dict. "
                "Is {type(self.observation_space))} instead."
            )

        flattened_obs_size = self.get_flattened_obs_size()

        # Model only outputs policy logits,
        # values are accessed via the self.value_function
        self.values = None

        num_fc_layers = len(fc_dims)

        input_dims = [flattened_obs_size] + fc_dims[:-1]
        output_dims = fc_dims

        self.fc_dict = nn.ModuleDict()
        for fc_layer in range(num_fc_layers):
            self.fc_dict[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )

        # policy network (list of heads)
        policy_heads = [None for _ in range(len(action_space))]
        self.output_dims = []  # Network output dimension(s)

        for idx, act_space in enumerate(action_space):
            output_dim = act_space.n
            self.output_dims += [output_dim]
            policy_heads[idx] = nn.Linear(fc_dims[-1], output_dim)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        self.vf_head = nn.Linear(fc_dims[-1], 1)

        # used for action masking
        self.action_mask = None

    def get_flattened_obs_size(self):
        """Get the total size of the observation after flattening."""
        if isinstance(self.observation_space, Box):
            obs_size = np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_size = 0
            for key in sorted(self.observation_space):
                if key == _ACTION_MASK:
                    pass
                else:
                    obs_size += np.prod(self.observation_space[key].shape)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return int(obs_size)

    def get_flattened_obs(self, obs):
        """Get the flattened observation (ignore the action masks)."""
        if isinstance(self.observation_space, Box):
            return self.reshape_and_flatten(obs)
        if isinstance(self.observation_space, Dict):
            flattened_obs_dict = {}
            for key in sorted(self.observation_space):
                assert key in obs
                if key == _ACTION_MASK:
                    self.action_mask = self.reshape_and_flatten_obs(obs[key])
                else:
                    flattened_obs_dict[key] = self.reshape_and_flatten_obs(obs[key])
            flattened_obs = torch.cat(list(flattened_obs_dict.values()), dim=-1)
            return flattened_obs
        raise NotImplementedError("Observation space must be of Box or Dict type")

    @staticmethod
    def reshape_and_flatten_obs(obs):
        """Flatten observation."""
        assert len(obs.shape) >= 2
        batch_dim = obs.shape[0]
        return obs.reshape(batch_dim, -1)

    @override(TorchModelV2)
    def value_function(self):
        """Returns the estimated value function."""
        return self.values.reshape(-1)

    @staticmethod
    def apply_logit_mask(logits, mask):
        """
        Mask values of 1 are valid actions.
        Add huge negative values to logits with 0 mask values.
        """
        logit_mask = torch.ones_like(logits) * -10000000
        logit_mask = logit_mask * (1 - mask)
        return logits + logit_mask

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """You should implement forward() of forward_rnn() in your subclass."""
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()

        # Note: restoring original obs
        # as RLlib does not seem to be doing it automatically!
        original_obs = restore_original_dimensions(
            input_dict["obs"], self.obs_space.original_space, "torch"
        )

        obs = self.get_flattened_obs(original_obs)

        # Feed through the FC layers
        for layer in range(len(self.fc_dict)):
            output = self.fc_dict[str(layer)](obs)
            obs = output
        logits = output

        # Compute the action probabilities and the value function estimate
        # Apply action mask to the logits as well.
        action_masks = [None for _ in range(len(self.output_dims))]
        if self.action_mask is not None:
            start = 0
            for idx, dim in enumerate(self.output_dims):
                action_masks[idx] = self.action_mask[..., start : start + dim]
                start = start + dim
        action_logits = [
            self.apply_logit_mask(ph(logits), action_masks[idx])
            for idx, ph in enumerate(self.policy_head)
        ]
        self.values = self.vf_head(logits)[..., 0]

        concatenated_action_logits = torch.cat(action_logits, dim=-1)
        return torch.reshape(concatenated_action_logits, [-1, self.num_outputs]), state


ModelCatalog.register_custom_model("torch_linear", TorchLinear)
