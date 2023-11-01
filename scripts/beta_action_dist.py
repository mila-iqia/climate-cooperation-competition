import torch.distributions as td
import numpy as np
import torch
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.models.action_dist import ActionDistribution
import gymnasium as gym
from typing import Optional
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
torch, _ = try_import_torch()

class BetaActionDistribution(TorchDistributionWrapper):
    # refer to the TorchDiagGaussian Class in miniconda3\envs\ai4gcc\lib\python3.7\site-packages\ray\rllib\models\torch\torch_action_dist.py
    @override(ActionDistribution)
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        *,
        action_space: Optional[gym.spaces.Space] = None
    ):
        super().__init__(inputs, model)
        # Assuming the first half of inputs is alpha, and the second half is beta.
        alpha, beta = torch.chunk(inputs, 2, dim=1)
        self.dist = torch.distributions.Beta(alpha, beta)
        self.zero_action_dim = action_space and action_space.shape == ()


    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        sample = super().sample()
        if self.zero_action_dim:
            return torch.squeeze(sample, dim=-1)
        return sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        alpha, beta = torch.chunk(self.inputs, 2, dim=1)
        mean = alpha / (alpha + beta)
        self.last_sample = mean
        return self.last_sample
    
    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return super().logp(actions).sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape, dtype=np.int32) * 2

    @override(ActionDistribution)
    def sampled_action_logp(self) -> TensorType:
        assert self.last_sample is not None
        return self.logp(self.last_sample)
    # def sampled_action_logp(self):
    #     if self.last_sample is None:
    #         # Use an example shape for the dummy batch. Here, I assume inputs shape is (batch_size, num_actions * 2).
    #         dummy_batch_size = self.inputs.shape[0]
    #         return torch.tensor([-float('inf')] * dummy_batch_size).squeeze(-1)
        
    #     logp = self.logp(self.last_sample)
    #     return logp

    # def sample(self, deterministic=False):
    #     if deterministic:
    #         action = self.deterministic_sample()
    #     else:
    #         action = self.dist.sample()

    #     self.last_sample = action  # Logging the sampled action
    #     return action