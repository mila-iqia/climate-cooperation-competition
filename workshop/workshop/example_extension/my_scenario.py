"""Your scenario: subclass `Rice` and override the hooks you need.

Example: a free-trade bloc. Bloc members
  1. cannot put import tariffs on each other          (generate_action_masks)
  2. observe whether they are in the bloc             (generate_observation)
  3. optionally share part of their reward            (generate_rewards)
and it logs the mitigation of bloc members vs outsiders (generate_info).

Everything here runs inside `jax.jit`: use `jnp` for array maths, and remember
that jax arrays are immutable (`x.at[i].set(v)` returns a *new* array).
"""

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from rice_jax.utils import i_to_agent_str

from rice_jax import Rice


class FreeTradeBloc(Rice):
    # Fields become constructor arguments: make_env(FreeTradeBloc, bloc_members=(0, 2))
    bloc_members: tuple[int, ...] = eqx.field(static=True, default=(0, 1))
    # 0 = every region maximises its own reward, 1 = bloc members all get the bloc average
    bloc_reward_sharing: float = 0.0

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        """Which discrete action levels are allowed (1) or not (0).

        Same structure as `self.action_space`: {region: {action_name: mask}}.
        """
        # Start with default masks, so defaults stay in place:
        mask = super().generate_action_masks(state)

        only_level_zero = jnp.arange(self.num_discrete_action_levels) == 0
        for i in self.bloc_members:
            region = i_to_agent_str(i)
            # Shape (num_regions, levels): row j = tariff on imports from region j
            tariff_mask = jnp.asarray(mask[region]["import_tariff"])
            for j in self.bloc_members:
                tariff_mask = tariff_mask.at[j].set(only_level_zero)
            mask[region]["import_tariff"] = tariff_mask

        return mask

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Change the observation for the environments
        Output should be a dict of {i_to_agent_str(i): ...}, i.e. a dict where
        the first level keys are the agent index (action space and rewards should
        also follow this structure)
        """
        obs = super().generate_observation(state)
        for i in range(self.num_regions):
            obs[i_to_agent_str(i)]["in_bloc"] = jnp.float32(i in self.bloc_members)
        return obs

    def generate_rewards(self, new_state: dict, old_state: dict) -> dict[str, Any]:
        """Per-region reward for the step old_state -> new_state.
        Same shaped dict as obs and action space"""
        rewards = super().generate_rewards(new_state, old_state)

        bloc = [i_to_agent_str(i) for i in self.bloc_members]
        bloc_mean = sum(rewards[r] for r in bloc) / len(bloc)
        w = self.bloc_reward_sharing
        for r in bloc:
            rewards[r] = (1 - w) * rewards[r] + w * bloc_mean

        return rewards

    def generate_info(self, state: dict, actions: dict, rewards: dict) -> dict:
        """Extra values to log each step. They end up in the `info` of every
        `env.step`: in PPO's `log_function` during training, and in rollouts."""
        info = super().generate_info(state, actions, rewards)

        mitigation = state["mitigation_rates_all_regions"]  # one rate per region
        bloc = np.array(self.bloc_members)
        outsiders = np.setdiff1d(np.arange(self.num_regions), bloc)
        info["bloc_mitigation"] = mitigation[bloc].mean()
        info["outsider_mitigation"] = mitigation[outsiders].mean()

        return info
