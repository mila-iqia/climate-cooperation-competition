from typing import Any

import chex
import jax.numpy as jnp
import numpy as np

from rice_jax import Rice
from rice_jax.utils import i_to_agent_str


class OptimalMitigation(Rice):
    """Sets a mimum and maximum mitigation rate for all agents through the action mask

    env = OptimalMitigation(minimum_mitigation_rate=y, maximum_mitigation_rate=x)
    """

    # Both are inclusive:
    minimum_mitigation_rate: int = 9
    maximum_mitigation_rate: int = 9

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        action_mask = super().generate_action_masks(state)  # get default

        actions = jnp.arange(self.num_discrete_action_levels)
        min_mask = actions >= self.minimum_mitigation_rate
        max_mask = actions <= self.maximum_mitigation_rate
        min_max_mask = min_mask * max_mask

        for agent_id in range(self.num_regions):
            mitigation_mask = action_mask[i_to_agent_str(agent_id)]["mitigation_rate"]
            mitigation_mask = mitigation_mask * min_max_mask

        return action_mask


class BasicClub(Rice):
    club_mitigation_rate: int = 8
    promote_free_trade_among_club_members: bool = True
    # NOTE: this will be updated later with more targeted region_ids
    club_members_ = [0, 2, 4, 5, 6, 7, 15, 8, 12]

    @property
    def club_members(self) -> np.ndarray:
        return np.array(
            [
                region_id
                for region_id in range(self.num_regions)
                if region_id in self.club_members_
            ]
        )

    @property
    def non_club_members(self) -> list:
        return np.array(
            [
                region_id
                for region_id in range(self.num_regions)
                if region_id not in self.club_members
            ]
        )

    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = super().generate_action_masks(state)  # get default

        # Update action mask for each club member
        for agent_id in self.club_members:
            agent_str = i_to_agent_str(agent_id)

            # Force club members to mitigate a minimum of self.club_mitigation_rate
            club_member_mitigation_mask = (
                jnp.arange(self.num_discrete_action_levels) >= self.club_mitigation_rate
            )
            action_mask[agent_str]["mitigation_rate"] = club_member_mitigation_mask

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                self.club_mitigation_rate - state["mitigation_rates_all_regions"]
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

            # Optional: promote free trade among club members
            # Only allow "no-tariff" among club members
            if self.promote_free_trade_among_club_members:
                action_mask[agent_str]["import_tariff"].at[self.club_members].set(
                    jnp.arange(self.num_discrete_action_levels) == 0
                )

        return action_mask

    def generate_observation(self, state: dict) -> chex.Array:
        """Add a club membership indicator to the observation"""
        obs = super().generate_observation(state)
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)
            is_club_member = agent_id in self.club_members
            obs[agent_str] = jnp.concatenate(
                [obs[agent_str], jnp.array([is_club_member])]
            )

        return obs
