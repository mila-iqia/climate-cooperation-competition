from typing import Any

import chex
import jax.numpy as jnp
import numpy as np

from rice_jax import Rice
from rice_jax.utils import i_to_agent_str

MITIGATION_RATE_ACTION_INDEX = 1


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
    club_members_ = [0, 1, 2, 4, 5, 6, 7, 15, 8, 12]

    def __check_init__(self):
        assert False, "This scenario is not implemented"

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

        ### First the mitigation rate actions for the club members
        # set all club members mitigation to False
        action_mask = action_mask.at[
            self.club_members, MITIGATION_RATE_ACTION_INDEX, :
        ].set(False)
        # Then set only the appropriate mitigation rate (and above) for club members to True
        action_mask = action_mask.at[
            self.club_members, MITIGATION_RATE_ACTION_INDEX, self.club_mitigation_rate :
        ].set(True)

        ### Next the import tariffs for the non-club members by the club members
        # NOTE: import tarrifs are the final actions in the action space
        first_tariff_action_index = len(self.action_nvec) - self.num_regions
        last_tariff_action_index = len(self.action_nvec)
        non_club_member_tariff_action_indices = (
            first_tariff_action_index + self.non_club_members
        )

        if self.promote_free_trade_among_club_members:
            club_member_tariff_action_indices = (
                first_tariff_action_index + self.club_members
            )

            # set all club member tariffs to False
            action_mask = action_mask.at[
                self.club_members, first_tariff_action_index:, :
            ].set(False)

            # Then set only "no-tariff" to true for club members
            action_mask = action_mask.at[
                self.club_members, club_member_tariff_action_indices[:, None], 0
            ].set(True)

        min_tariff_amount_per_region = (
            self.club_mitigation_rate - state.mitigation_rates_all_regions
        ).astype(int)
        min_tariff_amount_per_region = min_tariff_amount_per_region.clip(min=0)
        mask_per_region = (
            jnp.arange(self.num_discrete_action_levels)
            >= min_tariff_amount_per_region[:, None]
        )
        action_mask = action_mask.at[
            self.club_members[:, None], non_club_member_tariff_action_indices
        ].set(mask_per_region[self.non_club_members])

        return action_mask

    def generate_observation(self, state: dict) -> chex.Array:
        """Add a club membership indicator to the observation"""
        obs = super().generate_observation(state)
        club_member_indicator = jnp.isin(
            np.arange(self.num_regions), self.club_members
        )[:, None]
        return jnp.concatenate([obs, club_member_indicator], axis=-1)
