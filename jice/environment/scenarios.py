import chex
import numpy as np
import jax.numpy as jnp

from jice.environment import Rice
from jice.environment.base_and_wrappers import EnvState
from jice.environment.rice import EnvState


MITIGATION_RATE_ACTION_INDEX = 1


class OptimalMitigation(Rice):

    # Both are inclusive:
    minimum_mitigation_rate: int = 9
    maximum_mitigation_rate: int = 9

    def generate_action_masks(self, state: EnvState) -> chex.Array:
        action_mask = super().generate_action_masks(state)  # get default

        action_mask = action_mask.at[
            :, MITIGATION_RATE_ACTION_INDEX, self.maximum_mitigation_rate + 1 :
        ].set(False)
        action_mask = action_mask.at[
            :, MITIGATION_RATE_ACTION_INDEX, : self.minimum_mitigation_rate
        ].set(False)

        return action_mask


class BasicClub(Rice):
    club_mitigation_rate: int = 8
    promote_free_trade_among_club_members: bool = True
    # NOTE: this will be updated later with more targeted region_ids
    club_members_ = [1, 2, 3, 4, 15, 8, 12]

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

    def generate_action_masks(self, state: EnvState) -> chex.Array:
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
