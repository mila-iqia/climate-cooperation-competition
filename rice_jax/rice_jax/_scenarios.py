from typing import Any
import jaxnasium as jym

import chex
import jax.numpy as jnp
import numpy as np
from jaxnasium import Discrete, MultiDiscrete
import equinox as eqx
import jax
import optax

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
                self.club_mitigation_rate
                - (
                    state["mitigation_rates_all_regions"]
                    * self.num_discrete_action_levels
                )
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
            # Simply add a feature to each agent observation
            obs[agent_str]["is_club_member"] = is_club_member

        return obs


class MaxExport(Rice):
    """Scenario where each region is forced to choose the maximum export limit permitted.

    This builds on the base export mask that already constrains export_ratio by
    physical capacity. After computing the default mask we reduce the
    export_limit action space to a single choice: the highest-discretization
    level still allowed by that mask.
    """

    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
        mask = super().generate_action_masks(state)

        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)
            if "export_limit" in mask[agent_str]:
                export_mask = mask[agent_str]["export_limit"]
                # find largest allowed index in a JIT‑friendly way
                # reverse mask and take argmax (returns 0 if no True values)
                rev_idx = jnp.argmax(export_mask[::-1])
                # compute corresponding forward index
                max_idx = export_mask.shape[0] - 1 - rev_idx
                # ensure the mask actually contained a True
                has_allowed = jnp.any(export_mask)
                # build one-hot: if none allowed, leave mask unchanged
                new_mask = jnp.where(
                    has_allowed,
                    jnp.eye(export_mask.shape[0], dtype=export_mask.dtype)[max_idx],
                    export_mask,
                )
                mask[agent_str]["export_limit"] = new_mask
        return mask


class BasicClubTariffAmbition(Rice):
    @property
    def action_space(self):
        N_REGIONS = self.num_regions
        N_DISCRETIZATION = self.num_discrete_action_levels

        action_space = super().action_space

        for agent_id in range(self.num_regions):
            agent_key = i_to_agent_str(agent_id)
            # remove standard propose
            action_space[agent_key].pop("proposal_ask", None)
            action_space[agent_key].pop("proposal_promise", None)
            # add new propose
            action_space[agent_key]["proposal"] = Discrete(N_DISCRETIZATION)

        return action_space


    def step_propose(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        proposals = actions["proposal"]

        state["proposals"] = proposals
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")

        proposals = state["proposals"]
        proposal_decisions = actions["proposal_decisions"].T

        proposal_decisions_ = (proposal_decisions > 0).astype(jnp.bool_)

        accepted_mitigation_rates = proposals * proposal_decisions_.astype(jnp.int_)

        lower_bound_mitigation_rates = jnp.max(accepted_mitigation_rates, axis=1)
        state["proposal_decisions"] = proposal_decisions_
        state["minimum_mitigation_rate_all_regions"] = lower_bound_mitigation_rates
        return state

    def _get_initial_state(self, key):
        # this fn gets called in reset_env of parent class before generate_observation_and_action_mask
        # so we can do scenario-specific state alterations here

        # Get default initial state
        state = super()._get_initial_state(key)

        # Scenario alterations:
        state.pop("promised_mitigation_rate")
        state.pop("requested_mitigation_rate")

        state["proposals"] = jnp.zeros((self.num_regions))

        return state

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Format observations for each agent by concatenating global, public and private features."""

        global_features = [
            # World values observed by all regions
            "activity_timestep",
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "global_temperature_boxes",
            "global_carbon_reservoirs",
            "global_cumulative_emissions",
            "global_cumulative_land_emissions",
            "global_alpha",
            "global_emissions",
            "global_acc_pert_carb_stock",
        ]

        public_features = [  # NOTE: commented some things out here and (partly) moved to private
            # Per agent values observed by all regions
            "mitigation_rates_all_regions",
            # "capital_all_regions",
            # "capital_depreciation_all_regions",
            # "labor_all_regions",
            # "gross_output_all_regions",
            # "investment_all_regions",
            # "aggregate_consumption",
            # "savings_all_regions",
            # "export_limit_all_regions",
            # "current_balance_all_regions",
            # "import_tariffs",
            "proposals",
        ]

        private_features = [
            # Per agent values whereby each agent only observes its own values
            "production_factor_all_regions",
            "intensity_all_regions",
            # "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            # "social_welfare_all_regions",
            # "reward_all_regions",
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            # "savings_all_regions",
            # "export_limit_all_regions",
            # "current_balance_all_regions",
            # "import_tariffs",
        ]
        bilateral_features = []  # Bilateral features are only observed by two regions
        if self.negotiation_on:
            global_features.append("negotiation_stage")
            private_features.append("minimum_mitigation_rate_all_regions")
            bilateral_features = ["proposal_decisions"]

        obs = {
            i_to_agent_str(agent_id): {
                **{feature: state[feature] for feature in global_features},
                **{feature: state[feature] for feature in public_features},
                **{feature: state[feature][agent_id] for feature in private_features},
                **{
                    feature: jnp.concatenate(
                        [state[feature][agent_id], state[feature].T[agent_id]]
                    )
                    for feature in bilateral_features
                },
            }
            for agent_id in range(self.num_regions)
        }

        # We flatten and concat everything (per agent) in the `generate_observation_and_action_mask` function
        return obs

    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = super().generate_action_masks(state)  # get default

        # Update action masks
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)

            # Minimum mitigation mask for own proposals is set via minimum mitigation rate in the state
            # and therefore already handled in parent class

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                (
                    state["minimum_mitigation_rate_all_regions"][agent_id]
                    - (state["mitigation_rates_all_regions"])
                )
                * self.num_discrete_action_levels
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )

            # When other regions are above own MMR, then the min_tariff should now all be 1s
            # Those we set to "only allow 0 tariff"
            min_tariff_amount_per_region_mask = jnp.where(
                jnp.all(min_tariff_amount_per_region_mask, axis=1)[:, None],
                (jnp.arange(self.num_discrete_action_levels) == 0)[None, :],
                min_tariff_amount_per_region_mask,
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

        return action_mask
    
class BasicClubTariffAmbitionFixedSavings(BasicClubTariffAmbition):
    """
    This scenario is identical to BasicClubTariffAmbition, but the savings
    rate action is removed from the agent's action space and fixed to a 
    constant value of 0.2 for all regions.
    """

    fixed_savings_rate: float = 0.2

    @property
    def action_space(self):
        # Get the action space from the parent class (BasicClubTariffAmbition)
        # by explicitly calling the property's getter function.
        action_space = BasicClubTariffAmbition.action_space.fget(self)

        # Remove the 'savings_rate' action for all agents, as it is now fixed
        for agent_id in range(self.num_regions):
            agent_key = i_to_agent_str(agent_id)
            if "savings_rate" in action_space[agent_key]:
                action_space[agent_key].pop("savings_rate")

        return action_space

    def process_actions(self, actions: dict, state: dict):
        # To call the overridden parent method, we must be explicit, as super()
        # does not work with a regular method if the class has property overrides.
        processed_actions = BasicClubTariffAmbition.process_actions(self, actions, state)

        # Manually add the fixed savings rate to the processed actions dictionary.
        # This creates an array of shape (num_regions,) with the fixed value.
        processed_actions["savings_rate"] = jnp.full(
            (self.num_regions,), self.fixed_savings_rate
        )

        return processed_actions
    
    def generate_action_masks_base(self, state: dict[str, Any]) -> dict[str, Any]:
        """Should output the same structure as `self.action_space`
        1: Allowed action, 0: disallowed action
        """

        NUM_REGIONS = self.num_regions
        DISCRETE_ACTION_LEVELS = self.num_discrete_action_levels

        def allow_all_actions_in_action_space(action_space):
            if isinstance(action_space, MultiDiscrete):
                num_actions = np.array(action_space.nvec).shape[0]
                num_discrete_actions = np.array(action_space.nvec)[0]
                return np.ones((num_actions, num_discrete_actions))
            elif isinstance(action_space, Discrete):
                num_discrete_actions = int(action_space.n)
                return np.ones((num_discrete_actions,))
            else:
                raise ValueError(f"Unknown action space: {action_space}")

        # Allow each action as a base
        mask = jax.tree.map(allow_all_actions_in_action_space, self.action_space)

        # Disallow actions on own region "self"
        for a_id in range(NUM_REGIONS):
            agent_str = i_to_agent_str(a_id)
            # Set diagonal elements to 0 for import actions (except first element)
            mask[agent_str]["import_bid"][a_id][1:] = 0
            mask[agent_str]["import_tariff"][a_id][1:] = 0
            if self.negotiation_on and "proposal_ask" in mask[agent_str]:
                mask[agent_str]["proposal_ask"][a_id][1:] = 0
                mask[agent_str]["proposal_promise"][a_id][1:] = 0

        # Minimum mitigation rate masking
        minimum_mitigation_rate_all = state["minimum_mitigation_rate_all_regions"]
        for agent_id in range(self.num_regions):
            min_mitigation_rate_agent = (
                minimum_mitigation_rate_all[agent_id] * self.num_discrete_action_levels
            )
            mask[i_to_agent_str(agent_id)]["mitigation_rate"] = (
                jnp.arange(self.num_discrete_action_levels) >= min_mitigation_rate_agent
            )

        if self.action_window_size > 0:

            def create_windowed_mask(prev_actions):
                MAX_DIFF = self.action_window_size
                POSSIBLE_ACTIONS = jnp.arange(DISCRETE_ACTION_LEVELS)
                return jnp.abs(POSSIBLE_ACTIONS - prev_actions) <= MAX_DIFF

            # Only allow actions around the previous action for `savings` and `mitigation` rate actions
            # prev_savings_actions = jnp.round(
            #     state["savings_all_regions"] * DISCRETE_ACTION_LEVELS
            # )
            prev_mitigation_actions = jnp.round(
                state["mitigation_rates_all_regions"] * DISCRETE_ACTION_LEVELS
            )
            for agent_id in range(NUM_REGIONS):
                # _savings_mask = create_windowed_mask(prev_savings_actions[agent_id])
                # mask[i_to_agent_str(agent_id)]["savings_rate"] = (
                #     mask[i_to_agent_str(agent_id)]["savings_rate"] * _savings_mask
                # )  # Multiply with existing mask to not overwrite

                _mitigation_mask = create_windowed_mask(
                    prev_mitigation_actions[agent_id]
                )
                agent_mitigation_mask = (
                    mask[i_to_agent_str(agent_id)]["mitigation_rate"] * _mitigation_mask
                )  # Multiply with existing mask to not overwrite

                # If the mitigation rate mask is now all 0s, that means the minimum mitigation rate is outside the window
                # in that case, the mitigation rate should still MOVE to the minimum mitigation rate
                # i.e. the upper half of the _mitigation_mask should be
                is_action_available = jnp.any(agent_mitigation_mask)
                move_to_minimum_within_window = (
                    prev_mitigation_actions[agent_id]
                    < jnp.arange(DISCRETE_ACTION_LEVELS)
                ) * _mitigation_mask
                mask[i_to_agent_str(agent_id)]["mitigation_rate"] = jax.lax.select(
                    is_action_available,
                    agent_mitigation_mask,
                    move_to_minimum_within_window,
                )

        return mask
    
    def generate_action_masks(self, state: dict) -> chex.Array:
        action_mask = self.generate_action_masks_base(state)  # get default

        # Update action masks
        for agent_id in range(self.num_regions):
            agent_str = i_to_agent_str(agent_id)

            # Minimum mitigation mask for own proposals is set via minimum mitigation rate in the state
            # and therefore already handled in parent class

            # Now we put a minimum tariff on everyone below the club mitigation rate
            # (for club members the minimum should be 0 since they always mitigate the club rate)
            min_tariff_amount_per_region = (
                (
                    state["minimum_mitigation_rate_all_regions"][agent_id]
                    - (state["mitigation_rates_all_regions"])
                )
                * self.num_discrete_action_levels
            ).clip(min=0)
            min_tariff_amount_per_region_mask = (
                jnp.arange(self.num_discrete_action_levels)
                >= min_tariff_amount_per_region[:, None]
            )

            # When other regions are above own MMR, then the min_tariff should now all be 1s
            # Those we set to "only allow 0 tariff"
            min_tariff_amount_per_region_mask = jnp.where(
                jnp.all(min_tariff_amount_per_region_mask, axis=1)[:, None],
                (jnp.arange(self.num_discrete_action_levels) == 0)[None, :],
                min_tariff_amount_per_region_mask,
            )
            action_mask[agent_str]["import_tariff"] = min_tariff_amount_per_region_mask

        return action_mask
    