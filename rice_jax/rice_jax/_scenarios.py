from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np

from rice_jax import Rice
from rice_jax.utils import i_to_agent_str
from jaxnasium import Discrete, MultiDiscrete

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
            # Simply add a feature to each agent observation
            obs[agent_str]["is_club_member"] = is_club_member

        return obs
    

class BasicClubTariffAmbition(Rice):
    
    @property
    def action_space(self):

        N_REGIONS = self.num_regions
        N_DISCRETIZATION = self.num_discrete_action_levels

        action_space = super().action_space

        for agent_id in range(self.num_regions):
            agent_key = i_to_agent_str(agent_id)
            #remove standard propose 
            action_space[agent_key].pop("proposal_ask", None)
            action_space[agent_key].pop("proposal_promise", None)
            #add new propose 
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
        #breakpoint()
        proposal_decisions = actions["proposal_decisions"].T

        proposal_decisions_ = (proposal_decisions > 0).astype(jnp.bool_)

        accepted_mitigation_rates = proposals * proposal_decisions_.astype(jnp.int_)

        lower_bound_mitigation_rates = jnp.max(accepted_mitigation_rates, axis=1)
        state["proposal_decisions"] = proposal_decisions_
        state["minimum_mitigation_rate_all_regions"] = lower_bound_mitigation_rates
        return state
    
    def reset_env(self, key):
        if self.temperature_calibration == "base":
            global_temperature = jnp.array(
                [self.region_params.xT_AT_0, self.region_params.xT_LO_0]
            )
        elif self.temperature_calibration == "FaIR":
            global_temperature = jnp.array(
                [self.region_params.xT_AT_0_FaIR, self.region_params.xT_LO_0_FaIR]
            )
        elif self.temperature_calibration == "DFaIR":
            global_temperature = jnp.array(
                [
                    self.region_params.xT_LO_0 + self.region_params.xT_UO_0,
                    self.region_params.xT_LO_0,
                ]
            )
        else:
            raise ValueError(
                f"Unknown temperature calibration: {self.temperature_calibration}"
            )

        # fmt: off
        state = {
            "current_timestep": 0,
            "activity_timestep": 0,
            "current_simulation_year": self.start_year,
            # Climate states
            "global_temperature": global_temperature,
            "global_carbon_mass": jnp.array(
                [
                    self.region_params.xM_AT_0,
                    self.region_params.xM_UP_0,
                    self.region_params.xM_LO_0,
                ]
            ).astype(jnp.float32),
            "global_exogenous_emissions": 0.0,  # Originally an array (jnp.zeros(1))
            "global_land_emissions": 0.0,  # jnp.zeros(1)
            "intensity_all_regions": self.region_params.xsigma_0,
            "mitigation_rates_all_regions": self.region_params.xmitigation_0,
            # Additional climate states
            "global_alpha": jnp.array(self.region_params.xalpha_0, dtype=jnp.float32),
            "global_carbon_reservoirs": jnp.array(
                [
                    self.region_params.xM_R1_0,
                    self.region_params.xM_R2_0,
                    self.region_params.xM_R3_0,
                    self.region_params.xM_R4_0,
                ]
            ),
            "global_cumulative_emissions": jnp.array(self.region_params.xEcum_0, dtype=jnp.float32),
            "global_cumulative_land_emissions": jnp.array(self.region_params.xEcumL_0, dtype=jnp.float32),
            "global_emissions": jnp.array(self.region_params.xEInd_0 + self.region_params.xEL_0),
            "global_acc_pert_carb_stock": jnp.array(
                self.region_params.xEcum_0
                + self.region_params.xEcumL_0
                - (
                    self.region_params.xM_R1_0
                    + self.region_params.xM_R2_0
                    + self.region_params.xM_R3_0
                    + self.region_params.xM_R4_0
                )
            ),
            "global_temperature_boxes": jnp.array([self.region_params.xT_LO_0, self.region_params.xT_UO_0]),
            # Economic states
            "production_all_regions": jnp.zeros(self.num_regions),
            "gross_output_all_regions": jnp.zeros(self.num_regions),
            "aggregate_consumption": jnp.zeros(self.num_regions),
            "investment_all_regions": jnp.zeros(self.num_regions),
            "capital_all_regions": self.region_params.xK_0,
            "capital_depreciation_all_regions": jnp.zeros(self.num_regions),
            "labor_all_regions": self.region_params.xL_0,
            "production_factor_all_regions": self.region_params.xA_0,
            "current_balance_all_regions": jnp.zeros(self.num_regions),
            "abatement_cost_all_regions": jnp.zeros(self.num_regions),
            "damages_all_regions": jnp.zeros(self.num_regions),
            "utility_all_regions": jnp.zeros(self.num_regions),
            "utility_times_welfloss_all_regions": jnp.zeros(self.num_regions),
            # Trade states
            "import_tariffs": jnp.zeros((self.num_regions, self.num_regions)),
            "normalized_import_bids_all_regions": jnp.zeros((self.num_regions, self.num_regions)),
            "import_bids_all_regions": self.region_params.ximport,
            "import_tariffs_all_regions": jnp.zeros((self.num_regions, self.num_regions)),
            "imports_minus_tariffs": jnp.zeros((self.num_regions, self.num_regions)),
            "export_limit_all_regions": self.region_params.xexport,
            "savings_all_regions": self.region_params.xsaving_0,
            # Negotiation states
            "negotiation_stage": 0,
            "minimum_mitigation_rate_all_regions": jnp.zeros(self.num_regions),
            # "promised_mitigation_rate": jnp.zeros((self.num_regions, self.num_regions)), #REMOVED FOR SCENARIO
            # "requested_mitigation_rate": jnp.zeros((self.num_regions, self.num_regions)), #REMOVED FOR SCENARIO
            "proposal_decisions": jnp.zeros((self.num_regions, self.num_regions), dtype=jnp.bool),
        }
        # fmt: on

        #SCENARIO SPECIFIC STATEs
        state["proposals"] = jnp.zeros((self.num_regions))

        obs_dict = self.generate_observation_and_action_mask(state)

        
        return obs_dict, state

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
            "proposals"
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
            bilateral_features = [

                "proposal_decisions"
            ]

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
    
    def generate_action_masks(self, state: dict[str, Any]) -> dict[str, Any]:
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

        def get_tariff_mask_value(agent_mmr, other_agent_mmr):
            if other_agent_mmr < agent_mmr:
                return agent_mmr - other_agent_mmr
            else:
                return 0
        
        # Allow each action as a base
        mask = jax.tree.map(allow_all_actions_in_action_space, self.action_space)

        # Minimum mitigation rate masking
        minimum_mitigation_rate_all = state["minimum_mitigation_rate_all_regions"]

        # Disallow actions on own region "self"
        for a_id in range(NUM_REGIONS):
            agent_str = i_to_agent_str(a_id)
            # Set diagonal elements to 0 for import actions (except first element)
            mask[agent_str]["import_bid"][a_id][1:] = 0
            mask[agent_str]["import_tariff"][a_id][1:] = 0

            #mask mitigation rate
            min_mitigation_rate_agent = minimum_mitigation_rate_all[a_id] * self.num_discrete_action_levels
            mask[agent_str]["mitigation_rate"] = (
                jnp.arange(self.num_discrete_action_levels) >= min_mitigation_rate_agent
            )

            # #mask tariff rates
            # for other_id in range(NUM_REGIONS):
            #     if a_id == other_id:
            #         pass
            #     else:
            #         #check if agent in club
            #         other_agent_mmr = minimum_mitigation_rate_all[other_id] * self.num_discrete_action_levels
            #         #if agent not in club tariff the difference
                    
            #         if other_agent_mmr < min_mitigation_rate_agent:
            #             difference = min_mitigation_rate_agent - other_agent_mmr
            #             mask[agent_str]["import_tariff"][other_id] = [0] * difference + [1]*(self.num_discrete_action_levels-difference)
            #         #free trade
            #         else:
            #             mask[agent_str]["import_tariff"][other_id][1:] = 0

            # inside: for a_id in range(NUM_REGIONS):
            agent_str = i_to_agent_str(a_id)
            minimum_mitigation_rate_all = state["minimum_mitigation_rate_all_regions"]

            # Convert fractional minima to discrete indices (same semantics as your commented multiplication)
            min_mit_agent_idx = jnp.floor(minimum_mitigation_rate_all[a_id] * DISCRETE_ACTION_LEVELS).astype(jnp.int32)  # scalar
            other_agent_mmr_idx = jnp.floor(minimum_mitigation_rate_all * DISCRETE_ACTION_LEVELS).astype(jnp.int32)     # shape (NUM_REGIONS,)

            # Compute difference = max(0, min_agent_idx - other_agent_mmr_idx) for each other agent
            diff_vec = jnp.clip(min_mit_agent_idx - other_agent_mmr_idx, 0, DISCRETE_ACTION_LEVELS)  # shape (NUM_REGIONS,)

            # Per-index vector for building masks
            indices = jnp.arange(DISCRETE_ACTION_LEVELS)  # shape (N,)
            # If other_agent_mmr < min_mit_agent: mask = [0]*diff + [1]*(N-diff)  -> indices >= diff
            masks_if_tariff = indices[None, :] >= diff_vec[:, None]  # shape (NUM_REGIONS, N), bool
            # Else (free trade): only allow index 0 (same as mask[...][1:] = 0)
            masks_if_free = (indices == 0)[None, :].repeat(NUM_REGIONS, axis=0)  # shape (NUM_REGIONS, N), bool

            # Condition per other agent
            condition = other_agent_mmr_idx < min_mit_agent_idx  # shape (NUM_REGIONS,), bool
            final_masks = jnp.where(condition[:, None], masks_if_tariff, masks_if_free)  # shape (NUM_REGIONS, N)

            # Preserve the already-set diagonal behaviour (do not overwrite the self -> self row)
            # earlier in generate_action_masks you set mask[agent_str]["import_tariff"][a_id][1:] = 0
            # so keep that row as-is:
            final_masks = final_masks.at[a_id].set(mask[agent_str]["import_tariff"][a_id])

            # Assign back
            mask[agent_str]["import_tariff"] = final_masks
                    



        if self.action_window_size > 0:

            def create_windowed_mask(prev_actions):
                MAX_DIFF = self.action_window_size
                POSSIBLE_ACTIONS = jnp.arange(DISCRETE_ACTION_LEVELS)
                return jnp.abs(POSSIBLE_ACTIONS - prev_actions) <= MAX_DIFF

            # Only allow actions around the previous action for `savings` and `mitigation` rate actions
            prev_savings_actions = jnp.round(
                state["savings_all_regions"] * DISCRETE_ACTION_LEVELS
            )
            prev_mitigation_actions = jnp.round(
                state["mitigation_rates_all_regions"] * DISCRETE_ACTION_LEVELS
            )
            for agent_id in range(NUM_REGIONS):
                savings_mask = mask[i_to_agent_str(agent_id)]["savings_rate"]
                _savings_mask = create_windowed_mask(prev_savings_actions[agent_id])
                savings_mask = savings_mask * _savings_mask  # Multiply to not overwrite

                mitigation_mask = mask[i_to_agent_str(agent_id)]["mitigation_rate"]
                _mitigation_mask = create_windowed_mask(
                    prev_mitigation_actions[agent_id]
                )
                mitigation_mask = mitigation_mask * _mitigation_mask  # ""

        return mask