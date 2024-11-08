import chex
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Optional


from jice.environment import Rice
from jice.environment.base_and_wrappers import EnvState
from jice.environment.rice import EnvState, Actions
from dataclasses import replace, asdict


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
    


@chex.dataclass(frozen=True)
class ActionsOpt:
    savings_rate: chex.Array  # one action (per region)
    mitigation_rate: chex.Array  # one action (per region)
    export_limit: chex.Array  # one action (per region)
    import_bids: chex.Array  # num_regions actions (-1(optional)) (per region)
    import_tariff: chex.Array  # num_regions actions (-1(optional)) (per region)
    opt: chex.Array  # one action (per region)
    proposed_mitigation_rates: chex.Array  # one action (per region)
    
    # Optional fields with default values
    # promised_mitigation_rate: Optional[chex.Array] = None
    # requested_mitigation_rate: Optional[chex.Array] = None
    proposal_decisions: Optional[chex.Array] = None
 

@chex.dataclass
class EnvStateOpt(EnvState):
    current_timestep: int # The RL timestep
    activity_timestep: int # The timestep in the simulation (can be different from RL timestep if negotiation is on)
    current_simulation_year: int

    # climate states
    global_temperature: chex.Array
    global_carbon_mass: chex.Array
    global_exogenous_emissions: chex.Array
    global_land_emissions: chex.Array
    intensity_all_regions: chex.Array
    mitigation_rates_all_regions: chex.Array
    global_temperature_boxes: chex.Array

    # additional climate states for carbon model
    global_alpha: int  # or float?
    global_carbon_reservoirs: chex.Array
    global_cumulative_emissions: chex.Array
    global_cumulative_land_emissions: int  # or float?
    global_emissions: int  # or float?
    global_acc_pert_carb_stock: int  # or float?

    # economic states
    production_all_regions: chex.Array
    gross_output_all_regions: chex.Array
    aggregate_consumption: chex.Array
    investment_all_regions: chex.Array
    capital_all_regions: chex.Array
    capital_depreciation_all_regions: chex.Array
    labor_all_regions: chex.Array
    production_factor_all_regions: chex.Array
    current_balance_all_regions: chex.Array
    abatement_cost_all_regions: chex.Array
    # mitigation_cost_all_regions: chex.Array
    damages_all_regions: chex.Array
    utility_all_regions: chex.Array
    # social_welfare_all_regions: chex.Array

    # trade states
    # tariffs: chex.Array
    import_tariffs: chex.Array
    normalized_import_bids_all_regions: chex.Array
    import_bids_all_regions: chex.Array
    imports_minus_tariffs: chex.Array
    export_limit_all_regions: chex.Array

    savings_all_regions: chex.Array
    utility_times_welfloss_all_regions: (
        chex.Array
    )  # this is basically what used to be "rewards_all_regions"

    # # negotiation states
    negotiation_stage: chex.Array
    minimum_mitigation_rate_all_regions: chex.Array
    opts_all_regions: chex.Array
    proposed_mitigation_rates: chex.Array
    proposal_decisions: chex.Array    

class OptIn(Rice):
    """
    Questions for Koen:
    - if we're strict typing,
         would i need actually to overwrite every single function bc the 
         dataclasses are fixed. 
    - How do we set the number of possible values per action, id like opt to be 0/1
    """

    @property
    def action_index(self):
        # Action indices
        SAVINGS_RATE_INDEX = 0
        MITIGATION_RATE_INDEX = 1
        EXPORT_LIMIT_INDEX = 2
        IMPORT_BID_INDEX_START = 3
        IMPORT_BID_INDEX_END = IMPORT_BID_INDEX_START + self.num_regions - self.reduce_action_space_size
        IMPORT_TARIFF_INDEX_START = IMPORT_BID_INDEX_END
        IMPORT_TARIFF_INDEX_END = IMPORT_TARIFF_INDEX_START + self.num_regions - self.reduce_action_space_size
        OPT_INDEX = IMPORT_TARIFF_INDEX_END
        PROPOSAL_INDEX_START = OPT_INDEX + 1
        PROPOSAL_INDEX_END = PROPOSAL_INDEX_START + (self.num_regions * 2)
        DECISION_INDEX_START = PROPOSAL_INDEX_END
        DECISION_INDEX_END = DECISION_INDEX_START + self.num_regions
        return {
            "savings_rate": SAVINGS_RATE_INDEX,
            "mitigation_rate": MITIGATION_RATE_INDEX,
            "export_limit": EXPORT_LIMIT_INDEX,
            "import_bid_start": IMPORT_BID_INDEX_START,
            "import_bid_end": IMPORT_BID_INDEX_END,
            "import_tariff_start": IMPORT_TARIFF_INDEX_START,
            "import_tariff_end": IMPORT_TARIFF_INDEX_END,
            "opts": OPT_INDEX,
            "proposal_start": PROPOSAL_INDEX_START,
            "proposal_end": PROPOSAL_INDEX_END,
            "decision_start": DECISION_INDEX_START,
            "decision_end": DECISION_INDEX_END,
        }
    
    ###
    ## Helper and environment functions
    ###
    @property
    def action_nvec(self) -> chex.Array:
        # num_actions = len(Actions.__annotations__)
        num_regions = self.num_regions
        import_bids_nvec = [self.num_discrete_action_levels] * (
            num_regions - self.reduce_action_space_size
        )
        import_tariff_nvec = [self.num_discrete_action_levels] * (
            num_regions - self.reduce_action_space_size
        )
        actions_nvec = [
                [self.num_discrete_action_levels],  # savings_rate
                [self.num_discrete_action_levels],  # mitigation_rate
                [self.num_discrete_action_levels],  # export_limit
                import_bids_nvec,
                import_tariff_nvec,
            ]

        if self.negotiation_on:
            opts_nvec = [self.num_discrete_action_levels]
            proposal_nvec = [self.num_discrete_action_levels] * 2 * num_regions
            # TODO: decision_nvec needs to be [2] * num_regions
            # But the current setup is not able to handle varying length outputs
            decision_nvec = [self.num_discrete_action_levels] * num_regions 
            actions_nvec += [opts_nvec, proposal_nvec, decision_nvec]

        return np.concatenate(actions_nvec)

    def step_opt(
            self, state: EnvStateOpt, actions: ActionsOpt
    ) -> Tuple[chex.Array, EnvStateOpt]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        
        opts = ActionsOpt.opt
        opts_binary = jnp.where(opts < 5, 0, jnp.where(opts > 5, 1, opts))

        return replace(
            state,
            opts=opts_binary
        )

    def step_propose(
        self, state: EnvState, actions: Actions
    ) -> Tuple[chex.Array, EnvState]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        proposed_mitigation_rates = actions.proposed_mitigation_rates
        opted_in_regions = state.opts

        #only include proposals from regions who've opted should be saved to the sate
        proposed_mitigation_rates_opt_ins = proposed_mitigation_rates * opted_in_regions
        return replace(
            state,
            proposed_mitigation_rates=proposed_mitigation_rates_opt_ins,
        )

    def step_evaluate_proposals(
        self, state: EnvState, actions: Actions
    ) -> Tuple[chex.Array, EnvState]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        
        proposed_mitigation_rates = state.proposed_mitigation_rate
        proposal_decisions = actions.proposal_decisions.T

        accepted_mitigation_rates = proposed_mitigation_rates * proposal_decisions
        lower_bound_mitigation_rates = jnp.max(accepted_mitigation_rates, axis=1)
        #only opting in regions have a lower bound
        opted_in_regions = state.opts
        lower_bound_mitigation_rates_opt_ins = lower_bound_mitigation_rates * opted_in_regions
        return replace(
            state,
            proposal_decisions=proposal_decisions,
            minimum_mitigation_rate_all_regions=lower_bound_mitigation_rates_opt_ins,
        )
    
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:

        obs_dict, state = super().reset_env(key)
        state = EnvStateOpt(
            **state,
            opts_all_regions = jnp.zeros(self.num_regions),
            proposed_mitigation_rates=jnp.zeros(self.num_regions),
        )
        obs_dict = self.generate_observation_and_action_mask(state)
        return obs_dict, state

    def generate_action_masks(self, state: EnvStateOpt) -> chex.Array:
        """This function is typically overwritten by a scenario"""
        default_action_mask = jnp.ones(  # allow everything
            (
                self.num_regions,
                self.action_nvec.shape[0],
                self.num_discrete_action_levels,
            ),
            dtype=jnp.bool,
        )
        action_mask = default_action_mask
        
        #get where all minimum mitigation rates
        min_mitigation_rate_diff = state.minimum_mitigation_rate_all_regions[:,None]\
        - state.minimum_mitigation_rate_all_regions[None,:]
        #this gives a 0 for all regions i,j where i < j and 1 otherwise
        min_mitigation_rate_vec_mask = jnp.clip(min_mitigation_rate_diff,0,1)

        tariff_values = min_mitigation_rate_diff * min_mitigation_rate_vec_mask
        opts = state.opts_all_regions #(num_regions)
        tariff_values_opt_in = tariff_values*opts


        # min_tariff_idx = self.action_index()["import_tariff_start"]*self.num_discrete_action_levels
        # min_tariff_end = self.action_index()["import_tariff_start"]+self.num_regions*self.num_discrete_action_levels

        tariff_mask = tariff_values_opt_in[:, :, None] <= jnp.arange(self.num_discrete_action_levels)[None, None, :]

        action_mask = action_mask.at[
            :,self.action_index["import_tariff_start"] : self.action_index["import_tariff_end"]
        ].set(tariff_mask)

        minimum_mitigation_rate = state.minimum_mitigation_rate_all_regions
        action_mask = action_mask.at[
            :, self.action_index["mitigation_rate"]
        ].set(
            jnp.arange(self.num_discrete_action_levels) >= minimum_mitigation_rate[:, None]
        )
        # action_mask = action_mask * min_mitigation_mask

        return action_mask

        # min_mitigation_rate_diff = min_mitigation_rate_diff[:,None,:] #inserts an extra axis
        # min_mitigation_rate_diff = jnp.broadcast_to(min_mitigation_rate_diff,
        #                                              (self.num_regions,
        #                                                self.action_nvec.shape[0],
        #                                                self.num_discrete_action_levels))

        
        # if self.action_window_size > 0:
        #     # Only allow actions around the previous action
        #     # For the actions: Savings_rate and Mitigation_rate
        #     action_window_mask = default_action_mask.copy()
        #     prev_savings_action = jnp.round(state.savings_all_regions * self.num_discrete_action_levels).astype(jnp.int32)
        #     prev_mitigation_action = jnp.round(state.mitigation_rates_all_regions  * self.num_discrete_action_levels).astype(jnp.int32)
        #     action_window_mask = action_window_mask.at[
        #         :, self.action_index["savings_rate"]
        #     ].set(
        #         jnp.abs(np.arange(self.num_discrete_action_levels) - prev_savings_action[:, None]).astype(jnp.int32) <= self.action_window_size
        #     )
        #     action_window_mask = action_window_mask.at[
        #         :, self.action_index["mitigation_rate"]
        #     ].set(
        #         jnp.abs(np.arange(self.num_discrete_action_levels) - prev_mitigation_action[:, None]).astype(jnp.int32) <= self.action_window_size
        #     )
        #     action_mask = action_mask * action_window_mask
        


    def step_env(
        self,
        key: chex.PRNGKey,
        prev_state: EnvState,
        raw_actions: chex.Array,
        negotiation_stage: int,
    ) -> Tuple[chex.PyTreeDef, EnvState, float, bool, dict]:
        
        state = replace(
            prev_state,
            current_timestep=prev_state.current_timestep + 1,
        )

        actions = self.process_actions(raw_actions, state)

        if not self.negotiation_on:
            negotiation_stage = 0
        if negotiation_stage == 0:
            state = self.step_climate_and_economy(state, actions)
        elif negotiation_stage == 1:
            state = self.step_opt(state,actions)
        elif negotiation_stage == 2:
            state = self.step_propose(state, actions)
        elif negotiation_stage == 3:
            state = self.step_evaluate_proposals(state, actions)

        obs_dict = self.generate_observation_and_action_mask(state)
        reward = self.generate_rewards(
            state, prev_state
        ) # NOTE: rewards is zero for proposel steps
        done, discount = self.generate_terminated_truncated_discount(state)
        info = self.generate_info(state, actions)

        return (obs_dict, reward, done, discount, info), state

class BasicClub(Rice):
    club_mitigation_rate: int = 8
    promote_free_trade_among_club_members: bool = True
    # NOTE: this will be updated later with more targeted region_ids
    club_members_ = [0, 1, 2, 4, 5, 6, 7, 15, 8, 12]

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

    def generate_observation(self, state: EnvState) -> chex.Array:
        """Add a club membership indicator to the observation"""
        obs = super().generate_observation(state)
        club_member_indicator = jnp.isin(
            np.arange(self.num_regions), self.club_members
        )[:, None]
        return jnp.concatenate([obs, club_member_indicator], axis=-1)