import chex
import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional


from jice.environment import Rice
from jice.environment.base_and_wrappers import EnvState, MultiDiscrete
from jice.environment.rice import EnvState, Actions

from dataclasses import replace, asdict


MITIGATION_RATE_ACTION_INDEX = 1
OBSERVATIONS = "observations"
ACTION_MASK = "action_mask"
NORMALIZATION_FACTORS = {
    "agent_ids": 1,
    "activity_timestep": 1e2,
    "global_temperature": 1e1,
    "global_carbon_mass": 1e4,
    "global_exogenous_emissions": 1,
    "global_land_emissions": 1,
    "global_temperature_boxes": 1e1,
    "global_carbon_reservoirs": 1e4,
    "global_cumulative_emissions": 1e4,
    "global_cumulative_land_emissions": 1e4,
    "global_alpha": 1e4,
    "global_emissions": 1e4,
    "global_acc_pert_carb_stock": 1e4,
    "capital_all_regions": 1e4,
    "capital_depreciation_all_regions": 1,
    "labor_all_regions": 1e4,
    "gross_output_all_regions": 1e3,
    "investment_all_regions": 1e3,
    "aggregate_consumption": 1e3,
    "savings_all_regions": 1e-1,
    "mitigation_rates_all_regions": 1e-1,
    "export_limit_all_regions": 1e-1,
    "current_balance_all_regions": 1e3,
    "import_tariffs": 1e2,
    "production_factor_all_regions": 1e2,
    "intensity_all_regions": 1e-1,
    "mitigation_cost_all_regions": 1,
    "damages_all_regions": 1,
    "abatement_cost_all_regions": 1,
    "production_all_regions": 1e3,
    "utility_all_regions": 1,
    "social_welfare_all_regions": 1,
    "utility_times_welfloss_all_regions": 1,

    # negotiation states
    "negotiation_stage": 1,
    "minimum_mitigation_rate_all_regions": 1e1,
    "promised_mitigation_rate": 1e1,
    "requested_mitigation_rate": 1e1,
    "proposal_decisions": 1,
    "proposed_mitigation_rates":1,
    "opts_all_regions":1
}

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
class EnvStateOpt:
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
        PROPOSAL_INDEX = OPT_INDEX + 1
        # PROPOSAL_INDEX_END = PROPOSAL_INDEX_START + (self.num_regions)
        DECISION_INDEX_START = PROPOSAL_INDEX + 1
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
            "proposals": PROPOSAL_INDEX,
            "decision_start": DECISION_INDEX_START,
            "decision_end": DECISION_INDEX_END,
        }

    @property
    def STEP_STAGES(self):
        if self.negotiation_on:
            return 4 # step_climate_and_economy, step_opt, step_propose, step_evaluate_proposals
        return 1

    @property
    def episode_length(self):
        simulation_timesteps = self.region_params.xN 
        return simulation_timesteps * self.STEP_STAGES
    
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
            proposal_nvec = [self.num_discrete_action_levels] * num_regions
            # TODO: decision_nvec needs to be [2] * num_regions
            # But the current setup is not able to handle varying length outputs
            decision_nvec = [self.num_discrete_action_levels] * num_regions 
            actions_nvec += [opts_nvec, proposal_nvec, decision_nvec]

        return np.concatenate(actions_nvec)

    @property
    def action_space(self) -> MultiDiscrete:
        return MultiDiscrete(self.action_nvec)

    def step_opt(
            self, state: EnvStateOpt, actions: ActionsOpt
    ) -> Tuple[chex.Array, EnvStateOpt]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        opts = actions.opt

        opts_binary = jnp.where(opts < 5, 0.0, jnp.where(opts > 5, 1.0, opts)).astype(jnp.float32)
        return replace(
            state,
            opts_all_regions=opts_binary
        )

    def step_propose(
        self, state: EnvStateOpt, actions: ActionsOpt
    ) -> Tuple[chex.Array, EnvStateOpt]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        proposed_mitigation_rates = actions.proposed_mitigation_rates
        opted_in_regions = state.opts_all_regions

        #only include proposals from regions who've opted should be saved to the sate

        proposed_mitigation_rates_opt_ins = proposed_mitigation_rates * opted_in_regions
        return replace(
            state,
            proposed_mitigation_rates=proposed_mitigation_rates_opt_ins,
        )

    def step_evaluate_proposals(
        self, state: EnvStateOpt, actions: ActionsOpt
    ) -> Tuple[chex.Array, EnvStateOpt]:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        proposed_mitigation_rates = state.proposed_mitigation_rates
        proposal_decisions = actions.proposal_decisions.T

        accepted_mitigation_rates = proposed_mitigation_rates * proposal_decisions
        lower_bound_mitigation_rates = jnp.max(accepted_mitigation_rates, axis=1)

        #only opting in regions have a lower bound
        opted_in_regions = state.opts_all_regions
        lower_bound_mitigation_rates_opt_ins = lower_bound_mitigation_rates * opted_in_regions
        return replace(
            state,
            proposal_decisions=proposal_decisions,
            minimum_mitigation_rate_all_regions=lower_bound_mitigation_rates_opt_ins,
        )
    
    def step_climate_and_economy(
        self, state: EnvStateOpt, actions: ActionsOpt
    ) -> Tuple[chex.Array, EnvStateOpt]:

        damages = self.calc_damages(state)
        abatement_costs = self.calc_abatement_costs(state, actions)  #
        productions = self.calc_productions(state)
        gross_outputs = self.calc_gross_outputs(
            damages, abatement_costs, productions
        )  #
        investments = self.calc_investments(gross_outputs, actions)  #
        gov_balances_post_interest = self.calc_gov_balances_post_interest(state)
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)
        gross_imports = self.calc_gross_imports(
            state,
            actions,
            gross_outputs,
            investments,
            debt_ratios,
        )

        tariff_revenues, net_imports = self.calc_trade_sanctions(
            state, gross_imports, actions
        )
        welfloss_multipliers = self.calc_welfloss_multiplier(
            state, gross_outputs, gross_imports, net_imports
        )
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports
        )
        utilities = self.calc_utilities(state, consumptions)  #
        # social_welfare = self.calc_social_welfares(state, utilities) #
        labors = self.calc_labors(state)
        capitals = self.calc_capitals(state, investments)
        production_factors = self.calc_production_factors(state)
        gov_balances_post_trade = self.calc_gov_balances_post_trade(
            gov_balances_post_interest, gross_imports
        )
        carbon_intensities = self.calc_carbon_intensities(state)

        global_carbon_mass = self.calc_global_carbon_mass(
            state, productions, actions.mitigation_rate
        )
        global_temperature, global_exogenous_emissions, global_temperature_boxes = self.calc_global_temperature(
            state
        )

        current_simulation_year = self.calc_current_simulation_year(state)

        utility_times_welfloss = utilities * welfloss_multipliers

        state: EnvStateOpt = replace(
            state,
            activity_timestep=state.activity_timestep + 1,
            # actions
            savings_all_regions=actions.savings_rate,
            import_tariffs=actions.import_tariff,
            export_limit_all_regions=actions.export_limit,
            import_bids_all_regions=actions.import_bids,
            mitigation_rates_all_regions=actions.mitigation_rate,
            # others
            damages_all_regions=damages,
            aggregate_consumption=consumptions,
            abatement_cost_all_regions=abatement_costs,
            production_all_regions=productions,
            gross_output_all_regions=gross_outputs,
            investment_all_regions=investments,
            current_balance_all_regions=gov_balances_post_trade,
            imports_minus_tariffs=net_imports,
            utility_all_regions=utilities,
            # social_welfare_all_regions=social_welfare,
            labor_all_regions=labors,
            capital_all_regions=capitals,
            production_factor_all_regions=production_factors,
            intensity_all_regions=carbon_intensities,
            global_carbon_mass=global_carbon_mass,
            global_temperature=global_temperature,
            global_exogenous_emissions=global_exogenous_emissions,
            global_temperature_boxes=global_temperature_boxes,
            current_simulation_year=current_simulation_year,
            utility_times_welfloss_all_regions=utility_times_welfloss,
        )
        return state
    
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:

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

        state = EnvStateOpt(
            current_timestep=0,
            activity_timestep=0,
            current_simulation_year=self.start_year,
            # Climate states
            global_temperature=global_temperature,
            global_carbon_mass=jnp.array(
                [
                    self.region_params.xM_AT_0,
                    self.region_params.xM_UP_0,
                    self.region_params.xM_LO_0,
                ]
            ).astype(jnp.float32),
            global_exogenous_emissions=0.0,  # NOTE: this is an array in the original (jnp.zeros(1))
            global_land_emissions=jnp.zeros(1),
            intensity_all_regions=self.region_params.xsigma_0,
            mitigation_rates_all_regions=self.region_params.xmitigation_0,
            # additional climate states for carbon and temperature model
            global_alpha=jnp.array(self.region_params.xalpha_0),
            global_carbon_reservoirs=jnp.array(
                [
                    self.region_params.xM_R1_0,
                    self.region_params.xM_R2_0,
                    self.region_params.xM_R3_0,
                    self.region_params.xM_R4_0,
                ]
            ),
            global_cumulative_emissions=jnp.array([self.region_params.xEcum_0]),
            global_cumulative_land_emissions=jnp.array(self.region_params.xEcumL_0),
            global_emissions=jnp.array(
                self.region_params.xEInd_0 + self.region_params.xEL_0
            ),
            global_acc_pert_carb_stock=jnp.array(
                self.region_params.xEcum_0
                + self.region_params.xEcumL_0
                - (
                    self.region_params.xM_R1_0
                    + self.region_params.xM_R2_0
                    + self.region_params.xM_R3_0
                    + self.region_params.xM_R4_0
                )
            ),
            global_temperature_boxes=jnp.array(
                [self.region_params.xT_LO_0, self.region_params.xT_UO_0]
            ),
            # economic states
            production_all_regions=jnp.zeros(self.num_regions),
            gross_output_all_regions=jnp.zeros(self.num_regions),
            aggregate_consumption=jnp.zeros(self.num_regions),
            investment_all_regions=jnp.zeros(self.num_regions),
            capital_all_regions=self.region_params.xK_0,
            capital_depreciation_all_regions=jnp.zeros(self.num_regions),
            labor_all_regions=self.region_params.xL_0,
            production_factor_all_regions=self.region_params.xA_0,
            current_balance_all_regions=jnp.zeros(self.num_regions),
            abatement_cost_all_regions=jnp.zeros(self.num_regions),
            # mitigation_cost_all_regions=jnp.zeros(self.num_regions),
            damages_all_regions=jnp.zeros(self.num_regions),
            utility_all_regions=jnp.zeros(self.num_regions),
            # social_welfare_all_regions=jnp.zeros(self.num_regions),
            utility_times_welfloss_all_regions=jnp.zeros(
                self.num_regions
            ),  # this is basically what used to be "rewards_all_regions"
            # trade states
            import_tariffs=jnp.zeros((self.num_regions, self.num_regions)),
            normalized_import_bids_all_regions=jnp.zeros(
                (self.num_regions, self.num_regions)
            ),
            import_bids_all_regions=self.region_params.ximport,
            imports_minus_tariffs=jnp.zeros((self.num_regions, self.num_regions)),
            export_limit_all_regions=self.region_params.xexport,
            savings_all_regions=self.region_params.xsaving_0,

            # negotiation states
            negotiation_stage=0,
            minimum_mitigation_rate_all_regions=jnp.zeros(self.num_regions),
            proposal_decisions=jnp.zeros((self.num_regions, self.num_regions), dtype=jnp.bool),
            opts_all_regions = jnp.zeros(self.num_regions, dtype=jnp.float32),
            proposed_mitigation_rates=jnp.zeros(self.num_regions),
        )

        obs_dict = self.generate_observation_and_action_mask(state)
        return obs_dict, state
    
    def generate_observation(self, state: EnvStateOpt) -> chex.Array:
        """
        Format observations for each agent by concatenating global, public
        and private features.
        """

        global_features = {
            "activity_timestep": jnp.array([state.activity_timestep]),
            "global_temperature": state.global_temperature,
            "global_carbon_mass": state.global_carbon_mass,
            "global_exogenous_emissions": jnp.array([state.global_exogenous_emissions]),
            "global_land_emissions": state.global_land_emissions,
            "global_temperature_boxes": state.global_temperature_boxes,
            "global_carbon_reservoirs": state.global_carbon_reservoirs,
            "global_cumulative_emissions": state.global_cumulative_emissions,
            "global_cumulative_land_emissions": jnp.array(
                [state.global_cumulative_land_emissions]
            ),
            "global_alpha": jnp.array([state.global_alpha]),
            "global_emissions": jnp.array([state.global_emissions]),
            "global_acc_pert_carb_stock": jnp.array([state.global_acc_pert_carb_stock]),
        }
        public_features = {
            # "capital_all_regions": state.capital_all_regions,
            # "capital_depreciation_all_regions": state.capital_depreciation_all_regions,
            # "labor_all_regions": state.labor_all_regions,
            # "gross_output_all_regions": state.gross_output_all_regions,
            # "investment_all_regions": state.investment_all_regions,
            # "aggregate_consumption": state.aggregate_consumption,
            # "savings_all_regions": state.savings_all_regions,
            "mitigation_rates_all_regions": state.mitigation_rates_all_regions,
            "opts_all_regions":state.opts_all_regions,
            "proposed_mitigation_rates": state.proposed_mitigation_rates,
            # "export_limit_all_regions": state.export_limit_all_regions,
            # "current_balance_all_regions": state.current_balance_all_regions,
            # "import_tariffs": state.import_tariffs.flatten(),
        }
        agent_ids = np.arange(self.num_regions)
        binary_agent_ids = ((agent_ids[:, None] & (1 << np.arange(self.num_regions.bit_length()))) > 0).astype(int)[:, ::-1]
        private_features = {
            "agent_ids": binary_agent_ids,
            "production_factor_all_regions": state.production_factor_all_regions,
            "intensity_all_regions": state.intensity_all_regions,
            # "mitigation_cost_all_regions": state.mitigation_cost_all_regions,
            "damages_all_regions": state.damages_all_regions,
            "abatement_cost_all_regions": state.abatement_cost_all_regions,
            "production_all_regions": state.production_all_regions,
            "utility_all_regions": state.utility_all_regions,
            # "social_welfare_all_regions": state.social_welfare_all_regions,
            # "utility_times_welfloss_all_regions": state.utility_times_welfloss_all_regions,

            "capital_all_regions": state.capital_all_regions,
            "capital_depreciation_all_regions": state.capital_depreciation_all_regions,
            "labor_all_regions": state.labor_all_regions,
            "gross_output_all_regions": state.gross_output_all_regions,
            "investment_all_regions": state.investment_all_regions,
            "aggregate_consumption": state.aggregate_consumption,
        }

        # Features concerning two regions
        bilateral_features = {}

        if self.negotiation_on:
            global_features["negotiation_stage"] = jnp.array([state.negotiation_stage])

            private_features["minimum_mitigation_rate_all_regions"] = state.minimum_mitigation_rate_all_regions

            bilateral_features = {
                "proposal_decisions": state.proposal_decisions,
            }


            # bilateral_features += [
            #     "promised_mitigation_rate",
            #     "requested_mitigation_rate",
            #     "proposal_decisions",
            # ]

        # Normalization:
        # assert all norm factors are present
        feature_keys = set(global_features.keys()) | set(public_features.keys()) | set(private_features.keys()) | set(bilateral_features.keys())
        assert feature_keys.issubset(set(NORMALIZATION_FACTORS.keys())), f"Missing normalization factors for {feature_keys - set(NORMALIZATION_FACTORS.keys())}"
        norm_factors = {k: v for k, v in NORMALIZATION_FACTORS.items() if k in feature_keys}

        normalized_features = jax.tree.map(
            lambda x, y: x / y,
            {**global_features, **public_features, **private_features, **bilateral_features},
            norm_factors,
        )

        global_public_features = {
            k: v
            for k, v in normalized_features.items()
            if k in {**global_features, **public_features}.keys()
        }
        global_public_features = jnp.concat(jax.tree.leaves(global_public_features))
        global_public_features_per_agent = jnp.broadcast_to(
            global_public_features, (self.num_regions, global_public_features.shape[0])
        )

        private_features = {
            k: v for k, v in normalized_features.items() if k in private_features.keys()
        }
        private_features_per_agent = jnp.column_stack(jax.tree.leaves(private_features))

        observations = [global_public_features_per_agent, private_features_per_agent]

        if self.negotiation_on:
            bilateral_features = {
                k: v for k, v in normalized_features.items() if k in bilateral_features.keys()
            }
            bilateral_features = jnp.hstack(jax.tree.leaves(bilateral_features))
            observations += [bilateral_features]

        return jnp.concatenate(observations, axis=1)
    


    def generate_observation_and_action_mask(self, state: EnvStateOpt) -> chex.Array:
        observations = self.generate_observation(state)
        action_masks = self.generate_action_masks(state)
        return {OBSERVATIONS: observations, ACTION_MASK: action_masks}

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

        return action_mask
        
    def process_actions(self, actions: chex.Array, state: EnvState) -> Actions:
        # actions is currently structured as (num_regions, num_actions)
        actions = actions.T  # (num_actions, num_regions)

        def add_diagonal_of_zeros(x: chex.Array):
            """
            Takes an ((n, n-1)) matrix and adds a 0s diagonal to it
            Output shape is ((n, n))
            This is helpful because it allows us to insert a 0 for an agent interacting with itself
            @example:
                [[2, 3],
                [1, 3],
                [1, 2]]
                ->
                [[0, 2, 3],
                [1, 0, 3],
                [1, 2, 0]]
            """
            # NOTE: see warning in the "Actions" class.
            n, m = x.shape
            assert n == m + 1, f"Expected x to have shape ((n, n-1)), but got {x.shape}"

            output = jnp.zeros(n * n, dtype=x.dtype)
            indices = (
                np.eye(n, dtype=np.bool_).__invert__().flatten()
            )  # this is fixed, so use Numpy
            output = output.at[indices].set(x.flatten())

            return output.reshape((n, n))

        def set_diagonal_to_zeros(x: chex.Array):
            """
            Takes an ((n, n)) matrix and sets the diagonal to 0
            """
            n, m = x.shape
            assert n == m, f"Expected x to have shape ((n, n)), but got {x.shape}"

            output = x.at[np.eye(n).astype(jnp.bool)].set(0)
            return output

        savings_rate_actions = actions[self.action_index["savings_rate"]]
        mitigation_rate_actions = actions[self.action_index["mitigation_rate"]]
        export_limit_actions = actions[self.action_index["export_limit"]]
        import_bid_actions = actions[self.action_index["import_bid_start"]:self.action_index["import_bid_end"]].T
        import_tariff_actions = actions[self.action_index["import_tariff_start"]:self.action_index["import_tariff_end"]].T

        # action windows
        if self.action_window_size > 0:
            # clip actions to be within the action window
            prev_savings_action = jnp.round(state.savings_all_regions * self.num_discrete_action_levels).astype(jnp.int32)
            prev_mitigation_action = jnp.round(state.mitigation_rates_all_regions  * self.num_discrete_action_levels).astype(jnp.int32)
            savings_rate_actions = jnp.clip(savings_rate_actions, prev_savings_action - self.action_window_size, prev_savings_action + self.action_window_size)
            mitigation_rate_actions = jnp.clip(mitigation_rate_actions, prev_mitigation_action - self.action_window_size, prev_mitigation_action + self.action_window_size)

        ### Set mitigation rate at min. mitigation rate
        ## This is for now also be enforced in the action mask
        # NOTE: this can possibly clash with the action window
        min_mitigation_rate = state.minimum_mitigation_rate_all_regions * self.num_discrete_action_levels
        mitigation_rate_actions = jnp.maximum(mitigation_rate_actions, min_mitigation_rate)
            

        if self.reduce_action_space_size:
            import_bid_actions = add_diagonal_of_zeros(import_bid_actions)
            import_tariff_actions = add_diagonal_of_zeros(import_tariff_actions)
        else: # set the diagonal to 0:
            import_bid_actions = set_diagonal_to_zeros(import_bid_actions)
            import_tariff_actions = set_diagonal_to_zeros(import_tariff_actions)
        if self.disable_trading:
            export_limit_actions = jnp.zeros_like(export_limit_actions)
            import_bid_actions = jnp.zeros_like(import_bid_actions)
            import_tariff_actions = jnp.zeros_like(import_tariff_actions)
        if not self.negotiation_on:
            return Actions(
                savings_rate=savings_rate_actions / self.num_discrete_action_levels,
                mitigation_rate=mitigation_rate_actions / self.num_discrete_action_levels,
                export_limit=export_limit_actions / self.num_discrete_action_levels,
                import_bids=import_bid_actions / self.num_discrete_action_levels,
                import_tariff=import_tariff_actions / self.num_discrete_action_levels,
            )
        else:
            proposal_actions = actions[self.action_index["proposals"]]
            decision_actions = actions[self.action_index["decision_start"]:self.action_index["decision_end"]].T
            opts = actions[self.action_index["opts"]]
            #decision_actions = set_diagonal_to_zeros(decision_actions)
            return ActionsOpt(
                savings_rate=savings_rate_actions / self.num_discrete_action_levels,
                mitigation_rate=mitigation_rate_actions / self.num_discrete_action_levels,
                export_limit=export_limit_actions / self.num_discrete_action_levels,
                import_bids=import_bid_actions / self.num_discrete_action_levels,
                import_tariff=import_tariff_actions / self.num_discrete_action_levels,
                opt = opts,
                proposed_mitigation_rates = proposal_actions / self.num_discrete_action_levels,
                proposal_decisions=decision_actions >= (self.num_discrete_action_levels / 2), # TODO
            )

    def step_env(
        self,
        key: chex.PRNGKey,
        prev_state: EnvStateOpt,
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