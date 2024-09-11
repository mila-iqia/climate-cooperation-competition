import jax
import jax.experimental
import jax.numpy as jnp
import chex
from typing import Tuple
from gymnax.environments.spaces import Box
from types import SimpleNamespace
from dataclasses import replace, asdict
import numpy as np
import equinox as eqx

from jice.environment.base_and_wrappers import JaxBaseEnv, EnvState, MultiDiscrete

# TODO:
# - BUG export-limit 0 is always learned
# - negotiation


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
}


# jax.config.update("jax_disable_jit", True)
@chex.dataclass(frozen=True)
class Actions:
    # NOTE minus one is not in the original; but bid/tarif on self does not make sense
    # this can be toggled with the reduce_action_space_size flag
    # reduces the action space, but since a single model is used for all agents,
    # the action number may mean something different for each agent
    savings_rate: chex.Array  # one action (per region)
    mitigation_rate: chex.Array  # one action (per region)
    export_limit: chex.Array  # one action (per region)
    import_bids: chex.Array  # num_regions actions (-1(optional)) (per region)
    import_tariff: chex.Array  # num_regions actions (-1(optional)) (per region)


@chex.dataclass
class EnvState:
    current_timestep: int
    activity_timestep: int
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
    mitigation_cost_all_regions: chex.Array
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
    # negotiation_stage: chex.Array
    # minimum_mitigation_rate_all_regions: chex.Array
    # promised_mitigation_rate: chex.Array
    # requested_mitigation_rate: chex.Array
    # proposal_decisions: chex.Array


class Rice(JaxBaseEnv):
    """
    Rice model environment written in JAX.
    Optionally takes in a set of parameters to override the default ones.
    """

    region_params: SimpleNamespace = eqx.field(static=True)

    num_regions: int = 3
    scenario: str = "default"
    diff_reward_mode: bool = False
    relative_reward_mode: bool = True
    # action_type: str = "discrete" # NOTE: continuous not implemented
    num_discrete_action_levels: int = 10
    train_env: bool = (
        False  # if True, the environment will not create extensive "info" dicts on each step
    )
    reduce_action_space_size: bool = (
        False  # removes irrelevant actions from the action space (i.e. tarriff on itself). #NOTE: not sure if this confuses learning, so made this optional
    )

    disable_trading: bool = False # trade actions always 0, actions are not removed from the action space
    negotiation_on: bool = False
    dmg_function: str = "base"
    temperature_calibration: str = "base"
    carbon_model: str = "base"
    apply_welfloss: bool = True
    apply_welfgain: bool = True

    # trade params: were not part of a yaml
    init_capital_multiplier: float = 10.0
    balance_interest_rate: float = 0.1
    consumption_substitution_rate: float = 0.5
    preference_for_domestic: float = 0.5
    baseline_rewards: chex.Array = None

    # default discount factor, variable gamma can be returned from
    # "generate_terminated_truncated_discount" function
    init_gamma: float = 0.99  # discount factor

    @property
    def start_year(self):
        return self.region_params.xt_0

    @property
    def years_per_step(self):
        return self.region_params.xDelta

    @property
    def episode_length(self):
        return self.region_params.xN  # (max steps in episode)

    # if self.negotiation_on:
    #     # NOTE: need to increase number of steps
    #     raise NotImplementedError("Negotiation not implemented yet")

    def __check_init__(self):
        # eqx module function, may use to assert some things
        pass

    def __post_init__(self):
        
        # Baseline rewards:
        key = jax.random.PRNGKey(0)
        default_actions = jnp.zeros((self.num_regions, self.action_nvec.shape[0]))
        default_actions = default_actions.at[:, 0].set(2.5) # savings
        default_actions = default_actions.at[:, 1].set(0.0) # mitigation
        _, state = self.reset_env(key)
        rewards = []
        while True:
            (_, reward, done, _, _), state = self.step_env(key, state, default_actions)
            rewards.append(reward)
            if done:
                break

        object.__setattr__(self, 'baseline_rewards', jnp.array(rewards))

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

        state = EnvState(
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
            mitigation_cost_all_regions=jnp.zeros(self.num_regions),
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
        )

        obs_dict = self.generate_observation_and_action_mask(state)
        return obs_dict, state

    def step_env(
        self,
        key: chex.PRNGKey,
        prev_state: EnvState,
        actions: chex.Array,
    ) -> Tuple[chex.PyTreeDef, EnvState, float, bool, dict]:

        actions = self.process_actions(actions)

        state = replace(
            prev_state,
            activity_timestep=prev_state.activity_timestep + 1,
            current_timestep=prev_state.current_timestep + 1,
        )

        if self.negotiation_on:
            raise NotImplementedError("Negotiation not implemented yet")
            state = self.step_propose()
            state = self.step_evaluate_proposals()

        state = self.step_climate_and_economy(state, actions)

        obs_dict = self.generate_observation_and_action_mask(state)
        reward = self.generate_rewards(
            state, prev_state
        ) # NOTE: rewards is zero for proposel steps
        done, discount = self.generate_terminated_truncated_discount(state)
        info = self.generate_info(state, actions)

        return (obs_dict, reward, done, discount, info), state

    def generate_observation_and_action_mask(self, state: EnvState) -> chex.Array:
        observations = self.generate_observation(state)
        action_masks = self.generate_action_masks(state)
        return {OBSERVATIONS: observations, ACTION_MASK: action_masks}

    def generate_observation(self, state: EnvState) -> chex.Array:
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
            "mitigation_cost_all_regions": state.mitigation_cost_all_regions,
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
        # bilateral_features = []

        if self.negotiation_on:  # TODO
            raise NotImplementedError("Negotiation not implemented yet")
            global_features += ["negotiation_stage"]
            public_features += []
            private_features += [
                "minimum_mitigation_rate_all_regions",
            ]
            bilateral_features += [
                "promised_mitigation_rate",
                "requested_mitigation_rate",
                "proposal_decisions",
            ]

        # Normalization:
        # assert all norm factors are present
        feature_keys = set(global_features.keys()) | set(public_features.keys()) | set(private_features.keys())
        assert feature_keys.issubset(set(NORMALIZATION_FACTORS.keys()))
        norm_factors = {k: v for k, v in NORMALIZATION_FACTORS.items() if k in feature_keys}

        normalized_features = jax.tree.map(
            lambda x, y: x / y,
            {**global_features, **public_features, **private_features},
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

        observations = jnp.concatenate(
            [global_public_features_per_agent, private_features_per_agent], axis=1
        )
        return observations

    def generate_action_masks(self, state: EnvState) -> chex.Array:
        """This function is typically overwritten by a scenario"""
        default_action_mask = jnp.ones(  # allow everything
            (
                self.num_regions,
                self.action_nvec.shape[0],
                self.num_discrete_action_levels,
            ),
            dtype=jnp.bool,
        )
        if self.negotiation_on:
            raise NotImplementedError("Negotiation not implemented yet")
        return default_action_mask

    def generate_rewards(self, new_state: EnvState, old_state: EnvState) -> chex.Array:
        
        reward = new_state.utility_times_welfloss_all_regions

        if self.diff_reward_mode:
            reward = reward - old_state.utility_times_welfloss_all_regions

        # if relative_reward, but no baseline_rewards, then we are building the baseline
        if self.relative_reward_mode and self.baseline_rewards is not None: 
            reward = reward - self.baseline_rewards[old_state.current_timestep]
        
        return reward
        

    def generate_terminated_truncated_discount(
        self, state: EnvState
    ) -> Tuple[bool, bool]:
        """Generate a done flag"""
        terminated = False  # termination only happens due to timesteps
        truncated = state.current_timestep >= self.episode_length
        done = terminated or truncated

        # TODO: variable gamma based on state (per agent)
        discount = jnp.ones((self.num_regions,)) * self.init_gamma
        return done, discount

    def generate_info(self, state: EnvState, actions: Actions) -> dict:
        if self.train_env:
            return {}  # Saving some computation during training
        else:
            info = asdict(state)
            keys = [key for key in info.keys()]
            per_region_keys = [key for key in keys if key.endswith("_all_regions")]
            per_region_keys += ["aggregate_consumption"]
            trade_states = [
                "import_tariffs",
                "normalized_import_bids_all_regions",
                "import_bids_all_regions",
                "imports_minus_tariffs",
            ]
            per_region_keys = set(per_region_keys) - set(trade_states)
            for key in per_region_keys:
                this_key_region_dict = {
                    region_id: info[key][region_id]
                    for region_id in range(info[key].shape[0])
                }
                info[key] = this_key_region_dict
            for key in trade_states:
                this_key_region_dict = {
                    f"from-{region_id}": {
                        f"to-{region_id_2}": info[key][region_id, region_id_2]
                        for region_id_2 in range(info[key].shape[1])
                    }
                    for region_id in range(info[key].shape[0])
                }
                info[key] = this_key_region_dict
            info["global_temperature"] = {
                "atmosphere": info["global_temperature"][0],
                "lower_ocean": info["global_temperature"][1],
            }
            info["global_carbon_mass"] = {
                "atmosphere": info["global_carbon_mass"][0],
                "upper_ocean": info["global_carbon_mass"][1],
                "lower_ocean": info["global_carbon_mass"][2],
            }

            # actions
            info["actions"] = {key: {} for key in actions.__annotations__.keys()}
            for action_key in info["actions"].keys():
                for region_id in range(self.num_regions):
                    info["actions"][action_key][region_id] = actions.__getattribute__(
                        action_key
                    )[region_id]

            return info

    def process_actions(self, actions: chex.Array) -> Actions:
        # actions is currently structured as (num_regions, num_actions)
        actions = actions.T  # (num_actions, num_regions)

        @jax.jit
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

        if not self.negotiation_on:
            savings_rate_actions = actions[0]
            mitigation_rate_actions = actions[1]
            export_limit_actions = actions[2]

            if self.reduce_action_space_size:
                import_bid_actions = actions[3 : 3 + (self.num_regions - 1)].T
                import_tariff_actions = actions[3 + (self.num_regions - 1) :].T
                import_bid_actions = add_diagonal_of_zeros(import_bid_actions)
                import_tariff_actions = add_diagonal_of_zeros(import_tariff_actions)
            else:
                import_bid_actions = actions[3 : 3 + self.num_regions].T
                import_tariff_actions = actions[3 + self.num_regions :].T
                # set the diagonal to 0:
                import_bid_actions = import_bid_actions.at[
                    np.eye(self.num_regions).astype(jnp.bool)
                ].set(0)
                import_tariff_actions = import_tariff_actions.at[
                    np.eye(self.num_regions).astype(jnp.bool)
                ].set(0)
            if self.disable_trading:
                export_limit_actions = jnp.zeros_like(export_limit_actions)
                import_bid_actions = jnp.zeros_like(import_bid_actions)
                import_tariff_actions = jnp.zeros_like(import_tariff_actions)
            return Actions(
                savings_rate=savings_rate_actions / self.num_discrete_action_levels,
                mitigation_rate=mitigation_rate_actions / self.num_discrete_action_levels,
                export_limit=export_limit_actions / self.num_discrete_action_levels,
                import_bids=import_bid_actions / self.num_discrete_action_levels,
                import_tariff=import_tariff_actions / self.num_discrete_action_levels,
            )
        else:
            raise NotImplementedError("Negotiation not implemented yet")

    def step_climate_and_economy(
        self, state: EnvState, actions: Actions
    ) -> Tuple[chex.Array, EnvState]:

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
        global_temperature, global_exogenous_emissions = self.calc_global_temperature(
            state
        )

        current_simulation_year = self.calc_current_simulation_year(state)

        utility_times_welfloss = utilities * welfloss_multipliers

        state: EnvState = replace(
            state,
            # actions
            savings_all_regions=actions.savings_rate,
            import_tariffs=actions.import_tariff,
            export_limit_all_regions=actions.export_limit,
            import_bids_all_regions=actions.import_bids,
            mitigation_rates_all_regions=actions.mitigation_rate,
            # others
            damages_all_regions=damages,
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
            current_simulation_year=current_simulation_year,
            utility_times_welfloss_all_regions=utility_times_welfloss,
        )
        return state

    def step_propose(
        self, state: EnvState, actions: Actions
    ) -> Tuple[chex.Array, EnvState]:
        raise NotImplementedError("Negotiation not implemented yet")

    def step_evaluate_proposals(
        self, state: EnvState, actions: Actions
    ) -> Tuple[chex.Array, EnvState]:
        raise NotImplementedError("Negotiation not implemented yet")

    ### Rice specific functions
    ## Part of step_climate_and_economy()
    ###
    def calc_damages(self, state: EnvState) -> chex.Array:
        prev_atmospheric_temperature = state.global_temperature[0]

        # NOTE: this function returns the (1 - damages) as a percentage of production?

        if self.dmg_function == "base":
            # Isnt this supposedly like in the original one of nordhaus?
            damages = 1 / (
                1
                + self.region_params.xa_1 * prev_atmospheric_temperature
                + self.region_params.xa_2
                * jnp.power(prev_atmospheric_temperature, self.region_params.xa_3)
            )
        elif self.dmg_function == "updated":
            damages = 1 - (0.7438 * (prev_atmospheric_temperature**2)) / 100
            damages = jnp.broadcast_to(damages, (self.num_regions,))
        else:
            raise ValueError(
                f"Unknown damage function: {self.dmg_function.dmg_function}"
            )

        return damages

    def calc_abatement_costs(self, state: EnvState, actions: Actions) -> chex.Array:

        def calc_mitigation_costs():
            mitigation_costs = (
                self.region_params.xp_b
                / (1000 * self.region_params.xtheta_2)
                * jnp.power(
                    1 - self.region_params.xdelta_pb, state.activity_timestep - 1
                )
                * state.intensity_all_regions
            )
            return mitigation_costs

        mitigations_rates_all_agents = actions.mitigation_rate
        mitigation_costs = calc_mitigation_costs()
        abatement_costs = mitigation_costs * jnp.pow(
            mitigations_rates_all_agents, self.region_params.xtheta_2
        )
        # abatement_costs = mitigation_costs * mitigations_rates_all_agents
        # NOTE: the whitepaper multiplies this by production and the savings rate

        return abatement_costs

    def calc_productions(self, state: EnvState) -> chex.Array:
        productions = (
            state.production_factor_all_regions
            * jnp.power(state.capital_all_regions, self.region_params.xgamma)
            * jnp.power(state.labor_all_regions / 1000, 1 - self.region_params.xgamma)
        )
        return productions

    def calc_gross_outputs(
        self, damages: chex.Array, abatement_costs: chex.Array, productions: chex.Array
    ) -> chex.Array:
        gross_outputs = damages * (1 - abatement_costs) * productions
        return gross_outputs

    def calc_investments(
        self, gross_outputs: chex.Array, actions: Actions
    ) -> chex.Array:
        investments = actions.savings_rate * gross_outputs
        return investments

    def calc_gov_balances_post_interest(self, state: EnvState) -> chex.Array:
        gov_balances_post_interest = state.current_balance_all_regions * (
            1 + self.balance_interest_rate
        )
        return gov_balances_post_interest

    def calc_debt_ratios(self, gov_balances_post_interest: chex.Array) -> chex.Array:
        gov_balances = gov_balances_post_interest
        debt_ratios = (
            gov_balances * self.init_capital_multiplier / self.region_params.xK_0
        )
        # We scale the debt ratios by factor 10 and then clip it?

        debt_ratios = jnp.clip(debt_ratios, -1.0, 0.0)  # NOTE does this make sense?
        return debt_ratios

    def calc_gross_imports(
        self,
        state: EnvState,
        actions: Actions,
        gross_outputs: chex.Array,
        investments: chex.Array,
        debt_ratios: chex.Array,
    ) -> chex.Array:

        def calc_normalized_import_bids(potential_import_bids):
            normalized_import_bids_all_regions = jnp.zeros(
                (self.num_regions, self.num_regions)
            )

            max_export_rate = actions.export_limit

            def calc_max_exports():
                return jnp.where(
                    max_export_rate * gross_outputs <= gross_outputs - investments,
                    max_export_rate * gross_outputs,
                    gross_outputs - investments,
                )

            max_export_all_regions = calc_max_exports()
            desired_exports_from_each_region = jnp.sum(potential_import_bids, axis=0)
            # NOTE: this is the original. But it seems like region export is set to 0
            # if max_export > desired_export. https://github.com/mila-iqia/climate-cooperation-competition/issues/46
            # return jnp.where(
            #     desired_exports_from_each_region > max_export_all_regions,
            #     potential_import_bids / desired_exports_from_each_region * max_export_all_regions,
            #     normalized_import_bids_all_regions,
            # )

            # FIX?
            return jnp.where(
                desired_exports_from_each_region > max_export_all_regions,
                potential_import_bids
                / desired_exports_from_each_region
                * max_export_all_regions,
                potential_import_bids,
            )

        import_bids_all_regions = actions.import_bids

        potential_import_bids = jnp.zeros((self.num_regions, self.num_regions))

        # NOTE: original contains some writeable bugfix and empties the bid to itself
        ## We instead deal with this in the process_actions() function
        total_import_bids = jnp.sum(import_bids_all_regions, axis=1)
        potential_import_bids = jnp.where(
            total_import_bids * gross_outputs > gross_outputs,
            import_bids_all_regions / total_import_bids * gross_outputs,
            import_bids_all_regions * gross_outputs,
        )
        potential_import_bids *= 1 + debt_ratios

        normalized_import_bids_all_regions = calc_normalized_import_bids(
            potential_import_bids
        )
        return normalized_import_bids_all_regions

    def calc_trade_sanctions(
        self, state: EnvState, gross_imports: chex.Array, actions: Actions
    ) -> Tuple[chex.Array, chex.Array]:
        # NOTE: Original used: self.get_prev_state("import_tariffs_all_regions")
        # this delays the action one step? here, this is changed to current action
        net_imports = gross_imports * (
            1 - actions.import_tariff
        )  # NOTE: replaced from prev_state to curr_state
        tariff_revenues = (
            gross_imports * actions.import_tariff
        )  # NOTE: replaced from prev_state to curr_state
        return tariff_revenues, net_imports

    def calc_welfloss_multiplier(
        self,
        state: EnvState,
        gross_outputs: chex.Array,
        gross_imports: chex.Array,
        net_imports: chex.Array,
        welfare_loss_per_unit_tariff: float = None,
        welfare_gain_per_unit_exported=None,
    ) -> chex.Array:
        if not self.apply_welfloss:
            return np.ones((self.num_regions))

        if welfare_loss_per_unit_tariff is None:
            welfare_loss_per_unit_tariff = 0.4  # From Nordhaus 2015
        if welfare_gain_per_unit_exported is None:
            welfare_gain_per_unit_exported = 0.4

        welfloss = jnp.ones((self.num_regions)) - (
            (gross_imports.sum(axis=0) / gross_outputs)
            * state.import_tariffs.sum(axis=0)  # TODO: again, original used prev_state
            * welfare_loss_per_unit_tariff
        )
        if self.apply_welfgain:
            welfloss += (
                net_imports.sum(axis=0) / gross_outputs * welfare_gain_per_unit_exported
            )
        return welfloss

    def calc_consumptions(
        self,
        gross_outputs: chex.Array,
        investments: chex.Array,
        gross_imports: chex.Array,
        net_imports: chex.Array,
    ) -> chex.Array:
        total_exports = gross_imports.sum(axis=0)

        domestic_consumption = jnp.maximum(  # Consumption cannot be negative
            gross_outputs - investments - total_exports, 0
        )

        c_dom_pref = self.preference_for_domestic * (
            domestic_consumption**self.consumption_substitution_rate
        )
        preference_for_imported = np.array(
            [  # Remains fixed throughout the run; so np.
                (1 - self.preference_for_domestic) / (self.num_regions - 1)
            ]
            * self.num_regions
        )

        c_for_pref = jnp.sum(
            preference_for_imported
            * jnp.pow(net_imports.sum(axis=1), self.consumption_substitution_rate)
        )

        consumptions = (c_dom_pref + c_for_pref) ** (
            1 / self.consumption_substitution_rate
        )  # CES function

        return consumptions

    def calc_utilities(self, state: EnvState, consumptions: chex.Array) -> chex.Array:
        scaled_labor_all_regions = state.labor_all_regions / 1000.0
        utilities = (
            scaled_labor_all_regions
            * (
                jnp.power(
                    consumptions / scaled_labor_all_regions + 1e-0,
                    1 - self.region_params.xalpha,
                )
                - 1
            )
            / (1 - self.region_params.xalpha)
        )
        return utilities

    def calc_social_welfares(
        self, state: EnvState, utilities: chex.Array
    ) -> chex.Array:
        social_welfares = utilities / (
            jnp.power(
                1 + self.region_params.xrho,
                self.region_params.xDelta * state.activity_timestep,
            )
        )
        return social_welfares

    def calc_capitals(self, state: EnvState, investments: chex.Array) -> chex.Array:
        capital_depreciation = jnp.power(
            1 - self.region_params.xdelta_K, self.region_params.xDelta
        )
        capitals = capital_depreciation * state.capital_all_regions + (
            self.region_params.xDelta * investments
        )
        return capitals

    def calc_labors(self, state: EnvState) -> chex.Array:
        labors = state.labor_all_regions * jnp.power(
            (1 + self.region_params.xL_a) / (1 + state.labor_all_regions),
            self.region_params.xl_g,
        )
        return labors

    def calc_production_factors(self, state: EnvState) -> chex.Array:
        production_factors = state.production_factor_all_regions * (
            jnp.exp(0.0033)
            + self.region_params.xg_A
            * jnp.exp(
                -self.region_params.xdelta_A
                * self.region_params.xDelta
                * (state.activity_timestep - 1)
            )
        )
        return production_factors

    def calc_gov_balances_post_trade(
        self, gov_balances_post_interest: chex.Array, gross_imports: chex.Array
    ) -> chex.Array:
        trade_balance = self.region_params.xDelta * (
            jnp.sum(gross_imports, axis=0) - jnp.sum(gross_imports, axis=1)
        )
        gov_balances_post_trade = gov_balances_post_interest + trade_balance
        return gov_balances_post_trade

    def calc_carbon_intensities(self, state: EnvState) -> chex.Array:
        carbon_intensity = state.intensity_all_regions * jnp.exp(
            -self.region_params.xg_sigma
            * jnp.power(
                1 - self.region_params.xdelta_sigma,
                self.region_params.xDelta * (state.activity_timestep - 1),
            )
            * self.region_params.xDelta
        )
        return carbon_intensity

    def calc_global_carbon_mass(
        self, state: EnvState, productions: chex.Array, mitigation_rates: chex.Array
    ) -> chex.Array:

        def calc_land_emissions():
            """Obtain the amount of land emissions."""
            e_l0 = self.region_params.xE_L0
            delta_el = self.region_params.xdelta_EL

            global_land_emissions = (
                e_l0
                * jnp.power(1 - delta_el, state.activity_timestep - 1)
                / self.num_regions
            )
            return global_land_emissions

        if self.carbon_model == "base":
            global_land_emissions = calc_land_emissions()
            # (original) TODO: fix aux_m treatment
            aux_m_all_regions = (
                state.intensity_all_regions * (1 - mitigation_rates) * productions
                + global_land_emissions
            )

            """Get the carbon mass level."""
            sum_aux_m = np.sum(aux_m_all_regions)
            prev_global_carbon_mass = state.global_carbon_mass
            global_carbon_mass = jnp.dot(
                jnp.asarray(self.region_params.xPhi_M), prev_global_carbon_mass
            ) + jnp.dot(jnp.asarray(self.region_params.xB_M), sum_aux_m)

        elif self.carbon_model in ["FaIR", "AR5", "DFaIR"]:
            raise NotImplementedError(
                f"Carbon model {self.carbon_model} not implemented in jax yet."
            )
            prev_global_land_emissions = state.global_land_emissions
            prev_global_emissions = state.global_emissions
            prev_global_carbon_reservoirs = state.global_carbon_reservoirs
            prev_global_cumulative_emissions = state.global_cumulative_emissions
            prev_global_cumulative_land_emissions = (
                state.global_cumulative_land_emissions
            )
            prev_global_temperature = state.global_temperature
            prev_global_acc_pert_carb_stock = state.global_acc_pert_carb_stock

            a = np.array(
                [
                    self.region_params.xM_a0,
                    self.region_params.xM_a1,
                    self.region_params.xM_a2,
                    self.region_params.xM_a3,
                ]
            )
            tau = np.array(
                [
                    self.region_params.xM_t0,
                    self.region_params.xM_t1,
                    self.region_params.xM_t2,
                    self.region_params.xM_t3,
                ]
            )
            C0 = self.region_params.xM_AT_1750

            irf0, irC, irT = (
                self.all_regions_params[0]["irf0"],
                self.all_regions_params[0]["irC"],
                self.all_regions_params[0]["irT"],
            )

            # DAE determines given concentrations and temperature how much the reservoirs can absorb
            if self.carbon_model in ["FaIR", "DFaIR"]:
                raise NotImplementedError(
                    "The newton function is not implemented yet. (in jax); hence FaIR carbon model is not implemented yet."
                )
                prev_global_alpha = state.global_alpha

                def DAE_(oneoveralpha):
                    b = a * tau * (1 - np.exp(-100 * oneoveralpha / tau))
                    return np.sum(b) - oneoveralpha * (
                        irf0
                        + irC * prev_global_acc_pert_carb_stock
                        + irT * prev_global_temperature[0]
                    )

                global_alpha = 1 / newton(DAE_, x0=1 / prev_global_alpha)
                assert np.isclose(
                    0, DAE_(1 / global_alpha), rtol=1e-2
                ), f"DAE not solved correctly."
                assert (
                    0.01 <= global_alpha <= 100
                ), f"Value out of bounds: {global_alpha} is not within [0.01, 100]"

            elif self.carbon_model == "AR5":
                global_alpha = 1

            if save_state:
                self.set_state("global_alpha", global_alpha)

            # conversion 5/3.67 = 1.36388
            conv = self.region_params.xB_M
            global_land_emissions = calc_land_emissions()
            # TODO: fix aux_m treatment
            aux_m_all_regions = (
                state.intensity_all_regions * (1 - mitigation_rates) * productions
                + global_land_emissions
            )  # NOTE: aux_m_all_regions was saved to state, but never used outside this function # Maybe logging?

            """Get the carbon mass level."""
            sum_aux_m = jnp.sum(aux_m_all_regions)
            # In case, we want to prescribe the emissions to investigate the behavior of the temperature and carbon model
            # if self.prescribed_emissions is not None:
            #     sum_aux_m = self.prescribed_emissions[self.activity_timestep]
            if save_state:
                self.set_state("global_emissions", np.sum(aux_m_all_regions))

            global_carbon_reservoirs = np.zeros(4)
            global_cumulative_emissions = (
                prev_global_cumulative_emissions
                + (prev_global_emissions - prev_global_land_emissions) * conv
            )

            if save_state:
                self.set_state(
                    "global_cumulative_emissions", global_cumulative_emissions
                )

            global_cumulative_land_emissions = (
                prev_global_cumulative_land_emissions
                + prev_global_land_emissions * self.num_regions * conv
            )
            if save_state:
                self.set_state(
                    "global_cumulative_land_emissions", global_cumulative_land_emissions
                )

            if self.carbon_model in ["AR5", "FaIR"]:
                global_carbon_reservoirs = prev_global_carbon_reservoirs ** np.exp(
                    -5 / (global_alpha * tau)
                ) + a * sum_aux_m / 5 * conv * (
                    np.exp(-1 / (global_alpha * tau))
                    - np.exp(-6 / (global_alpha * tau))
                ) / (
                    1 - np.exp(-1 / (global_alpha * tau))
                )
            elif self.carbon_model == "DFaIR":
                global_carbon_reservoirs = prev_global_carbon_reservoirs * np.exp(
                    -5 / (tau * global_alpha)
                ) + a * sum_aux_m / 5 * conv * tau * global_alpha * (
                    1 - np.exp(-5 / (global_alpha * tau))
                )
            if save_state:
                self.set_state("global_carbon_reservoirs", global_carbon_reservoirs)

            global_acc_pert_carb_stock = (
                global_cumulative_emissions + global_cumulative_land_emissions
            ) - jnp.sum(global_carbon_reservoirs)
            if save_state:
                self.set_state("global_acc_pert_carb_stock", global_acc_pert_carb_stock)

            global_carbon_mass = C0 + sum((global_carbon_reservoirs))
        else:
            raise NotImplementedError(
                f"Carbon model {self.carbon_model} not implemented."
            )

        return global_carbon_mass

    def calc_global_temperature(self, state: EnvState) -> chex.Array:

        def calc_exogenous_emissions():
            """Obtain the amount of exogeneous emissions."""
            f_0 = self.region_params.xf_0
            f_1 = self.region_params.xf_1
            t_f = self.region_params.xt_f

            exogenous_emissions = f_0 + jnp.minimum(
                f_1 - f_0, (f_1 - f_0) / t_f * (state.activity_timestep - 1)
            )
            return exogenous_emissions

        if self.temperature_calibration == "base":
            global_exogenous_emissions = (
                calc_exogenous_emissions()
            )  # also exogenous forcings
            prev_carbon_mass = state.global_carbon_mass
            prev_global_temperature = state.global_temperature
            # (original) TODO: why the zero index?
            # (original) global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = prev_carbon_mass[0]
            phi_t = jnp.asarray(self.region_params.xPhi_T)
            b_t = jnp.asarray(self.region_params.xB_T)
            f_2x = jnp.asarray(self.region_params.xF_2x)
            atmospheric_carbon_mass = jnp.asarray(self.region_params.xM_AT_1750)

            global_temperature = jnp.dot(phi_t, prev_global_temperature) + jnp.dot(
                b_t,
                f_2x
                * jnp.log(prev_atmospheric_carbon_mass / atmospheric_carbon_mass)
                / jnp.log(2)
                + global_exogenous_emissions,
            )

            return global_temperature, global_exogenous_emissions

        elif self.temperature_calibration == "FaIR":
            global_exogenous_emissions = calc_exogenous_emissions()
            prev_carbon_mass = state.global_carbon_mass
            prev_global_temperature = state.global_temperature
            # (original) TODO: why the zero index?
            # (original) global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = prev_carbon_mass[0]
            atmospheric_carbon_mass = np.array(self.region_params.xM_AT_1750)

            t_2x = self.region_params.xT_2x
            f_2x = self.region_params.xF_2x

            xT_1 = self.region_params.xT_1
            xT_2 = f_2x / t_2x
            xT_3 = self.region_params.xT_3
            xT_4 = self.region_params.xT_4

            forcings = (
                f_2x
                * jnp.log(prev_atmospheric_carbon_mass / atmospheric_carbon_mass)
                / jnp.log(2)
                + global_exogenous_emissions
            )

            # update global atmospheric temperature in 4 smaller steps
            global_temperature_short = prev_global_temperature[0]
            for _ in range(4):  # TODO: this might be doable in one go?
                global_temperature_short = global_temperature_short + 1 / xT_1 * (
                    (forcings - xT_2 * global_temperature_short)
                    - xT_3 * (global_temperature_short - prev_global_temperature[1])
                )
            global_temperature = jnp.array(
                [
                    global_temperature_short,
                    prev_global_temperature[1]
                    + 5
                    * xT_3
                    / xT_4
                    * (prev_global_temperature[0] - prev_global_temperature[1]),
                ]
            )

            return global_temperature, global_exogenous_emissions

        elif self.temperature_calibration == "DFaIR":
            raise NotImplementedError(
                f"Temperature calibration {self.temperature_calibration} not implemented for jax yet."
            )
            global_exogenous_emissions = self.calc_exogenous_emissions()
            prev_carbon_mass = self.get_prev_state("global_carbon_mass")
            prev_global_temperature = self.get_prev_state("global_temperature")
            prev_global_temperature_boxes = self.get_prev_state(
                "global_temperature_boxes"
            )

            # TODO: why the zero index?
            # global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = prev_carbon_mass[0]
            atmospheric_carbon_mass = np.array(
                self.all_regions_params[0]["xM_AT_1750"]
            )  # Equilibrium atmospheric carbon mass
            f_2x = np.array(self.all_regions_params[0]["xF_2x"])

            d = np.array(
                [
                    self.all_regions_params[0]["xT_LO_rt"],
                    self.all_regions_params[0]["xT_UO_rt"],
                ]
            )
            teq = np.array(
                [
                    self.all_regions_params[0]["xT_LO_tq"],
                    self.all_regions_params[0]["xT_UO_tq"],
                ]
            )

            forcings = (
                f_2x
                * np.log(prev_atmospheric_carbon_mass / atmospheric_carbon_mass)
                / np.log(2)
                + global_exogenous_emissions
            )

            global_temperature_boxes = prev_global_temperature_boxes * np.exp(
                -5 / d
            ) + teq * forcings * (1 - np.exp(-5 / d))

            if save_state:
                self.set_state("global_temperature_boxes", global_temperature_boxes)

            global_temperature = np.array([np.sum(global_temperature_boxes), 0])

            if save_state:
                self.set_state("global_temperature", global_temperature)

            return global_temperature

        else:
            raise ValueError(
                f"Unknown temperature calibration: {self.temperature_calibration}"
            )

    def calc_current_simulation_year(self, state: EnvState) -> chex.Array:
        return state.current_simulation_year + self.region_params.xDelta

    ###
    ## Helper and environment functions
    ###
    @property
    def action_nvec(self) -> chex.Array:
        if not self.negotiation_on:
            num_actions = len(Actions.__annotations__)
            num_regions = self.num_regions
            import_bids_nvec = [self.num_discrete_action_levels] * (
                num_regions - self.reduce_action_space_size
            )
            import_tariff_nvec = [self.num_discrete_action_levels] * (
                num_regions - self.reduce_action_space_size
            )
            actions_nvec = np.concatenate(
                [
                    [self.num_discrete_action_levels],  # savings_rate
                    [self.num_discrete_action_levels],  # mitigation_rate
                    [self.num_discrete_action_levels],  # export_limit
                    import_bids_nvec,
                    import_tariff_nvec,
                ]
            )
            return actions_nvec
        else:
            raise NotImplementedError("Negotiation not implemented yet")

    @property
    def action_space(self) -> MultiDiscrete:
        return MultiDiscrete(self.action_nvec)

    def observation_space(self) -> Box:
        obs_dict, _ = self.reset(jax.random.PRNGKey(0))
        obs = obs_dict[OBSERVATIONS]
        return Box(-9999, 9999, shape=obs.shape, dtype=obs.dtype)
