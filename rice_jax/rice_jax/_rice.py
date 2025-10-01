from types import SimpleNamespace
from typing import Any, Callable, Literal, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jaxnasium as jym
import numpy as np
import optax
from jaxnasium import Discrete, MultiDiscrete

from .utils import empty_info_log_fn, i_to_agent_str, solve_for_alpha


class Rice(jym.Environment):
    """
    Rice model environment written in JAX.
    Optionally takes in a set of parameters to override the default ones.
    """

    region_params: SimpleNamespace = eqx.field(static=True)

    num_regions: Literal[3, 7, 20] = 3
    diff_reward_mode: bool = True
    relative_reward_mode: bool = False
    num_discrete_action_levels: int = 10
    action_window_size: int = 0  # 0 = No action windows
    disable_trading: bool = False
    negotiation_on: bool = False

    dmg_function: Literal["base", "updated"] = "base"
    temperature_calibration: Literal["base", "FaIR", "DFaIR"] = "base"
    carbon_model: Literal["base", "FaIR", "DFaIR", "AR5"] = "base"
    apply_welfloss: bool = True
    apply_welfgain: bool = True

    # trade params: were not part of a yaml
    init_capital_multiplier: float = 10.0
    balance_interest_rate: float = 0.1
    consumption_substitution_rate: float = 0.5
    preference_for_domestic: float = 0.5

    baseline_rewards: chex.Array = None
    log_info_fn: Callable[..., dict] = empty_info_log_fn

    # default discount factor, variable gamma can be returned from
    # "generate_terminated_truncated_discount" function
    init_gamma: float = 0.99  # discount factor
    _multi_agent: bool = True  # Must set to True

    @property
    def start_year(self):
        return self.region_params.xt_0

    @property
    def years_per_step(self):
        return self.region_params.xDelta

    @property
    def episode_length(self):
        simulation_timesteps = self.region_params.xN
        if self.negotiation_on:
            # 2 extra steps for negotiation
            simulation_timesteps = simulation_timesteps * 3
        return simulation_timesteps

    def __check_init__(self):
        # eqx module function, may use to assert some things
        pass

    def __post_init__(self):
        # Baseline rewards:
        if not self.relative_reward_mode:
            return
        key = jax.random.PRNGKey(0)

        default_actions = self.sample_action(key)
        default_actions = optax.tree.zeros_like(default_actions)
        for agent_id in range(self.num_regions):
            agent = i_to_agent_str(agent_id)
            default_actions[agent]["savings_rate"] = 2.5
            default_actions[agent]["mitigation_rate"] = 0.0

        # Get baseline rewards with default actions
        _, state = self.reset_env(key)
        rewards = []
        while True and self.relative_reward_mode:
            (_, reward, terminated, truncated, _), state = self.step_env(
                key, state, default_actions
            )
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break

        object.__setattr__(self, "baseline_rewards", jnp.array(rewards))

    def reset_env(self, key: chex.PRNGKey) -> Tuple[dict, dict]:
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
            "promised_mitigation_rate": jnp.zeros((self.num_regions, self.num_regions)),
            "requested_mitigation_rate": jnp.zeros((self.num_regions, self.num_regions)),
            "proposal_decisions": jnp.zeros((self.num_regions, self.num_regions), dtype=jnp.bool),
        }
        # fmt: on

        obs_dict = self.generate_observation_and_action_mask(state)
        return obs_dict, state

    def step_env(
        self, key: chex.PRNGKey, prev_state: dict, actions: dict
    ) -> Tuple[jym.TimeStep, dict]:
        state = prev_state.copy()

        pre_processed_actions = actions  # For easier logging
        state["current_timestep"] = prev_state["current_timestep"] + 1
        actions = self.process_actions(actions, state)
        if not self.negotiation_on:
            state = self.step_climate_and_economy(state, actions)
        else:
            negotiation_stage = state["current_timestep"] % 3
            state = jax.lax.switch(
                negotiation_stage,
                [
                    lambda: self.step_climate_and_economy(state, actions),
                    lambda: self.step_propose(state, actions),
                    lambda: self.step_evaluate_proposals(state, actions),
                ],
            )

        obs_dict = self.generate_observation_and_action_mask(state)
        reward = self.generate_rewards(state, prev_state)  # proposal step rewards = 0
        terminated, truncated = self.generate_terminated_truncated(state)
        info = self.generate_info(state, pre_processed_actions)
        # info["ENV_GAMMA"] = self.generate_discount(state)

        return (obs_dict, reward, terminated, truncated, info), state

    def generate_observation_and_action_mask(
        self, state: dict[str, Any]
    ) -> dict[str, jym.AgentObservation]:
        observations = self.generate_observation(state)

        # Make sure everything is 1d and flat
        # because accidental 2d observations will be trained via a CNN
        observations = jax.tree.map(
            lambda o: jnp.reshape(jnp.atleast_1d(o), (-1)), observations
        )

        # Concatenate each agent's observations into a single array
        # map_one_level essentially applies a function to each agent('s observation in this case)
        # NOTE: this isn't strictly neccesary, as the algorithm can also deal with this; but it cleans up the network.
        observations = jym.tree.map_one_level(
            lambda o: jnp.concatenate(jax.tree.leaves(o)), observations
        )

        action_masks = self.generate_action_masks(state)

        # Return as a AgentObservation such that algorithms properly deal with the action masks
        return jym.tree.map_one_level(
            lambda o, m: jym.AgentObservation(observation=o, action_mask=m),
            observations,
            action_masks,
        )

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
            "mitigation_rates_all_regions"
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
                "promised_mitigation_rate",
                "requested_mitigation_rate",
                "proposal_decisions",
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

        # NOTE: We skip normalization, as it can be done via a wrapper or in the algorithm itself
        # by keeping track of running statistics

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

        # Allow each action as a base
        mask = jax.tree.map(allow_all_actions_in_action_space, self.action_space)

        # Disallow actions on own region "self"
        for a_id in range(NUM_REGIONS):
            agent_str = i_to_agent_str(a_id)
            # Set diagonal elements to 0 for import actions (except first element)
            mask[agent_str]["import_bid"][a_id][1:] = 0
            mask[agent_str]["import_tariff"][a_id][1:] = 0
            if self.negotiation_on:
                mask[agent_str]["proposal_ask"][a_id][1:] = 0
                mask[agent_str]["proposal_promise"][a_id][1:] = 0

        # Minimum mitigation rate masking
        minimum_mitigation_rate_all = state["minimum_mitigation_rate_all_regions"]
        for agent_id in range(self.num_regions):
            min_mitigation_rate_agent = minimum_mitigation_rate_all[agent_id]
            mask[i_to_agent_str(agent_id)]["mitigation_rate"] = (
                jnp.arange(self.num_discrete_action_levels) >= min_mitigation_rate_agent
            )

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
                _savings_mask = create_windowed_mask(prev_savings_actions[agent_id])
                mask[i_to_agent_str(agent_id)]["savings_rate"] = (
                    mask[i_to_agent_str(agent_id)]["savings_rate"] * _savings_mask
                )  # Multiply with existing mask to not overwrite

                _mitigation_mask = create_windowed_mask(
                    prev_mitigation_actions[agent_id]
                )
                mask[i_to_agent_str(agent_id)]["mitigation_rate"] = (
                    mask[i_to_agent_str(agent_id)]["mitigation_rate"] * _mitigation_mask
                )  # Multiply with existing mask to not overwrite

        return mask

    def generate_rewards(self, new_state: dict, old_state: dict) -> dict[str, float]:
        reward = new_state["utility_times_welfloss_all_regions"]

        if self.diff_reward_mode:
            reward = reward - old_state["utility_times_welfloss_all_regions"]

        # if relative_reward, but no baseline_rewards, then we are building the baseline
        if self.relative_reward_mode and self.baseline_rewards is not None:
            reward = reward - self.baseline_rewards[old_state["current_timestep"]]

        return {i_to_agent_str(i): reward[i] for i in range(self.num_regions)}

    def generate_terminated_truncated(self, state: dict) -> Tuple[bool, bool]:
        """Generate a terminated and truncated flag"""
        terminated = False  # never terminate, only truncate (stop due to time limit)
        truncated = state["current_timestep"] >= self.episode_length

        return terminated, truncated

    # def generate_discount(self, state: dict) -> dict[str, float]:
    #     """Generate a discount factor. Make sure the algorithm uses this"""
    #     # Currently we just use a static discount factor raised to the number of years per step
    #     discount = self.init_gamma
    #     discount = discount**self.years_per_step
    #     return {i_to_agent_str(i): discount for i in range(self.num_regions)}

    def generate_info(self, state, actions) -> dict:
        return self.log_info_fn(state, actions)

    def process_actions(self, actions: dict[str, Any], state: dict[str, chex.Array]):
        """ """

        # First: actions arrive as {agent1: {action1: value, action2: value, ...}, agent2: ...}
        # We tranpose this to {action1: {agent1: value, agent2: value, ...}, action2: ...}
        # NOTE: this is only possible like this because all agents output the same actions
        all_actions, agent_structure = eqx.tree_flatten_one_level(actions)
        action_structure = jax.tree.structure(all_actions[0])
        actions = jax.tree.transpose(agent_structure, action_structure, actions)

        # NOTE: This is a good place to enforce invalid actions (min_mitigation_rate, action windows, etc.)
        # But this is enforced in the action mask as well, so we leave it out here for now.

        if self.disable_trading:
            actions["export_limit"] = optax.tree.zeros_like(actions["export_limit"])
            actions["import_bid"] = optax.tree.zeros_like(actions["import_bid"])
            actions["import_tariff"] = optax.tree.zeros_like(actions["import_tariff"])

        # Div each action by the number of discrete action levels
        # actions["proposal_decision"] is just 0 / 1, so gets special treatment here
        if "proposal_decision" in actions:
            _proposal_decisions = actions["proposal_decision"].copy()
            actions = jax.tree.map(
                lambda x: x / self.num_discrete_action_levels, actions
            )
            actions["proposal_decision"] = _proposal_decisions
        else:
            actions = jax.tree.map(
                lambda x: x / self.num_discrete_action_levels, actions
            )

        # Subsequently, we convert each action to a array:
        # {action_name:{agent1: action_value, agent2: action_value, ...}, ...} --> {action_name: Array([action_value, action_value, ...])}
        # for easier vectorized operations down the line
        actions = jym.tree.map_one_level(jym.tree.stack, actions)

        return actions

    def step_climate_and_economy(self, state: dict[str, Any], actions: dict[str, Any]):
        damages = self.calc_damages(state)
        abatement_costs = self.calc_abatement_costs(state, actions)
        productions = self.calc_productions(state)
        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(gross_outputs, actions)
        gov_balances_post_interest = self.calc_gov_balances_post_interest(state)
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)
        gross_imports = self.calc_gross_imports(
            state, actions, gross_outputs, investments, debt_ratios
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

        global_carbon_mass, carbon_updates = self.calc_global_carbon_mass(
            state, productions, actions["mitigation_rate"]
        )
        # TODO: calc_global_temperature should already have the new global_carbon_mass (nameing: prev_global_carbon_mass is also misleading in the function)
        global_temperature, global_exogenous_emissions, global_temperature_boxes = (
            self.calc_global_temperature(state, global_carbon_mass)
        )

        current_simulation_year = self.calc_current_simulation_year(state)

        utility_times_welfloss = utilities * welfloss_multipliers

        state = state.copy()
        state.update(
            {
                "activity_timestep": state["activity_timestep"] + 1,
                # actions
                "savings_all_regions": actions["savings_rate"],
                "mitigation_rates_all_regions": actions["mitigation_rate"],
                "export_limit_all_regions": actions["export_limit"],
                "import_bids_all_regions": actions["import_bid"],
                "import_tariffs_all_regions": actions["import_tariff"],
                # others
                "damages_all_regions": damages,
                "aggregate_consumption": consumptions,
                "abatement_cost_all_regions": abatement_costs,
                "production_all_regions": productions,
                "gross_output_all_regions": gross_outputs,
                "investment_all_regions": investments,
                "current_balance_all_regions": gov_balances_post_trade,
                "imports_minus_tariffs": net_imports,
                "utility_all_regions": utilities,
                # "social_welfare_all_regions": social_welfare,
                "labor_all_regions": labors,
                "capital_all_regions": capitals,
                "production_factor_all_regions": production_factors,
                "intensity_all_regions": carbon_intensities,
                "global_carbon_mass": global_carbon_mass,
                "global_temperature": global_temperature,
                "global_exogenous_emissions": global_exogenous_emissions,
                "global_temperature_boxes": global_temperature_boxes,
                "current_simulation_year": current_simulation_year,
                "utility_times_welfloss_all_regions": utility_times_welfloss,
            }
        )
        # carbon updates
        state.update(carbon_updates)

        return state

    def step_propose(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")
        promised_mitigation_rate = actions["proposal_promise"]
        requested_mitigation_rate = actions["proposal_ask"]

        state["promised_mitigation_rate"] = promised_mitigation_rate
        state["requested_mitigation_rate"] = requested_mitigation_rate
        return state

    def step_evaluate_proposals(self, state: dict, actions: dict) -> dict:
        if not self.negotiation_on:
            raise ValueError("Negotiation is not enabled")

        promised_mitigation_rates = state["promised_mitigation_rate"]
        requested_mitigation_rates = state["requested_mitigation_rate"]
        proposal_decisions = actions["proposal_decision"].T

        outgoing_accepted_mitigation_rates = (
            promised_mitigation_rates * proposal_decisions
        )
        incoming_accepted_mitigation_rates = (
            requested_mitigation_rates * proposal_decisions
        )
        # NOTE: The original Rice-N adds the two arrays?
        combined_max_accepted_mitigation_rates = jnp.maximum(
            outgoing_accepted_mitigation_rates, incoming_accepted_mitigation_rates.T
        )
        lower_bound_mitigation_rates = jnp.max(
            combined_max_accepted_mitigation_rates, axis=1
        )

        state["proposal_decisions"] = proposal_decisions.astype(jnp.bool)
        state["minimum_mitigation_rate_all_regions"] = lower_bound_mitigation_rates
        return state

    ### Rice specific functions
    ## Part of step_climate_and_economy()
    ###
    def calc_damages(self, state: dict) -> chex.Array:
        prev_atmospheric_temperature = state["global_temperature"][0]

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
            raise ValueError(f"Unknown damage function: {self.dmg_function}")

        return damages

    def calc_abatement_costs(self, state: dict, actions: dict) -> chex.Array:
        def calc_mitigation_costs():
            mitigation_costs = (
                self.region_params.xp_b
                / (1000 * self.region_params.xtheta_2)
                * jnp.power(
                    1 - self.region_params.xdelta_pb, state["activity_timestep"] - 1
                )
                * state["intensity_all_regions"]
            )
            return mitigation_costs

        mitigations_rates_all_agents = actions["mitigation_rate"]
        mitigation_costs = calc_mitigation_costs()
        abatement_costs = mitigation_costs * jnp.pow(
            mitigations_rates_all_agents, self.region_params.xtheta_2
        )
        # abatement_costs = mitigation_costs * mitigations_rates_all_agents
        # NOTE: the whitepaper multiplies this by production and the savings rate

        return abatement_costs

    def calc_productions(self, state: dict) -> chex.Array:
        productions = (
            state["production_factor_all_regions"]
            * jnp.power(state["capital_all_regions"], self.region_params.xgamma)
            * jnp.power(
                state["labor_all_regions"] / 1000, 1 - self.region_params.xgamma
            )
        )
        return productions

    def calc_gross_outputs(
        self, damages: chex.Array, abatement_costs: chex.Array, productions: chex.Array
    ) -> chex.Array:
        gross_outputs = damages * (1 - abatement_costs) * productions
        return gross_outputs

    def calc_investments(self, gross_outputs: chex.Array, actions: dict) -> chex.Array:
        investments = actions["savings_rate"] * gross_outputs
        return investments

    def calc_gov_balances_post_interest(self, state: dict) -> chex.Array:
        gov_balances_post_interest = state["current_balance_all_regions"] * (
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
        state: dict,
        actions: dict,
        gross_outputs: chex.Array,
        investments: chex.Array,
        debt_ratios: chex.Array,
    ) -> chex.Array:
        def calc_normalized_import_bids(potential_import_bids):
            normalized_import_bids_all_regions = jnp.zeros(
                (self.num_regions, self.num_regions)
            )

            max_export_rate = actions["export_limit"]

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

        import_bids_all_regions = actions["import_bid"]

        potential_import_bids = jnp.zeros((self.num_regions, self.num_regions))

        # NOTE: original contains some writeable bugfix and empties the bid to itself
        ## We instead deal with this in the action masking / process actions  function
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
        self, state: dict, gross_imports: chex.Array, actions: dict
    ) -> Tuple[chex.Array, chex.Array]:
        # NOTE: Original used: self.get_prev_state("import_tariffs_all_regions")
        # this delays the action one step? here, this is changed to current step (current action)
        net_imports = gross_imports * (1 - actions["import_tariff"])
        tariff_revenues = gross_imports * actions["import_tariff"]
        return tariff_revenues, net_imports

    def calc_welfloss_multiplier(
        self,
        state: dict,
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
            * state["import_tariffs"].sum(
                axis=0
            )  # TODO: again, original used prev_state, here state is used
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

    def calc_utilities(self, state: dict, consumptions: chex.Array) -> chex.Array:
        scaled_labor_all_regions = state["labor_all_regions"] / 1000.0
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

    def calc_social_welfares(self, state: dict, utilities: chex.Array) -> chex.Array:
        social_welfares = utilities / (
            jnp.power(
                1 + self.region_params.xrho,
                self.region_params.xDelta * state["activity_timestep"],
            )
        )
        return social_welfares

    def calc_capitals(self, state: dict, investments: chex.Array) -> chex.Array:
        capital_depreciation = jnp.power(
            1 - self.region_params.xdelta_K, self.region_params.xDelta
        )
        capitals = capital_depreciation * state["capital_all_regions"] + (
            self.region_params.xDelta * investments
        )
        return capitals

    def calc_labors(self, state: dict) -> chex.Array:
        labors = state["labor_all_regions"] * jnp.power(
            (1 + self.region_params.xL_a) / (1 + state["labor_all_regions"]),
            self.region_params.xl_g,
        )
        return labors

    def calc_production_factors(self, state: dict) -> chex.Array:
        production_factors = state["production_factor_all_regions"] * (
            jnp.exp(0.0033)
            + self.region_params.xg_A
            * jnp.exp(
                -self.region_params.xdelta_A
                * self.region_params.xDelta
                * (state["activity_timestep"] - 1)
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

    def calc_carbon_intensities(self, state: dict) -> chex.Array:
        carbon_intensity = state["intensity_all_regions"] * jnp.exp(
            -self.region_params.xg_sigma
            * jnp.power(
                1 - self.region_params.xdelta_sigma,
                self.region_params.xDelta * (state["activity_timestep"] - 1),
            )
            * self.region_params.xDelta
        )
        return carbon_intensity

    def calc_global_carbon_mass(
        self, state: dict, productions: chex.Array, mitigation_rates: chex.Array
    ) -> Tuple[chex.Array, dict]:
        prev_global_carbon_mass = state["global_carbon_mass"]
        carbon_updates = {}

        def calc_land_emissions():
            """Obtain the amount of land emissions."""
            e_l0 = self.region_params.xE_L0
            delta_el = self.region_params.xdelta_EL

            global_land_emissions = (
                e_l0
                * jnp.power(1 - delta_el, state["activity_timestep"] - 1)
                / self.num_regions
            )
            return global_land_emissions

        if self.carbon_model == "base":
            global_land_emissions = calc_land_emissions()
            # (original) TODO: fix aux_m treatment
            aux_m_all_regions = (
                state["intensity_all_regions"] * (1 - mitigation_rates) * productions
                + global_land_emissions
            )

            """Get the carbon mass level."""
            sum_aux_m = np.sum(aux_m_all_regions)
            global_carbon_mass = jnp.dot(
                jnp.asarray(self.region_params.xPhi_M), prev_global_carbon_mass
            ) + jnp.dot(jnp.asarray(self.region_params.xB_M), sum_aux_m)

        elif self.carbon_model in ["FaIR", "AR5", "DFaIR"]:
            carbon_model_params = {
                "a": jnp.array(
                    [
                        self.region_params.xM_a0,
                        self.region_params.xM_a1,
                        self.region_params.xM_a2,
                        self.region_params.xM_a3,
                    ]
                ),
                "tau": jnp.array(
                    [
                        self.region_params.xM_t0,
                        self.region_params.xM_t1,
                        self.region_params.xM_t2,
                        self.region_params.xM_t3,
                    ]
                ),
                "C0": self.region_params.xM_AT_1750,
                "irf0": self.region_params.irf0,
                "irC": self.region_params.irC,
                "irT": self.region_params.irT,
                "conv": jnp.array(
                    1.36388
                ),  # jnp.array(self.region_params),  # conversion 5/3.67 = 1.36388
            }

            # DAE determines given concentrations and temperature how much the reservoirs can absorb
            if self.carbon_model in ["FaIR", "DFaIR"]:
                # TODO: Plot the alpha values for diagnostics (if constantly 0.1 or 100 apparently we fail to solve the DAE). root_find has also throw option.
                global_alpha = solve_for_alpha(
                    state["global_alpha"],
                    carbon_model_params["a"],
                    carbon_model_params["tau"],
                    carbon_model_params["irf0"],
                    carbon_model_params["irC"],
                    carbon_model_params["irT"],
                    state["global_acc_pert_carb_stock"],
                    state["global_temperature"][0],
                )
            elif self.carbon_model == "AR5":
                global_alpha = 1.0

            carbon_updates["global_alpha"] = global_alpha

            global_land_emissions = calc_land_emissions()
            carbon_updates["global_land_emissions"] = global_land_emissions
            # (original) TODO: fix aux_m treatment
            aux_m_all_regions = (
                state["intensity_all_regions"] * (1 - mitigation_rates) * productions
                + global_land_emissions
            )

            """Get the carbon mass level."""
            sum_aux_m = jnp.sum(aux_m_all_regions)
            # In case, we want to prescribe the emissions to investigate the behavior of the temperature and carbon model
            # if self.prescribed_emissions is not None:
            #     sum_aux_m = self.prescribed_emissions[self.activity_timestep]
            carbon_updates["global_emissions"] = sum_aux_m

            global_cumulative_emissions = (
                state["global_cumulative_emissions"]
                + (state["global_emissions"] - state["global_land_emissions"])
                * carbon_model_params["conv"]
            )

            carbon_updates["global_cumulative_emissions"] = global_cumulative_emissions

            global_cumulative_land_emissions = (
                state["global_cumulative_land_emissions"]
                + state["global_land_emissions"]
                * self.num_regions
                * carbon_model_params["conv"]
            )
            carbon_updates["global_cumulative_land_emissions"] = (
                global_cumulative_land_emissions
            )

            if self.carbon_model in ["AR5", "FaIR"]:
                # Roll out of 5 intermediate steps reformulated with partial geometric series identity. Exponential as exponential is not a mistake.
                global_carbon_reservoirs = state["global_carbon_reservoirs"] ** jnp.exp(
                    -5 / (global_alpha * carbon_model_params["tau"])
                ) + carbon_model_params["a"] * sum_aux_m / 5 * carbon_model_params[
                    "conv"
                ] * (
                    jnp.exp(-1 / (global_alpha * carbon_model_params["tau"]))
                    - jnp.exp(-6 / (global_alpha * carbon_model_params["tau"]))
                ) / (1 - jnp.exp(-1 / (global_alpha * carbon_model_params["tau"])))
            elif self.carbon_model == "DFaIR":
                global_carbon_reservoirs = state["global_carbon_reservoirs"] * jnp.exp(
                    -5 / (carbon_model_params["tau"] * global_alpha)
                ) + carbon_model_params["a"] * sum_aux_m / 5 * carbon_model_params[
                    "conv"
                ] * carbon_model_params["tau"] * global_alpha * (
                    1 - jnp.exp(-5 / (global_alpha * carbon_model_params["tau"]))
                )
            carbon_updates["global_carbon_reservoirs"] = global_carbon_reservoirs

            global_acc_pert_carb_stock = (
                global_cumulative_emissions + global_cumulative_land_emissions
            ) - jnp.sum(global_carbon_reservoirs)
            carbon_updates["global_acc_pert_carb_stock"] = global_acc_pert_carb_stock

            atmospheric_carbon_mass = carbon_model_params["C0"] + jnp.sum(
                global_carbon_reservoirs
            )
            global_carbon_mass = prev_global_carbon_mass.at[0].set(
                atmospheric_carbon_mass
            )
        else:
            raise NotImplementedError(
                f"Carbon model {self.carbon_model} not implemented."
            )

        return global_carbon_mass, carbon_updates

    def calc_global_temperature(
        self, state: dict, global_carbon_mass: chex.Array
    ) -> chex.Array:
        global_temperature_boxes = state[
            "global_temperature_boxes"
        ]  # only changed in DFaIR

        def calc_exogenous_emissions():
            """Obtain the amount of exogeneous emissions."""
            f_0 = self.region_params.xf_0
            f_1 = self.region_params.xf_1
            t_f = self.region_params.xt_f

            exogenous_emissions = f_0 + jnp.minimum(
                f_1 - f_0, (f_1 - f_0) / t_f * (state["activity_timestep"] - 1)
            )
            return exogenous_emissions

        if self.temperature_calibration == "base":
            global_exogenous_emissions = calc_exogenous_emissions()
            prev_global_temperature = state["global_temperature"]
            # (original) TODO: why the zero index?
            # (original) global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = global_carbon_mass.at[0].get()
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

            return (
                global_temperature,
                global_exogenous_emissions,
                global_temperature_boxes,
            )

        elif self.temperature_calibration == "FaIR":
            global_exogenous_emissions = calc_exogenous_emissions()
            prev_global_temperature = state["global_temperature"]
            # (original) TODO: why the zero index?
            # (original) global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = global_carbon_mass.at[0].get()
            atmospheric_carbon_mass = jnp.array(self.region_params.xM_AT_1750)

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
            global_temperature_short = prev_global_temperature.at[0].get()
            for _ in range(4):  # TODO: this might be doable in one go?
                global_temperature_short = global_temperature_short + 1 / xT_1 * (
                    (forcings - xT_2 * global_temperature_short)
                    - xT_3
                    * (global_temperature_short - prev_global_temperature.at[1].get())
                )
            global_temperature = jnp.array(
                [
                    global_temperature_short,
                    prev_global_temperature.at[1].get()
                    + 5
                    * xT_3
                    / xT_4
                    * (
                        prev_global_temperature.at[0].get()
                        - prev_global_temperature.at[1].get()
                    ),
                ]
            )

            return (
                global_temperature,
                global_exogenous_emissions,
                global_temperature_boxes,
            )

        elif self.temperature_calibration == "DFaIR":
            global_exogenous_emissions = calc_exogenous_emissions()
            prev_global_temperature = state["global_temperature"]
            prev_global_temperature_boxes = state["global_temperature_boxes"]

            # (original) TODO: why the zero index?
            # (original) global_exogenous_emissions = global_exogenous_emissions[0]
            prev_atmospheric_carbon_mass = global_carbon_mass.at[0].get()
            atmospheric_carbon_mass = jnp.array(self.region_params.xM_AT_1750)

            f_2x = self.region_params.xF_2x
            d = jnp.array([self.region_params.xT_LO_rt, self.region_params.xT_UO_rt])
            teq = jnp.array(
                [
                    self.region_params.xT_LO_tq,
                    self.region_params.xT_UO_tq,
                ]
            )
            forcings = (
                f_2x
                * jnp.log(prev_atmospheric_carbon_mass / atmospheric_carbon_mass)
                / jnp.log(2)
                + global_exogenous_emissions
            )
            global_temperature_boxes = prev_global_temperature_boxes * jnp.exp(
                -5 / d
            ) + teq * forcings * (1 - jnp.exp(-5 / d))

            global_temperature = jnp.array([np.sum(global_temperature_boxes), 0])

            return (
                global_temperature,
                global_exogenous_emissions,
                global_temperature_boxes,
            )

        else:
            raise ValueError(
                f"Unknown temperature calibration: {self.temperature_calibration}"
            )

    def calc_current_simulation_year(self, state: dict) -> chex.Array:
        return state["current_simulation_year"] + self.region_params.xDelta

    @property
    def action_space(self) -> dict[str, jym.Space]:
        N_REGIONS = self.num_regions
        N_DISCRETIZATION = self.num_discrete_action_levels
        actions = {
            "import_bid": MultiDiscrete([N_DISCRETIZATION] * N_REGIONS),
            "import_tariff": MultiDiscrete([N_DISCRETIZATION] * N_REGIONS),
            #
            "savings_rate": Discrete(N_DISCRETIZATION),
            "mitigation_rate": Discrete(N_DISCRETIZATION),
            "export_limit": Discrete(N_DISCRETIZATION),
        }
        if self.negotiation_on:
            # 2 actions per region (accept/reject)
            actions["proposal_ask"] = MultiDiscrete([N_DISCRETIZATION] * N_REGIONS)
            actions["proposal_promise"] = MultiDiscrete([N_DISCRETIZATION] * N_REGIONS)
            actions["proposal_decision"] = MultiDiscrete([2] * N_REGIONS)  # Yes /No

        # Return the actions for each region
        return {i_to_agent_str(i): actions for i in range(N_REGIONS)}

    @property
    def observation_space(self) -> jym.Box:
        obs, _ = self.reset(jax.random.PRNGKey(0))
        single_agent_obs = obs[i_to_agent_str(0)].observation
        return {
            i_to_agent_str(i): jym.Box(
                low=-9999,
                high=9999,
                shape=single_agent_obs.shape,
                dtype=single_agent_obs.dtype,
            )
            for i in range(self.num_regions)
        }
