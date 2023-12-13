# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


"""
Regional Integrated model of Climate and the Economy (RICE)
"""
import logging
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import yaml


_PUBLIC_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_SMALL_NUM = 1e-0  # small number added to handle consumption blow-up

sys.path = [_PUBLIC_REPO_DIR] + sys.path

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

_FEATURES = "features"
_ACTION_MASK = "action_mask"

class Rice(gym.Env):
    name = "Rice"

    def __init__(
        self,
        num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
        negotiation_on=False,
        scenario="default"  # If True then negotiation is on, else off
    ):
        self.global_state = {}

        self.set_discrete_action_levels(num_discrete_action_levels)
        self.set_dtypes()


        self.set_all_region_params()
        self.set_trade_params()
        self.num_regions = len(self.all_regions_params)
        self.num_agents = self.num_regions  # for env wrapper


        self.start_year = self.get_start_year()
        self.end_year = self.calc_end_year()
        self.current_simulation_year = None
        self.current_timestep = None
        self.activity_timestep = None
        self.negotiation_on = negotiation_on
        self.apply_welfloss = True
        if self.negotiation_on:
            self.negotiation_stage = 0
            self.num_negotiation_stages = 2
        self.set_episode_length(negotiation_on)

        self.observation_space = None
        self.set_possible_actions()
        self.total_possible_actions = self.calc_total_possible_actions(
            self.negotiation_on
        )
        self.action_space = self.get_action_space()

        self.set_default_agent_action_mask()

    def get_start_year(self):
        return self.all_regions_params[0]["xt_0"]

    def reset(self, *, seed=None, options=None):

        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state('timestep')
        self.reset_state('activity_timestep')

        # climate states
        self.reset_state('global_temperature')
        self.reset_state('global_carbon_mass')
        self.reset_state('global_exogenous_emissions')
        self.reset_state('global_land_emissions')
        self.reset_state('intensity_all_regions')
        self.reset_state('mitigation_rates_all_regions')

        # economic states
        self.reset_state('production_all_regions')
        self.reset_state('gross_output_all_regions')
        self.reset_state('aggregate_consumption')
        self.reset_state('investment_all_regions')
        self.reset_state('capital_all_regions')
        self.reset_state('capital_depreciation_all_regions')
        self.reset_state('labor_all_regions')
        self.reset_state('production_factor_all_regions')
        self.reset_state('current_balance_all_regions')
        self.reset_state('abatement_cost_all_regions')
        self.reset_state('mitigation_cost_all_regions')
        self.reset_state('damages_all_regions')
        self.reset_state('utility_all_regions')
        self.reset_state('social_welfare_all_regions')
        self.reset_state('reward_all_regions')

        # trade states
        self.reset_state('tariffs')
        self.reset_state('import_tariffs')
        self.reset_state('normalized_import_bids_all_regions')
        self.reset_state('import_bids_all_regions')
        self.reset_state('imports_minus_tariffs')
        self.reset_state('export_limit_all_regions')

        # negotiation states
        self.reset_state('negotiation_stage')
        self.reset_state('savings_all_regions')
        self.reset_state('minimum_mitigation_rate_all_regions')
        self.reset_state('promised_mitigation_rate')
        self.reset_state('requested_mitigation_rate')
        self.reset_state('proposal_decisions')

        info = { region:{} for region in range(self.num_regions)}  # for the new ray rllib env format
        return self.get_observations(), info

    def step(self, actions):
        self.current_timestep += 1
        self.set_state("timestep", self.current_timestep, dtype=self.int_dtype)

        self.set_current_global_state_to_past_global_state()

        if self.negotiation_on:
            self.set_negotiation_stage()
            if self.is_proposal_stage():
                return self.step_propose(actions)

            elif self.is_evaluation_stage():
                return self.step_evaluate_proposals(actions)

        return self.step_climate_and_economy(actions)

    def step_propose(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        promised_mitigation_rates = self.get_actions("promised_mitigation_rate", actions)
        self.set_state("promised_mitigation_rate", np.array(promised_mitigation_rates))
        requested_mitigation_rates = self.get_actions("requested_mitigation_rate", actions)
        self.set_state("requested_mitigation_rate", np.array(requested_mitigation_rates))

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info

    def step_evaluate_proposals(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=2)
        self.is_valid_actions_dict(actions)

        proposal_decisions = self.get_actions("proposal_decisions", actions)

        self.set_state("proposal_decisions", proposal_decisions)

        for region_id in range(self.num_regions):
            min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)

            self.set_state(
                "minimum_mitigation_rate_all_regions", min_mitigation, region_id
                )

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}
        return observations, rewards, terminateds, truncateds, info

    def step_climate_and_economy(self, actions=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)

        actions_dict = {
            "savings_all_regions" : self.get_actions("savings", actions),
            "mitigation_rate_all_regions" : self.get_actions("mitigation_rate", actions),
            "export_limit_all_regions" : self.get_actions("export_limit", actions),
            "import_bids_all_regions" : self.get_actions("import_bids", actions),
            "import_tariffs_all_regions" : self.get_actions("import_tariffs", actions),
        }

        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(actions_dict["mitigation_rate_all_regions"])
        productions = self.calc_productions()

        gross_outputs = self.calc_gross_outputs(damages, abatement_costs, productions)
        investments = self.calc_investments(gross_outputs, actions_dict["savings_all_regions"])

        gov_balances_post_interest = self.calc_gov_balances_post_interest()
        debt_ratios = self.calc_debt_ratios(gov_balances_post_interest)

        # TODO: self.set_global_state("tariffs", self.global_state["import_tariffs"]["value"][self.current_timestep])
        # TODO: fix dependency on gross_output_all_regions
        # TODO: government should reuse tariff revenue
        gross_imports = self.calc_gross_imports(actions_dict['import_bids_all_regions'], gross_outputs, investments, debt_ratios)

        tariff_revenues, net_imports = self.calc_trade_sanctions(gross_imports)
        welfloss_multipliers = self.calc_welfloss_multiplier(gross_outputs, gross_imports)

        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports, net_imports)
        utilities = self.calc_utilities(consumptions)

        self.calc_social_welfares(utilities)
        self.calc_rewards(utilities, welfloss_multipliers)

        self.calc_capitals(investments)
        self.calc_labors()
        self.calc_production_factors()
        self.calc_gov_balances_post_trade(gov_balances_post_interest, gross_imports)

        self.calc_carbon_intensities()
        self.calc_global_carbon_mass(productions)
        self.calc_global_temperature()

        current_simulation_year = self.calc_current_simulation_year()
        observations = self.get_observations()
        rewards = self.get_rewards()
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds = {"__all__": current_simulation_year == self.end_year}
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds = {"__all__": current_simulation_year == self.episode_length}
        info = {}

        return observations, rewards, terminateds, truncateds, info

    def calc_carbon_intensities(self, save_state=True):
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            carbon_intensity = self.get_prev_state("intensity_all_regions", region_id=region_id) * np.exp(
            -regional_params["xg_sigma"]
            * pow(
                1 - regional_params["xdelta_sigma"],
                regional_params["xDelta"] * (self.activity_timestep - 1),
            )
            * regional_params["xDelta"]
            )
            if save_state:
                self.set_state("intensity_all_regions", carbon_intensity, region_id=region_id)

    def calc_production_factors(self, save_state=True):
        production_factors = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            production_factors[region_id] = self.get_prev_state("production_factor_all_regions", region_id=region_id) * (
                np.exp(0.0033)
                + regional_params["xg_A"]
                * np.exp(
                    -regional_params["xdelta_A"]
                    * regional_params["xDelta"]
                    * (self.activity_timestep - 1)
                )
            )

            if save_state:
                self.set_state("production_factor_all_regions", production_factors[region_id], region_id=region_id)

    def calc_labors(self, save_state=True):
        labors = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            labors[region_id] = self.get_prev_state("labor_all_regions", region_id=region_id) * pow(
                (1 + regional_params["xL_a"]) / (1 + self.get_prev_state("labor_all_regions", region_id=region_id)),
                regional_params["xl_g"],
            )
            if save_state:
                self.set_state("labor_all_regions", labors[region_id], region_id=region_id) #TODO check all save_states for region_id
        return labors

    def calc_capitals(self, investments, save_state=True):
        capitals = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            x_delta_k = regional_params["xdelta_K"]
            x_delta = regional_params["xDelta"]
            capital_depreciation = pow(1 - x_delta_k, x_delta)
            if save_state:
                self.set_state("capital_depreciation_all_regions", capital_depreciation, region_id=region_id)
            capitals[region_id] =  (
                capital_depreciation * self.get_prev_state("capital_all_regions", region_id=region_id)
                + regional_params["xDelta"] * investments[region_id]
            )

            if save_state:
                self.set_state("capital_all_regions", capitals[region_id], region_id=region_id)

        return capitals

    def calc_rewards(self, utilities, welfloss_multipliers, save_state=True):
        rewards = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            rewards[region_id] = utilities[region_id] * welfloss_multipliers[region_id]
            self.set_state("reward_all_regions", utilities[region_id], region_id=region_id)
        return rewards

    def calc_gov_balances_post_trade(self, gov_balances, gross_imports, save_state=True):
        gov_balances_post_trade = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            trade_balance = regional_params["xDelta"] * (
                    np.sum(gross_imports[:, region_id])
                    - np.sum(gross_imports[region_id, :])
                )
            gov_balances_post_trade[region_id] = gov_balances[region_id] + trade_balance

            if save_state:
                self.set_state("current_balance_all_regions", gov_balances_post_trade[region_id], region_id=region_id)

        return gov_balances_post_trade

    def calc_social_welfares(self, utilities, save_state=True):
        social_welfares = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            rho = regional_params["xrho"]
            delta = regional_params["xDelta"]
            """Compute social welfare"""
            social_welfares[region_id] =  utilities[region_id] / pow(1 + rho, delta * self.activity_timestep)
            if save_state:
                self.set_state("social_welfare_all_regions", social_welfares[region_id], region_id=region_id)
        return social_welfares

    def calc_utilities(self, consumptions, save_state=True):
        utilities = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            utilities[region_id] =  (
                (self.get_prev_state("labor_all_regions",region_id) / 1000.0)
                * (pow(consumptions[region_id] / (self.get_prev_state("labor_all_regions",region_id) / 1000.0) + _SMALL_NUM, 1 - regional_params["xalpha"]) - 1)
                / (1 - regional_params["xalpha"])
            )
            if save_state:
                self.set_state("utility_all_regions", utilities[region_id], region_id=region_id)
        return utilities

    def calc_consumptions(self, gross_outputs, investments, gross_imports, net_imports, save_state=True):
        consumptions = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            total_exports = np.sum(gross_imports[:, region_id])
            assert (
                gross_outputs[region_id] - investments[region_id] - total_exports > -1e-5
            ), "consumption cannot be negative."
            domestic_consumption =  max(0.0, gross_outputs[region_id] - investments[region_id] - total_exports)

            c_dom_pref = self.preference_for_domestic * (
                domestic_consumption**self.consumption_substitution_rate
            )
            c_for_pref = np.sum(
                self.preference_for_imported
                * pow(net_imports[region_id, :], self.consumption_substitution_rate)
            )

            consumptions[region_id] = (c_dom_pref + c_for_pref) ** (
                1 / self.consumption_substitution_rate
            )  # CES function

            # TODO: fix for region-specific state saving
            if save_state:
                self.set_state("aggregate_consumption", consumptions[region_id], region_id=region_id)
        return consumptions

    def calc_debt_ratios(self, gov_balances, save_state=True):
        debt_ratios = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            gov_balance = gov_balances[region_id]
            regional_params = self.all_regions_params[region_id]
            debt_ratio = (
                gov_balance * self.init_capital_multiplier / regional_params["xK_0"]
            )
            debt_ratio = min(0.0, debt_ratio)
            debt_ratio = max(-1.0, debt_ratio)
            debt_ratios[region_id] = np.array(debt_ratio).astype(self.float_dtype)
        if save_state:
            self.set_state("debt_ratio_all_regions", debt_ratios[region_id], region_id=region_id)

        return debt_ratios

    def calc_gov_balances_post_interest(self, save_state=True):
        gov_balances_post_interest = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            gov_balances_post_interest[region_id] = self.get_prev_state("current_balance_all_regions", region_id) * (1 + self.balance_interest_rate)
            if save_state:
                self.set_state("current_balance_all_regions", gov_balances_post_interest[region_id], region_id=region_id)
        return gov_balances_post_interest

    def calc_investments(self, gross_outputs, savings, save_state=True):
        investments = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            investments[region_id] = savings[region_id] * gross_outputs[region_id]
            if save_state:
                self.set_state("investment_all_regions", investments[region_id], region_id=region_id)
        return investments

    def calc_gross_outputs(self, damages, abatement_costs, productions, save_state=True):
        gross_outputs = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            gross_outputs[region_id] =  damages[region_id] * (1 - abatement_costs[region_id]) * productions[region_id]

            if save_state:
                self.set_state("gross_output_all_regions", gross_outputs[region_id], region_id=region_id)

        return gross_outputs

    def calc_productions(self, save_state=True):
        productions = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            productions[region_id] = (
            self.get_prev_state("production_factor_all_regions", region_id)
            * pow(self.get_prev_state("capital_all_regions", region_id), self.all_regions_params[region_id]["xgamma"])
            * pow(self.get_prev_state("labor_all_regions", region_id) / 1000, 1 - self.all_regions_params[region_id]["xgamma"])
            )

        if save_state:
            self.set_state("production_all_regions", productions[region_id], region_id=region_id)

        return productions

    def calc_damages(self, save_state=True):
        damages = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            prev_atmospheric_temperature = self.get_prev_state("global_temperature")[0]
            damages[region_id] =  1 / (
                1
                + self.all_regions_params[region_id]["xa_1"] * prev_atmospheric_temperature
                + self.all_regions_params[region_id]["xa_2"]
                * pow(prev_atmospheric_temperature, self.all_regions_params[region_id]["xa_3"])
            )

            if save_state:
                self.set_state("damages_all_regions", damages[region_id], region_id=region_id)

        return damages

    def calc_abatement_costs(self, mitigation_rates_all_regions, save_state=True):
        mitigation_costs = self.calc_mitigation_costs()
        abatement_costs = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            abatement_costs[region_id] = mitigation_costs[region_id] * pow(mitigation_rates_all_regions[region_id], self.all_regions_params[region_id]["xtheta_2"])
            if save_state:
                self.set_state("abatement_cost_all_regions", abatement_costs[region_id], region_id=region_id)
        return abatement_costs

    def calc_mitigation_costs(self, save_state=True):
        mitigation_costs = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            regional_params = self.all_regions_params[region_id]
            mitigation_costs[region_id] = (
            regional_params["xp_b"]
            / (1000 * regional_params["xtheta_2"])
            * pow(1 - regional_params["xdelta_pb"], self.activity_timestep - 1)
            * self.get_prev_state("intensity_all_regions", region_id)
        )

            if save_state:
                self.set_state("mitigation_cost_all_regions", mitigation_costs[region_id], region_id=region_id)

        return mitigation_costs

    def calc_gross_imports(self, import_bids, gross_outputs, investments, debt_ratios, save_state=True):
        potential_import_bids = np.zeros((self.num_regions, self.num_regions), dtype=self.float_dtype)

        for region_id in range(self.num_regions):
            gross_output = gross_outputs[region_id]
            debt_ratio = debt_ratios[region_id]
            potential_import_bids = np.zeros((self.num_regions, self.num_regions), dtype=self.float_dtype)

            import_bids[region_id][region_id] = 0

            total_import_bids = np.sum(import_bids[region_id])
            if total_import_bids * gross_output > gross_output:
                potential_import_bids[region_id] = (
                    import_bids[region_id]
                    / total_import_bids
                    * gross_output
                )
            else:
                potential_import_bids[region_id] = (
                    import_bids[region_id] * gross_output
                )

            potential_import_bids[region_id] *= 1 + debt_ratio


        normalized_import_bids_all_regions = self.calc_normalized_import_bids(
            potential_import_bids,
            gross_outputs,
            investments)

        if save_state:
            self.set_state("normalized_import_bids_all_regions", normalized_import_bids_all_regions)
        return normalized_import_bids_all_regions

    def calc_trade_sanctions(self, gross_imports, save_state=True):
        import_tariffs = self.get_prev_state("import_tariffs")
        net_imports = np.zeros((self.num_regions, self.num_regions), dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            # TODO: calculate using arrays?
            for exporting_region in range(self.num_regions):
                net_imports[region_id, exporting_region] = \
                    gross_imports[region_id, exporting_region] * \
                    (1 - import_tariffs[region_id, exporting_region])

        if save_state:
            self.set_state("imports_minus_tariffs", net_imports)

        tariff_revenues = np.zeros((self.num_regions, self.num_regions), dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            for exporting_region in range(self.num_regions):
                tariff_revenues[region_id, exporting_region] = \
                    gross_imports[region_id, exporting_region] * \
                    import_tariffs[region_id, exporting_region]

        if save_state:
            self.set_state("tariff_revenues", tariff_revenues)

        return tariff_revenues, net_imports

    def calc_welfloss_multiplier(self, gross_outputs, gross_imports, welfare_loss_per_unit_tariff=None, save_state=True):
        """Calculate the welfare loss multiplier of exporting region due to being tariffed."""
        if not self.apply_welfloss:
            return np.zeros((self.num_regions), dtype=self.float_dtype)

        if welfare_loss_per_unit_tariff is None:
            welfare_loss_per_unit_tariff = 0.4 # From Nordhaus 2015

        import_tariffs = self.get_prev_state("import_tariffs")
        welfloss = np.zeros((self.num_regions), dtype=self.float_dtype)

        for region_id in range(self.num_regions):
            for exporting_region in range(self.num_regions):
                welfloss[region_id] += \
                    (gross_imports[region_id, exporting_region] / gross_outputs[region_id]) * \
                        import_tariffs[region_id, exporting_region] * welfare_loss_per_unit_tariff

        if save_state:
            self.set_state("welfloss", welfloss)

        return welfloss

    def calc_exogenous_emissions(self, save_state=True):
        """Obtain the amount of exogeneous emissions."""
        f_0 = self.all_regions_params[0]["xf_0"]
        f_1 = self.all_regions_params[0]["xf_1"]
        t_f = self.all_regions_params[0]["xt_f"]

        exogenous_emissions = f_0 + min(f_1 - f_0, (f_1 - f_0) / t_f * (self.activity_timestep - 1))
        if save_state:
            self.set_state("global_exogenous_emissions", exogenous_emissions)
        return exogenous_emissions

    def calc_land_emissions(self, save_state=True):
        """Obtain the amount of land emissions."""
        e_l0 = self.all_regions_params[0]["xE_L0"]
        delta_el = self.all_regions_params[0]["xdelta_EL"]

        global_land_emissions = e_l0 * pow(1 - delta_el, self.activity_timestep - 1) / self.num_regions
        if save_state:
            self.set_state("global_land_emissions", global_land_emissions)
        return global_land_emissions

    def calc_max_exports(self, x_max, gross_output, investment):
        """Determine the maximum potential exports."""
        if x_max * gross_output <= gross_output - investment:
            return x_max * gross_output
        return gross_output - investment

    def calc_global_temperature(
        self,
        save_state=True,
    ):

        global_exogenous_emissions = self.calc_exogenous_emissions()
        prev_carbon_mass = self.get_prev_state("global_carbon_mass")
        prev_global_temperature = self.get_prev_state("global_temperature")
        # TODO: why the zero index?
        # global_exogenous_emissions = global_exogenous_emissions[0]
        prev_atmospheric_carbon_mass = prev_carbon_mass[0]
        phi_t = np.array(self.all_regions_params[0]["xPhi_T"])
        b_t = np.array(self.all_regions_params[0]["xB_T"])
        f_2x = np.array(self.all_regions_params[0]["xF_2x"])
        atmospheric_carbon_mass = np.array(self.all_regions_params[0]["xM_AT_1750"])

        global_temperature = np.dot(phi_t, np.asarray(prev_global_temperature)) + np.dot(
            b_t,
            f_2x * np.log(prev_atmospheric_carbon_mass / atmospheric_carbon_mass) / np.log(2) + global_exogenous_emissions,
        )

        if save_state:
            self.set_state("global_temperature",global_temperature)

        return global_temperature

    def calc_global_carbon_mass(
        self,
        productions,
        save_state=True
    ):
        global_land_emissions = self.calc_land_emissions()
        # prev_global_carbon_mass = self.get_prev_global_state("global_carbon_mass")[0]
        mitigation_rates = self.get_state("mitigation_rates_all_regions")
        # TODO: fix aux_m treatment
        aux_m_all_regions = np.zeros(self.num_regions, dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            prev_intensity = self.get_prev_state("intensity_all_regions", region_id=region_id)

            aux_m_all_regions[region_id] = prev_intensity * (1 - mitigation_rates[region_id]) * productions[region_id] + global_land_emissions
            self.set_state("aux_m_all_regions", aux_m_all_regions[region_id], region_id=region_id)


        """Get the carbon mass level."""
        sum_aux_m = np.sum(aux_m_all_regions)
        prev_global_carbon_mass = self.get_prev_state("global_carbon_mass")
        global_carbon_mass = np.dot(
            self.all_regions_params[0]["xPhi_M"], np.asarray(prev_global_carbon_mass)
        )

        global_carbon_mass += np.dot(self.all_regions_params[0]["xB_M"], sum_aux_m)

        if save_state:
            self.set_state("global_carbon_mass", global_carbon_mass)

        return global_carbon_mass

    def calc_possible_actions(self, action_type):
        if action_type == "savings":
            return [self.num_discrete_action_levels]
        if action_type == "mitigation_rate":
            return [self.num_discrete_action_levels]
        if action_type == "export_limit":
            return [self.num_discrete_action_levels]
        if action_type == "import_bids":
            return [self.num_discrete_action_levels] * self.num_regions
        if action_type == "import_tariffs":
            return [self.num_discrete_action_levels] * self.num_regions

        if action_type == "proposal":
            return [self.num_discrete_action_levels] * 2 * self.num_regions

        if action_type == 'proposal_decisions':
            return [2] * self.num_regions

    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = (
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
            )

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions
                + self.evaluation_possible_actions
            )

        return total_possible_actions

    def calc_uniform_foreign_preferences(self):
        return [
            (1 - self.preference_for_domestic) / (self.num_regions - 1)
        ] * self.num_regions

    def calc_end_year(self):
        return (
            self.start_year
            + self.all_regions_params[0]["xDelta"]
            * self.all_regions_params[0]["xN"]
        )

    def calc_mitigation_rate_lower_bound(self, region_id):
        outgoing_accepted_mitigation_rates = (
            self.get_outgoing_accepted_mitigation_rates(region_id)
        )
        incoming_accepted_mitigation_rates = (
            self.get_incoming_accepted_mitigation_rates(region_id)
        )

        min_mitigation = max(
            outgoing_accepted_mitigation_rates
            + incoming_accepted_mitigation_rates
        )

        return min_mitigation

    def calc_normalized_import_bids(self, potential_import_bids_all_regions, gross_outputs, investments):
        normalized_import_bids_all_regions = np.zeros((self.num_regions, self.num_regions), dtype=self.float_dtype)
        for region_id in range(self.num_regions):
            max_export_rate = self.get_state("export_limit_all_regions",region_id=region_id)

            max_exports_from_region_id = self.calc_max_exports(
                max_export_rate,
                gross_outputs[region_id],
                investments[region_id])

            desired_exports_from_region_id = np.sum(normalized_import_bids_all_regions[:, region_id])

            if desired_exports_from_region_id > max_exports_from_region_id:
                for exporting_region in range(self.num_regions):
                    normalized_import_bids_all_regions[exporting_region][region_id] = \
                    (potential_import_bids_all_regions[exporting_region][region_id]
                        / desired_exports_from_region_id
                        * max_exports_from_region_id
                    )

        return normalized_import_bids_all_regions

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()
            if self.negotiation_on:
                minimum_mitigation_rate = int(
                    round(
                        self.global_state[
                            "minimum_mitigation_rate_all_regions"
                        ]["value"][self.current_timestep, region_id]
                        * self.num_discrete_action_levels
                    )
                )
                mitigation_mask = np.array(
                    [0 for _ in range(minimum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels
                            - minimum_mitigation_rate
                        )
                    ]
                )
                mask_start = sum(self.savings_possible_actions)
                mask_end = mask_start + sum(
                    self.mitigation_rate_possible_actions
                )
                mask[mask_start:mask_end] = mitigation_mask
            mask_dict[region_id] = mask

        return mask_dict

    def calc_current_simulation_year(self):

        self.current_simulation_year += self.all_regions_params[0]["xDelta"]
        return self.current_simulation_year

    def calc_activity_timestep(self):
        self.activity_timestep += 1
        self.set_state(
            key="activity_timestep",
            value=self.activity_timestep,
            timestep=self.current_timestep,
            dtype=self.int_dtype,
        )
        self.is_valid_activity_timestep()
        return self.activity_timestep

    def get_incoming_accepted_mitigation_rates(self, region_id):
        return [
            self.global_state["requested_mitigation_rate"]["value"][
                self.current_timestep, j, region_id
            ]
            * self.global_state['proposal_decisions']["value"][
                self.current_timestep, region_id, j
            ]
            for j in range(self.num_regions)
        ]

    def get_outgoing_accepted_mitigation_rates(self, region_id):
        return [
            self.global_state["promised_mitigation_rate"]["value"][
                self.current_timestep, region_id, j
            ]
            * self.global_state["proposal_decisions"]["value"][
                self.current_timestep, j, region_id
            ]
            for j in range(self.num_regions)
        ]

    def get_action_space(self):
        return {
            str(region_id): MultiDiscrete(self.total_possible_actions)
            for region_id in range(self.num_regions)
        }

    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index(
                "mitigation_rate"
            )
            return [
                actions[region_id][mitigation_rate_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "export_limit":
            export_action_index = self.get_actions_index("export_limit")
            return [
                actions[region_id][export_action_index]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_bids":
            tariffs_action_index = self.get_actions_index("import_bids")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index
                    + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index
                    + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "promised_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")
            num_proposal_actions = len(self.proposal_possible_actions)

            value = [
                actions[
                    region_id][
                        proposal_actions_index_start : proposal_actions_index_start + num_proposal_actions : 2
                        ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]
            return value

        if action_type == "requested_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")
            num_proposal_actions = len(self.proposal_possible_actions)

            return [
                actions[region_id][
                    proposal_actions_index_start
                    + 1 : proposal_actions_index_start
                    + num_proposal_actions : 2
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == 'proposal_decisions':
            proposal_decisions_index_start = self.get_actions_index(
                'proposal_decisions'
            )
            num_evaluation_actions = len(self.evaluation_possible_actions)

            proposal_decisions = np.array(
                [
                    actions[region_id][
                        proposal_decisions_index_start : proposal_decisions_index_start
                        + num_evaluation_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )
            for region_id in range(self.num_regions):
                proposal_decisions[region_id, region_id] = 0

            return proposal_decisions

    def get_actions_index(self, action_type):
        if action_type == "savings":
            return 0
        if action_type == "mitigation_rate":
            return len(self.savings_possible_actions)
        if action_type == "export_limit":
            return len(self.savings_possible_actions) + len(
                self.mitigation_rate_possible_actions
            )
        if action_type == "import_bids":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
            )
        if action_type == "import_tariffs":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
            )

        if action_type == "proposal":
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
            )

        if action_type == 'proposal_decisions':
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.proposal_possible_actions
            )

    def is_evaluation_stage(self):
        return self.negotiation_stage == 2

    def is_proposal_stage(self):
        return self.negotiation_stage == 1

    def is_valid_activity_timestep(self):
        if not self.negotiation_on:
            assert self.current_timestep == self.activity_timestep

    def is_valid_negotiation_stage(self, negotiation_stage):
        if self.negotiation_on:
            assert self.negotiation_stage == negotiation_stage
        if not self.negotiation_on:
            assert negotiation_stage == 0, "Negotiation is not on, so why is negotiation_stage anything other than 0?"

    def is_valid_actions_dict(self, actions):
        assert isinstance(actions, dict)
        assert len(actions) == self.num_regions

    def set_actions_in_global_state(self, actions_dict):
        for (action_name, action_value) in actions_dict.items():
            self.set_state(
                key=action_name,
                value=action_value,
                timestep=self.current_timestep,
                dtype=self.float_dtype,
            )

    def set_dtypes(self):
        self.float_dtype = np.float32
        self.int_dtype = np.int32

    def set_discrete_action_levels(self, num_discrete_action_levels):
        assert (
            num_discrete_action_levels > 1
        ), "the number of action levels should be > 1."
        self.num_discrete_action_levels = num_discrete_action_levels

    def set_all_region_params(self):
        param_path = os.path.join(_PUBLIC_REPO_DIR, "region_yamls")
        num_regions, raw_params = self.read_rice_param_yamls(param_path)
        self.num_regions = num_regions
        self.region_specific_params = raw_params["_RICE_CONSTANT"]
        self.common_params = raw_params["_DICE_CONSTANT"]
        self.all_regions_params = self.merge_to_regional_param_dict(
            self.common_params, self.region_specific_params
        )

    def set_possible_actions(self):
        self.savings_possible_actions = self.calc_possible_actions("savings")
        self.mitigation_rate_possible_actions = self.calc_possible_actions(
            "mitigation_rate"
        )
        self.export_limit_possible_actions = self.calc_possible_actions(
            "export_limit"
        )
        self.import_bids_possible_actions = self.calc_possible_actions(
            "import_bids"
        )
        self.import_tariff_possible_actions = self.calc_possible_actions(
            "import_tariffs"
        )

        if self.negotiation_on:
            self.proposal_possible_actions = self.calc_possible_actions(
                "proposal"
            )
            self.evaluation_possible_actions = self.calc_possible_actions(
                'proposal_decisions'
            )

    def set_default_agent_action_mask(self):
        self.possible_actions_length = sum(self.total_possible_actions)
        self.default_agent_action_mask = np.ones(
            self.possible_actions_length, dtype=self.int_dtype
        )

    def set_episode_length(self, negotiation_on):
        self.episode_length = self.all_regions_params[00]["xN"]

        if negotiation_on:
            self.episode_length += self.common_params["xN"] * (
                self.num_negotiation_stages + 1
            )

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 0.5
        self.preference_for_domestic = 0.5
        self.preference_for_imported = self.calc_uniform_foreign_preferences()

        # Typecasting
        self.consumption_substitution_rate = np.array(
            [self.consumption_substitution_rate]
        ).astype(self.float_dtype)
        self.preference_for_domestic = np.array(
            [self.preference_for_domestic]
        ).astype(self.float_dtype)
        self.preference_for_imported = np.array(
            self.preference_for_imported, dtype=self.float_dtype
        )

    def set_negotiation_stage(self):
        # Note: The '+1` below is for the climate_and_economy_simulation_step
        self.negotiation_stage = self.current_timestep % (
            self.num_negotiation_stages + 1
        )
        self.set_state(
            "negotiation_stage",
            self.negotiation_stage,
            self.current_timestep,
            dtype=self.int_dtype,
        )

    def set_current_global_state_to_past_global_state(self):
        for key in self.global_state:
            if key != "reward_all_regions":
                self.global_state[key]["value"][self.current_timestep] = \
                    self.global_state[key]["value"][self.current_timestep - 1].copy()

    def get_observations(self):
        """
        Format observations for each agent by concatenating global, public
        and private features.
        The observations are returned as a dictionary keyed by region index.
        Each dictionary contains the features as well as the action mask.
        """
        # Observation array features

        # Global features that are observable by all regions
        global_features = [
            "global_temperature",
            "global_carbon_mass",
            "global_exogenous_emissions",
            "global_land_emissions",
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "capital_all_regions",
            "capital_depreciation_all_regions",
            "labor_all_regions",
            "gross_output_all_regions",
            "investment_all_regions",
            "aggregate_consumption",
            "savings_all_regions",
            "mitigation_rates_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
            "production_factor_all_regions",
            "intensity_all_regions",
            "mitigation_cost_all_regions",
            "damages_all_regions",
            "abatement_cost_all_regions",
            "production_all_regions",
            "utility_all_regions",
            "social_welfare_all_regions",
            "reward_all_regions",
        ]

        # Features concerning two regions
        bilateral_features = []

        if self.negotiation_on:
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

        shared_features = np.array([])
        for feature in global_features + public_features:
            shared_features = np.append(
                shared_features,
                self.flatten_array(
                    self.global_state[feature]["value"][self.current_timestep]
                    / self.global_state[feature]["norm"]
                ),
            )

        # Form the feature dictionary, keyed by region_id.
        features_dict = {}
        for region_id in range(self.num_regions):
            # Add a region indicator array to the observation
            region_indicator = np.zeros(
                self.num_regions, dtype=self.float_dtype
            )
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert (
                    self.global_state[feature]["value"].shape[1]
                    == self.num_regions
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            for feature in bilateral_features:
                assert (
                    self.global_state[feature]["value"].shape[1]
                    == self.num_regions
                )
                assert (
                    self.global_state[feature]["value"].shape[2]
                    == self.num_regions
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )
                all_features = np.append(
                    all_features,
                    self.flatten_array(
                        self.global_state[feature]["value"][
                            self.current_timestep, :, region_id
                        ]
                        / self.global_state[feature]["norm"]
                    ),
                )

            features_dict[region_id] = all_features

        # Fetch the action mask dictionary, keyed by region_id.
        action_mask_dict = self.calc_action_mask()

        # Form the observation dictionary keyed by region id.
        obs_dict = {}
        for region_id in range(self.num_regions):
            obs_dict[region_id] = {
                _FEATURES: features_dict[region_id],
                _ACTION_MASK: action_mask_dict[region_id],
            }

        return obs_dict

    def get_rewards(self):
        # regions Ids must be strings
        return {str(region_id): self.get_state("reward_all_regions", region_id=region_id) for region_id in range(self.num_regions)}

    def reset_state(self, key):
        # timesteps
        if key == 'timestep': self.set_state(key, value=self.current_timestep, dtype=self.int_dtype, norm=1e2)
        if key == 'activity_timestep': self.set_state(key, value=self.activity_timestep, dtype=self.int_dtype)

        # scalars
        if key == 'negotiation_stage': self.set_state(key, value=np.zeros(1,), dtype=self.int_dtype)
        if key in ['global_land_emissions', 'global_exogenous_emissions']:
            self.set_state(key, value=np.zeros(1,))

        # num_regions vectors
        if key in ['minimum_mitigation_rate_all_regions', 'reward_all_regions', 'social_welfare_all_regions',
                   'utility_all_regions', 'abatement_cost_all_regions', 'damages_all_regions',
                   'mitigation_cost_all_regions', 'export_limit_all_regions', 'mitigation_rates_all_regions',
                   'savings_all_regions', 'capital_depreciation_all_regions']:
            self.set_state(key, value=np.zeros(self.num_regions))
        if key in ['production_all_regions', 'investment_all_regions', 'gross_output_all_regions',
                   'current_balance_all_regions', 'aggregate_consumption']:
            self.set_state(key, value=np.zeros(self.num_regions), norm=1e3)
        region_ids = range(self.num_regions)
        params = self.all_regions_params
        if key == 'intensity_all_regions':
            self.set_state(key, value=np.array([params[region]["xsigma_0"] for region in region_ids]), norm=1e-1)

        if key == 'production_factor_all_regions':
            self.set_state(key, value=np.array([params[region]["xA_0"] for region in region_ids]), norm=1e2, )

        if key == 'labor_all_regions':
            self.set_state(key, value=np.array([params[region]["xL_0"] for region in region_ids]), norm=1e4, )

        if key == 'capital_all_regions':
            self.set_state(key, value=np.array([params[region]["xK_0"] for region in region_ids]), norm=1e4, )

        if key == 'global_temperature':
            self.set_state(key, value=np.array([params[0]["xT_AT_0"], params[0]["xT_LO_0"]]), norm=1e1)

        if key == 'global_carbon_mass':
            self.set_state(key, value=np.array([params[0]["xM_AT_0"], params[0]["xM_UP_0"], params[0]["xM_LO_0"]]), norm=1e4)

        # num_regions x num_regions matrices
        if key in ['proposal_decisions', 'requested_mitigation_rate', 'promised_mitigation_rate']:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)))
        if key in ['imports_minus_tariffs', 'desired_imports', 'import_tariffs', 'tariffs',
                   'normalized_import_bids_all_regions']:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)), norm=1e2)

    def get_state(self, key=None, region_id=None, timestep=None):
        assert key in self.global_state, f"Invalid key '{key}' in global state!"
        if timestep is None:
            timestep = self.current_timestep
        if region_id is None:
            return self.global_state[key]["value"][timestep].copy()
        return self.global_state[key]["value"][timestep, region_id].copy()

    def get_prev_state(self, key, region_id=None):
        return self.get_state(
            key, region_id=region_id, timestep=self.current_timestep - 1,
        )

    def set_state(self,
        key=None,
        value=None,
        timestep=None,
        norm=None,
        region_id=None,
        dtype=None,
    ):
        """
        Set a specific slice of the environment global state with a key and value pair.
        The value is set for a specific timestep, and optionally, a specific region_id.
        Optionally, a normalization factor (used for generating observation),
        and a datatype may also be provided.
        """
        assert key is not None
        assert value is not None
        if timestep is None:
            timestep = self.current_timestep
        if norm is None:
            norm = 1.0
        if dtype is None:
            dtype = self.float_dtype

        if isinstance(value, list):
            value = np.array(value, dtype=dtype)
        elif isinstance(value, (float, np.floating)):
            value = np.array([value], dtype=self.float_dtype)
        elif isinstance(value, (int, np.integer)):
            value = np.array([value], dtype=self.int_dtype)
        else:
            assert isinstance(value, np.ndarray)

        if key not in self.global_state:
            logging.info(f"Adding {key} to global state.")
            if region_id is None:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,)
                        + value.shape,
                        dtype=dtype
                    ),
                    "norm": norm,
                }
            else:
                self.global_state[key] = {
                    "value": np.zeros(
                        (self.episode_length + 1,)
                        + (self.num_regions,)
                        + value.shape,
                        dtype=dtype,
                    ),
                    "norm": norm,
                }

        # Set the value
        if region_id is None:
            self.global_state[key]["value"][timestep] = value
        else:
            self.global_state[key]["value"][timestep, region_id] = value

    def read_rice_param_yamls(self, yamls_folder=None):
        """Helper function to read yaml data and set environment configs."""
        assert yamls_folder is not None
        dice_params = self.read_yaml_data(str(os.path.join(yamls_folder, "default.yml")))
        file_list = sorted(os.listdir(yamls_folder))  #
        yaml_files = []
        for file in file_list:
            if file[-4:] == ".yml" and file != "default.yml":
                yaml_files.append(file)

        rice_params = []
        for file in yaml_files:
            rice_params.append(self.read_yaml_data(os.path.join(yamls_folder, file)))

        # Overwrite rice params
        num_regions = len(rice_params)
        for k in dice_params["_RICE_CONSTANT"].keys():
            dice_params["_RICE_CONSTANT"][k] = [
                dice_params["_RICE_CONSTANT"][k]
            ] * num_regions
        for idx, param in enumerate(rice_params):
            for k in param["_RICE_CONSTANT"].keys():
                dice_params["_RICE_CONSTANT"][k][idx] = param["_RICE_CONSTANT"][k]

        return num_regions, dice_params

    def read_yaml_data(self, yaml_file):
        """Helper function to read yaml configuration data."""
        with open(yaml_file, "r", encoding="utf-8") as file_ptr:
            file_data = file_ptr.read()
        file_ptr.close()
        data = yaml.load(file_data, Loader=yaml.FullLoader)
        return data

    @staticmethod
    def flatten_array(array):
        """Flatten a numpy array"""
        return np.reshape(array, -1)

    def merge_to_regional_param_dict(self, world, regional):
        """
        This function merges the world params dict into the regional params dict.
        Inputs:
            world: global params, dict, each value is common to all regions.
            regional: region-specific params, dict,
                      length of the values should equal the num of regions.
        Outputs:
            outs: list of dicts, each dict corresponding to a region
                  and comprises the global and region-specific parameters.
        """
        vals = regional.values()
        assert all(
            len(item) == self.num_regions for item in vals
        ), "The number of regions has to be consistent!"

        outs = []
        for region_id in range(self.num_regions):
            out = world.copy()
            for key, val in regional.items():
                out[key] = val[region_id]
            outs.append(out)
        return outs
