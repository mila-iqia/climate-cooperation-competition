from rice import *
import random

_FEATURES = "features"
_ACTION_MASK = "action_mask"

class WelfGain(Rice):

    """subset of prefs are high + added to observation also glo-fo-pref 
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="WelfGain"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.welfare_gain_per_unit_exported = .4


    def step_climate_and_economy(self, actions=None):
        self.calc_activity_timestep()
        self.is_valid_negotiation_stage(negotiation_stage=0)
        self.is_valid_actions_dict(actions)

        actions_dict = {
            "savings_all_regions" : self.get_actions("savings", actions),
            "mitigation_rates_all_regions" : self.get_actions("mitigation_rate", actions),
            "export_limit_all_regions" : self.get_actions("export_limit", actions),
            "import_bids_all_regions" : self.get_actions("import_bids", actions),
            "import_tariffs_all_regions" : self.get_actions("import_tariffs", actions),
        }

        self.set_actions_in_global_state(actions_dict)

        damages = self.calc_damages()
        abatement_costs = self.calc_abatement_costs(actions_dict["mitigation_rates_all_regions"])
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
        welfloss_multipliers = self.calc_welfloss_multiplier(gross_outputs, gross_imports, net_imports)
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


    def calc_welfloss_multiplier(self, gross_outputs, gross_imports, net_imports, welfare_loss_per_unit_tariff=None, save_state=True):
        """Calculate the welfare loss multiplier of exporting region due to being tariffed."""
        if not self.apply_welfloss:
            return np.ones((self.num_regions), dtype=self.float_dtype)

        if welfare_loss_per_unit_tariff is None:
            welfare_loss_per_unit_tariff = 0.4 # From Nordhaus 2015

        import_tariffs = self.get_prev_state("import_tariffs_all_regions")
        welfloss = np.ones((self.num_regions), dtype=self.float_dtype)

        for region_id in range(self.num_regions):
            for destination_region in range(self.num_regions):
                welfloss[region_id] -= \
                    (gross_imports[destination_region, region_id] / gross_outputs[region_id]) * \
                        import_tariffs[destination_region, region_id] * welfare_loss_per_unit_tariff
                
                welfloss[region_id] += \
                    (net_imports[destination_region, region_id]) / gross_outputs[region_id] * self.welfare_gain_per_unit_exported

                
        if save_state:
            self.set_state("welfloss", welfloss)

        return welfloss

class WelfGainNoTariff(WelfGain):
    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="WelfGainNoTariff"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()



                
            #no tariffs
            tariff_mask = [0] * self.num_discrete_action_levels * self.num_regions
            

            #mask tariff
            tariffs_mask_start = sum(self.savings_possible_actions
                                    + self.mitigation_rate_possible_actions
                                    + self.export_limit_possible_actions)
            tariff_mask_end = sum(self.calc_possible_actions("import_tariffs")) + tariffs_mask_start
            mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)


            mask_dict[region_id] = mask
            
        return mask_dict

    

    
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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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

class SubsetPrefsAndExport(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="SubsetPrefsAndExport"  
            ):
        self.global_state = {}
        self.exp_level = 8
        self.subset_size = 5
        self.set_all_region_params()
        self.num_regions = len(self.all_regions_params)
        self.num_agents = self.num_regions
        self.subset = random.sample(range(self.num_regions), self.subset_size)
        self.non_subset_pref = .01
        self.set_discrete_action_levels(num_discrete_action_levels)
        self.set_dtypes()


        
        
        self.set_trade_params()
          # for env wrapper


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


    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1.0
        self.preference_for_domestic = 0.1
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

    def calc_uniform_foreign_preferences(self):

        subset_pref = (1 - self.preference_for_domestic \
            - (self.num_regions - self.subset_size)*self.non_subset_pref)/self.subset_size
        return [subset_pref if idx in self.subset else self.non_subset_pref 
            for idx in range(self.num_regions)]
    
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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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

        #add preferences
        one_hot_prefs = np.array([0 if idx not in self.subset else 1 for idx in range(self.num_regions)])
        shared_features = np.append(shared_features,one_hot_prefs)


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

    def calc_action_mask(self):
        """
        Generate action masks.
        """

        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            mask_start = sum(self.savings_possible_actions \
                    + self.mitigation_rate_possible_actions)
            mask_end = mask_start + sum(self.calc_possible_actions("export_limit"))

            #high export for subset
            if region_id in self.subset:
                regional_export_mask = [0]*self.exp_level + [1]*(self.num_discrete_action_levels-self.exp_level)
            else:
                regional_export_mask = [1]*(self.num_discrete_action_levels-self.exp_level) + [0]*self.exp_level 

            mask[mask_start:mask_end] = np.array(regional_export_mask)

            mask_dict[region_id] = mask
            
        return mask_dict

class HighExpForPref(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="HighExpForPref"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()
        self.exp_level = 8

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1.0
        self.preference_for_domestic = 0.2
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
    
    def calc_action_mask(self):
        """
        Generate action masks.
        """

        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            
            mask_start = sum(self.savings_possible_actions \
                + self.mitigation_rate_possible_actions)
            mask_end = mask_start + sum(self.calc_possible_actions("export_limit"))
            regional_export_mask = [0]*self.exp_level + [1]*(self.num_discrete_action_levels-self.exp_level)
            mask[mask_start:mask_end] = np.array(regional_export_mask)

            mask_dict[region_id] = mask
            
        return mask_dict
    
class SubsetGlobalPrefs(Rice):

    """subset of prefs are high + added to observation also glo-fo-pref 
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="SubsetGlobalPrefs"  
            ):
        self.global_state = {}
        self.exp_level = 8
        self.subset_size = 5
        self.set_all_region_params()
        self.num_regions = len(self.all_regions_params)
        self.num_agents = self.num_regions
        self.subset = random.sample(range(self.num_regions), self.subset_size)
        self.non_subset_pref = .01
        self.set_discrete_action_levels(num_discrete_action_levels)
        self.set_dtypes()


        
        
        self.set_trade_params()
          # for env wrapper


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


    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1
        self.preference_for_domestic = 0.1
        self.preference_for_imported = 1-self.preference_for_domestic
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
            c_for_pref = self.preference_for_imported * np.sum(
               pow(net_imports[region_id, :], self.consumption_substitution_rate)
            )

            print(c_dom_pref, c_for_pref)

            consumptions[region_id] = (c_dom_pref + c_for_pref) ** (
                1 / self.consumption_substitution_rate
            )  # CES function
            # TODO: fix for region-specific state saving
            if save_state:
                self.set_state("aggregate_consumption", consumptions[region_id], region_id=region_id)
        return consumptions

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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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

        #add preferences
        one_hot_prefs = np.array([0 if idx not in self.subset else 1 for idx in range(self.num_regions)])
        shared_features = np.append(shared_features,one_hot_prefs)


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

class SubsetPrefs(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="SubsetPrefs"  
            ):
        self.global_state = {}
        self.exp_level = 8
        self.subset_size = 5
        self.set_all_region_params()
        self.num_regions = len(self.all_regions_params)
        self.num_agents = self.num_regions
        self.subset = random.sample(range(self.num_regions), self.subset_size)
        self.non_subset_pref = .01
        self.set_discrete_action_levels(num_discrete_action_levels)
        self.set_dtypes()


        
        
        self.set_trade_params()
          # for env wrapper


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


    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1.0
        self.preference_for_domestic = 0.1
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

    def calc_uniform_foreign_preferences(self):

        subset_pref = (1 - self.preference_for_domestic \
            - (self.num_regions - self.subset_size)*self.non_subset_pref)/self.subset_size
        return [subset_pref if idx in self.subset else self.non_subset_pref 
            for idx in range(self.num_regions)]
    
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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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

        #add preferences
        one_hot_prefs = np.array([0 if idx not in self.subset else 1 for idx in range(self.num_regions)])
        shared_features = np.append(shared_features,one_hot_prefs)


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

class SubsetExportGlobalPrefs(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="SubsetExportGlobalPrefs"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()
        self.exp_level = 8
        self.subset_size = 5
        self.subset = random.sample(range(self.num_regions), self.subset_size)

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1
        self.preference_for_domestic = 0.1
        self.preference_for_imported = 1-self.preference_for_domestic
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
            c_for_pref = self.preference_for_imported * np.sum(
               pow(net_imports[region_id, :], self.consumption_substitution_rate)
            )

            print(c_dom_pref, c_for_pref)

            consumptions[region_id] = (c_dom_pref + c_for_pref) ** (
                1 / self.consumption_substitution_rate
            )  # CES function
            # TODO: fix for region-specific state saving
            if save_state:
                self.set_state("aggregate_consumption", consumptions[region_id], region_id=region_id)
        return consumptions

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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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

        #add preferences
        one_hot_prefs = np.array([0 if idx not in self.subset else 1 for idx in range(self.num_regions)])
        shared_features = np.append(shared_features,one_hot_prefs)


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
    
    def calc_action_mask(self):
        """
        Generate action masks.
        """

        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            mask_start = sum(self.savings_possible_actions \
                    + self.mitigation_rate_possible_actions)
            mask_end = mask_start + sum(self.calc_possible_actions("export_limit"))

            #high export for subset
            if region_id in self.subset:
                regional_export_mask = [0]*self.exp_level + [1]*(self.num_discrete_action_levels-self.exp_level)
            else:
                regional_export_mask = [1]*(self.num_discrete_action_levels-self.exp_level) + [0]*self.exp_level 

            mask[mask_start:mask_end] = np.array(regional_export_mask)
            mask_dict[region_id] = mask
            
        return mask_dict

class SubsetExport(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="SubsetExport"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()
        self.exp_level = 8
        self.subset_size = 5
        self.subset = random.sample(range(self.num_regions), self.subset_size)

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1.0
        self.preference_for_domestic = 0.2
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
            "timestep",
        ]

        # Public features that are observable by all regions
        public_features = [
            "gross_output_all_regions",
            "investment_all_regions",
            "export_limit_all_regions",
            "current_balance_all_regions",
            "tariffs",
        ]

        # Private features that are private to each region.
        private_features = [
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
    
    def calc_action_mask(self):
        """
        Generate action masks.
        """

        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            mask_start = sum(self.savings_possible_actions \
                    + self.mitigation_rate_possible_actions)
            mask_end = mask_start + sum(self.calc_possible_actions("export_limit"))

            #high export for subset
            if region_id in self.subset:
                regional_export_mask = [0]*self.exp_level + [1]*(self.num_discrete_action_levels-self.exp_level)
            else:
                regional_export_mask = [1]*(self.num_discrete_action_levels-self.exp_level) + [0]*self.exp_level 

            mask[mask_start:mask_end] = np.array(regional_export_mask)
            mask_dict[region_id] = mask
            
        return mask_dict
    


class MaxTrade(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="MaxTrade"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 0.95
        self.preference_for_domestic = 0.1
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
    
class MaxTradeGlobalPreferences(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="MaxTradeGlobalPreferences"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 1
        self.preference_for_domestic = 0.1
        self.preference_for_imported = 1-self.preference_for_domestic
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
            c_for_pref = self.preference_for_imported * np.sum(
               pow(net_imports[region_id, :], self.consumption_substitution_rate)
            )

            consumptions[region_id] = (c_dom_pref + c_for_pref) ** (
                1 / self.consumption_substitution_rate
            )  # CES function
            # TODO: fix for region-specific state saving
            if save_state:
                self.set_state("aggregate_consumption", consumptions[region_id], region_id=region_id)
        return consumptions

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

class MinTrade(Rice):

    """Scenario where agents have a high preference for foreign consumption
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 

        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="MinTrade"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.set_trade_params()

    def set_trade_params(self):
        # TODO : add to yaml

        self.init_capital_multiplier = 10.0
        self.balance_interest_rate = 0.1
        self.consumption_substitution_rate = 0.9
        self.preference_for_domestic = 0.8
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

    

class OptimalMitigation(Rice):

    """Scenario where all agents mitigate to a given extent
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 
    
        Attributes:
        - maximum_mitigation_rate: the rate rate all agents will mitigate to.
        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="OptimalMitigation"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.maximum_mitigation_rate = 9

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()

            mitigation_mask = np.array(
                    [0 for _ in range(self.maximum_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels
                            - self.maximum_mitigation_rate
                        )
                    ]
                )

            mask_start = self.get_actions_index("mitigation_rate")
            mask_end = mask_start + sum(self.calc_possible_actions("mitigation_rate"))
            mask[mask_start:mask_end] = mitigation_mask
            mask_dict[region_id] = mask

        return mask_dict
    
class BasicClub(Rice):

    """Scenario where a subset of regions mitigate to a set amount, 
        other agents get a tariff based on the difference between their mitigation rate
        and the club rate
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 
    
        Attributes:
        - club_mitigation_rate: the rate rate all agents will mitigate to.
        - club_members: subset of states in club
        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="BasicClub"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.club_mitigation_rate = 7
        #Note: this will be updated later with more targeted region_ids
        self.club_members = [1,2,3,4,15,8,12]

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()

            #club members mitigate
            if region_id in self.club_members:
                mask = self.default_agent_action_mask.copy()
                #mask mitigation
                mitigation_mask = np.array(
                        [0 for _ in range(self.club_mitigation_rate)]
                        + [
                            1
                            for _ in range(
                                self.num_discrete_action_levels
                                - self.club_mitigation_rate
                            )
                        ]
                    )

                mitigation_mask_start = sum(self.savings_possible_actions)
                
                mitigation_mask_end = mitigation_mask_start + sum(self.calc_possible_actions("mitigation_rate"))
                mask[mitigation_mask_start:mitigation_mask_end] = mitigation_mask
                
                #tariff non club members
                tariff_mask = []
                for other_region_id in range(self.num_regions):
                    # if other region is self or in club
                    if (other_region_id == region_id) or (other_region_id in self.club_members):
                        # minimize tariff for free trade
                        regional_tariff_mask = [1] + [0] * (self.num_discrete_action_levels-1)
                    else:
                        other_region_mitigation_rate = self.get_state("mitigation_rates_all_regions",
                                                                       region_id=other_region_id)
                        #min tariff by difference between mitigation rate and club mitigation rate
                        tariff_rate = int(self.club_mitigation_rate - other_region_mitigation_rate)
                        regional_tariff_mask = [0] * tariff_rate \
                            + [1] * (self.num_discrete_action_levels-tariff_rate)
                    tariff_mask.extend(regional_tariff_mask)

                #mask tariff
                tariffs_mask_start = sum(self.savings_possible_actions
                                        + self.mitigation_rate_possible_actions
                                        + self.export_limit_possible_actions)
                tariff_mask_end = sum(self.calc_possible_actions("import_tariffs")) + tariffs_mask_start
                mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)


            mask_dict[region_id] = mask
            
        return mask_dict
    


    
class ExportAction(Rice):

    """Scenario where a subset of regions mitigate to a set amount, 
        other agents get a tariff based on the difference between their mitigation rate
        and the club rate
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 
    
        Attributes:
        - club_mitigation_rate: the rate rate all agents will mitigate to.
        - club_members: subset of states in club
        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="ExportAction"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        #Note: this will be updated later with more targeted region_ids
        
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
        
        if action_type == "export_regions":
            return [2] * self.num_regions
        
    def calc_total_possible_actions(self, negotiation_on):

        total_possible_actions = (
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.export_regions_possible_actions
            )

        if negotiation_on:
            total_possible_actions += (
                self.proposal_possible_actions
                + self.evaluation_possible_actions
            )

        return total_possible_actions

    def set_possible_actions(self):

        super().set_possible_actions()
        self.export_regions_possible_actions = self.calc_possible_actions(
            "export_regions"
        )

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
        
        if action_type == "export_regions":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
                + len(self.export_regions_possible_actions)
            )

        if action_type == "proposal":
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.export_limit_possible_actions
            )

        if action_type == 'proposal_decisions':
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.export_limit_possible_actions
                + self.proposal_possible_actions
            )
        
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
        
        if action_type == "export_regions":
            export_regions_index_start = self.get_actions_index("export_regions")
            num_export_regions_actions = len(self.export_regions_possible_actions)

            export_regions = np.array(
                [
                    actions[region_id][
                        export_regions_index_start : export_regions_index_start 
                        + num_export_regions_actions
                    ]
                    for region_id in range(self.num_regions)
                ]
            )

            for region_id in range(self.num_regions):
                export_regions[region_id, region_id] = 0
            return export_regions
        
    def set_actions_in_global_state(self, actions_dict):
        for (action_name, action_value) in actions_dict.items():
            self.set_state(
                key=action_name,
                value=action_value,
                timestep=self.current_timestep,
                dtype=self.float_dtype,
            )

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
        if key in ['proposal_decisions','export_regions_all_regions', 'requested_mitigation_rate', 'promised_mitigation_rate']:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)))
        if key in ['imports_minus_tariffs', 'desired_imports', 'import_tariffs', 'tariffs',
                   'normalized_import_bids_all_regions']:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)), norm=1e2)

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
        self.reset_state('export_regions_all_regions')
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
            "export_regions_all_regions" : self.get_actions("export_regions", actions)
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
        self.export_regions_possible_actions = self.calc_possible_actions(
            "export_regions"
        )

        if self.negotiation_on:
            self.proposal_possible_actions = self.calc_possible_actions(
                "proposal"
            )
            self.evaluation_possible_actions = self.calc_possible_actions(
                'proposal_decisions'
            )
        
    def get_importable_regions(self, region_id):
        """
        Get the output of the export_region action for a given region
        """
        
        export_regions = self.get_state("export_regions_all_regions")
        open_for_trade = export_regions[:, region_id]
        return open_for_trade

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

        if action_type == "export_regions":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
                + len(self.import_tariff_possible_actions)
            )
 

        if action_type == "proposal":
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.export_regions_possible_actions
            )

        if action_type == 'proposal_decisions':
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.export_regions_possible_actions
                + self.proposal_possible_actions
            )




    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()
            open_for_trade = self.get_importable_regions(region_id)

            imports_mask = []

            for other_region in range(self.num_regions):
                if other_region != region_id:
                    if open_for_trade[other_region] == 1:
                        imports_mask.extend([1]*self.num_discrete_action_levels)
                    else:
                        imports_mask.extend([0]*self.num_discrete_action_levels)
                else:
                    imports_mask.extend([0]*self.num_discrete_action_levels)
            mask_dict[region_id] = mask
            
            mask_start = sum(self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions)

            mask_end = mask_start + sum(self.calc_possible_actions("import_bids"))
            mask[mask_start:mask_end] = np.array(imports_mask)

        return mask_dict
    
class TariffTest(ExportAction):

    """Scenario where a subset of regions mitigate to a set amount, 
        other agents get a tariff based on the difference between their mitigation rate
        and the club rate
    
        Arguments:
        - num_discrete_action_levels (int):  the number of discrete levels for actions, > 1
        - negotiation_on (boolean): whether negotiation actions are available to agents
        - scenario (str): name of scenario 
    
        Attributes:
        - club_mitigation_rate: the rate rate all agents will mitigate to.
        - club_members: subset of states in club
        """
    

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="TariffTest"  
            ):
        super().__init__(num_discrete_action_levels,negotiation_on,scenario)
        self.tariff_rate = 9
        #Note: this will be updated later with more targeted region_ids

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
        self.reset_state('export_regions_all_regions')
        self.reset_state('export_limit_all_regions')

        # negotiation states
        self.reset_state('negotiation_stage')
        self.reset_state('savings_all_regions')
        self.reset_state('minimum_mitigation_rate_all_regions')
        self.reset_state('promised_mitigation_rate')
        self.reset_state('requested_mitigation_rate')
        self.reset_state('proposal_decisions')

        self.target_region = np.random.randint(low=0, high=self.num_agents)
        self.tariff_begin_year = np.random.randint(low=self.start_year+10, high=self.end_year-10)

        info = { region:{} for region in range(self.num_regions)}  # for the new ray rllib env format
        return self.get_observations(), info

    def calc_action_mask(self):
        """
        Generate action masks.
        """

        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()

            open_for_trade = self.get_importable_regions(region_id)

            imports_mask = []

            for other_region in range(self.num_regions):
                if other_region != region_id:
                    if open_for_trade[other_region] == 1:
                        imports_mask.extend([1]*self.num_discrete_action_levels)
                    else:
                        imports_mask.extend([0]*self.num_discrete_action_levels)
                else:
                    imports_mask.extend([0]*self.num_discrete_action_levels)
            
            
            mask_start = sum(self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions)
            mask_end = mask_start + sum(self.calc_possible_actions("import_bids"))
            mask[mask_start:mask_end] = np.array(imports_mask)

            #only tariff if after a random tariff year
            if region_id == self.target_region and self.current_simulation_year >= self.tariff_begin_year:
                mask = self.default_agent_action_mask.copy()
                #tariff all importers
                tariff_mask = []
                for other_region_id in range(self.num_regions):
                    # Do not tariff self
                    if (other_region_id == region_id) :
                        # minimize tariff for free trade
                        regional_tariff_mask = [1] + [0] * (self.num_discrete_action_levels-1)
                    else:
                        #tariff to rate.
                        regional_tariff_mask = [0] * self.tariff_rate \
                            + [1] * (self.num_discrete_action_levels-self.tariff_rate)
                    tariff_mask.extend(regional_tariff_mask)

                #mask tariff
                tariffs_mask_start = sum(self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_tariff_possible_actions)
                tariff_mask_end = sum(self.calc_possible_actions("import_tariffs")) + tariffs_mask_start
                mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask
            
        return mask_dict