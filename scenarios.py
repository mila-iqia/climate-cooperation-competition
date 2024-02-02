from rice import *

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

            mask_start = sum(self.savings_possible_actions)
            mask_end = mask_start + sum(
                    self.mitigation_rate_possible_actions
                )
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
                mitigation_mask_end = mitigation_mask_start + sum(
                        self.mitigation_rate_possible_actions
                    )
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
                tariffs_mask_start = self.get_actions_index("import_tariffs")
                tariff_mask_end = self.num_regions * self.num_discrete_action_levels + tariffs_mask_start
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
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
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
        
    def get_importable_regions(self, region_id):
        """
        Get the output of the export_region action for a given region
        """
        
        export_regions = self.get_state("export_regions_all_regions")
        open_for_trade=export_regions[:, region_id]
        return open_for_trade




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
            
            mask_start = sum(self.export_limit_possible_actions)
            mask_end = mask_start + sum(self.import_bids_possible_actions)
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
            mask_dict[region_id] = mask
            
            mask_start = sum(self.export_limit_possible_actions)
            mask_end = mask_start + sum(self.import_bids_possible_actions)
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
                tariffs_mask_start = self.get_actions_index("import_tariffs")
                tariff_mask_end = self.num_regions * self.num_discrete_action_levels + tariffs_mask_start
                mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)

            mask_dict[region_id] = mask
            
        return mask_dict
