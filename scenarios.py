from rice import *
import random
from math import ceil
_FEATURES = "features"
_ACTION_MASK = "action_mask"

class CarbonLeakage(Rice):

    """
    Scenario to test whether carbon leakage occurs.

    Carbon leakage is an increase of emissions in one region as a 
    result of a policy to decrease emissions in another region

    We can check for carbon leakage at the policy level (do they change their mitigation rate)
    and at the emissions level (do their absolute emissions increase)

    Followup experiment
    - create a random club of a given minimum mitigation rate
    - run the rollout with the club and measure emissions of non-club members and measure mitigation rates of non-club members
    - reset the env and re-run the env without the club (self.control = True) and measure the same
    - compare the emissions / mitigation rates of the non-club members in the presence and absence of the club
    - NOTE: it may be that emissions need to be normalized w.r.t. emissions as carbon
    """

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=False, # If True then negotiation is on, else off
                 scenario="CarbonLeakage",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None
            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions)
        
        #if its the control group, don't apply the club rules
        
        self.control = False
        self.training = True
        self.minimum_mitigation_rate = 8
        self.club_size = ceil(self.num_regions/2)
        self.club_members = random.sample(range(0, self.num_regions + 1), self.club_size)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        #recreate club each time
        if self.training:
            self.club_members = random.sample(range(0, self.num_regions + 1), self.club_size)

        #during training, switch up control conditions
        if self.training:
            if random.uniform(0,1) < 0.3:
                self.control = True
            else:
                self.control = False
        return obs, info


    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()

            if region_id in self.club_members and not self.control:

                mitigation_mask = np.array(
                        [0 for _ in range(self.minimum_mitigation_rate)]
                        + [
                            1
                            for _ in range(
                                self.num_discrete_action_levels
                                - self.minimum_mitigation_rate
                            )
                        ]
                    )

                mask_start = sum(self.savings_possible_actions)
                mask_end = mask_start + sum(
                        self.mitigation_rate_possible_actions
                    )
                mask[mask_start:mask_end] = mitigation_mask
            else:
                pass
            mask_dict[region_id] = mask
            
        return mask_dict





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
                 scenario="OptimateMitigation",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None

            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions)
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
    
class MinimalMitigation(Rice):

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
                 scenario="MinimalMitigation",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None

            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions)
        self.maximum_mitigation_rate = 1

    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):
            mask = self.default_agent_action_mask.copy()

            mitigation_mask = np.array(
                    [1 for _ in range(self.maximum_mitigation_rate)]
                    + [
                        0
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

    """Scenario where regions propose minimum mitigation rates, that are either accepted or rejected
    agents commit to the maximum of their accepted mitigation rates
    agents impose 0 tariff on club members
    outside club members get tariffed proportional to the diff between their tariffs. 
    
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
                 negotiation_on=True, # If True then negotiation is on, else off
                 scenario="BasicClub",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None

            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions)

    def calc_possible_actions(self, action_type):
        if self.action_space_type == "discrete":
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
                return [self.num_discrete_action_levels] 

            if action_type == "proposal_decisions":
                return [2] * self.num_regions
            
    def get_actions(self, action_type, actions):
        if action_type == "savings":
            savings_actions_index = self.get_actions_index("savings")
            return [
                actions[region_id][savings_actions_index]
                / self.num_discrete_action_levels  # TODO: change this for savings levels?
                for region_id in range(self.num_regions)
            ]

        if action_type == "mitigation_rate":
            mitigation_rate_action_index = self.get_actions_index("mitigation_rate")
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
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposed_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")

            return [
                actions[region_id][proposal_actions_index_start]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]

        if action_type == "proposal_decisions":
            proposal_decisions_index_start = self.get_actions_index(
                "proposal_decisions"
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
            
    def step_propose(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        proposed_mitigation_rates = self.get_actions(
            "proposed_mitigation_rate", actions
        )
        self.set_state("proposed_mitigation_rate", np.array(proposed_mitigation_rates))

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info
    
    def reset_state(self, key):
        
        if key == "proposed_mitigation_rate":
            self.set_state(key, value=np.zeros(self.num_regions))
        else:
            super().reset_state(key)

    def reset(self, *, seed=None, options=None):


        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state("timestep")
        self.reset_state("activity_timestep")

        # climate states
        self.reset_state("global_temperature")
        self.reset_state("global_carbon_mass")
        self.reset_state("global_exogenous_emissions")
        self.reset_state("global_land_emissions")
        self.reset_state("intensity_all_regions")
        self.reset_state("mitigation_rates_all_regions")

        # additional climate states for carbon and temperature model
        self.reset_state("global_alpha")
        self.reset_state("global_carbon_reservoirs")
        self.reset_state("global_cumulative_emissions")
        self.reset_state("global_cumulative_land_emissions")
        self.reset_state("global_emissions")
        self.reset_state("global_acc_pert_carb_stock")
        self.reset_state('global_temperature_boxes')

        # economic states
        self.reset_state("production_all_regions")
        self.reset_state("gross_output_all_regions")
        self.reset_state("aggregate_consumption")
        self.reset_state("investment_all_regions")
        self.reset_state("capital_all_regions")
        self.reset_state("capital_depreciation_all_regions")
        self.reset_state("labor_all_regions")
        self.reset_state("production_factor_all_regions")
        self.reset_state("current_balance_all_regions")
        self.reset_state("abatement_cost_all_regions")
        self.reset_state("mitigation_cost_all_regions")
        self.reset_state("damages_all_regions")
        self.reset_state("utility_all_regions")
        self.reset_state("social_welfare_all_regions")
        self.reset_state("reward_all_regions")

        # trade states
        self.reset_state("tariffs")
        self.reset_state("import_tariffs")
        self.reset_state("normalized_import_bids_all_regions")
        self.reset_state("import_bids_all_regions")
        self.reset_state("imports_minus_tariffs")
        self.reset_state("export_limit_all_regions")
        self.reset_state('export_regions_all_regions')

        # negotiation states
        self.reset_state("negotiation_stage")
        self.reset_state("savings_all_regions")
        self.reset_state("minimum_mitigation_rate_all_regions")
        self.reset_state("proposed_mitigation_rate")
        self.reset_state("promised_mitigation_rate")
        self.reset_state("requested_mitigation_rate")
        self.reset_state("proposal_decisions")

        info = {
            region: {} for region in range(self.num_regions)
        }  # for the new ray rllib env format
        return self.get_observations(), info
    
    def calc_mitigation_rate_lower_bound(self, region_id):

        #get all proposed_mitigation rates
        current_proposals = self.global_state["proposed_mitigation_rate"]["value"][self.current_timestep]
        proposal_decisions = [
            self.global_state["proposal_decisions"]["value"][
                self.current_timestep, j, region_id
            ]
            for j in range(self.num_regions)
        ]


        #remove all rejected mitigation rates
        accepted_proposals = current_proposals*proposal_decisions
        max_prop = max(accepted_proposals)

        #return max of accepted
        return max(accepted_proposals)

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
        
    def calc_action_mask(self):
        """
        Generate action masks.
        """
        mask_dict = {region_id: None for region_id in range(self.num_regions)}
        for region_id in range(self.num_regions):

            mask = self.default_agent_action_mask.copy()


            #minimum commitment
            min_mitigation_rate = int(self.get_state("minimum_mitigation_rate_all_regions",
                            region_id=region_id,
                                timestep=self.current_timestep)*self.num_discrete_action_levels)
            
            #mask mitigation
            mitigation_mask = np.array(
                    [0 for _ in range(min_mitigation_rate)]
                    + [
                        1
                        for _ in range(
                            self.num_discrete_action_levels
                            - min_mitigation_rate
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

                #get other regions mitigation commitment
                other_mitigation_rate = self.get_state("minimum_mitigation_rate_all_regions",
                            region_id=other_region_id,
                                timestep=self.current_timestep)

                # if other region is self or in club
                if (other_region_id == region_id) or (other_mitigation_rate >=min_mitigation_rate):
                    # minimize tariff for free trade
                    regional_tariff_mask = [1] + [0] * (self.num_discrete_action_levels-1)
                else:
                    
                    #min tariff by difference between mitigation rate and club mitigation rate
                    tariff_rate = int(min_mitigation_rate - other_mitigation_rate)
                    regional_tariff_mask = [0] * tariff_rate \
                        + [1] * (self.num_discrete_action_levels-tariff_rate)
                tariff_mask.extend(regional_tariff_mask)

            #mask tariff
            tariffs_mask_start = self.get_actions_index("import_tariffs")
            tariff_mask_end = self.num_regions * self.num_discrete_action_levels + tariffs_mask_start
            mask[tariffs_mask_start:tariff_mask_end] = np.array(tariff_mask)


            mask_dict[region_id] = mask
            
        return mask_dict
    
class BasicClubFixedMembers(Rice):

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
        self.club_mitigation_rate = 9
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
                 scenario="ExportAction",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None

            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions)
        
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
        
        if action_type == "export_regions":
            return (
                len(self.savings_possible_actions)
                + len(self.mitigation_rate_possible_actions)
                + len(self.export_limit_possible_actions)
                + len(self.import_bids_possible_actions)
                + len(self.export_regions_possible_actions)
            )
        else:
            return super().get_actions_index(action_type)

        
    def get_actions(self, action_type, actions):

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
        else:
            return super().get_actions(action_type, actions)
        
    def set_actions_in_global_state(self, actions_dict):
        for (action_name, action_value) in actions_dict.items():
            self.set_state(
                key=action_name,
                value=action_value,
                timestep=self.current_timestep,
                dtype=self.float_dtype,
            )

    def reset_state(self, key):
        
        if key != "export_regions_all_regions":
            super().reset_state(key)
        else:
            self.set_state(key, value=np.zeros((self.num_regions, self.num_regions)))

    def reset(self, *, seed=None, options=None):


        self.current_timestep = 0
        self.activity_timestep = 0
        self.current_simulation_year = self.start_year
        self.reset_state("timestep")
        self.reset_state("activity_timestep")

        # climate states
        self.reset_state("global_temperature")
        self.reset_state("global_carbon_mass")
        self.reset_state("global_exogenous_emissions")
        self.reset_state("global_land_emissions")
        self.reset_state("intensity_all_regions")
        self.reset_state("mitigation_rates_all_regions")

        # additional climate states for carbon model
        self.reset_state("global_alpha")
        self.reset_state("global_carbon_reservoirs")
        self.reset_state("global_cumulative_emissions")
        self.reset_state("global_cumulative_land_emissions")
        self.reset_state("global_emissions")
        self.reset_state("global_acc_pert_carb_stock")

        # economic states
        self.reset_state("production_all_regions")
        self.reset_state("gross_output_all_regions")
        self.reset_state("aggregate_consumption")
        self.reset_state("investment_all_regions")
        self.reset_state("capital_all_regions")
        self.reset_state("capital_depreciation_all_regions")
        self.reset_state("labor_all_regions")
        self.reset_state("production_factor_all_regions")
        self.reset_state("current_balance_all_regions")
        self.reset_state("abatement_cost_all_regions")
        self.reset_state("mitigation_cost_all_regions")
        self.reset_state("damages_all_regions")
        self.reset_state("utility_all_regions")
        self.reset_state("social_welfare_all_regions")
        self.reset_state("reward_all_regions")

        # trade states
        self.reset_state("tariffs")
        self.reset_state("import_tariffs")
        self.reset_state("normalized_import_bids_all_regions")
        self.reset_state("import_bids_all_regions")
        self.reset_state("imports_minus_tariffs")
        self.reset_state("export_limit_all_regions")
        self.reset_state('export_regions_all_regions')

        # negotiation states
        self.reset_state("negotiation_stage")
        self.reset_state("savings_all_regions")
        self.reset_state("minimum_mitigation_rate_all_regions")
        self.reset_state("promised_mitigation_rate")
        self.reset_state("requested_mitigation_rate")
        self.reset_state("proposal_decisions")

        info = {
            region: {} for region in range(self.num_regions)
        }  # for the new ray rllib env format
        return self.get_observations(), info

    
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
            "global_carbon_reservoirs",
            "global_cumulative_emissions",
            "global_cumulative_land_emissions",
            "global_alpha",
            "global_emissions",
            "global_acc_pert_carb_stock",
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
            "export_regions_all_regions",
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
            region_indicator = np.zeros(self.num_regions, dtype=self.float_dtype)
            region_indicator[region_id] = 1

            all_features = np.append(region_indicator, shared_features)

            for feature in private_features:
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert (
                    np.isnan(all_features).sum() == 0
                ), f"NaN in the features: {feature}"
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
                assert self.global_state[feature]["value"].shape[1] == self.num_regions
                assert self.global_state[feature]["value"].shape[2] == self.num_regions
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
            open_for_trade = self.get_importable_regions(region_id)

            imports_mask = []

            for other_region in range(self.num_regions):
                if other_region != region_id:
                    if open_for_trade[other_region] == 1:
                        imports_mask.extend([1]*self.num_discrete_action_levels)
                    else:
                        imports_mask.extend([1] + [0]*(self.num_discrete_action_levels-1))
                else:
                    imports_mask.extend([1] + [0]*(self.num_discrete_action_levels-1))
            mask_dict[region_id] = mask
            
            mask_start = sum(self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions)

            mask_end = mask_start + sum(self.calc_possible_actions("import_bids"))
            mask[mask_start:mask_end] = np.array(imports_mask)

        return mask_dict
    
    def get_importable_regions(self, region_id):
        """
        Get the output of the export_region action for a given region
        """
        
        export_regions = self.get_state("export_regions_all_regions")
        open_for_trade = export_regions[:, region_id]
        return open_for_trade
    
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
            "export_regions_all_regions" : self.get_actions("export_regions", actions)
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
            gross_outputs, investments, gross_imports, net_imports
        )
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
