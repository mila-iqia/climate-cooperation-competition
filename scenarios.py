from rice_discrete import *

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
    

