# Make a new scenario

A scenario is a subclass of the `RICE` class. This allows us to work with different variations in a configurable way. So if you have an idea for a new scenario, simply add it to the `scenarios.py` . 

For example:

```jsx
class TestScenario(Rice):
    """
    This is an example of a different scenario. 
    It has all the same properties and methods of Rice
    """

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=True, # If True then negotiation is on, else off
                 scenario="TestScenario",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None,
                 pct_reward=False,
                 clubs_enabled = False,
                 club_members = [],
                 action_window = True,
                 relative_reward = False
            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions,
                pct_reward=pct_reward,
                clubs_enabled = clubs_enabled,
                club_members = club_members,
                action_window = action_window,
                relative_reward=relative_reward)
```

Note, the class parameters correspond to the “env” component of the config yaml. 

### Modifying attributes:

Lets say that we want to have a scenario that disables welfloss. Then simply add it to the init after the `super()` call.

For example:

```jsx
 class TestScenario(Rice):
    """
    This is an example of a different scenario. 
    It has all the same properties and methods of Rice
    """

    def __init__(self,
                 num_discrete_action_levels=10,  # the number of discrete levels for actions, > 1
                 negotiation_on=True, # If True then negotiation is on, else off
                 scenario="BasicClubOptIn",
                 action_space_type="discrete",  # or "continuous"
                 dmg_function="base",
                 carbon_model="base",
                 temperature_calibration="base",
                 prescribed_emissions=None,
                 pct_reward=False,
                 clubs_enabled = False,
                 club_members = [],
                 action_window = True,
                 relative_reward = False
            ):
        super().__init__(negotiation_on=negotiation_on,  # If True then negotiation is on, else off
                scenario=scenario,
                num_discrete_action_levels=num_discrete_action_levels, 
                action_space_type=action_space_type,  # or "continuous"
                dmg_function=dmg_function,
                carbon_model=carbon_model,
                temperature_calibration=temperature_calibration,
                prescribed_emissions=prescribed_emissions,
                pct_reward=pct_reward,
                clubs_enabled = clubs_enabled,
                club_members = club_members,
                action_window = action_window,
                relative_reward=relative_reward)
        
        #disable welfloss
        self.apply_welfloss = False
```

Then if you want to train a model with this variety, you’ll need to do 2 things:

**Add scenario to config in** `rice_rllib_discrete.yaml` .

```jsx
# Environment configuration
env:
    negotiation_on: True # flag to indicate whether negotiation is allowed or not
    scenario: "TestScenario" # key that maps to either Rice or and alternate Rice class
    num_discrete_action_levels: 10
    action_space_type: "discrete"
    dmg_function: "base"
    carbon_model: "base"
    temperature_calibration: "base"
    pct_reward: False
    clubs_enabled: True
    club_members: []
    action_window: True
    relative_reward: False
    ....
```

**Add the class name to** `train_with_rllib.py` .

```jsx
#scenarios
SCENARIO_MAPPING = {
    "default":Rice,
    "OptimalMitigation":OptimalMitigation,
    "MinimalMitigation":MinimalMitigation,
    "BasicClub":BasicClub,
		"TestScenario":TestScenario
}
.....
class EnvWrapper(MultiAgentEnv):
    """
    The environment wrapper class.
    """

    def __init__(self, env_config=None):
        super().__init__()

        env_config_copy = env_config.copy()
        if env_config_copy is None:
            env_config_copy = {}
        source_dir = env_config_copy.get("source_dir", None)
        if "source_dir" in env_config_copy:
            del env_config_copy["source_dir"]
        if source_dir is None:
            source_dir = PUBLIC_REPO_DIR
        assert isinstance(env_config_copy, dict)
        self.env = SCENARIO_MAPPING[env_config["scenario"]](**env_config_copy)
```

The `SCENARIO_MAPPING` gets invoked during the creation of the environment wrapper. If you’ve saved the yaml config, then the scenario will be passed to the wrapper via the `env_config` . 

Begin training by executing `python scripts/train_with_rllib.py` 

### Adding / Modifying Steps and Actions

Adding action is more involved than simply modifying attributes. 

*Adding a step*

Actions can be found in the step function:

```jsx
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

        return self.step_climate_and_economy(action
```

Note, if negotiation is on, then there are different steps before the main `step_climate_and_economy` . Lets say I want to add a step prior to the proposal stage called “opt”, with the idea that regions need to opt-in to negotiations before they can participate in them. Then I’d had it here:

```jsx
    def step(self, actions):
        self.current_timestep += 1
        self.set_state("timestep", self.current_timestep, dtype=self.int_dtype)

        self.set_current_global_state_to_past_global_state()

        if self.negotiation_on:
            self.set_negotiation_stage()

            if self.is_opt_stage():
                return self.**step_opt**(actions)

            if self.is_proposal_stage():
                return self.step_propose(actions)

            elif self.is_evaluation_stage():
                return self.step_evaluate_proposals(actions)

        return self.step_climate_and_economy(actions)
```

The env needs to keep track of the order of steps prior to the main climate and economy step. Furthermore, the number of steps in the total episode depends on the number of negotiation steps. So first reset the `num_negotiation_steps` and `episode length` after the `super()` in the subclass initiation.  

```jsx
				#extra negotiation stage added
        self.num_negotiation_stages=3
        #reset length
        self.set_episode_length(self.negotiation_on)
```

Each phase of negotiation has a step number counting up from 1. Verified by these functions

```jsx
    def is_evaluation_stage(self):
        return self.negotiation_stage == 3

    def is_proposal_stage(self):
        return self.negotiation_stage == 2
    
    def is_opt_stage(self):
        return self.negotiation_stage == 1
```

These checks are called prior to executing the step function. 

*Adding an action that corresponds to that step*

Update the possible actions function:

```jsx
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
            **if action_type == "opt":
                return [2]**
            if action_type == "proposal":
                return [self.num_discrete_action_levels] 
            if action_type == "proposal_decisions":
                return [2] * self.num_regions
```

Update the set_possible_actions function:

```jsx
def set_possible_actions(self):
        self.savings_possible_actions = self.calc_possible_actions("savings")
        self.mitigation_rate_possible_actions = self.calc_possible_actions(
            "mitigation_rate"
        )
        self.export_limit_possible_actions = self.calc_possible_actions("export_limit")
        self.import_bids_possible_actions = self.calc_possible_actions("import_bids")
        self.import_tariff_possible_actions = self.calc_possible_actions(
            "import_tariffs"
        )

        if self.negotiation_on:
            self.proposal_possible_actions = self.calc_possible_actions("proposal")
            self.opt_possible_actions = self.calc_possible_actions("opt")
            self.evaluation_possible_actions = self.calc_possible_actions(
                "proposal_decisions"
            )
```

Actions are essentially one long vector with subsets of the vector corresponding to different actions. So the start index of each action is managed by this function that also needs to be updated:

```jsx
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
        
        if action_type == "opt":
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
                + self.opt_possible_actions
            )

        if action_type == "proposal_decisions":
            return len(
                self.savings_possible_actions
                + self.mitigation_rate_possible_actions
                + self.export_limit_possible_actions
                + self.import_bids_possible_actions
                + self.import_tariff_possible_actions
                + self.opt_possible_actions
                + self.proposal_possible_actions
            )
```

Then you also need to update the function that extracts the action values from the action vector:

```jsx
def get_actions(self, action_type, actions):
        ......
        if action_type == "import_tariffs":
            tariffs_action_index = self.get_actions_index("import_tariffs")
            return [
                actions[region_id][
                    tariffs_action_index : tariffs_action_index + self.num_regions
                ]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]
        
        **if action_type == "opt":
            opt_action_index = self.get_actions_index("opt")
            return [
                actions[region_id][opt_action_index]
                for region_id in range(self.num_regions)
            ]**

        if action_type == "proposed_mitigation_rate":
            proposal_actions_index_start = self.get_actions_index("proposal")

            return [
                actions[region_id][proposal_actions_index_start]
                / self.num_discrete_action_levels
                for region_id in range(self.num_regions)
            ]
						....
```

Now that you have the ability to get values from an action and you’ve set up the step logic, you can actually add the meat to the step_opt function

```jsx
    def step_opt(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=1)
        self.is_valid_actions_dict(actions)

        **opt_ins = self.get_actions("opt", actions)
        self.set_state("opts", np.array(opt_ins))**

        observations = self.get_observations()
        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}

        return observations, rewards, terminateds, truncateds, info
```

Note, not much happens in this step other than extracting the “opt-ins / opt-outs” from the action vector. However those opt ins and outs can be used by later steps. For example, here only agents who have opted in have the ability to form climate agreements.

```jsx
def step_evaluate_proposals(self, actions=None):
        self.is_valid_negotiation_stage(negotiation_stage=3)
        self.is_valid_actions_dict(actions)

        proposal_decisions = self.get_actions("proposal_decisions", actions)
        **opts = self.get_state("opts")**
        for region_id in range(self.num_regions):
            **if opts[region_id] == 0:**
                proposal_decisions[region_id] = np.zeros(shape=proposal_decisions[region_id].shape)
            else:
                pass
        self.set_state("proposal_decisions", proposal_decisions)
        for region_id in range(self.num_regions):
            #only regions that opt in set mmr's
            **if opts[region_id] == 1:**
                min_mitigation = self.calc_mitigation_rate_lower_bound(region_id)
            else:
                min_mitigation = 0
            self.set_state(
                key="minimum_mitigation_rate_all_regions",
                 value= min_mitigation,
                  region_id= region_id
            )

        observations = self.get_observations()

        rewards = {region_id: 0.0 for region_id in range(self.num_regions)}
        terminateds = {region_id: 0 for region_id in range(self.num_regions)}
        terminateds["__all__"] = 0
        truncateds = {region_id: 0 for region_id in range(self.num_regions)}
        truncateds["__all__"] = 0
        info = {}
        return observations, rewards, terminateds, truncateds, info
```

Up at the top of the tutorial I’ve stated already how to train your particular subclass. So give it a go!

After training you can run:

`python scripts/evaluate_submission.py -r Submissions/ZIPFILE.zip`

The name of the zip is output in the console after training.