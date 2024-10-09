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

### Adding / Modifying Actions

To add an action a number of