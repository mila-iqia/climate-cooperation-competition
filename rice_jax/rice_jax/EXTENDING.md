# Extending Rice-N-Jax (Scenarios)

This file contains a brief guide on how to create extensions (scenarios) to Rice-N-Jax. Typically, you simply suclass the base `Rice` class and overwrite/add/remove any code you see fit. This overview lists what some commonly overwritten methods are expected to return, as it differs from how Rice-N-Jax behaved previously.

## Action Space

The action space is now simply a container (PyTree) of Spaces, where the first level of this container indicates the agents:

```python

N_DISCRETIZATION = num_discrete_action_levels

action_space = {
	"region_01": {
            "import_bid": MultiDiscrete([N_DISCRETIZATION] * N_REGIONS),
            "import_tariff": MultiDiscrete([N_DISCRETIZATION] * N_REGIONS),
            "savings_rate": Discrete(N_DISCRETIZATION),
            "mitigation_rate": Discrete(N_DISCRETIZATION),
            "export_limit": Discrete(N_DISCRETIZATION),
        },
	"region_02: {
		…
	}
	...
}

```

The action spaces in standard Rice are homogeneous, but in theory extensions allow for different action spaces per agent. Furthermore, we now define the action space as a dictionary, but they do not have to be. They could be Tuples, Lists, or any user-defined pytree or even nested structures -- as long as the "first-level" of the container is the agent axis. Typically a dictionary is the easiest and clearest to work with though.

### Using actions in the environment

Actions passed to the `step()` (or `step_env()`) function are passed as the same structure as the action space:

```python
# Example 
# n_regions = 3

def step_env(self, key: chex.PRNGKey, prev_state: dict, actions: dict):
	print(actions)
	# ///// example:
	# actions = {
    #   "region_01": {
    #       "import_bid": np.array([3, 2, 8]),
    #       "import_tariff": np.array([3, 1, 5]),
    #       "savings_rate": 8,
    #       "mitigation_rate": 5,
    #       "export_limit": 4,
    #   },
	#   "region_02: {
	#	    …
	#   }
	# ...
	# }
```

### Processing actions

In the step function, `process_actions()` is still called. This used to be an unclear annoying function to get everything in the right shape. Now it is still there for the following:

1. Dividing all the actions by the num_discrete_action_levels. I.e.: 8 --> 0.8
2. Tranposing the action dict from {agent1: {action1: value, action2: value, ...}, agent2: ...} to {action1: {agent1: value, agent2: value, ...}, action2: ...}
3. Each action is then still stacked into an array: {action1: Array([agent1_action1, agent2_action1, ...]), action2: ...}. This is mostly so that you can easily do vectorized operations down the line instead of constant for loops.


### How to alter

In a subclass, just overwrite the action space.

```python

class SomeScenario(Rice):

…

    @property
    def action_space(self):
        action_space = super().action_space # Get standard action_space
        
        # Add a new action to each agent
        for agent_id in range(self.num_regions):
            action_space[
                i_to_agent_str(agent_id)
            ]["new_action"] = Discrete(10)

        return action_space

```

Again, it's perfectly possible to only add actions to only some agents. Removing actions here is also possible. But note that the standard actions are used in the `step_climate_and_economy()` function, so removing actions requires some additional work.

#### Action masking

If you add or remove actions from the action space, the `generate_action_masks` function should additionally be updated to reflect the changes. This function should just output the same structure as the action_space but with a boolean array for all available (1) and unavailble (0) actions.


## Observation Space

Similar to the action space, the observation space should output a container where the first level is the agent axis. Note that this first-level container structure must be exactly the same as the output of the action_space (as well as the reward function).

 If you return additional features (or less features) from the `generate_observation()` function, the models will automatically pick that up.

```python

class SomeScenario(Rice):

…

def generate_observation(self, state: dict[str, Any]):
	observations = super().generate_observation(state) # Get default obs

	# print(observations)
	# {agent1: {feature1: int, feature2: array, feature3: float}, agent2: {...}, ...}

	# Update obs and return
    # Example: give agent0 an additional random observation
    seed = jax.random.seed(0)
    observations[i_to_agent_str(0)] = jax.random.normal(seed)

    return oberservations

```

As you may notice, `generate_observation`, can  output a nested dict of observations (with the top-level of the dict being the agent axis). This nested dict structure would work fine, but to simplify the network the observations are automatically concatenated into a single array (per agent) in `generate_observation_and_action_mask`. This happens after the `generate_observation()` call, so you typically do not need to worry about this.

> Note `generate_observation_and_action_mask` also flattens each array and makes sure they are 1d. If you nééd (partly) 2d observations that should, for example, be processed by a CNN, you should tweak the `generate_observation_and_action_mask` function.

### What about `def observation_space(self)`

The observation_space property of the environment simply calls `step()` and infers the shape of the observation space from the output of that call. As such, you should typically require no additional setup here.

## State space

The state space has been moved from a dataclass to a simple dictionary, as was already the case in Numpy Rice. This loses some type safety but extending the state space was quite cumbersome with a dataclass. Perhaps important to note here is that, unlike the obs/action space, the state space is not a per-agent dict, but a per-feature dict, just like Numpy Rice. 

Features that contain information for each agent are still simple arrays as it was before. This is because you are likely to operate on all these features in a vectorized operation and then do not need to loop over each agent feature.

```
# example
state = {
	"current_timestep": 0,
	"production_all_regions": jnp.array(...) # shape: (self.num_regions,)
    ...
}
```

There aren't any limitations on what features can be, you can just add features and make sure that `step_env()` and `reset_env()` returns them in the state variable.
Ensure that both functions return the exact some structure of the state (all keys must match).

## Other shape requirements

The reward function should also output a container with the same top-level agent axis as the action/observation space.

```python

def generate_rewards(self, new_state: dict, old_state: dict) -> dict[str, float]:
    reward = new_state["utility_times_welfloss_all_regions"]
    return {i_to_agent_str(i): reward[i] for i in range(self.num_regions)} # Return dict of reward per agent

```

