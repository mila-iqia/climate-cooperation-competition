from gymnasium.spaces import MultiDiscrete
from rice_discrete import Rice  # Assuming the class is in rice.py
import numpy as np

def test_discrete_action_space():
    # 1. Create an instance of the Rice environment.
    env = Rice()

    # 2. Modify the action space of the environment to use the continuous Box space.
    env.action_space = {
        region_id: MultiDiscrete(nvec=env.total_possible_actions)
        for region_id in range(env.num_regions)
    }

    # 3. Reset the environment (initialize necessary variables).
    env.reset()

    # 4. Sample an action from the modified action space.
    actions = {region_id: env.action_space[region_id].sample() for region_id in range(env.num_regions)}

    # 5. Call the step method with the sampled actions.
    observations, rewards, terminateds, truncateds, info = env.step(actions)

# Execute the test function
test_discrete_action_space()