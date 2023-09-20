from gym.spaces import Box
from rice_cont import Rice  # Assuming the class is in rice.py
import numpy as np

def test_continuous_action_space():
    # 1. Create an instance of the Rice environment.
    env = Rice()

    # 2. Modify the action space of the environment to use the continuous Box space.
    env.action_space = {
        region_id: Box(low=0, high=1, shape=(len(env.actions_nvec),), dtype=np.float32)
        for region_id in range(env.num_regions)
    }

    # 3. Reset the environment (initialize necessary variables).
    env.reset()

    # 4. Sample an action from the modified action space.
    actions = {region_id: env.action_space[region_id].sample() for region_id in range(env.num_regions)}

    # 5. Call the step method with the sampled actions.
    obs, rew, done, info = env.step(actions)

# Execute the test function
test_continuous_action_space()
