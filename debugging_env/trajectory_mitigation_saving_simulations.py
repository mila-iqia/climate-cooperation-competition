import numpy as np
import wandb
import time
from rice import Rice
from copy import deepcopy
rice_env = Rice()
rice_env.reset()
total_actions = rice_env.total_possible_actions
regions = list(range(rice_env.num_regions))


class RiceAction:
    def __init__(self, dict):
        self.actions = np.zeros(len(total_actions))

        for k, v in dict.items():
            start_idx = rice_env.get_actions_index(k)
            end_idx = start_idx + rice_env.get_actions_len(k)
            self.actions[start_idx:end_idx] = v


def get_state_at_time_t(global_state, t, nb_region=27):
    state_dict = {}
    for k, v in global_state.items():
        value_shape = v["value"].shape
        if len(value_shape) == 2:
            all_timesteps, nb_item = value_shape
            if nb_item == 1:
                state_dict[k] = v["value"][t, 0]
            elif nb_item != nb_region:
                for i in range(nb_item):
                    state_dict[f"{k}_{i}"] = v["value"][t, i]
            else:
                for i in range(nb_item):
                    state_dict[f"{k}_region_{i}"] = v["value"][t, i]
        elif len(value_shape) == 3:
            all_timesteps, nb_item1, nb_item2 = value_shape
            if nb_item2 == 1:
                for i in range(nb_item1):
                    state_dict[f"{k}_region_{i}"] = v["value"][t, i]
            elif nb_item2 == nb_region:
                for i in range(nb_item1):
                    for j in range(nb_item2):
                        state_dict[f"{k}_region_{i}_{j}"] = v["value"][t, i, j]
    return state_dict


def run_single_experiment(is_wandb=False, mitigation_rate=None, savings_rate=None):
    if is_wandb:
        # Initialize a wandb run
        run = wandb.init()

    # Create the Rice environment
    env = Rice()
    env.reset()
    if mitigation_rate is not None:
        run.config.mitigation_rate = mitigation_rate
    if savings_rate is not None:
        run.config.savings_rate = savings_rate
    if is_wandb:
        # Access the sweep parameters from wandb.config
        mitigation_rate = run.config.mitigation_rate
        savings_rate = run.config.savings_rate
        # Create a unique name for the experiment based on the parameters
        experiment_name = f"mitigation_{mitigation_rate}_savings_{savings_rate}"
        wandb.run.name = experiment_name

    if isinstance(mitigation_rate, (list, tuple, np.ndarray)):
        mitigation_rates = list(deepcopy(mitigation_rate))
        m_r = mitigation_rates.pop(0)
    else:
        m_r = mitigation_rate
    if isinstance(savings_rate, (list, tuple, np.ndarray)):
        savings_rates = list(deepcopy(savings_rate))
        s_r = savings_rates.pop(0)
    else:
        s_r = savings_rate
    while True:  # Or some other condition to terminate the loop
        ind_actions = RiceAction(
            {"savings": s_r, "mitigation_rate": m_r}
        ).actions

        actions = {region_id: ind_actions for region_id in regions}
        obs, rew, done, truncated, info = env.step(actions)
        if done["__all__"]:
            break

        current_timestep = env.current_timestep
        current_states = get_state_at_time_t(env.global_state, current_timestep)
        current_states.update(
            {
                "Mitigation rate": mitigation_rate,
                "Savings rate": savings_rate,
                "Year": current_timestep * 5,
            }
        )
        if isinstance(mitigation_rate, (list, tuple, np.ndarray)) and len(mitigation_rates) > 0:
            m_r = mitigation_rates.pop(0)
        if isinstance(savings_rate, (list, tuple, np.ndarray)) and len(savings_rates) > 0:
            s_r = savings_rates.pop(0)
        # Log the current states, not the empty 'result' variable
        if is_wandb:
            wandb.log(current_states)
    if is_wandb:    
        wandb.finish()
if __name__ == "__main__":
    run_single_experiment(is_wandb=True, mitigation_rate = [i/10 for i in range(1, 11) for _ in range(2)], savings_rate=0.1)