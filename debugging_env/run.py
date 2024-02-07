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


def run_single_experiment():
    # Initialize a wandb run
    run = wandb.init()

    # Access the sweep parameters from wandb.config
    mitigation_rate = run.config.mitigation_rate
    savings_rate = run.config.savings_rate
    pliability = run.config.pliability
    damage_type = run.config.damage_type
    abatement_cost_type = run.config.abatement_cost_type
    debugging_folder = run.config.debugging_folder
    # Create the Rice environment
    env = Rice(dmg_function=damage_type, abatement_cost_type=abatement_cost_type, pliability=pliability, debugging_folder=debugging_folder)
    env.reset()
    # Create a unique name for the experiment based on the parameters
    if debugging_folder == "2_region":
        num_region = 2
    elif debugging_folder == "region_yamls" or debugging_folder is None:
        num_region = 27
    else:
        raise ValueError("Invalid debugging folder")
    experiment_name = f"m_{mitigation_rate}_s_{savings_rate}_p_{pliability}_d_{damage_type}_a_{abatement_cost_type}_v_{num_region}"
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
                "Mitigation rate": m_r,
                "Savings rate": s_r,
                "Year": current_timestep * 5,
            }
        )
        if isinstance(mitigation_rate, (list, tuple, np.ndarray)) and len(mitigation_rates) > 0:
            m_r = mitigation_rates.pop(0)
        if isinstance(savings_rate, (list, tuple, np.ndarray)) and len(savings_rates) > 0:
            s_r = savings_rates.pop(0)
        # Log the current states, not the empty 'result' variable
        wandb.log(current_states)
    wandb.finish()


sweep_config = {
    "method": "grid",  # or 'random' for random search
    "metric": {
        "name": "loss",  # Replace with your metric
        "goal": "minimize",  # or 'maximize'
    },
    "parameters": {
        "mitigation_rate": {
            "values": [0,5,9,[9,9,9,9,9,9,9,9,9,9,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,9,9,9,9,9,9,9,9,9,9],[0,9,0,9,0,9,0,9,0,9,0,9,0,9,0,9,0,9,0], [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], [0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]]
        },
        "savings_rate": {"values": [1]},  # Or any other values you want to try
        "pliability": {"values": [0]},
        "damage_type": {"values": ["updated"]},
        "abatement_cost_type": {"values": ["base_abatement", "path_dependent"]},
        # "debugging_folder": {"values": ["2_region", "region_yamls"]},
        "debugging_folder": {"values": ["region_yamls"]},
        
    },
    # "parameters": {
    #     "mitigation_rate": {
    #         "values": []
    #     },
    #     "savings_rate": {"values": [1]},  # Or any other values you want to try
    #     "pliability": {"values": [0.5, 0.7, 0.9]},
    #     "damage_type": {"values": ["base", "updated"]},
    #     "abatement_cost_type": {"values": ["path_dependent"]},
        
    # },
}

sweep_id = wandb.sweep(
    sweep_config, project="ricen-abatement-function-debugging", entity="tianyuzhang"
)


def main():
    wandb.agent(sweep_id, run_single_experiment)


if __name__ == "__main__":
    main()
