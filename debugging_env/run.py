import numpy as np
import wandb
import time
from rice import Rice

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

    # Create the Rice environment
    env = Rice()
    env.reset()

    # Access the sweep parameters from wandb.config
    mitigation_rate = run.config.mitigation_rate
    savings_rate = run.config.savings_rate

    # Create a unique name for the experiment based on the parameters
    experiment_name = f"mitigation_{mitigation_rate}_savings_{savings_rate}"
    wandb.run.name = experiment_name

    while True:  # Or some other condition to terminate the loop
        ind_actions = RiceAction(
            {"savings": savings_rate, "mitigation_rate": mitigation_rate}
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
            "values": list(np.array(range(10)) / 100) + [0.3, 0.5, 0.7, 0.9]
        },
        "savings_rate": {"values": [0.1]},  # Or any other values you want to try
    },
}

sweep_id = wandb.sweep(
    sweep_config, project="rice_model_simulations", entity="tianyuzhang"
)


def main():
    wandb.agent(sweep_id, run_single_experiment)


if __name__ == "__main__":
    main()
