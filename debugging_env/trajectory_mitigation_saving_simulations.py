import numpy as np
import wandb
import time
from rice import Rice
from copy import deepcopy
import pickle
from fire import Fire
import random
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


with open(
    "/home/work/climate-cooperation-competition/debugging_env/sequences.pkl", "rb"
) as f:
    sequences = pickle.load(f)

# randomly sample one sequence from the list of sequences
random_idx = np.random.randint(0, len(sequences))
dice = random.random()
# dice = 0.7
if dice < 0.3333:
    type_ = "increasing"
    lst = sequences[random_idx]
elif dice < 0.6666:
    type_ = "decreasing"
    lst = sequences[random_idx][::-1]
else:
    type_ = "random"
    lst = np.random.randint(0, 9 + 1, 19).tolist()


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


def run_single_experiment(
    is_wandb=False,
    mitigation_rate=None,
    savings_rate=None,
    pliability=None,
    damage_type=None,
    abatement_cost_type=None,
    debugging_folder=None,
    carbon_model=None,
    prescribed_emissions=None,
    temperature_calibration=None,
    mitigation_type=None,
):
    if is_wandb:
        # Initialize a wandb run
        run = wandb.init(project="ricen-fair-movingdiff-mitigation-pct-relative-reward")
        wandb.config.update(locals())
    # Create the Rice environment
    if temperature_calibration is None:
        if carbon_model == "base":
            temperature_calibration = "base"
        elif carbon_model in ["FaIR", "AR5"]:
            temperature_calibration = "FaIR"
    env = Rice(
        dmg_function=damage_type,
        abatement_cost_type=abatement_cost_type,
        pliability=pliability,
        debugging_folder=debugging_folder,
        carbon_model=carbon_model,
        prescribed_emissions=prescribed_emissions,
        temperature_calibration=temperature_calibration,
    )
    env.reset()
    if isinstance(mitigation_rate, (list, tuple, np.ndarray)):
        mitigation_rate_str = "".join(str(num) for num in mitigation_rate)
    elif isinstance(mitigation_rate, (int, float)):
        mitigation_rate_str = str(mitigation_rate)
    else:
        raise ValueError(f"Invalid mitigation_rate type {type(mitigation_rate)}")

    if isinstance(savings_rate, (list, tuple, np.ndarray)):
        savings_rate_str = "".join(str(num) for num in savings_rate)
    elif isinstance(savings_rate, (int, float)):
        savings_rate_str = str(savings_rate)
    else:
        raise ValueError(f"Invalid savings_rate type {type(savings_rate)}")

    if is_wandb:
        if mitigation_rate is not None:
            run.config.mitigation_rate = mitigation_rate
        if savings_rate is not None:
            run.config.savings_rate = savings_rate
        # Access the sweep parameters from wandb.config
        mitigation_rate = run.config.mitigation_rate
        savings_rate = run.config.savings_rate
        # Create a unique name for the experiment based on the parameters
        if debugging_folder == "2_region":
            num_region = 2
        elif debugging_folder == "region_yamls" or debugging_folder is None:
            num_region = 3
        else:
            raise ValueError("Invalid debugging folder")
        experiment_name = f"m_{mitigation_rate}_s_{savings_rate}_p_{pliability}_d_{damage_type}_a_{abatement_cost_type}_v_{num_region}_c_{carbon_model}_pe_{prescribed_emissions}_tc_{temperature_calibration}"
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
    all_states = []
    while True:  # Or some other condition to terminate the loop
        current_timestep = env.current_timestep
        current_states = get_state_at_time_t(
            env.global_state, current_timestep, nb_region=env.num_regions
        )
        current_states.update(
            {
                "Mitigation rate": m_r,
                "Savings rate": s_r,
                "Year": current_timestep * 5,
            }
        )
        if (
            isinstance(mitigation_rate, (list, tuple, np.ndarray))
            and len(mitigation_rates) > 0
        ):
            m_r = mitigation_rates.pop(0)
        if (
            isinstance(savings_rate, (list, tuple, np.ndarray))
            and len(savings_rates) > 0
        ):
            s_r = savings_rates.pop(0)
        # Log the current states, not the empty 'result' variable
        if is_wandb:
            wandb.log(current_states)
        all_states.append(current_states)
        ind_actions = RiceAction({"savings": s_r, "mitigation_rate": m_r}).actions

        actions = {region_id: ind_actions for region_id in regions}
        obs, rew, done, truncated, info = env.step(actions)
        if done["__all__"]:
            # save a copy of current_states as json file locally
            with open(
                f"debugging_env/save_results/m_{mitigation_rate_str}_s_{savings_rate_str}_p_{pliability}_d_{damage_type}_a_{abatement_cost_type}_v_{env.num_regions}_c_{carbon_model}_pe_{prescribed_emissions}_tc_{temperature_calibration}.json",
                "w",
            ) as f:
                json.dump(all_states, f, cls=NumpyEncoder, indent=4)
            break
    if is_wandb:
        wandb.finish()


if __name__ == "__main__":
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], savings_rate=1)
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,1,1,1,1,1,1,1,1,1,9,9,9,9,9,9,9,9,9], savings_rate=1)
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9], savings_rate=1)
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,9,0,9,0,9,0,9,0,9,0,9,0,9,0,9,0,9,0], savings_rate=1)
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,0,0,0,0,0,0,0,0,9,9,9,9,9,9,9,9,9,9], savings_rate=1)
    # run_single_experiment(is_wandb=False, mitigation_rate = [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], savings_rate=1, pliability=0.9, damage_type="updated", abatement_cost_type="path_dependent")
    # run_single_experiment(is_wandb=True, mitigation_rate = [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9], savings_rate=1, pliability=0, damage_type="updated", abatement_cost_type="path_dependent")
    run_single_experiment(
        is_wandb=True,
        mitigation_rate=lst,
        savings_rate=2.5,
        pliability=0.4,
        damage_type="updated",
        abatement_cost_type="path_dependent",
        debugging_folder=None,
        carbon_model="base",
        prescribed_emissions=None,
        temperature_calibration=None,
        mitigation_type=type_,
    )
    # run_single_experiment(
    #     is_wandb=True,
    #     mitigation_rate=[0] * 19,
    #     savings_rate=2.5,
    #     pliability=0.4,
    #     damage_type="updated",
    #     abatement_cost_type="path_dependent",
    #     debugging_folder=None,
    #     carbon_model="base",
    #     prescribed_emissions=None,
    #     temperature_calibration=None,
    #     mitigation_type=type_,
    # )
