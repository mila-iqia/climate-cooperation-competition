import numpy as np
import time
import os
import json
from fixed_paths import PUBLIC_REPO_DIR
import logging
from tqdm import tqdm
from datetime import datetime
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
        json.dump(metrics, f, cls=NumpyArrayEncoder)

def run_carbon_leakage(trainer_obj=None,
                       condition="treatment",
                        episode_states=None,
                          file_name = None,
                          seed = None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None

    

    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    #ensure no instance of no club
    env_object.training = False
    obs, _ = env_object.reset()  
    metrics = {}
    env = env_object.env

    #enable or disable the presence of clubs
    env.control = condition == "control"
    if not env.control:
        club_members = env.club_members
    else:
        club_members = []
    obs, _ = env.reset()   
    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_agents):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    for timestep in range(env.episode_length):

        actions = {}
        # TODO: Consider using the `compute_actions` (instead of `compute_action`)
        # API below for speed-up when there are many agents.
        for region_id in range(env.num_agents):
            if (
                len(agent_states[region_id]) == 0
            ):  # stateless, with a linear model, for example

                
                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_actions(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
                
        obs, rewards, done, truncateds, info = env.step(actions)
        if done["__all__"]:
            break
            
    metrics["global_state"] = env.global_state
    metrics["condition"] = condition
    metrics["club_members"] = club_members
    #save
    # Get the current script's directory
    current_directory = os.path.dirname(__file__)
    # Construct the path to the 'eval' directory
    eval_directory = os.path.join(current_directory, "experiments", 'carbon_leakage')
    # Ensure the path is absolute
    eval_directory = os.path.abspath(eval_directory)
    formatted_datetime = datetime.now()\
        .strftime("%Y%m%d%H%M%S")
    file_name = "|".join([str(cm) for cm in club_members]) + f"_cnd_{condition}"
    name = f"cl_{file_name}_{formatted_datetime}.json"
    # Define the file name and construct the full file path
    file_path = os.path.join(eval_directory, name)
    with open(file_path, "w") as f:
        json.dump(metrics, f, cls=NumpyArrayEncoder)

def run_carbon_leakage_variable(trainer_obj=None,
                       condition="treatment",
                        episode_states=None,
                          file_name = None, 
                          seed = None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None

    

    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    conditions = ["control", "treatment"]
    # Fetch the env object from the trainer
    env_object = trainer_obj.workers.local_worker().env
    first_club = env_object.env.club_members
    #ensure no instance of no club
    env_object.training = False
    obs, _ = env_object.reset()  
    env = env_object.env

    metrics = {}
    #enable or disable the presence of clubs
    env.control = condition == "control"
    if not env.control:
        club_members = env.club_members
    else:
        club_members = []
    obs, _ = env.reset()   
    agent_states = {}
    policy_ids = {}
    policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    for region_id in range(env.num_agents):
        policy_ids[region_id] = policy_mapping_fn(region_id)
        agent_states[region_id] = trainer_obj.get_policy(
            policy_ids[region_id]
        ).get_initial_state()

    for timestep in range(env.episode_length):

        actions = {}
        # TODO: Consider using the `compute_actions` (instead of `compute_action`)
        # API below for speed-up when there are many agents.
        for region_id in range(env.num_agents):
            if (
                len(agent_states[region_id]) == 0
            ):  # stateless, with a linear model, for example

                
                actions[region_id] = trainer_obj.compute_single_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )
            else:  # stateful
                (
                    actions[region_id],
                    agent_states[region_id],
                    _,
                ) = trainer_obj.compute_actions(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )



                
        obs, rewards, done, truncateds, info = env.step(actions)
        if done["__all__"]:
            break
            
    metrics["global_state"] = env.global_state
    metrics["condition"] = condition
    metrics["club_members"] = club_members
    metrics["mmr"] = env.minimum_mitigation_rate
    metrics["seed"]=seed
    #save
    # Get the current script's directory
    current_directory = os.path.dirname(__file__)
    # Construct the path to the 'eval' directory
    eval_directory = os.path.join(current_directory, "experiments", 'carbon_leakage_variable')
    # Ensure the path is absolute
    eval_directory = os.path.abspath(eval_directory)
    formatted_datetime = datetime.now()\
        .strftime("%Y%m%d%H%M%S")
    file_name = "|".join([str(cm) for cm in club_members]) + f"_cnd_{condition}_sd_{seed}"
    name = f"cl_{file_name}_{formatted_datetime}.json"
    # Define the file name and construct the full file path
    file_path = os.path.join(eval_directory, name)
    with open(file_path, "w") as f:
        json.dump(metrics, f, cls=NumpyArrayEncoder)
