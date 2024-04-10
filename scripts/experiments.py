

import numpy as np
import time
import os
import json
from fixed_paths import PUBLIC_REPO_DIR
import logging

def fetch_episode_states_tariff_test(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    # Fetch the env object from the trainer
    # try:
    #     env_object = trainer_obj.workers.local_worker().env
    #     obs = env_object.reset()
    # except:
    #     envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    #     env_object = envs[1].env 
    #     obs, info = env_object.reset()

    envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    env_object = envs[1].env
    obs, _ = env_object.reset() 
    env = env_object

    #choose one agent to receive a tariff from all other agents
    pariah_id = np.random.randint(0,env_object.num_agents) 
    tariff_rates = [7,6,8,9,5]
    groups = ["pariah", "control"]
    
    metrics = []

    for tariff_rate in tariff_rates:
        for group in groups:
            envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
            env_object = envs[1].env 
            env = env_object
            obs,_ = env.reset()
            logging.info(f"conducting {group} with tariff rate {tariff_rate}")

            agent_states = {}
            policy_ids = {}
            # policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
            policy_mapping_fn = trainer_obj.config["policy_mapping_fn"]
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
                        ) = trainer_obj.compute_action(
                            obs[region_id],
                            agent_states[region_id],
                            policy_id=policy_ids[region_id],
                        )

                    if group == "pariah":
                        #all other regions
                        if region_id != pariah_id:
                            #heavily tariff the pariah
                            actions[region_id][env_object.get_actions_index("import_tariffs")+pariah_id] = tariff_rate
                            #everyone must import max from pariah
                            actions[region_id][env_object.get_actions_index("import_bids")+pariah_id] = 9

                
                                #get tariffs aginst agents
                average_tariffs = np.mean([actions[region_id][env_object.get_actions_index("import_tariffs")+pariah_id] for region_id in range(env.num_agents) if region_id !=pariah_id])
                export_action = actions[pariah_id][env_object.get_actions_index("export_limit")]
                
                average_imports = np.mean([actions[region_id][env_object.get_actions_index("import_bids")+pariah_id] for region_id in range(env.num_agents) if region_id !=pariah_id])

                
                obs, rewards, done, truncateds, info = env.step(actions)
                reward = env.get_state("reward_all_regions",region_id=pariah_id, timestep=timestep)
                labor = env.get_state("labor_all_regions",region_id=pariah_id, timestep=timestep)
                welfloss = env.get_state("welfloss",region_id=pariah_id, timestep=timestep)
                metrics.append({
                    f"{group}_tariffs_{tariff_rate}":float(average_tariffs),
                    f"{group}_reward_{tariff_rate}":float(reward),
                    f"{group}_stepreward_{tariff_rate}":float(rewards[pariah_id]),
                    f"{group}_labor_{tariff_rate}":float(labor),
                    f"{group}_exports_{tariff_rate}":float(export_action),
                    f"{group}_imports_{tariff_rate}":float(average_imports),
                    f"{group}_welfloss_{tariff_rate}":float(welfloss),
                })


                #end given run
                if done["__all__"]:
                    print("done")
                    break
    #save to disk
    current_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f"fr_{pariah_id}_{current_time}.json"

    with open(os.path.join(PUBLIC_REPO_DIR,"scripts","experiments", "tariff", file_name), "w") as f:
        json.dump(metrics, f)

def fetch_episode_states_trade_preference(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    # Fetch the env object from the trainer
    # try:
    #     env_object = trainer_obj.workers.local_worker().env
    #     obs = env_object.reset()
    # except:
    #     envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    #     env_object = envs[1].env 
    #     obs, info = env_object.reset()

    envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    env_object = envs[1].env
    obs, _ = env_object.reset() 
    env = env_object

    #choose one agent to receive a tariff from all other agents
    dom_prefs = [x/10 for x in range(0,10)]
    
    metrics = []

    for dom_pref in dom_prefs:
            

        envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
        env_object = envs[1].env 
        env = env_object
        
        env.preference_for_domestic = dom_pref
        env.consumption_substitution_rate = 1-dom_pref
        env.preference_for_imported = env.calc_uniform_foreign_preferences()

        # Typecasting
        env.consumption_substitution_rate = np.array(
            [env.consumption_substitution_rate]
        ).astype(env.float_dtype)
        env.preference_for_domestic = np.array(
            [env.preference_for_domestic]
        ).astype(env.float_dtype)
        env.preference_for_imported = np.array(
            env.preference_for_imported, dtype=env.float_dtype
        )
        obs,_ = env.reset()

        agent_states = {}
        policy_ids = {}
        # policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
        policy_mapping_fn = trainer_obj.config["policy_mapping_fn"]
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
                    ) = trainer_obj.compute_action(
                        obs[region_id],
                        agent_states[region_id],
                        policy_id=policy_ids[region_id],
                    )



            
                            #get tariffs aginst agents
            average_tariffs = np.mean([np.mean(actions[region_id][env_object.get_actions_index("import_tariffs"):env_object.get_actions_index("import_tariffs")+len(env_object.calc_possible_actions("import_tariffs"))]) for region_id in range(env.num_agents)])
            average_exports = np.mean([actions[region_id][env_object.get_actions_index("export_limit")] for region_id in range(env.num_agents)])
            average_imports = np.mean([np.mean(actions[region_id][env_object.get_actions_index("import_bids"):env_object.get_actions_index("import_bids")+len(env_object.calc_possible_actions("import_bids"))]) for region_id in range(env.num_agents)])

            
            obs, rewards, done, truncateds, info = env.step(actions)
            metrics.append({
                f"tariffs":float(average_tariffs),
                f"exports":float(average_exports),
                f"imports":float(average_imports),
                "timestep":timestep,
                "pref":dom_pref,
            })


            #end given run
            if done["__all__"]:
                print("done")
                break
    #save to disk
    current_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f"dp_{current_time}.json"

    with open(os.path.join(PUBLIC_REPO_DIR,"scripts","experiments", "forpref", file_name), "w") as f:
        json.dump(metrics, f)

def fetch_episode_states_get_imports(trainer_obj=None, episode_states=None):
    """
    Helper function to rollout the env and fetch env states for an episode.
    """
    assert trainer_obj is not None
    assert episode_states is not None
    assert isinstance(episode_states, list)
    assert len(episode_states) > 0

    # Fetch the env object from the trainer
    # try:
    #     env_object = trainer_obj.workers.local_worker().env
    #     obs = env_object.reset()
    # except:
    #     envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    #     env_object = envs[1].env 
    #     obs, info = env_object.reset()

    envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    env_object = envs[1].env
    obs, _ = env_object.reset() 
    env = env_object

    #choose one agent to receive a tariff from all other agents
    
    metrics = []


            

    envs = trainer_obj.workers.foreach_worker(lambda worker: worker.env)
    env_object = envs[1].env 
    env = env_object
    
    
    obs,_ = env.reset()

    agent_states = {}
    policy_ids = {}
    # policy_mapping_fn = trainer_obj.config["multiagent"]["policy_mapping_fn"]
    policy_mapping_fn = trainer_obj.config["policy_mapping_fn"]
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
                ) = trainer_obj.compute_action(
                    obs[region_id],
                    agent_states[region_id],
                    policy_id=policy_ids[region_id],
                )



        
                        #get tariffs aginst agents
        average_tariffs = np.mean([np.mean(actions[region_id][env_object.get_actions_index("import_tariffs"):env_object.get_actions_index("import_tariffs")+len(env_object.calc_possible_actions("import_tariffs"))]) for region_id in range(env.num_agents)])
        average_exports = np.mean([actions[region_id][env_object.get_actions_index("export_limit")] for region_id in range(env.num_agents)])
        average_imports = np.mean([np.mean(actions[region_id][env_object.get_actions_index("import_bids"):env_object.get_actions_index("import_bids")+len(env_object.calc_possible_actions("import_bids"))]) for region_id in range(env.num_agents)])
        regional_imports = [np.mean(actions[region_id][env_object.get_actions_index("import_bids"):env_object.get_actions_index("import_bids")+len(env_object.calc_possible_actions("import_bids"))]) for region_id in range(env.num_agents)]
        print("average")
        print(regional_imports)
        print("region by region")
        for region_id in range(env.num_agents):
            print(region_id)
            print(actions[region_id][env_object.get_actions_index("import_bids"):env_object.get_actions_index("import_bids")+len(env_object.calc_possible_actions("import_bids"))])
        obs, rewards, done, truncateds, info = env.step(actions)
        metrics.append({
            f"tariffs":float(average_tariffs),
            f"exports":float(average_exports),
            f"imports":float(average_imports),
            f"regional_imports":regional_imports,
            "timestep":timestep,
        })


        #end given run
        if done["__all__"]:
            print("done")
            break
    #save to disk
    current_time = time.strftime("%H:%M:%S", time.localtime())
    file_name = f"regionalimports_{current_time}.json"

    with open(os.path.join(PUBLIC_REPO_DIR,"scripts","experiments", "forpref", file_name), "w") as f:
        json.dump(metrics, f)