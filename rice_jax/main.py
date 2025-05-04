import argparse
import os
import time
from dataclasses import replace
from typing import Any, Dict

import equinox as eqx
import jax
import jymkit as jym
import numpy as np
import optax
import yaml
from jaxtyping import PRNGKeyArray
from jymkit.algorithms import PPO

from rice_jax import BasicClub, OptimalMitigation, Rice
from rice_jax.util import (  # noqa: F401
    load_region_yamls,
    log_episode_stats_to_wandb,
    plot_data,
)

SETTINGS_YAML_PATH = "./rice_jax/config_yamls/"


def play_single_episode(key: PRNGKeyArray, env: Rice, agent: PPO) -> None:
    """
    Play an episode in the environment using the agent.
    """

    # log_state_keys = []
    log_exclude_keys = [
        "returned_episode_returns",
        "returned_episode_lengths",
        "returned_episode",
        "_TERMINAL_OBSERVATION",
        "DISCOUNT",
    ]

    def do_step(carry, _):
        key, obs, state = carry
        keys = jax.random.split(key, 3)
        action = agent.get_action(keys[0], obs)
        (obs, reward, _, _, info), state = env.step(keys[1], state, action)
        info = {k: v for k, v in info.items() if k not in log_exclude_keys}
        return (keys[2], obs, state), info

    env = eqx.tree_at(lambda x: x.log_state_in_info, env, True)
    obs, state = env.reset(key)
    _, info_stack = jax.lax.scan(
        do_step,
        (key, obs, state),
        None,
        length=env.episode_length,
    )
    return info_stack


def build_rice_scenario(yaml_file: Dict[str, Any]) -> Rice:
    region_params = load_region_yamls(yaml_file["env_settings"]["num_regions"])
    env_settings = {
        "region_params": region_params,
        **yaml_file["env_settings"],
    }

    if env_settings["scenario"] == "default":
        env = Rice(**env_settings)
    elif env_settings["scenario"] == "optimal_mitigation":
        env = OptimalMitigation(**env_settings)
    elif env_settings["scenario"] == "basic_club":
        env = BasicClub(**env_settings)
    else:
        raise ValueError(f"Scenario {env_settings['scenario']} not recognized")

    return jym.LogWrapper(env)


def train_new_agent(seed: PRNGKeyArray, yaml_file: Dict[str, Any], env: Rice) -> PPO:
    import wandb

    wandb.init(
        project="jice",
        # config=config,
        # entity="ai4gcc-gaia",
        # reinit=True,
        # tags=["eval_run"],
    )

    def log_training_to_wandb_fn(data, iteration):
        num_envs = data["timestep"].shape[-1]
        return_values = data["returned_episode_returns"][data["returned_episode"]]
        timesteps = data["timestep"][data["returned_episode"]] * num_envs

        avg_return_values = np.mean(np.array(return_values), axis=0)
        avg_return_values_per_agent = list(avg_return_values)
        wandb.log(
            {
                "avg_return_per_agent": {
                    f"agent_{i}": avg_return_values_per_agent[i]
                    for i in range(len(avg_return_values_per_agent))
                },
                "total_avg_return": np.sum(avg_return_values),
                "training timestep": timesteps[-1],
            }
        )

    SAVE_MODEL_PATH = "saved_models/"
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    agent = PPO(log_function=log_training_to_wandb_fn, **args["trainer_settings"])

    # Set up Learning Rate & Entropy Schedule
    NUM_UPDATES = agent.num_iterations * agent.num_minibatches * agent.update_epochs
    learning_rate = optax.linear_schedule(
        init_value=agent.learning_rate, end_value=0.0, transition_steps=NUM_UPDATES
    )
    ent_coef = optax.linear_schedule(
        init_value=agent.ent_coef, end_value=0.01, transition_steps=NUM_UPDATES
    )
    agent: PPO = replace(agent, learning_rate=learning_rate, ent_coef=ent_coef)
    agent = agent.train(seed, env)

    # Saving the agent
    t = time.time()
    name = f"{args['env_settings']['scenario']}_{args['env_settings']['num_regions']}"
    model_name = f"{name}_{int(t)}"
    print(f"saving model to {SAVE_MODEL_PATH}{model_name}")
    agent.save(f"{SAVE_MODEL_PATH}{model_name}.eqx")

    return agent


if __name__ == "__main__":
    ### Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", help="Overwrite yaml train steps", default=1e6, type=int)
    parser.add_argument("-y", "--yaml", help="Yaml settings file", default="default")
    parser.add_argument("-l", "--load_model", help="Path to model file", default=None)
    command_line_args = parser.parse_args()

    yaml_file_path = os.path.join(SETTINGS_YAML_PATH, f"{command_line_args.yaml}.yml")
    args = yaml.safe_load(open(yaml_file_path, "r"))

    # merge the yaml file with the command line arguments
    args["trainer_settings"]["total_timesteps"] = command_line_args.t
    args["load_model"] = command_line_args.load_model

    #### ---- #####

    env = build_rice_scenario(args)
    seed = jax.random.PRNGKey(args["seed"])

    # Load or train an agent
    if args["load_model"]:
        print(f"Loading model from {args['load_model']}")
        agent = PPO.load(args["load_model"], env)
    else:
        print("Training new agent...")
        agent = train_new_agent(seed, args, env)

    ## Evaluate the agent -> this function only retrieves final (avg) episode rewards
    # avg_rewards = agent.evaluate(seed, env, num_eval_episodes=20)

    # Play a couple of episodes and obtain the states throughout
    NUM_EPISODES = 3
    episode_logs = jax.vmap(play_single_episode, in_axes=(0, None, None))(
        jax.random.split(seed, NUM_EPISODES), env, agent
    )
    log_episode_stats_to_wandb(episode_logs, args)
