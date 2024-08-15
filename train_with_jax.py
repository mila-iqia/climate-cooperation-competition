import jax
from typing import Dict, Callable, Tuple, Any
import wandb
import argparse
import equinox as eqx
import time
import os

from jice.util import load_region_yamls, log_episode_stats_to_wandb
from jice.algorithms import (
    build_random_trainer,
    build_ppo_trainer,
    BaseTrainerParams,
    PpoTrainerParams,
)
from jice.environment import Rice, OptimalMitigation, BasicClub

SAVE_MODEL_PATH = "jice/saved_models/"
if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="Algorithm to train with", default="ppo")
parser.add_argument("-nw", "--no_wandb", help="Log to wandb", action="store_true")
parser.add_argument("-s", "--seed", help="Random seed", default=42, type=int)
parser.add_argument("-sk", "--skip_training", help="Skip training", action="store_true")
parser.add_argument("-l", "--load_model", help="Path to model file", default=None)
parser.add_argument(
    "-sc",
    "--scenario",
    help="Scenario to train on",
    default="default",
    choices=["default", "optimal_mitigation", "basic_club"],
)
parser.add_argument(
    "-n",
    "--num_regions",
    help="Number of regions",
    default=7,
    type=int,
    choices=[3, 7, 20],
)
args = parser.parse_args()


def build_env_scenario(yaml_file: Dict[str, Any], init_gamma: float = 0.99) -> Rice:
    region_params = load_region_yamls(yaml_file["env_settings"]["num_regions"])
    env_settings = {
        "region_params": region_params,
        "init_gamma": init_gamma,
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

    return env


def build_trainer(yaml_file: Dict[str, Any]) -> Tuple[Callable, dict]:
    if args.algorithm == "random":
        print("Using random agent...")
        trainer_params = BaseTrainerParams(**yaml_file["trainer_settings"])
        env = build_env_scenario(yaml_file)
        trainer = build_random_trainer(env=env, trainer_params=trainer_params)
    elif args.algorithm == "ppo":
        print("Using PPO agent...")
        trainer_params = PpoTrainerParams(**yaml_file["trainer_settings"])
        env = build_env_scenario(yaml_file, trainer_params.gamma)
        trainer = build_ppo_trainer(
            env=env, trainer_params=trainer_params, load_model=args.load_model
        )
    merged_settings = {**args.__dict__, **env.__dict__, **trainer_params.__dict__}
    return trainer, merged_settings


yaml_file = {
    "wandb": not args.no_wandb,
    "env_settings": {
        "num_regions": args.num_regions,  # [3, 7, 20]
        "train_env": True,
        "scenario": args.scenario,
    },
    "trainer_settings": {
        "num_log_episodes_after_training": 2,
        "num_envs": 4,
        "total_timesteps": 1e6,
        "trainer_seed": args.seed,
        "backend": "gpu",
        "skip_training": args.skip_training,
    },
}

trainer, merged_settings = build_trainer(yaml_file)

if yaml_file["wandb"] and not args.skip_training:
    # removing arrays, as they cause issues with wandb
    merged_settings = eqx.filter(merged_settings, eqx.is_array, inverse=True)
    wandb.init(
        project="jice", config=merged_settings, tags=["train_run"], entity="ai4gcc-gaia"
    )

seed = jax.random.PRNGKey(args.seed)

start_time = time.time()
print("Starting JAX compilation...")
trainer = jax.jit(trainer, backend=merged_settings["backend"]).lower(seed).compile()
print(
    f"JAX compilation finished in {time.time() - start_time} seconds, starting training..."
)
out = trainer(seed)
print("Training finished")
wandb.finish()

train_state = out["train_state"]
train_rewards = out["train_metrics"]
eval_logs = out["eval_logs"]
eval_rewards = out["eval_rewards"]

print(f"finished training")

if not args.skip_training:
    model_name = f"{args.algorithm}_{args.scenario}_{args.num_regions}_{time.time()}"
    print(f"saving model to {SAVE_MODEL_PATH}{model_name}")
    eqx.tree_serialise_leaves(f"{SAVE_MODEL_PATH}{model_name}.eqx", train_state)

if yaml_file["wandb"]:
    log_episode_stats_to_wandb(eval_logs, merged_settings)
