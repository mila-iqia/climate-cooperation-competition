import jax
from typing import Dict, Callable, Tuple, Any
from jice.util import load_region_yamls, log_episode_stats_to_wandb
from jice.algorithms import build_random_trainer, build_sac_trainer, build_ppo_trainer, BaseTrainerParams, PpoTrainerParams
import wandb
import argparse
import equinox as eqx
import time

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="Algorithm to train with", default="ppo")
parser.add_argument("-nw", "--no_wandb", help="Log to wandb", action="store_true")
parser.add_argument("-s", "--seed", help="Random seed", default=42, type=int)
args = parser.parse_args()	

def build_trainer(yaml_file: Dict[str, Any]) -> Tuple[Callable, dict]:
    region_params = load_region_yamls(yaml_file["env_settings"]["num_regions"])

    env_settings = {
        "region_params": region_params,
        **yaml_file["env_settings"]
    }

    if args.algorithm == "random":
        print("Using random agent...")
        trainer_params = BaseTrainerParams(**yaml_file["trainer_settings"])
        trainer, env = build_random_trainer(
            env_params=env_settings, 
            trainer_params=trainer_params
        )
    elif args.algorithm == "ppo":
        print("Using PPO agent...")
        trainer_params = PpoTrainerParams(**yaml_file["trainer_settings"])
        trainer, env = build_ppo_trainer(
            env_params=env_settings, 
            trainer_params=trainer_params
        )
    merged_settings = {
        **args.__dict__,
        **env._env.__dict__, 
        **trainer_params.__dict__
    }
    return trainer, merged_settings

yaml_file = {
    "wandb": not args.no_wandb,
    "env_settings": {
        "num_regions": 7, # [3, 7, 20]
        "train_env": True
    },
    "trainer_settings": {
        "num_log_episodes_after_training": 2, 
        "num_envs": 4,
        "total_timesteps": 2e6,
        "trainer_seed": args.seed,
        "backend": "gpu"
    }
}

trainer, merged_settings = build_trainer(yaml_file)

if yaml_file["wandb"]:
    # removing arrays, as they cause issues with wandb
    merged_settings = eqx.filter(merged_settings, eqx.is_array, inverse=True)
    wandb.init(project="jice", config=merged_settings, tags=["train_run"], entity="ai4gcc-gaia")

seed = jax.random.PRNGKey(args.seed)

start_time = time.time()
print("Starting JAX compilation...")
trainer = jax.jit(trainer, backend=merged_settings["backend"]).lower(seed).compile()
print(f"JAX compilation finished in {time.time() - start_time} seconds, starting training...")
out = trainer(seed)
print("Training finished")
wandb.finish()

train_state = out["train_state"]
train_rewards = out["train_metrics"]
eval_logs = out["eval_logs"]
eval_rewards = out["eval_rewards"]

print(f"finished training")

if yaml_file["wandb"]:
    log_episode_stats_to_wandb(eval_logs, merged_settings)