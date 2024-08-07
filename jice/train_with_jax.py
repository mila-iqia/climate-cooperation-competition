import jax
from typing import Dict, Callable, Tuple, Any
from jice.environment import RiceEnvParams
from jice.util import load_region_yamls, log_episode_stats_to_wandb
from jice.algorithms import build_random_trainer, build_sac_trainer, build_ppo_trainer, BaseTrainerParams, SacTrainerParams, PpoTrainerParams
import wandb
import argparse
import equinox as eqx

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="Algorithm to train with", default="ppo")
parser.add_argument("-nw", "--no_wandb", help="Log to wandb", action="store_true")
args = parser.parse_args()	

def build_trainer(yaml_file: Dict[str, Any]) -> Tuple[Callable, dict]:
    region_params = load_region_yamls(yaml_file["env_settings"]["num_regions"])
    env_settings = RiceEnvParams(
        region_params=region_params,
        **yaml_file["env_settings"],  
    ) 

    if args.algorithm == "random":
        print("Training random agent...")
        trainer_params = BaseTrainerParams(**yaml_file["trainer_settings"])
        trainer = build_random_trainer(
            # env_params=env_params, 
            env_params=env_settings, 
            trainer_params=trainer_params
        )
    elif args.algorithm == "sac":
        print("Training SAC agent...")
        trainer_params = SacTrainerParams(**yaml_file["trainer_settings"])
        trainer = build_sac_trainer(
            env_params=env_settings, 
            trainer_params=trainer_params
        )
    elif args.algorithm == "ppo":
        print("Training PPO agent...")
        trainer_params = PpoTrainerParams(**yaml_file["trainer_settings"])
        trainer = build_ppo_trainer(
            env_params=env_settings, 
            trainer_params=trainer_params
        )
    merged_settings = {
        **args.__dict__,
        **env_settings.__dict__, 
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
        "total_timesteps": 1e6,
        "trainer_seed": 42,
        "backend": "gpu"
    }
}

trainer, merged_settings = build_trainer(yaml_file)

if yaml_file["wandb"]:
    # removing arrays from settings, as they cause issues with wandb
    merged_settings = eqx.filter(merged_settings, eqx.is_array, inverse=True)
    wandb.init(project="jice", config=yaml_file, tags=["train_run"])

out = trainer(jax.random.PRNGKey(0))
wandb.finish()

train_state = out["train_state"]
train_rewards = out["train_metrics"]
eval_logs = out["eval_logs"]
eval_rewards = out["eval_rewards"]

print(f"finished training")

if yaml_file["wandb"]:
    log_episode_stats_to_wandb(eval_logs, merged_settings)