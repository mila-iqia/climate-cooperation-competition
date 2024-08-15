import os
import yaml
import jax.numpy as jnp
import wandb
import jax
from types import SimpleNamespace
import time
import wandb
import numpy as np


def logwrapper_callback(metric, num_envs: int, counter: int | None = None):
    if (
        counter is not None and np.random.rand() < 0.9
    ):  # prevent too much logging in random agent
        return
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]
    timesteps = metric["timestep"][metric["returned_episode"]] * num_envs
    if not np.any(metric["returned_episode"]):
        return
    for t in range(len(timesteps[-5:])):  # only show the last 5, to avoid printing
        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
    if wandb.run:
        try:
            losses = metric["loss_info"]
            total_loss, actor_loss, value_loss, entropy = jax.tree.map(jnp.mean, losses)
        except KeyError:
            total_loss, actor_loss, value_loss, entropy = None, None, None, None
        episode_returns_averaged = np.mean(np.array(return_values), axis=0)
        wandb.log(
            {
                "per_agent_episode_return": {
                    f"{agent_id}": episode_returns_averaged[agent_id]
                    for agent_id in range(len(episode_returns_averaged))
                },
                "total_episode_return_sum": np.sum(episode_returns_averaged),
                "total_loss": total_loss,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "training timestep": timesteps[-1],
            }
        )


def log_episode_stats_to_wandb(episode_stats, config):
    dummy_key = "current_timestep"  # any key that exists and contains scalars per timestep (not per region arrays)
    num_envs = len(episode_stats[dummy_key])
    num_steps = len(episode_stats[dummy_key][0])
    group_name = f"run_{time.time()}"
    for env in range(num_envs):
        run = wandb.init(
            project="jice",
            config=config,
            entity="ai4gcc-gaia",
            reinit=True,
            group=group_name,
            tags=["eval_run"],
        )
        for step in range(num_steps):
            step_dict = jax.tree_util.tree_map(lambda x: x[env][step], episode_stats)
            run.log(step_dict)
        run.finish()


def load_region_yamls(num_regions: int):
    assert num_regions in [3, 7, 20], "Supported number of regions are 3, 7, 20"
    yaml_file_directory = f"jice/region_yamls/"
    region_yamls = []
    for file in sorted(os.listdir(f"{yaml_file_directory}{num_regions}_regions")):
        if file.endswith(".yml"):
            with open(f"{yaml_file_directory}{num_regions}_regions/{file}", "r") as f:
                region = yaml.safe_load(f)
                region = region["_RICE_CONSTANT"]  # remove redundant key
                region_yamls.append(region)

    # ximport_ is an exception, sice it is an array for each region
    ximport_ = [region["ximport"] for region in region_yamls]
    ximport_ = [dict(sorted(x.items())) for x in ximport_]  # sort by region id
    ximport_ = [list(x.values()) for x in ximport_]

    region_params = {
        k: np.array([region[k] for region in region_yamls])
        for k in region_yamls[0].keys()
    }
    region_params["ximport"] = np.array(ximport_)

    # default params, that apply generally:
    with open(f"{yaml_file_directory}/default.yml", "r") as f:
        default_params = yaml.safe_load(f)
        dice_params = default_params["_DICE_CONSTANT"]
        rice_params = default_params["_RICE_CONSTANT"]

        # in dice_params, convert all lists (or lists of lists) to tuples
        def list_to_tuples(value):
            if isinstance(value, list):
                return tuple(list_to_tuples(v) for v in value)
            return value

        dice_params = {k: list_to_tuples(v) for k, v in dice_params.items()}

        default_params = {**dice_params, **rice_params}

    # merge the region_params, overwriting the key,values that are already present
    params = {**default_params, **region_params}

    # allow for dot notation
    params = SimpleNamespace(**params)

    return params
