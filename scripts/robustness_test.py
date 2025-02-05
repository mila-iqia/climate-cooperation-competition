import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path
import yaml
from fixed_paths import PUBLIC_REPO_DIR
from evaluate_submission import validate_dir, try_to_unzip_file, get_imports
from train_with_rllib import EnvWrapper, get_env_config, create_env_object, get_trainer_config, Callbacks
from ray.rllib.algorithms.a2c import A2CConfig

_path = Path(os.path.abspath(__file__))

def get_multiagent_policies_config(config_yaml=None, env_object=None):
    assert config_yaml is not None
    assert env_object is not None

    # Define all the policies here
    regions_policy_config = config_yaml["policy"]["regions"]
    multi_model = config_yaml["policy"]["multi_model"]
    env_config = config_yaml["env"]
    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    if multi_model and env_config["clubs_enabled"]:
        policies = {
            "regionsclub": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
            "regionsnonclub": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
        }

        club_members = env_config["club_members"]

        # Function mapping agent ids to policy ids.
        def policy_mapping_fn(agent_id=None, episode=None, worker=None, **kwargs):
            assert agent_id is not None
            if agent_id in club_members:
                return "regionsclub"
            else:
                return "regionsnonclub"

    else:
        policies = {
            "regions": (
                None,  # uses default policy
                env_object.observation_space[0],
                env_object.action_space["0"],
                regions_policy_config,
            ),
        }

        # Function mapping agent ids to policy ids.
        def policy_mapping_fn(agent_id=None, episode=None, worker=None, **kwargs):
            assert agent_id is not None
            return "regions"

    # Optional list of policies to train, or None for all policies.
    policies_to_train = None

    # Settings for Multi-Agent Environments
    multiagent_config = {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fn,
    }

    return multiagent_config

def get_multiagent_policies_config_(config_yaml=None, env_object=None, agent_to_retrain=None):
    assert config_yaml is not None
    assert env_object is not None

    # Define all the policies here
    regions_policy_config = config_yaml["policy"]["regions"]
    multi_model = config_yaml["policy"]["multi_model"]
    env_config = config_yaml["env"]
    # Map of type MultiAgentPolicyConfigDict from policy ids to tuples
    # of (policy_cls, obs_space, act_space, config). This defines the
    # observation and action spaces of the policies and any extra config.
    policies = {
        "base_policy": (
            None,  # uses default policy
            env_object.observation_space[0],
            env_object.action_space["0"],
            regions_policy_config,
        ),
        "retrain": (
            None,  # uses default policy
            env_object.observation_space[0],
            env_object.action_space["0"],
            regions_policy_config,
        ),
    }

    club_members = env_config["club_members"]

    # Function mapping agent ids to policy ids.
    def policy_mapping_fn(agent_id=None, episode=None, worker=None, **kwargs):
        assert agent_id is not None
        if agent_id ==agent_to_retrain:
            return "retrain"
        else:
            return "base_policy"


    # Optional list of policies to train, or None for all policies.
    policies_to_train = ["retrain"]

    # Settings for Multi-Agent Environments
    multiagent_config = {
        "policies": policies,
        "policies_to_train": policies_to_train,
        "policy_mapping_fn": policy_mapping_fn,
    }

    return multiagent_config

def get_rllib_config(config_yaml=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert config_yaml is not None
    assert env_class is not None

    env_config = get_env_config(config_yaml)
    assert isinstance(env_config, dict)

    env_object = create_env_object(env_class, env_config)

    multiagent_policies_config = get_multiagent_policies_config(
        config_yaml=config_yaml, env_object=env_object
    )

    trainer_config = get_trainer_config(config_yaml)

    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": config_yaml["env"],
        "framework": trainer_config["framework"],
        "multiagent": multiagent_policies_config,
        "num_workers": trainer_config["num_workers"],
        "num_gpus": trainer_config["num_gpus"],
        "num_cpus_per_worker": trainer_config["num_cpus_per_worker"],
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
        "train_batch_size": trainer_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config

def create_trainer_robustness(config_yaml=None, source_dir=None, seed=None):
    """
    Create the RLlib trainer.
    """

    # Create the A2C trainer.
    config_yaml["env"]["source_dir"] = source_dir

    if config_yaml["env"]["action_space_type"] == "discrete":
        from scripts.torch_models_discrete import TorchLinear
    elif config_yaml["env"]["action_space_type"] == "continuous":
        if "beta" in config_yaml["policy"]["regions"]["model"]["custom_model"].lower():
            from scripts.torch_models_cont_beta import CustomBetaPolicyModel
            from ray.rllib.models import ModelCatalog
            from beta_action_dist import BetaActionDistribution

            ModelCatalog.register_custom_action_dist(
                "beta_distribution", BetaActionDistribution
            )

            from beta_action_dist import BetaActionDistribution
        elif (
            "cont" in config_yaml["policy"]["regions"]["model"]["custom_model"].lower()
        ):
            from scripts.torch_models_cont import TorchLinear
        elif (
            "discrete"
            in config_yaml["policy"]["regions"]["model"]["custom_model"].lower()
        ):
            from scripts.torch_models_discrete import TorchLinear

    rllib_config = get_rllib_config(
        config_yaml=config_yaml,
        env_class=EnvWrapper,
        seed=seed,
    )

    config = A2CConfig()

    # config.num_agents = rllib_config["num_envs_per_worker"]

    config = config.training(train_batch_size=rllib_config["train_batch_size"])
    config = config.environment(disable_env_checking=True)
    config = config.multi_agent(
        policies=rllib_config["multiagent"]["policies"],
        policy_mapping_fn=rllib_config["multiagent"]["policy_mapping_fn"],
        policies_to_train=rllib_config["multiagent"]["policies_to_train"],
    )

    config = config.resources(num_gpus=rllib_config["num_gpus"])
    config = config.rollouts(
        num_rollout_workers=rllib_config["num_workers"],
        num_envs_per_worker=rllib_config["num_envs_per_worker"],
    )
    config = config.framework(rllib_config["framework"])
    config = config.environment(
        EnvWrapper,
        env_config=rllib_config["env_config"],
    )
    config = config.callbacks(Callbacks)

    config.seed = seed

    rllib_trainer = config.build()

    return rllib_trainer

def load_trainer(
    results_directory,
    framework,
    discrete = True,
    eval_seed=None,
):
    """
    Create the trainer and compute metrics.
    """
    assert results_directory is not None
    
    (
        create_trainer,
        load_model_checkpoints,
        fetch_episode_states,
        set_num_agents
    ) = get_imports(framework=framework)
    
    # Load a run configuration
    if discrete:
        yaml_path = f"rice_{framework}_discrete.yaml"
    else:
        yaml_path = f"rice_{framework}_cont.yaml"
    config_file = os.path.join(results_directory, yaml_path)
    
    try:
        assert os.path.exists(config_file)
    except Exception as err:
        logging.error(
            f"The run configuration is missing in {results_directory}."
        )
        raise err
    
    with open(config_file, "r", encoding="utf-8") as file_ptr:
        run_config = yaml.safe_load(file_ptr)
        #force eval on single worker
        run_config["trainer"]["num_workers"] = 0
        log_config = run_config["logging"]
    #update region yamls
    set_num_agents(run_config)
    

    # Copy the PUBLIC region yamls and rice_build.cu to the results directory.
    if not os.path.exists(os.path.join(results_directory, "region_yamls")):
        shutil.copytree(
            os.path.join(PUBLIC_REPO_DIR, "region_yamls"),
            os.path.join(results_directory, "region_yamls"),
        )
    if not os.path.exists(os.path.join(results_directory, "rice_build.cu")):
        shutil.copyfile(
            os.path.join(PUBLIC_REPO_DIR, "rice_build.cu"),
            os.path.join(results_directory, "rice_build.cu"),
        )
    
    # Create Trainer object
    try:
        trainer = create_trainer_robustness(
            run_config, source_dir=results_directory, seed=eval_seed
        )

    except Exception as err:
        logging.error(f"Could not create Trainer with the run_config provided.")
        raise err

    # Load model checkpoints
    try:
        trainer = load_model_checkpoints(trainer, results_directory)
    except Exception as err:
        logging.error(f"Could not load model checkpoints.")
        raise err

    return trainer, run_config

def get_rllib_config(config_yaml=None, env_class=None, seed=None):
    """
    Reference: https://docs.ray.io/en/latest/rllib-training.html
    """

    assert config_yaml is not None
    assert env_class is not None

    env_config = get_env_config(config_yaml)
    assert isinstance(env_config, dict)

    env_object = create_env_object(env_class, env_config)

    multiagent_policies_config = get_multiagent_policies_config(
        config_yaml=config_yaml, env_object=env_object
    )

    trainer_config = get_trainer_config(config_yaml)

    rllib_config = {
        # Arguments dict passed to the env creator as an EnvContext object (which
        # is a dict plus the properties: num_workers, worker_index, vector_index,
        # and remote).
        "env_config": config_yaml["env"],
        "framework": trainer_config["framework"],
        "multiagent": multiagent_policies_config,
        "num_workers": trainer_config["num_workers"],
        "num_gpus": trainer_config["num_gpus"],
        "num_cpus_per_worker": trainer_config["num_cpus_per_worker"],
        "num_envs_per_worker": trainer_config["num_envs_per_worker"],
        "train_batch_size": trainer_config["train_batch_size"],
    }
    if seed is not None:
        rllib_config["seed"] = seed

    return rllib_config

def reset_weights(trainer, agent):
    rllib_config = get_rllib_config(
        config_yaml=config_yaml,
        env_class=EnvWrapper,
        seed=seed,
    )
    config = config.multi_agent(
        policies=rllib_config["multiagent"]["policies"],
        policy_mapping_fn=rllib_config["multiagent"]["policy_mapping_fn"],
        policies_to_train=rllib_config["multiagent"]["policies_to_train"],
    )
    trainer.multiagent
    return trainer


if __name__ == "__main__":
    print("Training with RLlib...")

    # Read the run configurations specific to the environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="./Submissions/1680502535.zip",  # an example of a submission file
        help="The directory where all the submission files are saved. Can also be "
        "a zip-file containing all the submission files.",
    )
    args = parser.parse_args()

    results_dir = (
        try_to_unzip_file(args.results_dir)
        if args.results_dir.endswith(".zip")
        else args.results_dir
    )

    logging.info(f"Using submission files in {results_dir}")

    # Validate the submission directory
    framework, results_dir_is_valid, comment, discrete = validate_dir(results_dir)

    trainer, run_config = load_trainer(results_dir, framework)
    agent = 0
    trainer = reset_weights(trainer, agent)
    for i in range(10):
        print(i)
        trainer.train()
    
