import importlib.resources
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Annotated, Literal

import jax
import jaxnasium as jym
import tyro
from jaxnasium.algorithms import PPO

from _experiment_util import FixedActionAgent, load_agent, run_single_episode
from rice_jax import BasicClub, OptimalMitigation, Rice, BasicClubTariffAmbition, BasicClubTariffAmbitionFixedSavings
from rice_jax.utils import (  # noqa: F401
    create_plots,
    full_state_info_log_fn,
    load_region_yamls,
    log_episode_to_json,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

SETTINGS_YAML_PATH = importlib.resources.files("rice_jax").joinpath("./config_yamls/")


@dataclass
class EnvSettings:
    """The Rice environment settings."""

    num_regions: Literal[3, 7, 20] = 20
    diff_reward_mode: bool = True
    relative_reward_mode: bool = False
    num_discrete_action_levels: int = 10
    action_window_size: int = 1  # 0 = No action windows
    disable_trading: bool = False
    negotiation_on: bool = False

    dmg_function: Literal["base", "updated"] = "base"
    temperature_calibration: Literal["base", "FaIR", "DFaIR"] = "base"
    carbon_model: Literal["base", "FaIR", "DFaIR", "AR5"] = "base"
    apply_welfloss: bool = True
    apply_welfgain: bool = True

    # trade params
    init_capital_multiplier: float = 10.0
    balance_interest_rate: float = 0.1
    consumption_substitution_rate: float = 0.5
    preference_for_domestic: float = 0.5

    init_gamma: float = 0.99  # discount factor


@dataclass
class TrainerSettings:
    """The settings for the PPO trainer to be used."""

    total_timesteps: Annotated[
        int, tyro.conf.arg(aliases=("-t", "--total_timesteps"))
    ] = 1000000
    learning_rate: float = 2.5e-4
    anneal_learning_rate: bool | float = False
    ent_coef: float = 2.0
    anneal_ent_coef: bool | float = 0.05  # anneal to 0.05 over traing
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 0.5
    vf_coef: float = 0.5
    num_steps: int = 100
    num_minibatches: int = 4
    num_epochs: int = 4
    num_envs: int = 4
    normalize_observations: bool = True
    normalize_rewards: bool = False
    log_function: str = "tqdm"


@dataclass
class Config:
    """Main configuration for the rice_jax package."""

    seed: int = 0
    env_settings: EnvSettings = field(default_factory=lambda: EnvSettings())
    trainer_settings: TrainerSettings = field(default_factory=lambda: TrainerSettings())
    load_model: str | None = None
    scenario: Literal["default", "optimal_mitigation", "basic_club", "basic_club_tariff_ambition", "basic_club_tariff_ambition_fixed_savings", "max_export"] = "default"
    agent: Literal["fixed_action", "ppo"] = "ppo"
    # PQN, DQN, SAC also possible (although, TrainerSettings needs to be updated so not listed here (yet))


def build_rice_scenario(config: Config) -> Rice:
    region_params = load_region_yamls(config.env_settings.num_regions)
    env_settings = {
        **config.env_settings.__dict__,
        "region_params": region_params,
    }

    if config.scenario == "default":
        env = Rice(**env_settings)
    elif config.scenario == "optimal_mitigation":
        env = OptimalMitigation(**env_settings)
    elif config.scenario == "basic_club":
        env = BasicClub(**env_settings)
    elif config.scenario == "basic_club_tariff_ambition":
        env = BasicClubTariffAmbition(**env_settings)
    elif config.scenario == "basic_club_tariff_ambition_fixed_savings":
        env = BasicClubTariffAmbitionFixedSavings(**env_settings)
    elif config.scenario == "max_export":
        from rice_jax._scenarios import MaxExport

        env = MaxExport(**env_settings)
    else:
        raise ValueError(f"Scenario {env_settings['scenario']} not recognized")

    return jym.LogWrapper(env)


if __name__ == "__main__":
    # Parses command line arguments
    args = tyro.cli(Config)

    env = build_rice_scenario(args)
    seed = jax.random.PRNGKey(args.seed)

    # Load or train an agent
    if args.load_model:
        agent = load_agent(args.load_model)

    elif args.agent == "fixed_action":
        logger.info("Using fixed action agent...")
        agent = FixedActionAgent(env)
    elif args.agent == "ppo":
        logger.info("Using PPO agent...")
        agent = PPO(**args.trainer_settings.__dict__)
        agent = agent.train(seed, env)

        logger.info("Evaluating agent (only rewards)... ")
        avg_reward = agent.evaluate(seed, env, num_eval_episodes=10)
        logger.info(f"Average reward over 10 episodes: {avg_reward}")

    NUM_EPISODES = 3
    OUTPUT_DIR = "episode_logs"
    env = replace(env._env, log_info_fn=full_state_info_log_fn)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(
        f"Running {NUM_EPISODES} episodes to collect state logs per step. Logging to {OUTPUT_DIR}"
    )

    for episode_id in range(NUM_EPISODES):
        episode_logs = run_single_episode(seed, env, agent)

        # Log episode to JSON file
        log_filepath = log_episode_to_json(
            episode_logs,
            output_folder=OUTPUT_DIR,
            agent=agent,
            env=env,
            episode_id=episode_id,
            additional_metadata={"seed": int(seed[0])},  # Add seed for reproducibility
        )

        logger.info(f"Episode {episode_id} logs saved to: {log_filepath}")

        # # OPTIONALLY: Create the plots immediately:
        # # Create plots with default parameters (now creates a single combined plot)
        plot_files = create_plots(
            json_log_path=log_filepath,
            output_dir="plots",
            parameter_keys=[
                "global_temperature",  # Combined temperature plot
                "production_all_regions",
                "utility_all_regions",
                "actions.savings_rate",
                "actions.mitigation_rate",
                "gross_output_all_regions",
                "damages_all_regions",
            ],
            figsize=(12, 8),
            dpi=300,
        )
