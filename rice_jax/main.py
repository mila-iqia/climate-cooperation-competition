import importlib.resources
import logging
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Annotated, Literal

import jax
import numpy as np
import tyro
import yaml
from jaxnasium.algorithms import PPO

from _experiment_util import (
    FixedActionAgent,
    load_agent,
    run_single_episode,
    with_log_info_fn,
    wrap_rice_env,
)
from rice_jax import (
    BasicClub,
    BasicClubTariffAmbition,
    BasicClubTariffAmbitionFixedSavings,
    OptimalMitigation,
    Rice,
    RiceMRIO,
)
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
class MRIOSettings:
    """Settings for the RiceMRIO scenario."""

    num_regions: Literal[3, 7, 20] = 20
    # Root of the csv_asset directory (relative to the rice_jax/ working dir
    # or absolute).  Aggregated MRIO sub-paths and CountryClass CSVs are
    # derived automatically from num_regions.
    mrio_data_root: str = "../csv_asset"
    # AR(1) persistence weight ρ ∈ [0,1] for the destination-allocation logit
    # anchor.  ρ=0 → always re-anchor to 2016 MRIO baseline (default).
    # ρ=1 → fully adaptive (prev allocation becomes the new baseline).
    # See CBAM_GRADIENT_DESIGN.md § Approach D and Roberts & Tybout (1997).
    dest_alloc_persistence: float = 0.0
    # When True, the welfare-loss multiplier is resolved sector-by-sector:
    #   welfloss[r] = 1 - Σ_s (X_{r,s}^EU * σ_{r,s} * τ * α) / Y_r
    # Gives export_reallocation a per-sector reward gradient.  False = Approach B.
    sectoral_welfloss: bool = False
    # Remove savings_rate / mitigation_rate from the action space and fix them
    # to their hardcoded values (0.2 and 0.0 respectively), leaving
    # export_reallocation as the sole driver of differentiated reward.
    fixed_savings_rate: bool = False
    no_mitigation: bool = False
    # Sector granularity for the export_reallocation action space:
    #   "full"              — all 26 EORA sectors (default).
    #   "cbam-specific"     — 3 CBAM sectors separate + "non-CBAM" bucket (4 total).
    #   "simple"            — 2 sectors: "CBAM" and "non-CBAM".
    #   "emissions-specific" — 7 dirty sectors separate + "non-CBAM" (8 total).
    #   "emissions-simple"   — 2 sectors: dirty (CBAM + high-emissions) vs rest.
    sector_granularity: str = "full"


@dataclass
class EnvSettings:
    """The Rice environment settings."""

    num_regions: Literal[3, 7, 20] = 7
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
    preference_for_domestic: float = 0.9

    init_gamma: float = 0.99  # discount factor


@dataclass
class TrainerSettings:
    """PPO trainer settings (jaxnasium 0.1 schedule API)."""

    total_timesteps: Annotated[
        int, tyro.conf.arg(aliases=("-t", "--total_timesteps"))
    ] = 1_000_000
    learning_rate_start: float = 2.5e-4
    learning_rate_end: float | None = None  # None = constant LR
    ent_coef_start: float = 2.0
    ent_coef_end: float | None = 0.05  # None = constant entropy coef
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


def trainer_settings_to_ppo_kwargs(settings: TrainerSettings) -> dict:
    """Map CLI trainer settings to jaxnasium 0.1 PPO constructor kwargs."""
    return {
        "total_timesteps": settings.total_timesteps,
        "learning_rate_start": settings.learning_rate_start,
        "learning_rate_end": settings.learning_rate_end,
        "ent_coef_start": settings.ent_coef_start,
        "ent_coef_end": settings.ent_coef_end,
        "gamma": settings.gamma,
        "gae_lambda": settings.gae_lambda,
        "max_grad_norm": settings.max_grad_norm,
        "clip_coef": settings.clip_coef,
        "clip_coef_vf": settings.clip_coef_vf,
        "vf_coef": settings.vf_coef,
        "num_steps": settings.num_steps,
        "num_minibatches": settings.num_minibatches,
        "num_epochs": settings.num_epochs,
        "num_envs": settings.num_envs,
        "normalize_observations": settings.normalize_observations,
        "normalize_rewards": settings.normalize_rewards,
        "log_function": settings.log_function,
    }


@dataclass
class Config:
    """Main configuration for the rice_jax package."""

    seed: int = 0
    env_settings: EnvSettings = field(default_factory=lambda: EnvSettings())
    trainer_settings: TrainerSettings = field(default_factory=lambda: TrainerSettings())
    load_model: str | None = None
    # Path to a directory of numbered yaml files (e.g. rice_jax/cbam_yamls/setup_5).
    # When set, overrides num_regions-based yaml loading and infers num_regions
    # from the number of .yml files found in the directory.
    region_yamls_dir: str | None = None
    scenario: Literal[
        "default",
        "optimal_mitigation",
        "basic_club",
        "basic_club_tariff_ambition",
        "basic_club_tariff_ambition_fixed_savings",
        "max_export",
        "max_export_fixed_savings",
        "rice_mrio",
    ] = "default"
    mrio_settings: MRIOSettings = field(default_factory=lambda: MRIOSettings())
    agent: Literal["fixed_action", "ppo"] = "ppo"
    # PQN, DQN, SAC also possible (although, TrainerSettings needs to be updated so not listed here (yet))


def _load_region_yamls_from_dir(directory: str) -> tuple:
    """Load numbered yaml files from an arbitrary directory.

    Returns (region_params, num_regions) where region_params is a
    SimpleNamespace identical in structure to load_region_yamls().
    """
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".yml")],
        key=lambda f: int(os.path.splitext(f)[0]),
    )
    if not files:
        raise FileNotFoundError(f"No .yml files found in {directory}")

    region_yamls = []
    for fname in files:
        with open(os.path.join(directory, fname)) as f:
            doc = yaml.safe_load(f)
        region_yamls.append(doc["_RICE_CONSTANT"])

    ximport_ = [
        list(dict(sorted(r["ximport"].items(), key=lambda kv: int(kv[0]))).values())
        for r in region_yamls
    ]
    region_params = {
        k: np.array([r[k] for r in region_yamls])
        for k in region_yamls[0].keys()
        if k != "ximport"
    }
    region_params["ximport"] = np.array(ximport_)

    # Merge with default params (dice + rice constants) from the package default.yml
    yaml_file_directory = importlib.resources.files("rice_jax").joinpath(
        "./region_yamls/"
    )
    with open(f"{yaml_file_directory}/default.yml") as f:
        default_doc = yaml.safe_load(f)
    dice_params = default_doc["_DICE_CONSTANT"]
    rice_params_default = default_doc["_RICE_CONSTANT"]

    def list_to_tuples(v):
        return tuple(list_to_tuples(x) for x in v) if isinstance(v, list) else v

    dice_params = {k: list_to_tuples(v) for k, v in dice_params.items()}
    params = {**dice_params, **rice_params_default, **region_params}
    return SimpleNamespace(**params), len(files)


def build_rice_scenario(config: Config) -> Rice:
    if config.region_yamls_dir is not None:
        region_params, num_regions = _load_region_yamls_from_dir(
            config.region_yamls_dir
        )
    elif config.scenario == "rice_mrio":
        num_regions = config.mrio_settings.num_regions
        region_params = load_region_yamls(num_regions)
    else:
        num_regions = config.env_settings.num_regions
        region_params = load_region_yamls(num_regions)
    env_settings = {
        **config.env_settings.__dict__,
        "region_params": region_params,
        "num_regions": num_regions,
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
        from rice_jax.core.scenarios import MaxExport

        env = MaxExport(**env_settings)
    elif config.scenario == "max_export_fixed_savings":
        from rice_jax.core.scenarios import MaxExportFixedSavings

        env = MaxExportFixedSavings(**env_settings)
    elif config.scenario == "rice_mrio":
        env = RiceMRIO(
            **env_settings,
            mrio_data_root=config.mrio_settings.mrio_data_root,
            dest_alloc_persistence=config.mrio_settings.dest_alloc_persistence,
            sectoral_welfloss=config.mrio_settings.sectoral_welfloss,
            fixed_savings_rate=config.mrio_settings.fixed_savings_rate,
            no_mitigation=config.mrio_settings.no_mitigation,
            sector_granularity=config.mrio_settings.sector_granularity,
        )
    else:
        raise ValueError(f"Scenario {config.scenario} not recognized")

    return wrap_rice_env(env, for_training=True)


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
        agent = PPO(**trainer_settings_to_ppo_kwargs(args.trainer_settings))
        agent, _metrics = agent.train(seed, env)

        logger.info("Evaluating agent (only rewards)... ")
        avg_reward = agent.evaluate(seed, env, num_eval_episodes=10)
        logger.info(f"Average reward over 10 episodes: {avg_reward}")

    NUM_EPISODES = 3
    OUTPUT_DIR = "episode_logs"
    env = with_log_info_fn(env, full_state_info_log_fn)
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
