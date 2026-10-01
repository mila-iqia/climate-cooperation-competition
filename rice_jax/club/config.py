"""Factory + shared defaults for club-mediator experiments."""

from typing import Any

from rice_jax.utils import load_region_yamls

from .env import RiceClubMediator

DEFAULT_NUM_REGIONS = 7

_CLUB_ENV_DEFAULTS: dict[str, Any] = {
    "num_discrete_action_levels": 10,
    "diff_reward_mode": True,
    "negotiation_on": True,
    "mediator_reward_mode": "emissions",
    "fixed_club_params": None,
}

_CLUB_TRAIN_DEFAULTS: dict[str, Any] = {
    "total_timesteps": 200_000,
    "num_steps": 100,
    "num_envs": 4,
    "learning_rate_start": 3e-4,
    "learning_rate_end": 0.0,
    "num_minibatches": 4,
    "num_epochs": 8,
    "ent_coef_start": 0.01,
    "ent_coef_end": None,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 1.0,
    "clip_coef": 0.2,
    "clip_coef_vf": 0.5,
    "vf_coef": 0.5,
    "normalize_observations": True,
    "normalize_rewards": True,
}


def club_log_info_fn(state: dict, actions: dict, rewards=None, **kwargs) -> dict:
    """Per-step logger exposing club state for posthoc membership analysis."""
    return {
        "rewards": rewards,
        "club_membership": state["club_membership"],
        "club_min_mitigation": state["club_min_mitigation"],
        "club_tariff_rate": state["club_tariff_rate"],
        "mitigation_rates_all_regions": state["mitigation_rates_all_regions"],
        "global_emissions": state["global_emissions"],
    }


def make_club_env(
    num_regions: int = DEFAULT_NUM_REGIONS, **overrides: Any
) -> RiceClubMediator:
    kwargs = dict(_CLUB_ENV_DEFAULTS)
    kwargs.update(overrides)
    region_params = load_region_yamls(num_regions)
    return RiceClubMediator(
        region_params=region_params, num_regions=num_regions, **kwargs
    )


def club_train_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = dict(_CLUB_TRAIN_DEFAULTS)
    kwargs.update(overrides)
    return kwargs
