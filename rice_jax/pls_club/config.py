"""Factory + shared defaults for PLS club experiments."""

from typing import Any, Literal

import jax.numpy as jnp

from rice_jax.utils import load_region_yamls

from .env import PLSClubMediator, PLSNaiveClubRice

DEFAULT_NUM_REGIONS = 7

_SHIELD_DEFAULTS: dict[str, Any] = {
    "shield_strength": 0.8,
    "shield_mode": "constant",
    "shield_kappa": 10.0,
}

_NAIVE_ENV_DEFAULTS: dict[str, Any] = {
    "num_discrete_action_levels": 10,
    "diff_reward_mode": True,
    "negotiation_on": False,
    "club_min_rate": 0.5,
    "club_tariff": 0.5,
    **_SHIELD_DEFAULTS,
}

_MEDIATOR_ENV_DEFAULTS: dict[str, Any] = {
    "num_discrete_action_levels": 10,
    "diff_reward_mode": True,
    "negotiation_on": True,
    "mediator_reward_mode": "emissions",
    "fixed_club_params": None,
    **_SHIELD_DEFAULTS,
}

_PLS_TRAIN_DEFAULTS: dict[str, Any] = {
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


def pls_log_info_fn(state: dict, actions: dict, rewards=None, **kwargs) -> dict:
    """Per-step logger exposing membership/defection for posthoc analysis."""
    import_tariffs = state.get("import_tariffs")
    if import_tariffs is None:
        num_regions = state["club_membership"].shape[0]
        import_tariffs = jnp.zeros((num_regions, num_regions), dtype=jnp.float32)
    info = {
        "rewards": rewards,
        "club_membership": state["club_membership"],
        "club_defectors": state["club_defectors"],
        "mitigation_rates_all_regions": state["mitigation_rates_all_regions"],
        "global_emissions": state["global_emissions"],
        "import_tariffs": import_tariffs,
    }
    if "club_min_mitigation" in state:  # mediator variant only
        info["club_min_mitigation"] = state["club_min_mitigation"]
        info["club_tariff_rate"] = state["club_tariff_rate"]
    return info


def make_pls_env(
    variant: Literal["naive", "mediator"] = "naive",
    num_regions: int = DEFAULT_NUM_REGIONS,
    **overrides: Any,
) -> PLSNaiveClubRice | PLSClubMediator:
    if variant == "naive":
        cls, kwargs = PLSNaiveClubRice, dict(_NAIVE_ENV_DEFAULTS)
    elif variant == "mediator":
        cls, kwargs = PLSClubMediator, dict(_MEDIATOR_ENV_DEFAULTS)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    kwargs.update(overrides)
    region_params = load_region_yamls(num_regions)
    return cls(region_params=region_params, num_regions=num_regions, **kwargs)


def pls_train_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = dict(_PLS_TRAIN_DEFAULTS)
    kwargs.update(overrides)
    return kwargs
