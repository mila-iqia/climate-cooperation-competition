"""Small helpers to build and log a Rice scenario (defaults match rice_jax/main.py)."""

import dataclasses

import equinox as eqx
import jax
import jaxnasium as jym
from rice_jax.utils import (
    create_plots,
    full_state_info_log_fn,
    load_region_yamls,
    log_episode_to_json,
)

from rice_jax import Rice

# Added by LogWrapper / auto-reset; not part of the episode log
_WRAPPER_INFO_KEYS = (
    "returned_episode_returns",
    "returned_episode_lengths",
    "returned_episode",
    "_TERMINAL_OBSERVATION",
)


DEFAULT_PPO_PARAMS = {
    "learning_rate_start": 2.5e-4,
    "ent_coef_start": 0.01,
    "ent_coef_end": None,  # None = constant
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 1.0,
    "clip_coef": 0.2,
    "clip_coef_vf": 0.5,
    "vf_coef": 0.5,
    "num_steps": 100,  # steps per env per iteration
    "num_minibatches": 4,
    "num_epochs": 4,
    "normalize_observations": True,
    "normalize_rewards": False,
}


def make_env(env_cls=Rice, num_regions=3, **env_kwargs):
    """Create a (scenario) env; `env_kwargs` set any of its fields."""
    env = env_cls(
        region_params=load_region_yamls(num_regions),
        num_regions=num_regions,
        **{"action_window_size": 1, "preference_for_domestic": 0.9, **env_kwargs},
    )
    # Stack each region's discrete actions into one MultiDiscrete (faster PPO)
    env = jym.StackActionSpaceWrapper(env)
    # Record episode returns in the info (used for PPO's learning curves)
    return jym.LogWrapper(env)


def _set_log_info_fn(env, log_info_fn):
    """Replace `log_info_fn` on the Rice env inside the wrappers."""
    if isinstance(env, jym.Wrapper):
        return eqx.tree_at(
            lambda e: e._env, env, _set_log_info_fn(env._env, log_info_fn)
        )
    return dataclasses.replace(env, log_info_fn=log_info_fn)


@eqx.filter_jit  # compile once per env/agent, not on every call
def rollout(env, agent, key):
    """Play one episode with the (stochastic) policy; returns the infos stacked per step."""

    def step(carry, _):
        key, obs, state = carry
        key, action_key, step_key = jax.random.split(key, 3)
        action = agent.get_action(action_key, obs)
        (obs, _, _, _, info), state = env.step(step_key, state, action)
        info = {k: v for k, v in info.items() if k not in _WRAPPER_INFO_KEYS}
        return (key, obs, state), info

    obs, state = env.reset(key)
    _, infos = jax.lax.scan(step, (key, obs, state), None, length=env.episode_length)
    return infos


def log_episode(env, agent, key, output_dir="outputs"):
    """Play one episode, save the full state per step to JSON and plot it.

    Returns `(json_path, infos)`: `infos` holds every info key (including any your
    scenario adds in `generate_info`) stacked over the episode's steps.
    """
    env = _set_log_info_fn(env, full_state_info_log_fn)
    infos = rollout(env, agent, key)

    path = log_episode_to_json(
        infos, output_folder=f"{output_dir}/logs", agent=agent, env=env
    )
    create_plots(
        json_log_path=path,
        output_dir=f"{output_dir}/plots",
        parameter_keys=[
            "global_temperature",
            "gross_output_all_regions",
            "utility_all_regions",
            "actions.savings_rate",
            "actions.mitigation_rate",
        ],
    )
    return path, infos
