import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

import cloudpickle
import jax
import optax
from jaxnasium.algorithms import RLAlgorithm
from jaxtyping import Array, PRNGKeyArray

from rice_jax import Rice
from rice_jax.utils import i_to_agent_str

if TYPE_CHECKING:
    from main import Config  # Only imported for type checking

logger = logging.getLogger(__name__)


class FixedActionAgent:
    """
    A debug agent that takes a fixed action for all regions.
    Essentially just requires a `get_action` method.
    """

    def __init__(self, env: Rice):
        self.env = env
        self.state = 0  # Unused, but makes it compatible with RLAlgorithms

        random_action = env.sample_action(jax.random.PRNGKey(0))
        random_action_one_agent = random_action[i_to_agent_str(0)]
        zero_action_one_agent = optax.tree.zeros_like(random_action_one_agent)

        default_actions = zero_action_one_agent
        default_actions["savings_rate"] = 2.5
        default_actions["mitigation_rate"] = 0.0
        default_actions = {
            i_to_agent_str(i): default_actions for i in range(env.num_regions)
        }  # do this for all agents
        self.default_actions = default_actions

    def get_action(self, key: PRNGKeyArray, dummy_agent_state: Any, obs: Any) -> Array:
        return self.default_actions


def save_agent(agent: RLAlgorithm | FixedActionAgent, config: "Config") -> None:
    SAVE_MODEL_PATH = "saved_models/"
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    # Saving the agent
    _time = time.time()
    name = f"{config.scenario}_{config.env_settings.num_regions}"
    model_name = f"{name}_{int(_time)}"
    logger.info(f"saving model to {SAVE_MODEL_PATH}{model_name}.pkl")
    with open(f"{SAVE_MODEL_PATH}{model_name}.pkl", "wb") as f:
        cloudpickle.dump(agent, f)
    # agent.save(f"{SAVE_MODEL_PATH}{model_name}.eqx")


def load_agent(path: str) -> RLAlgorithm | FixedActionAgent:
    logger.info(f"loading model from {path}")
    with open(path, "rb") as f:
        return cloudpickle.load(f)


def run_single_episode(
    key: PRNGKeyArray, env: Rice, agent: RLAlgorithm | FixedActionAgent
) -> Array:
    """Play an episode in the environment using the agent.
    Returns the info dicts per step as a stacked dictionary.
    """

    # log_state_keys = []
    log_exclude_keys = [
        "returned_episode_returns",
        "returned_episode_lengths",
        "returned_episode",
        "_TERMINAL_OBSERVATION",
        "DISCOUNT",
    ]

    def do_step(carry, _):
        key, obs, state = carry
        keys = jax.random.split(key, 3)
        action = (
            agent.get_action(keys[0], obs)
            if not isinstance(agent, FixedActionAgent)
            else agent.get_action(keys[0], agent.state, obs)
        )
        (obs, reward, _, _, info), state = env.step(keys[1], state, action)
        info = {k: v for k, v in info.items() if k not in log_exclude_keys}
        return (keys[2], obs, state), info

    obs, state = env.reset(key)
    _, info_stack = jax.lax.scan(
        do_step,
        (key, obs, state),
        None,
        length=env.episode_length,
    )
    return info_stack


# ── Experiment directory helpers ─────────────────────────────────────────────
#
# Validation scripts call get_output_dir() / get_log_dir() instead of hard-
# coding "plots" / "training_logs".  When the CBAM_EXPERIMENT_DIR env-var is
# set (by run_cbam_experiment.py), outputs are redirected into the experiment
# folder; otherwise the original flat directories are used unchanged so every
# script still works standalone.

_EXPDIR_ENV_VAR = "CBAM_EXPERIMENT_DIR"


def get_experiment_dir() -> str | None:
    """Return the current experiment root directory, or None if unset."""
    return os.environ.get(_EXPDIR_ENV_VAR) or None


def get_output_dir(default: str = "plots") -> str:
    """Return the plots sub-directory for this run.

    If CBAM_EXPERIMENT_DIR is set, returns ``<experiment_dir>/plots``.
    Otherwise returns *default* (preserving the original behaviour).
    """
    base = get_experiment_dir()
    return os.path.join(base, "plots") if base else default


def get_log_dir(default: str = "training_logs") -> str:
    """Return the logs sub-directory for this run.

    If CBAM_EXPERIMENT_DIR is set, returns ``<experiment_dir>/logs``.
    Otherwise returns *default*.
    """
    base = get_experiment_dir()
    return os.path.join(base, "logs") if base else default


def save_run_config(config: dict) -> None:
    """Write *config* as ``config.json`` in the experiment directory.

    No-op if CBAM_EXPERIMENT_DIR is not set.
    """
    base = get_experiment_dir()
    if base is None:
        return
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2, default=str)

