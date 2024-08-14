import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, NamedTuple
import equinox as eqx

@chex.dataclass(frozen=True)
class EnvState:
    time: int

class TimeStep(NamedTuple):
    observation: chex.Array
    reward: Union[float, chex.Array]
    terminated: bool
    truncated: bool
    info: dict

class DiscountedTimeStep(NamedTuple):
    observation: chex.Array
    reward: Union[float, chex.Array]
    terminated: bool
    discount: Union[float, chex.Array]
    info: dict

class JaxBaseEnv(eqx.Module):
    """
        Base class for a JAX environment.
        This class inherits from eqx.Module, meaning it is a PyTree node and a dataclass.
        set params by setting the properties of the class.
        Much of the modules are inspired by the Gymnax base class.
    """

    # example_property: int = 0

    def __check_init__(self):
        """
            An equinox module that always runs on initialization.
            Can be used to check if parameters are set correctly, without overwriting __init__.
        """
        pass

    def step(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]) -> Tuple[Union[TimeStep, DiscountedTimeStep], EnvState]:
        """Performs step transitions in the environment."""

        (obs_step, reward, terminated, truncated, info), state_step = self.step_env(key, state, action)
        obs_reset, state_reset = self.reset_env(key) 

        # Auto-reset environment based on termination
        done = terminated or truncated
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.lax.cond(done, lambda: obs_reset, lambda: obs_step)

        return TimeStep(obs, reward, terminated, truncated, info), state

    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        obs, state = self.reset_env(key)
        return obs, state
    
    def reset_env(self, key: chex.PRNGKey) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset transition."""
        raise NotImplementedError

    def step_env(self, key: chex.PRNGKey, state: EnvState, action: Union[int, float, chex.Array]) -> Tuple[Union[TimeStep, DiscountedTimeStep], EnvState]:
        """Environment-specific step transition."""
        raise NotImplementedError
    
class MultiDiscrete(object):
    """
        Minimal implementation of a MultiDiscrete space.
        input nvec: array of integers representing the number of discrete values in each dimension
    """
    def __init__(self, nvec: chex.Array, dtype: jnp.dtype = jnp.int8, start: int = 0):
        assert len(nvec.shape) == 1 and nvec.shape[0] > 0, "nvec must be a 1D array with at least one element"
        assert jnp.all(nvec > 0), "All elements in nvec must be greater than 0"
        self.nvec = nvec
        self.shape = nvec.shape
        self.n = self.shape[0]
        self.num_action_types = self.shape[0]
        self.num_actions_per_type = nvec
        self.dtype = dtype
        self.start = start

        # check if all elements in nvec are the same
        if jnp.all(nvec == nvec[0]):
            self.uniform = True
        else:
            self.uniform = False

    def sample(self, key: chex.PRNGKey) -> chex.Array:
        return jax.random.randint(key, self.shape, self.start, self.nvec)
        
    def contains(self, x: chex.Array) -> bool:
        return x.shape == self.shape and jnp.all(x >= self.start) and jnp.all(x < self.nvec)


class JaxEnvWrapper(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    returned_episode_returns: float
    timestep: int

class LogWrapper(JaxEnvWrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[chex.Array, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state=env_state, 
            episode_returns=jnp.zeros(self._env.num_regions), 
            returned_episode_returns=jnp.zeros(self._env.num_regions), 
            timestep=0
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float, chex.Array],
    ) -> Tuple[Union[TimeStep, DiscountedTimeStep], LogEnvState]:
        (obs, reward, terminated, truncated, info), env_state = self._env.step(
            key, state.env_state, action
        )
        done = terminated | truncated
        new_episode_return = state.episode_returns + reward
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode"] = done
        info["timestep"] = state.timestep
        return TimeStep(obs, reward, terminated, truncated, info), state
