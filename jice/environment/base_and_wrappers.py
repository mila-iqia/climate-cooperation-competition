import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, Optional
from functools import partial
from gymnax.environments import environment
import equinox as eqx

@chex.dataclass
class EnvState:
    current_timestep: int

@chex.dataclass
class EnvParams:
    max_steps_in_episode: int = 1

class JaxBaseEnv(eqx.Module):
    """
        Base class for Jax-based environments.
        Implements the step function with auto-reset functionality.
        Child classes should implement the step_env and reset_env functions.   
        Do not override the step and reset functions.
    """

    @property
    def default_params(self) -> EnvParams:
        return NotImplementedError
    
    def __init__(self, params: Optional[EnvParams] = None):
        raise NotImplementedError("This is an abstract class")
    
    # @partial(jax.jit, static_argnums=(0))
    @eqx.filter_jit
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """
            Performs step transitions in the environment.
            Additionally performs an auto-reset of the environment if the episode is done.
        """
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)

        obs_step, state_step, reward, done, info = self.step_env(key, state, action, params)
        obs_reset, state_reset = self.reset_env(key_reset, params) 

        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_reset, state_step
        )
        obs = jax.lax.select(done, obs_reset, obs_step)

        return obs, state, reward, done, info

    @eqx.filter_jit
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state
    
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        raise NotImplementedError
    
class MultiDiscrete(object):
    """
        Minimal implementation of a MultiDiscrete space.
        input nvec: array of integers representing the number of discrete values in each dimension
    """
    def __init__(self, nvec: chex.Array, dtype: jnp.dtype = jnp.int8, start: jnp.int8 = 0):
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


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

@chex.dataclass(frozen=True)
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    returned_episode_returns: float
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns. From PureJaxRL"""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(
            env_state=env_state, 
            episode_returns=jnp.zeros(self._env.settings.num_regions), 
            returned_episode_returns=jnp.zeros(self._env.settings.num_regions), 
            timestep=0
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
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
        return obs, state, reward, done, info
