import jax
import jax.numpy as jnp
import equinox as eqx
import chex
from functools import partial
from dataclasses import replace
from jice.environment.rice import Rice, EnvParams as RiceEnvParams
from jice.environment.base_and_wrappers import LogWrapper
from util import logwrapper_callback

class RandomAgent(eqx.Module):
    action_space: any
    def __init__(self, action_space, **kwargs):
        self.action_space = action_space

    def __call__(self, key: chex.PRNGKey):
        return self.action_space.sample(key)

@chex.dataclass(frozen=True)
class BaseTrainerParams:
    num_envs: int = 20
    total_timesteps: int = 1e6
    trainer_seed: int = 0
    backend: str = "cpu" # or "gpu"
    num_log_episodes_after_training: int = 10

def build_random_trainer(
        env_params: RiceEnvParams,
        trainer_params: BaseTrainerParams = BaseTrainerParams(),
    ):
    env = Rice(env_params)
    env = LogWrapper(env)
    eval_env_params = replace(
        env_params,
        train_env=False,
    )
    eval_env = Rice(eval_env_params)

    rng = jax.random.PRNGKey(trainer_params.trainer_seed)

    num_agents = env.settings.num_regions
    rng, reset_key = jax.random.split(rng)

    reset_keys = jax.random.split(reset_key, trainer_params.num_envs)
    obs_v, env_state_v = jax.vmap(
        env.reset, in_axes=(0)
    )(reset_keys)

    agent = RandomAgent(env.action_space)

    @partial(jax.jit, backend=trainer_params.backend)
    def eval_func(key: chex.PRNGKey):
        def step_env(carry, _):
            rng, obs, env_state, done, episode_reward = carry
            rng, step_key, sample_key = jax.random.split(rng, 3)

            sample_keys = jax.random.split(sample_key, num_agents)
            actions = jax.vmap(agent)(sample_keys)
            obs, env_state, reward, done, info = eval_env.step(
                step_key, env_state, actions
            )
            episode_reward += reward

            return (rng, obs, env_state, done, episode_reward), info
        
        rng, reset_key = jax.random.split(key)
        obs_v, env_state_v = eval_env.reset(reset_key)
        done = False
        episode_reward = jnp.zeros(num_agents)

        carry, episode_stats = jax.lax.scan( # episode_length is fixed, so we can scan
            step_env,
            (rng, obs_v, env_state_v, done, episode_reward),
            None,
            eval_env.episode_length
        )

        return carry[-1], episode_stats
    
    @partial(jax.jit, backend=trainer_params.backend)
    def train_function(rng: chex.PRNGKey = rng):

        def env_step(runner_state, _):
            rng, obs, env_state, counter = runner_state
            rng, key = jax.random.split(rng)

            # split key into a 2d array of num_envs x num_agents
            action_keys = jax.random.split(key, num_agents * trainer_params.num_envs).reshape(
                (trainer_params.num_envs, num_agents, 2)
            )
            action = jax.vmap(jax.vmap(agent))(action_keys)

            step_key = jax.random.split(key, trainer_params.num_envs)
            obs_v, env_state, reward_v, terminated, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_key, env_state, action)

            jax.debug.callback(logwrapper_callback, info, trainer_params.num_envs, counter)

            return (rng, obs_v, env_state, counter + 1), reward_v

        rng, train_key = jax.random.split(rng)
        initial_train_runner_state = (train_key, obs_v, env_state_v, 0)
        train_runner_state, train_rewards = jax.lax.scan(
            env_step,
            initial_train_runner_state,
            None,
            length=trainer_params.total_timesteps
        )

        if trainer_params.num_log_episodes_after_training > 0:
            rng, eval_key = jax.random.split(rng)
            eval_keys = jax.random.split(eval_key, trainer_params.num_log_episodes_after_training)
            eval_rewards, eval_logs = jax.vmap(eval_func)(eval_keys)
        else: 
            eval_rewards = None
            eval_logs = None

        return {
            "train_state": train_runner_state,
            "train_metrics": train_rewards,
            "eval_rewards": eval_rewards,
            "eval_logs": eval_logs,
        }

    return train_function