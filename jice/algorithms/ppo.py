import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import chex
from typing import List
from functools import partial
from dataclasses import replace
from typing import NamedTuple
from jice.environment import LogWrapper
from util import logwrapper_callback
from jice.algorithms import BaseTrainerParams
from jice.algorithms.networks import ActorNetworkMultiDiscrete, CriticNetwork
from jice.environment import Rice

def create_ppo_networks(
        key,
        in_shape: int,
        actor_features: List[int],
        critic_features: List[int],
        actions_nvec: int,
    ):
    """Create PPO networks (actor critic)"""
    actor_key, critic1_key = jax.random.split(key)
    actor = ActorNetworkMultiDiscrete(actor_key, in_shape, actor_features, actions_nvec)
    critic = CriticNetwork(critic1_key, in_shape, critic_features)
    return actor, critic

@chex.dataclass(frozen=True)
class PpoTrainerParams(BaseTrainerParams):
    learning_rate: float = 2.5e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    clip_coef: float = 0.2
    clip_coef_vf: float = 10.0 # Depends on the reward scaling !
    ent_coef_start: float = 1.0
    ent_coef_end: float = 0.01
    # ent_coef: float = 0.1
    vf_coef: float = 0.25

    num_steps: int = 250 # steps per environment
    num_minibatches: int = 4 # Number of mini-batches
    update_epochs: int = 4 # K epochs to update the policy

    # to be filled in runtime in at init:
    batch_size: int = 0 # batch size (num_envs * num_steps)
    minibatch_size: int = 0 # mini-batch size (batch_size / num_minibatches)
    num_iterations: int = 0 # number of iterations (total_timesteps / num_steps / num_envs)

    def __post_init__(self):
        object.__setattr__(self, 'num_iterations', int(self.total_timesteps // self.num_steps // self.num_envs))
        object.__setattr__(self, 'minibatch_size', int(self.num_envs * self.num_steps // self.num_minibatches))
        object.__setattr__(self, 'batch_size', int(self.minibatch_size * self.num_minibatches))
    

@chex.dataclass(frozen=True)
class Transition:
    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    value: chex.Array
    log_prob: chex.Array
    info: chex.Array

class TrainState(NamedTuple):
    actor: eqx.Module
    critic: eqx.Module
    optimizer_state: optax.OptState

# Jit the returned function, not this function
def build_ppo_trainer(
        env: Rice,
        trainer_params: PpoTrainerParams = PpoTrainerParams(),
    ):
    config = trainer_params
    eval_env = eqx.tree_at(lambda x: x.train_env, env, False)
    env = LogWrapper(env)

    observation_space = env.observation_space()
    action_space = env.action_space
    actions_nvec = action_space.nvec
    num_agents = env.num_regions

    # rng keys
    rng = jax.random.PRNGKey(config.trainer_seed)
    rng, network_key, reset_key = jax.random.split(rng, 3)
    # networks
    actor, critic = create_ppo_networks(
        key=network_key, 
        in_shape=observation_space.shape[-1],
        actor_features=[256, 256], 
        critic_features=[256, 256], 
        actions_nvec=actions_nvec
    )

    number_of_update_steps = config.num_iterations * config.num_minibatches * config.update_epochs
    learning_rate_schedule = optax.linear_schedule(
        init_value=config.learning_rate, 
        end_value=0, 
        transition_steps=number_of_update_steps
    )
    ent_coef_schedule = optax.linear_schedule(
        init_value=config.ent_coef_start, 
        end_value=config.ent_coef_end, 
        transition_steps=number_of_update_steps
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(
            learning_rate=learning_rate_schedule if config.anneal_lr else config.learning_rate, 
            eps=1e-5
        ),
    )
    optimizer_state = optimizer.init({
        "actor": actor,
        "critic": critic
    })

    train_state = TrainState(
        actor=actor,
        critic=critic,
        optimizer_state=optimizer_state,
    )

    reset_key = jax.random.split(reset_key, config.num_envs)
    obs_v, env_state_v = jax.vmap(env.reset, in_axes=(0))(reset_key)

    @partial(jax.jit, backend=trainer_params.backend)
    def eval_func(key: chex.PRNGKey, train_state: TrainState):
        def step_env(carry, _):
            rng, obs_v, env_state, done, episode_reward = carry
            rng, step_key, sample_key = jax.random.split(rng, 3)

            action_dist = jax.vmap(train_state.actor)(obs_v)
            actions = action_dist.sample(seed=sample_key)
            (obs_v, reward, done, discount, info), env_state = eval_env.step(
                step_key, env_state, actions
            )
            episode_reward += reward

            return (rng, obs, env_state, done, episode_reward), info
        
        rng, reset_key = jax.random.split(key)
        obs, env_state = eval_env.reset(reset_key)
        done = False
        episode_reward = jnp.zeros(num_agents)

        # we know the episode length is fixed, so lets scan
        carry, episode_stats = jax.lax.scan(
            step_env,
            (rng, obs, env_state, done, episode_reward),
            None,
            eval_env.episode_length
        )

        return carry[-1], episode_stats

    @partial(jax.jit, backend=trainer_params.backend)
    def train_func(rng: chex.PRNGKey = rng):
        
        # functions prepended with _ are called in jax.lax.scan of train_step

        def _env_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state
            rng, sample_key, step_key = jax.random.split(rng, 3)

            action_dist = jax.vmap(jax.vmap(train_state.actor))(last_obs)
            value = jax.vmap(jax.vmap(train_state.critic))(last_obs)
            action, log_prob = action_dist.sample_and_log_prob(seed=sample_key)
            step_keys = jax.random.split(step_key, config.num_envs)
            (obsv, reward, done, discount, info), env_state = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, action)

            # broadcast discount such that it has a dimension per agent
            discount = jnp.broadcast_to(discount, (num_agents, discount.shape[0])).T

            transition = Transition(
                observation=last_obs,
                action=action,
                reward=reward,
                discount=discount,
                value=value,
                log_prob=log_prob,
                info=info
            )
            
            runner_state = (train_state, env_state, obsv, rng)
            return runner_state, transition
        
        def _calculate_gae(gae_and_next_values, transition):
            gae, next_value = gae_and_next_values
            value, reward, gamma = (
                transition.value,
                transition.reward,
                transition.discount,
            )
            delta = reward + gamma * next_value - value
            gae = delta + gamma * config.gae_lambda * gae
            return (gae, value), (gae, gae + value)
        
        def _update_epoch(update_state, _):
            """ Do one epoch of update"""

            @eqx.filter_value_and_grad(has_aux=True)
            def __ppo_los_fn(params, trajectory_minibatch, advantages, returns):
                observations = trajectory_minibatch.observation
                actions = trajectory_minibatch.action
                init_log_prob = trajectory_minibatch.log_prob.sum(axis=-1)
                init_value = trajectory_minibatch.value
                action_dist = jax.vmap(jax.vmap(params["actor"]))(observations)
                value = jax.vmap(jax.vmap(params["critic"]))(observations)                
                log_prob = action_dist.log_prob(actions)
                entropy = action_dist.entropy().mean()
                log_prob = log_prob.sum(axis=-1)

                # actor loss 
                ratio = jnp.exp(log_prob - init_log_prob)
                _advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                actor_loss1 = _advantages * ratio
                actor_loss2 = (
                    jnp.clip(
                        ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                    ) * _advantages
                )
                actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

                # critic loss
                value_pred_clipped = init_value + (
                    jnp.clip(
                        value - init_value, -config.clip_coef_vf, config.clip_coef_vf
                    )
                )
                value_losses = jnp.square(value - returns)
                value_losses_clipped = jnp.square(value_pred_clipped - returns)
                value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()

                ent_coef = ent_coef_schedule(optimizer_state[1][1].count)
                
                # Total loss
                total_loss = (
                    actor_loss 
                    + config.vf_coef * value_loss
                    - ent_coef * entropy
                )
                return total_loss, (actor_loss, value_loss, entropy)
            
            def __update_over_minibatch(train_state: TrainState, minibatch):
                trajectory_mb, advantages_mb, returns_mb = minibatch

                # train worker
                (total_loss, (actor_loss, value_loss, entropy)), grads = __ppo_los_fn({
                        "actor": train_state.actor,
                        "critic": train_state.critic
                    }, trajectory_mb, advantages_mb, returns_mb
                )
                updates, optimizer_state = optimizer.update(grads, train_state.optimizer_state)
                new_networks = optax.apply_updates({
                    "actor": train_state.actor,
                    "critic": train_state.critic
                }, updates)

                train_state = TrainState(
                    actor=new_networks["actor"],
                    critic=new_networks["critic"],
                    optimizer_state=optimizer_state
                )
                return train_state, (total_loss, actor_loss, value_loss, entropy)
            
            train_state, trajectory_batch, advantages, returns, rng = update_state
            rng, key = jax.random.split(rng)

            batch_idx = jax.random.permutation(key, config.batch_size)
            batch = (trajectory_batch, advantages, returns)
            
            # reshape (flatten)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]), batch
            )
            # take from the batch in a new order (the order of the randomized batch_idx)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, batch_idx, axis=0), batch
            )
            # split in minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
            )
            # update over minibatches
            train_state, losses = jax.lax.scan(
                __update_over_minibatch, train_state, minibatches
            )
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            return update_state, losses

        def train_step(runner_state, _):

            # Do rollout of single trajactory (num_steps)
            runner_state, trajectory_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # calculate gae
            train_state, env_state, last_obs, rng = runner_state
            last_value = jax.vmap(jax.vmap(train_state.critic))(last_obs)
            _, (advantages, returns) = jax.lax.scan(
                _calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory_batch,
                reverse=True,
                unroll=16
            )
    
            # Do update epochs
            update_state = (train_state, trajectory_batch, advantages, returns, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )

            train_state = update_state[0]
            metric = trajectory_batch.info
            metric["loss_info"] = loss_info
            rng = update_state[-1]

            jax.debug.callback(logwrapper_callback, metric, config.num_envs)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric 

        rng, key = jax.random.split(rng)
        runner_state = (train_state, env_state_v, obs_v, key)
        runner_state, metrics = jax.lax.scan(
            train_step, runner_state, None, config.num_iterations
        )
        trained_train_state = runner_state[0]
        rng = runner_state[-1]

        if trainer_params.num_log_episodes_after_training > 0:
            rng, eval_key = jax.random.split(rng)
            eval_keys = jax.random.split(eval_key, trainer_params.num_log_episodes_after_training)
            eval_rewards, eval_logs = jax.vmap(
                eval_func, in_axes=(0, None)
            )(eval_keys, trained_train_state)
        else: 
            eval_rewards = None
            eval_logs = None

        return {
            "train_state": trained_train_state,
            "train_metrics": metrics,
            "eval_rewards": eval_rewards,
            "eval_logs": eval_logs,
        }

    return train_func