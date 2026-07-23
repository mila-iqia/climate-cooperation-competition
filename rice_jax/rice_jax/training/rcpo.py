"""Reward Constrained Policy Optimization (RCPO) for CBAM training."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp

from jaxnasium import Environment
from jaxnasium.algorithms.utils import scan_callback

from .monitor import MonitoredPPO


class RCPOMonitoredPPO(MonitoredPPO):
    """MonitoredPPO with RCPO Lagrange-multiplier update for CBAM cost.

    Reference: Tessler et al. (2019), "Reward Constrained Policy Optimization",
    ICLR 2019, §5.2 (mean-value constraint).

    The penalised reward is  r̂ = r - λ·c  where c is the per-step CBAM
    cost and λ auto-tunes on a slower timescale than the policy:

        λ_{k+1} = max(0, λ_k + η_λ · (E[c] - α_target))

    Requirements on the environment:
    1. ``reward_mode="additive_cbam"`` on ``RiceMRIO``
    2. A ``log_info_fn`` that returns
       ``{"cbam_cost_all_regions": state["cbam_cost_all_regions"]}``
       (see :func:`rcpo_cbam_log_info_fn`).
    """

    rcpo_eta_lambda: float = eqx.field(static=True, default=5e-7)
    rcpo_alpha_target: float = eqx.field(static=True, default=0.01)

    def train(self, key, env: Environment, **hyperparams) -> RCPOMonitoredPPO:
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            self_: RCPOMonitoredPPO = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self_._collect_rollout(
                rollout_state, env
            )

            cbam_costs = trajectory_batch.info["cbam_cost_all_regions"]
            mean_cost = cbam_costs.mean()
            inner = env_state.env_state
            old_lambda = inner["cbam_lambda"].mean()
            new_lambda = jnp.maximum(
                0.0,
                old_lambda + self.rcpo_eta_lambda * (mean_cost - self.rcpo_alpha_target),
            )
            updated_inner = {
                **inner,
                "cbam_lambda": jnp.full_like(inner["cbam_lambda"], new_lambda),
            }
            env_state = eqx.tree_at(
                lambda s: s.env_state, env_state, updated_inner
            )

            base_info = {
                k: v
                for k, v in (trajectory_batch.info or {}).items()
                if k != "cbam_cost_all_regions"
            }
            action_mean = jax.tree.map(
                lambda a: a.mean(axis=(0, 1)), trajectory_batch.action
            )
            action_var = jax.tree.map(
                lambda a: a.var(axis=(0, 1)), trajectory_batch.action
            )
            reward_leaves = jax.tree.leaves(trajectory_batch.reward)
            reward_stack = jnp.stack([r.ravel() for r in reward_leaves])
            metric = {
                **base_info,
                "action_mean": action_mean,
                "action_var": action_var,
                "reward_mean": reward_stack.mean(),
                "reward_var": reward_stack.var(),
                "reward_sum": reward_stack.sum(),
                "cbam_lambda": new_lambda,
                "mean_cbam_cost": mean_cost,
            }

            trajectory_batch, updated_state = self_._postprocess_rollout(
                trajectory_batch, self_.state
            )
            updated_state = self_._update_agent_state(
                rng, updated_state, trajectory_batch
            )
            self_ = replace(self_, state=updated_state)

            runner_state = (self_, env_state, last_obs, rng)
            return runner_state, metric

        env = self.__check_env__(env, vectorized=True)
        self = replace(self, **hyperparams)

        if not self.is_initialized:
            self = self.init_state(key, env)

        obsv, env_state = env.reset(jax.random.split(key, self.num_envs))
        runner_state = (self, env_state, obsv, key)
        runner_state, _metrics = jax.lax.scan(
            train_iteration, runner_state, jnp.arange(self.num_iterations)
        )
        return runner_state[0]


def rcpo_cbam_log_info_fn(state: dict, actions: dict) -> dict:
    """log_info_fn for RCPO training: captures per-step CBAM cost.

    Pass this to the RiceMRIO constructor::

        env = RiceMRIO(..., log_info_fn=rcpo_cbam_log_info_fn)
    """
    return {"cbam_cost_all_regions": state["cbam_cost_all_regions"]}
