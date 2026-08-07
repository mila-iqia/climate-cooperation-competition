"""Reward Constrained Policy Optimization (RCPO) for CBAM training."""

from __future__ import annotations

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
from jaxnasium import Environment

from ..utils import actions_rewards_info_log_fn
from .monitor import LoggingPPO


class RCPOMonitoredPPO(LoggingPPO):
    """LoggingPPO with RCPO Lagrange-multiplier update for CBAM cost.

    Reference: Tessler et al. (2019), "Reward Constrained Policy Optimization",
    ICLR 2019, §5.2 (mean-value constraint).

    The penalised reward is  r̂ = r - λ·c  where c is the per-step CBAM
    cost (applied in ``RiceMRIO.generate_rewards`` under
    ``reward_mode="additive_cbam"``) and λ auto-tunes on a slower timescale
    than the policy:

        λ_{k+1} = max(0, λ_k + η_λ · (E[c] - α_target))

    Only ``_collect_rollout`` is overridden here (λ update + cost metrics).
    Metric summarization for log callbacks comes from :class:`LoggingPPO`.
    ``train()`` returns a ``PPOAgent`` (jaxnasium agent/trainer split).

    Requirements on the environment:
    1. ``reward_mode="additive_cbam"`` on ``RiceMRIO``
    2. ``log_info_fn=rcpo_cbam_log_info_fn`` (or equivalent that exposes
       ``cbam_cost_all_regions``)
    3. Training env wrapped in ``LogWrapper`` (as ``make_canonical_env`` does)
    """

    rcpo_eta_lambda: float = eqx.field(static=True, default=5e-7)
    rcpo_alpha_target: float = eqx.field(static=True, default=0.01)

    def _collect_rollout(self, agent, rollout_state, env: Environment, length=None):
        (env_state, last_obs, rng), trajectory_batch = super()._collect_rollout(
            agent, rollout_state, env, length=length
        )

        cbam_costs = trajectory_batch.info["cbam_cost_all_regions"]
        mean_cost = cbam_costs.mean()

        # LogWrapper state: .env_state is the (possibly vmapped) Rice state dict
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
        env_state = eqx.tree_at(lambda s: s.env_state, env_state, updated_inner)

        # Metric enrichment for log callbacks; drop bulky per-step cost tensor
        # (mean_cbam_cost is enough). actions/rewards stay until LoggingPPO
        # summarizes them in train_iteration.
        info = {
            k: v
            for k, v in (trajectory_batch.info or {}).items()
            if k != "cbam_cost_all_regions"
        }
        info["cbam_lambda"] = new_lambda
        info["mean_cbam_cost"] = mean_cost
        trajectory_batch = replace(trajectory_batch, info=info)

        return (env_state, last_obs, rng), trajectory_batch


def rcpo_cbam_log_info_fn(state: dict, actions: dict, **kwargs) -> dict:
    """log_info_fn for RCPO training: CBAM cost plus actions/rewards.

    Pass this to the RiceMRIO constructor::

        env = RiceMRIO(..., log_info_fn=rcpo_cbam_log_info_fn)
    """
    return {
        **actions_rewards_info_log_fn(state, actions, **kwargs),
        "cbam_cost_all_regions": state["cbam_cost_all_regions"],
    }
