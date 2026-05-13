"""training_monitor.py

A thin PPO subclass (MonitoredPPO) that exposes per-iteration action and reward
statistics during training, together with ready-made log-function factories.

Usage
-----
from training_monitor import MonitoredPPO, make_csv_log_fn

log_fn = make_csv_log_fn("training_stats.csv")
agent = MonitoredPPO(**ppo_kwargs, log_function=log_fn, log_interval=0.02)
agent = agent.train(key, env)

The CSV will contain one row per callback call with columns:
  iteration, timestep, reward_mean, reward_sum,
  action_mean_0, action_mean_1, ...,
  action_var_0,  action_var_1,  ...

``make_csv_log_fn`` tolerates multi-agent / pytree actions by flattening all
action leaves and computing a single mean/variance vector across agents and
action dimensions.
"""

import csv
import os
from dataclasses import replace
from functools import partial
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxnasium import Environment
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.utils import scan_callback


# ── MonitoredPPO ──────────────────────────────────────────────────────────────


class MonitoredPPO(PPO):
    """PPO subclass that adds action / reward statistics to the training metric.

    The additional keys injected into the metric dict each iteration are:

      ``action_mean``  — JAX array, mean of actions over (steps × envs), one
                         value per action dimension (or a pytree if action is
                         a pytree).
      ``action_var``   — same shape, variance over (steps × envs).
      ``reward_mean``  — scalar, mean reward across all steps and envs.
      ``reward_sum``   — scalar, sum of rewards across all steps and envs.

    These are then visible inside the ``log_function`` callback alongside the
    standard ``returned_episode_returns`` / ``returned_episode`` keys that
    ``jym.LogWrapper`` provides.
    """

    def train(self, key, env: Environment, **hyperparams) -> "MonitoredPPO":
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            # Inner self tracks the evolving equinox state; outer self holds
            # static hyperparams (num_envs, num_steps, etc.)
            self_: MonitoredPPO = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self_._collect_rollout(
                rollout_state, env
            )

            # ── Augmented metric ──────────────────────────────────────────
            # trajectory_batch.action and .reward may be pytrees (multi-agent
            # environments store per-agent values as dicts), so use
            # jax.tree for all aggregation.
            base_info = trajectory_batch.info or {}
            action_mean = jax.tree.map(
                lambda a: a.mean(axis=(0, 1)), trajectory_batch.action
            )
            action_var = jax.tree.map(
                lambda a: a.var(axis=(0, 1)), trajectory_batch.action
            )
            # Flatten reward pytree → single mean/sum over all agents & steps
            reward_leaves = jax.tree.leaves(trajectory_batch.reward)
            reward_stack = jnp.stack([r.ravel() for r in reward_leaves])
            metric = {
                **base_info,
                "action_mean": action_mean,
                "action_var": action_var,
                "reward_mean": reward_stack.mean(),
                "reward_var": reward_stack.var(),
                "reward_sum": reward_stack.sum(),
            }

            # ── Standard PPO post-processing + update ─────────────────────
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


# ── RCPOMonitoredPPO ──────────────────────────────────────────────────────────


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

    def train(self, key, env: Environment, **hyperparams) -> "RCPOMonitoredPPO":
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

            # ── RCPO λ update (Tessler et al. 2019, eq. 6) ───────────────
            # cbam_cost_all_regions shape: (num_steps, num_envs, num_regions)
            cbam_costs = trajectory_batch.info["cbam_cost_all_regions"]
            mean_cost = cbam_costs.mean()
            # env_state is a LogEnvState; inner RICE state is env_state.env_state
            inner = env_state.env_state
            old_lambda = inner["cbam_lambda"].mean()
            new_lambda = jnp.maximum(
                0.0,
                old_lambda + self.rcpo_eta_lambda * (mean_cost - self.rcpo_alpha_target),
            )
            updated_inner = {
                **inner,
                "cbam_lambda": jnp.full_like(
                    inner["cbam_lambda"], new_lambda
                ),
            }
            env_state = eqx.tree_at(
                lambda s: s.env_state, env_state, updated_inner
            )

            # ── Augmented metric ──────────────────────────────────────────
            # Exclude per-step cost array from scan output to save memory;
            # the aggregated scalar mean_cbam_cost is enough for logging.
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

            # ── Standard PPO post-processing + update ─────────────────────
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


# ── Log-function factories ────────────────────────────────────────────────────


def make_csv_log_fn(
    csv_path: str,
) -> Callable:
    """Return a log-function that appends one row per callback to *csv_path*.

    The CSV is created (with header) on the first call.  Safe to use across
    multiple training runs by choosing a unique path each time.

    Columns
    -------
    iteration, timestep, reward_mean, reward_sum,
    action_mean_0, action_mean_1, ...,
    action_var_0,  action_var_1,  ...

    If episode returns are available (env wrapped with ``LogWrapper``), also:
    ep_return_mean, ep_return_std
    """
    state = {"writer": None, "file": None, "initialized": False}

    def log_fn(data: dict, iteration: int):
        import numpy as _np

        # ── flatten action pytree → 1-D numpy arrays ─────────────────────
        def _flatten(tree):
            leaves = jax.tree.leaves(tree)
            return _np.concatenate([_np.array(l).ravel() for l in leaves])

        act_mean = _flatten(data.get("action_mean", _np.array([])))
        act_var = _flatten(data.get("action_var", _np.array([])))

        reward_mean = float(_np.array(data.get("reward_mean", _np.nan)))
        reward_var = float(_np.array(data.get("reward_var", _np.nan)))
        reward_sum = float(_np.array(data.get("reward_sum", _np.nan)))

        # timestep estimate from LogWrapper if available
        iter_int = int(_np.array(iteration))
        if "timestep" in data:
            ts_arr = _np.array(data["timestep"])
            timestep = int(ts_arr.max()) if ts_arr.size > 0 else iter_int
        else:
            timestep = iter_int

        # episodic returns from LogWrapper
        # returned_episode_returns may be a per-agent pytree (dict) in
        # multi-agent envs; flatten all leaves and index with the done mask.
        ep_return_mean = _np.nan
        ep_return_std = _np.nan
        if "returned_episode_returns" in data and "returned_episode" in data:
            done = _np.array(data["returned_episode"])
            if done.ndim > 0 and done.any():
                ret_leaves = jax.tree.leaves(data["returned_episode_returns"])
                ep_rets = _np.concatenate(
                    [_np.array(l)[done].ravel() for l in ret_leaves]
                )
                if ep_rets.size > 0:
                    ep_return_mean = float(ep_rets.mean())
                    ep_return_std = float(ep_rets.std())

        row = {
            "iteration": iter_int,
            "timestep": timestep,
            "reward_mean": round(reward_mean, 6),
            "reward_var": round(reward_var, 6) if not _np.isnan(reward_var) else "",
            "reward_sum": round(reward_sum, 4),
            "ep_return_mean": round(ep_return_mean, 4) if not _np.isnan(ep_return_mean) else "",
            "ep_return_std": round(ep_return_std, 4) if not _np.isnan(ep_return_std) else "",
        }

        # RCPO fields (present when using RCPOMonitoredPPO)
        cbam_lambda = data.get("cbam_lambda")
        if cbam_lambda is not None:
            row["cbam_lambda"] = round(float(_np.array(cbam_lambda)), 8)
        mean_cbam_cost = data.get("mean_cbam_cost")
        if mean_cbam_cost is not None:
            row["mean_cbam_cost"] = round(float(_np.array(mean_cbam_cost)), 8)

        # Absolute utility level (present when using cbam_and_utility_log_info_fn).
        # trajectory_batch.info["mean_utility_step"] has shape (num_steps, num_envs);
        # we average over both dims to get a single scalar per training iteration.
        mean_utility_step = data.get("mean_utility_step")
        if mean_utility_step is not None:
            row["mean_utility"] = round(float(_np.array(mean_utility_step).mean()), 6)

        # RICE economic decomposition scalars (same shape convention as mean_utility_step)
        for _key, _col in [
            ("mean_gross_output_step",   "mean_gross_output"),
            ("mean_consumption_step",    "mean_consumption"),
            ("mean_abatement_cost_step", "mean_abatement_cost"),
            ("mean_mitigation_step",     "mean_mitigation"),
        ]:
            _val = data.get(_key)
            if _val is not None:
                row[_col] = round(float(_np.array(_val).mean()), 6)

        # Per-region arrays (shape: num_steps × num_envs × NR after scan).
        # Written as <metric>_r0 .. _r{NR-1} by averaging over steps and envs.
        for _key in [
            "utility_per_region",
            "gross_output_per_region",
            "consumption_per_region",
            "abatement_cost_per_region",
            "mitigation_per_region",
        ]:
            _val = data.get(_key)
            if _val is not None:
                _arr = _np.array(_val)          # (num_steps, num_envs, NR)
                _region_means = _arr.mean(axis=tuple(range(_arr.ndim - 1)))  # (NR,)
                _col_prefix = _key.replace("_per_region", "")
                for _r, _v in enumerate(_region_means):
                    row[f"{_col_prefix}_r{_r}"] = round(float(_v), 6)

        row.update({
            **{f"action_mean_{i}": round(float(v), 6) for i, v in enumerate(act_mean)},
            **{f"action_var_{i}": round(float(v), 6) for i, v in enumerate(act_var)},
        })

        if not state["initialized"]:
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
            state["file"] = open(csv_path, "w", newline="")
            state["writer"] = csv.DictWriter(state["file"], fieldnames=list(row.keys()))
            state["writer"].writeheader()
            state["initialized"] = True

        state["writer"].writerow(row)
        state["file"].flush()

    return log_fn


def make_print_log_fn(
    action_labels: Optional[list[str]] = None,
    num_iterations: Optional[int] = None,
) -> Callable:
    """Return a log-function that prints a compact summary to stdout.

    Each line shows: iteration, reward_mean, optional ep_return, then
    per-action-dim  mean±std  (std = sqrt(var) over the rollout window).

    Parameters
    ----------
    action_labels:
        Optional list of names for each action dimension.
        If None, dimensions are labelled ``a0``, ``a1``, etc.
    num_iterations:
        If provided, a tqdm progress bar is created and advanced together
        with the printed stats.
    """
    tqdm_bar: list = []

    def log_fn(data: dict, iteration: int):
        import numpy as _np

        def _flatten(tree):
            leaves = jax.tree.leaves(tree)
            if not leaves:
                return _np.array([])
            return _np.concatenate([_np.array(l).ravel() for l in leaves])

        iter_int = int(_np.array(iteration))
        act_mean = _flatten(data.get("action_mean", []))
        act_var = _flatten(data.get("action_var", []))
        rew_mean = float(_np.array(data.get("reward_mean", _np.nan)))
        rew_std = float(_np.sqrt(_np.array(data.get("reward_var", _np.nan))))

        # Episode returns from LogWrapper
        # returned_episode_returns may be a per-agent pytree in multi-agent envs.
        ep_str = ""
        if "returned_episode_returns" in data and "returned_episode" in data:
            done = _np.array(data["returned_episode"])
            if done.ndim > 0 and done.any():
                ret_leaves = jax.tree.leaves(data["returned_episode_returns"])
                ep_rets = _np.concatenate(
                    [_np.array(l)[done].ravel() for l in ret_leaves]
                )
                if ep_rets.size > 0:
                    ep_str = f"  ep_ret={ep_rets.mean():.3f}±{ep_rets.std():.3f}"

        # RCPO info suffix
        rcpo_str = ""
        if "cbam_lambda" in data:
            lam = float(_np.array(data["cbam_lambda"]))
            cost = float(_np.array(data.get("mean_cbam_cost", _np.nan)))
            rcpo_str = f"  λ={lam:.6f} c̄={cost:.6f}"

        if act_mean.size > 0:
            labels = action_labels or [f"a{i}" for i in range(len(act_mean))]
            mean_parts = "  ".join(
                f"{l}={v:.3f}±{s:.3f}"
                for l, v, s in zip(labels, act_mean, _np.sqrt(act_var))
            )
            line = f"[{iter_int:5d}] rew={rew_mean:.4f}\u00b1{rew_std:.4f}{ep_str}{rcpo_str}  |  {mean_parts}"
        else:
            line = f"[{iter_int:5d}] rew={rew_mean:.4f}\u00b1{rew_std:.4f}{ep_str}{rcpo_str}"

        if num_iterations is not None:
            try:
                import tqdm.auto as _tqdm_mod
                if not tqdm_bar:
                    tqdm_bar.append(
                        _tqdm_mod.tqdm(total=num_iterations, desc="Training", unit=" iters")
                    )
                tqdm_bar[0].set_postfix_str(line)
                # Sync bar position directly from the scan-step index so the
                # percentage is correct regardless of log_interval / callback_interval.
                # iter_int is the raw scan-step passed by scan_callback; setting
                # .n directly keeps the display in sync with actual progress.
                tqdm_bar[0].n = min(iter_int + 1, num_iterations)
                tqdm_bar[0].refresh()
                return
            except ImportError:
                pass

        print(line)

    return log_fn


def make_combined_log_fn(*fns: Callable) -> Callable:
    """Combine multiple log functions into one (each receives the same data)."""
    def log_fn(data: dict, iteration: int):
        for fn in fns:
            fn(data, iteration)
    return log_fn
