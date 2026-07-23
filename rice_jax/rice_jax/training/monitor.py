"""MonitoredPPO and training log-function factories."""

from __future__ import annotations

import csv
import os
from dataclasses import replace
from typing import Callable, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxnasium import Environment
from jaxnasium.algorithms import PPO
from jaxnasium.algorithms.utils import scan_callback


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

    def train(self, key, env: Environment, **hyperparams) -> MonitoredPPO:
        @scan_callback(
            callback_fn=self.log_function,
            callback_interval=self.log_interval,
            n=self.num_iterations,
        )
        def train_iteration(runner_state, _):
            self_: MonitoredPPO = runner_state[0]
            rollout_state = runner_state[1:]
            (env_state, last_obs, rng), trajectory_batch = self_._collect_rollout(
                rollout_state, env
            )

            base_info = trajectory_batch.info or {}
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


def make_csv_log_fn(csv_path: str) -> Callable:
    """Return a log-function that appends one row per callback to *csv_path*."""
    state = {"writer": None, "file": None, "initialized": False}

    def log_fn(data: dict, iteration: int):
        def _flatten(tree):
            leaves = jax.tree.leaves(tree)
            return np.concatenate([np.array(l).ravel() for l in leaves])

        act_mean = _flatten(data.get("action_mean", np.array([])))
        act_var = _flatten(data.get("action_var", np.array([])))

        reward_mean = float(np.array(data.get("reward_mean", np.nan)))
        reward_var = float(np.array(data.get("reward_var", np.nan)))
        reward_sum = float(np.array(data.get("reward_sum", np.nan)))

        iter_int = int(np.array(iteration))
        if "timestep" in data:
            ts_arr = np.array(data["timestep"])
            timestep = int(ts_arr.max()) if ts_arr.size > 0 else iter_int
        else:
            timestep = iter_int

        ep_return_mean = np.nan
        ep_return_std = np.nan
        if "returned_episode_returns" in data and "returned_episode" in data:
            done = np.array(data["returned_episode"])
            if done.ndim > 0 and done.any():
                ret_leaves = jax.tree.leaves(data["returned_episode_returns"])
                ep_rets = np.concatenate(
                    [np.array(l)[done].ravel() for l in ret_leaves]
                )
                if ep_rets.size > 0:
                    ep_return_mean = float(ep_rets.mean())
                    ep_return_std = float(ep_rets.std())

        row = {
            "iteration": iter_int,
            "timestep": timestep,
            "reward_mean": round(reward_mean, 6),
            "reward_var": round(reward_var, 6) if not np.isnan(reward_var) else "",
            "reward_sum": round(reward_sum, 4),
            "ep_return_mean": round(ep_return_mean, 4) if not np.isnan(ep_return_mean) else "",
            "ep_return_std": round(ep_return_std, 4) if not np.isnan(ep_return_std) else "",
        }

        cbam_lambda = data.get("cbam_lambda")
        if cbam_lambda is not None:
            row["cbam_lambda"] = round(float(np.array(cbam_lambda)), 8)
        mean_cbam_cost = data.get("mean_cbam_cost")
        if mean_cbam_cost is not None:
            row["mean_cbam_cost"] = round(float(np.array(mean_cbam_cost)), 8)

        mean_utility_step = data.get("mean_utility_step")
        if mean_utility_step is not None:
            row["mean_utility"] = round(float(np.array(mean_utility_step).mean()), 6)

        for _key, _col in [
            ("mean_gross_output_step", "mean_gross_output"),
            ("mean_consumption_step", "mean_consumption"),
            ("mean_abatement_cost_step", "mean_abatement_cost"),
            ("mean_mitigation_step", "mean_mitigation"),
        ]:
            _val = data.get(_key)
            if _val is not None:
                row[_col] = round(float(np.array(_val).mean()), 6)

        for _key in [
            "utility_per_region",
            "gross_output_per_region",
            "consumption_per_region",
            "abatement_cost_per_region",
            "mitigation_per_region",
        ]:
            _val = data.get(_key)
            if _val is not None:
                _arr = np.array(_val)
                _region_means = _arr.mean(axis=tuple(range(_arr.ndim - 1)))
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
    """Return a log-function that prints a compact summary to stdout."""
    tqdm_bar: list = []

    def log_fn(data: dict, iteration: int):
        def _flatten(tree):
            leaves = jax.tree.leaves(tree)
            if not leaves:
                return np.array([])
            return np.concatenate([np.array(l).ravel() for l in leaves])

        iter_int = int(np.array(iteration))
        act_mean = _flatten(data.get("action_mean", []))
        act_var = _flatten(data.get("action_var", []))
        rew_mean = float(np.array(data.get("reward_mean", np.nan)))
        rew_std = float(np.sqrt(np.array(data.get("reward_var", np.nan))))

        ep_str = ""
        if "returned_episode_returns" in data and "returned_episode" in data:
            done = np.array(data["returned_episode"])
            if done.ndim > 0 and done.any():
                ret_leaves = jax.tree.leaves(data["returned_episode_returns"])
                ep_rets = np.concatenate(
                    [np.array(l)[done].ravel() for l in ret_leaves]
                )
                if ep_rets.size > 0:
                    ep_str = f"  ep_ret={ep_rets.mean():.3f}±{ep_rets.std():.3f}"

        rcpo_str = ""
        if "cbam_lambda" in data:
            lam = float(np.array(data["cbam_lambda"]))
            cost = float(np.array(data.get("mean_cbam_cost", np.nan)))
            rcpo_str = f"  λ={lam:.6f} c̄={cost:.6f}"

        if act_mean.size > 0:
            labels = action_labels or [f"a{i}" for i in range(len(act_mean))]
            mean_parts = "  ".join(
                f"{l}={v:.3f}±{s:.3f}"
                for l, v, s in zip(labels, act_mean, np.sqrt(act_var))
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
