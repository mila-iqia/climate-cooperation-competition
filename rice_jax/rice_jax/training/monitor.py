"""Training log-function factories and a thin PPO wrapper for compact metrics."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxnasium._environment import ORIGINAL_OBSERVATION_KEY
from jaxnasium.algorithms import PPO


def summarize_info_for_logging(info: dict | None) -> dict:
    """Reduce bulky per-step info to rollout aggregates (device-side).

    Replaces ``actions`` / ``rewards`` with ``action_mean`` / ``action_var`` /
    ``reward_mean`` / ``reward_var`` / ``reward_sum``, and drops the terminal
    observation copy that jaxnasium injects into every info dict. Intended to
    run inside ``train_iteration`` so ``jax.debug.callback`` only ships small
    arrays to the host.
    """
    info = info or {}
    out = {
        k: v
        for k, v in info.items()
        if k not in ("actions", "rewards", ORIGINAL_OBSERVATION_KEY)
    }

    actions = info.get("actions")
    if actions is not None:
        out["action_mean"] = jax.tree.map(lambda a: a.mean(axis=(0, 1)), actions)
        out["action_var"] = jax.tree.map(lambda a: a.var(axis=(0, 1)), actions)

    rewards = info.get("rewards")
    if rewards is not None:
        reward_leaves = jax.tree.leaves(rewards)
        reward_stack = jnp.stack([r.ravel() for r in reward_leaves])
        out["reward_mean"] = reward_stack.mean()
        out["reward_var"] = reward_stack.var()
        out["reward_sum"] = reward_stack.sum()

    return out


def episode_return_curve(metrics) -> np.ndarray:
    """Host 1-D mean-episode-return curve from jaxnasium ``train()`` metrics.

    With the default ``reduce_metrics_fn="mean"``, ``train`` returns per-iteration
    mean episode returns (a scalar array, or a per-agent pytree of them).  Multi-
    agent dicts are averaged across agents into a single learning curve.
    """
    host = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), metrics)
    if isinstance(host, dict):
        leaves = [v for v in host.values() if isinstance(v, np.ndarray) and v.size]
        if not leaves:
            return np.array([], dtype=float)
        return np.stack([np.asarray(v, dtype=float).ravel() for v in leaves], axis=0).mean(
            axis=0
        )
    return np.asarray(host, dtype=float).ravel()


class LoggingPPO(PPO):
    """Stock PPO that summarizes trajectory info before log callbacks.

    Only ``train_iteration`` is overridden: call the parent implementation,
    then :func:`summarize_info_for_logging` on the returned metric so host
    callbacks never receive full per-step ``actions`` / ``rewards``.

    Note: ``train()`` returns ``(agent, metrics)`` — a
    :class:`~jaxnasium.algorithms.ppo.PPOAgent` plus the reduced per-iteration
    training metrics (default: mean episode returns).
    """

    def train_iteration(self, runner_state, train_iter, *, env):
        runner_state, metric = super().train_iteration(
            runner_state, train_iter, env=env
        )
        return runner_state, summarize_info_for_logging(metric)


def _flatten_tree(tree) -> np.ndarray:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return np.array([])
    return np.concatenate([np.asarray(l).ravel() for l in leaves])


def _reduce_leading_axes(x, op: str):
    """Reduce over leading (steps, envs) axes when present."""
    arr = np.asarray(x)
    axes = (0, 1) if arr.ndim >= 2 else None
    return getattr(arr, op)(axis=axes)


def _rollout_action_reward_stats(
    data: dict,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Action/reward rollout stats for log callbacks.

    Prefers per-step ``actions`` / ``rewards`` from ``log_info_fn`` (mean/var
    over steps × envs). Falls back to pre-averaged ``action_mean`` /
    ``reward_mean`` keys when those are absent.
    """
    if data.get("actions") is not None:
        act_mean = _flatten_tree(
            jax.tree.map(lambda a: _reduce_leading_axes(a, "mean"), data["actions"])
        )
        act_var = _flatten_tree(
            jax.tree.map(lambda a: _reduce_leading_axes(a, "var"), data["actions"])
        )
    else:
        act_mean = _flatten_tree(data.get("action_mean", np.array([])))
        act_var = _flatten_tree(data.get("action_var", np.array([])))

    if data.get("rewards") is not None:
        reward_leaves = jax.tree.leaves(data["rewards"])
        reward_stack = np.stack([np.asarray(r).ravel() for r in reward_leaves])
        reward_mean = float(reward_stack.mean())
        reward_var = float(reward_stack.var())
        reward_sum = float(reward_stack.sum())
    else:
        reward_mean = float(np.array(data.get("reward_mean", np.nan)))
        reward_var = float(np.array(data.get("reward_var", np.nan)))
        reward_sum = float(np.array(data.get("reward_sum", np.nan)))

    return act_mean, act_var, reward_mean, reward_var, reward_sum


def make_csv_log_fn(csv_path: str) -> Callable:
    """Return a log-function that appends one row per callback to *csv_path*."""
    state = {"writer": None, "file": None, "initialized": False}

    def log_fn(data: dict, iteration: int):
        act_mean, act_var, reward_mean, reward_var, reward_sum = (
            _rollout_action_reward_stats(data)
        )

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
            "ep_return_mean": round(ep_return_mean, 4)
            if not np.isnan(ep_return_mean)
            else "",
            "ep_return_std": round(ep_return_std, 4)
            if not np.isnan(ep_return_std)
            else "",
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

        row.update(
            {
                **{
                    f"action_mean_{i}": round(float(v), 6)
                    for i, v in enumerate(act_mean)
                },
                **{
                    f"action_var_{i}": round(float(v), 6) for i, v in enumerate(act_var)
                },
            }
        )

        if not state["initialized"]:
            os.makedirs(
                os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".",
                exist_ok=True,
            )
            state["file"] = open(csv_path, "w", newline="")
            state["writer"] = csv.DictWriter(state["file"], fieldnames=list(row.keys()))
            state["writer"].writeheader()
            state["initialized"] = True

        state["writer"].writerow(row)
        state["file"].flush()

    return log_fn


def make_print_log_fn(
    action_labels: list[str] | None = None,
    num_iterations: int | None = None,
) -> Callable:
    """Return a log-function that prints a compact summary to stdout."""
    tqdm_bar: list = []

    def log_fn(data: dict, iteration: int):
        iter_int = int(np.array(iteration))
        act_mean, act_var, rew_mean, rew_var, _ = _rollout_action_reward_stats(data)
        rew_std = float(np.sqrt(rew_var)) if not np.isnan(rew_var) else float("nan")

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
                        _tqdm_mod.tqdm(
                            total=num_iterations, desc="Training", unit=" iters"
                        )
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
