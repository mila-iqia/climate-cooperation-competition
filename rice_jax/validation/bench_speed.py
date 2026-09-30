"""bench_speed.py

Diagnose training-speed slowdowns by comparing a set of env configurations
over a short fixed number of iterations.

Hypotheses tested:
  H1  no_mitigation=True  vs  False  (extra Discrete action + minimum-rate masking)
  H2  fixed_savings_rate=True  vs  False  (one more Discrete action)
  H3  zero_abatement_cost=True  vs  False  (extra computation inside step)
  H4  reward_mode="additive_cbam"  vs  "welfloss"

Each config is trained for BENCH_ITERS iterations (NOT timesteps) using the
same RCPO/PPO setup as validate_mitigation_incentive.  Wall-clock time is
measured and printed as iters/s.

Usage (from rice_jax/, rice-jax conda env):
    python validation/bench_speed.py
"""

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import jaxnasium as jym

from training_monitor import (
    RCPOMonitoredPPO,
    make_combined_log_fn,
    make_csv_log_fn,
    make_print_log_fn,
    rcpo_cbam_log_info_fn,
)
from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls

# ── Config ─────────────────────────────────────────────────────────────────

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT   = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", ".."))

NUM_REGIONS  = 9
EU_IDX       = 3
YAML_DIR     = _os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
MRIO_ROOT    = _os.path.join(_REPO_ROOT, "csv_asset")

# Small enough to compile + run in < 2 min per config
BENCH_ITERS = 30   # iterations (each = NUM_ENVS * NUM_STEPS steps)
NUM_ENVS    = 8
NUM_STEPS   = 100

_BASE = dict(
    num_regions               = NUM_REGIONS,
    mrio_data_root            = MRIO_ROOT,
    mrio_trade                = True,
    eu_region_idx             = EU_IDX,
    dest_alloc_persistence    = 0.55,
    dest_alloc_baseline_decay = 1.0,
    diff_reward_mode          = True,
    num_discrete_action_levels= 10,
    sector_granularity        = "emissions-simple",
    cbam_tariff_rate          = 0.80,
    cbam_randomize            = False,
    sectoral_welfloss         = True,
    welfare_loss_per_unit_tariff = 5.0,
)

_PPO = dict(
    total_timesteps     = BENCH_ITERS * NUM_ENVS * NUM_STEPS,
    num_steps           = NUM_STEPS,
    num_envs            = NUM_ENVS,
    learning_rate       = 3e-4,
    num_minibatches     = 4,
    num_epochs          = 8,
    ent_coef            = 0.01,
    gamma               = 0.99,
    gae_lambda          = 0.95,
    max_grad_norm       = 1.0,
    clip_coef           = 0.2,
    clip_coef_vf        = 0.5,
    vf_coef             = 0.5,
    normalize_observations = True,
    normalize_rewards   = True,
    log_interval        = 1.0,   # no per-iter printing
)

# ── Hypotheses ─────────────────────────────────────────────────────────────

CONFIGS = [
    # (label,  env_kwargs_overrides)
    # H1: mitigation action
    ("H1-no_mitigation=True  (baseline)",
     dict(no_mitigation=True,  fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),
    ("H1-no_mitigation=False (+ mitigation action)",
     dict(no_mitigation=False, fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),

    # H2: savings action (only meaningful when no_mitigation=True to isolate)
    ("H2-fixed_savings=True  (baseline)",
     dict(no_mitigation=True,  fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),
    ("H2-fixed_savings=False (+ savings action)",
     dict(no_mitigation=True,  fixed_savings_rate=False, reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),

    # H3: zero_abatement_cost computation overhead
    ("H3-zero_abatement=False (baseline)",
     dict(no_mitigation=False, fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),
    ("H3-zero_abatement=True  (extra rescaling)",
     dict(no_mitigation=False, fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=True)),

    # H4: reward_mode — additive_cbam includes mitigation-rate scaling in _compute_cbam
    ("H4-reward_mode=welfloss    (baseline)",
     dict(no_mitigation=False, fixed_savings_rate=True,  reward_mode="welfloss",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),
    ("H4-reward_mode=additive_cbam",
     dict(no_mitigation=False, fixed_savings_rate=True,  reward_mode="additive_cbam",
          log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False)),
]

# H5: log_interval — how often the Python callback fires (host sync cost)
# Uses the same env as H1 baseline. Tested separately to avoid polluting results
# with extra compile time.
LOG_INTERVAL_CONFIGS = [
    ("H5-log_interval=1.00  (null log,  rare sync)",   1.00, False),
    ("H5-log_interval=0.10  (null log,  10% sync)",    0.10, False),
    ("H5-log_interval=0.02  (null log,  like validate)",0.02, False),
    ("H5-log_interval=1.00  (real log,  rare sync)",   1.00, True),
    ("H5-log_interval=0.02  (real log,  like validate)",0.02, True),
]


def _build_env(overrides: dict):
    kwargs = {**_BASE, **overrides}
    env = RiceMRIO(
        region_params=load_region_yamls(NUM_REGIONS, yaml_dir=YAML_DIR),
        **kwargs,
    )
    return jym.LogWrapper(env)


def _null_log(metrics, info):
    pass


def _real_log_fn(num_iters: int):
    """Real log function matching validate_mitigation_incentive (print + CSV)."""
    csv_path = "/tmp/bench_speed_h5.csv"
    return make_combined_log_fn(
        make_print_log_fn(num_iterations=num_iters),
        make_csv_log_fn(csv_path),
    )


def _run_config(label: str, overrides: dict, key,
               log_interval: float = 1.0, use_real_log: bool = False) -> float:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")

    env = _build_env(overrides)

    log_fn = _real_log_fn(BENCH_ITERS) if use_real_log else _null_log
    ppo = RCPOMonitoredPPO(
        **{**_PPO, "log_interval": log_interval},
        log_function        = log_fn,
        rcpo_eta_lambda     = 1e-4,
        rcpo_alpha_target   = 0.005,
    )

    # Force JIT compilation by running a quick 1-iter warm-up
    warmup_ppo = replace(ppo,
        total_timesteps = 1 * NUM_ENVS * NUM_STEPS,
    )
    print("  [warm-up JIT compile...]", flush=True)
    t0 = time.perf_counter()
    warmup_ppo = warmup_ppo.train(key, env)
    t_compile = time.perf_counter() - t0
    print(f"  compile+1iter: {t_compile:.1f}s", flush=True)

    # Timed benchmark run
    print(f"  [timing {BENCH_ITERS} iters...]", flush=True)
    t0 = time.perf_counter()
    ppo = ppo.train(key, env)
    elapsed = time.perf_counter() - t0
    iters_per_sec = BENCH_ITERS / elapsed
    steps_per_sec = BENCH_ITERS * NUM_ENVS * NUM_STEPS / elapsed
    print(f"  RESULT: {elapsed:.1f}s total → {iters_per_sec:.2f} iters/s  ({steps_per_sec:.0f} steps/s)")
    return iters_per_sec


def main():
    key = jax.random.PRNGKey(0)
    print(f"\n{'='*60}")
    print("  bench_speed.py — training speed diagnostic")
    print(f"  BENCH_ITERS={BENCH_ITERS}  NUM_ENVS={NUM_ENVS}  NUM_STEPS={NUM_STEPS}")
    print(f"{'='*60}")

    results = []
    for label, overrides in CONFIGS:
        rate = _run_config(label, overrides, key)
        results.append((label, rate))

    # H5: log_interval sweep — use fixed env config (H1 baseline)
    h5_env_overrides = dict(
        no_mitigation=True, fixed_savings_rate=True, reward_mode="additive_cbam",
        log_info_fn=rcpo_cbam_log_info_fn, zero_abatement_cost=False,
    )
    print(f"\n{'='*60}")
    print("  H5: log_interval sweep (same env config for all)")
    print(f"{'='*60}")
    h5_results = []
    for label, log_interval, use_real_log in LOG_INTERVAL_CONFIGS:
        rate = _run_config(label, h5_env_overrides, key,
                           log_interval=log_interval, use_real_log=use_real_log)
        h5_results.append((label, rate))

    print(f"\n{'='*60}")
    print("  SUMMARY  (iters/s, higher = faster)")
    print(f"{'='*60}")
    for label, rate in results + h5_results:
        bar = "█" * int(rate * 3)
        print(f"  {rate:5.2f} iters/s  {bar}  {label}")
    print()


if __name__ == "__main__":
    main()
