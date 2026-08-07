"""canonical_config.py

Single source of truth for the CBAM-RICE paper-primary experimental setup.

Every headline experiment must import from this module rather than redefining
env kwargs inline. This is the chokepoint that makes the canonical setup
(audit §6.1) and seed protocol (audit §6.5) enforceable.

Structural choices fixed here are NOT free parameters. They are paper-frozen.
Continuous parameters that are legitimate sensitivity-analysis targets are
exposed via the SENSITIVITY_PARAMS whitelist; `make_canonical_env()` allows
overrides only for those, and raises on any other override.

Aggregation conventions
-----------------------
RoW (idx 0) and EU (idx 3) are excluded from "non-EU exporter" aggregates.
This is a display/aggregation choice (encoded in metrics.py), not a structural
setting — PKLs still store all 9 regions per metric so plots can be regenerated
with different exclusions without retraining.

Usage
-----
    from cbam.config.canonical_config import (
        make_canonical_env,
        CANONICAL_SEEDS,
        CANONICAL_TRAIN_KWARGS,
        EU_REGION_IDX,
        NON_EU_EXPORTER_IDXS,
    )

    env = make_canonical_env()                                # paper-primary
    env = make_canonical_env(dest_alloc_persistence=0.30)     # sensitivity arm
    env = make_canonical_env(num_regions=7)                   # raises
"""

from __future__ import annotations

import os as _os
from typing import Any

from _experiment_util import wrap_rice_env
from rice_jax import RiceMRIO
from rice_jax.training import rcpo_cbam_log_info_fn
from rice_jax.utils import load_region_yamls

# ── Paths ──────────────────────────────────────────────────────────────────

_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.abspath(_os.path.join(_THIS_DIR, "..", "..", ".."))

CANONICAL_YAML_DIR = _os.path.join(_REPO_ROOT, "cbam_yamls", "setup_vuln_9")
# Allow the data root to be overridden via an environment variable so training
# can use a restored MRIO bundle extracted to a non-default location without
# touching any code.  See cbam/scripts/restore_mrio_bundle.py.
#   export CBAM_MRIO_ROOT=/path/to/csv_asset   # before starting training
CANONICAL_MRIO_ROOT: str = _os.environ.get(
    "CBAM_MRIO_ROOT",
    _os.path.join(_REPO_ROOT, "csv_asset"),
)


# ── Region indexing (9-region vulnerability aggregation) ───────────────────

NUM_REGIONS: int = 9
EU_REGION_IDX: int = 3
ROW_REGION_IDX: int = 0

REGION_NAMES: dict[int, str] = {
    0: "RoW",
    1: "Russia+Eur.",
    2: "MENA",
    3: "EU",
    4: "SSA-Mining",
    5: "Americas",
    6: "SE Asia",
    7: "China",
    8: "India",
}

# All non-EU regions (includes RoW). Used when a claim is about EU vs the
# rest of the world.
NON_EU_IDXS: tuple[int, ...] = tuple(
    r for r in range(NUM_REGIONS) if r != EU_REGION_IDX
)

# Canonical "non-EU exporter" aggregation: excludes both RoW (catch-all,
# distorts scale) and EU (the importing region under CBAM).
# This is the default exclusion set for headline mitigation / dirty-share metrics.
NON_EU_EXPORTER_IDXS: tuple[int, ...] = tuple(
    r for r in NON_EU_IDXS if r != ROW_REGION_IDX
)


# ── Seed protocol ──────────────────────────────────────────────────────────

# Shared across every headline experiment. If a headline experiment uses a
# different seed set, it is by definition not headline-comparable.
CANONICAL_SEEDS: tuple[int, ...] = (0, 1, 2)


# ── Canonical env kwargs (paper-frozen) ────────────────────────────────────

# Defaults for the env. Everything listed here is the canonical paper choice.
# Anything in SENSITIVITY_PARAMS may be overridden via make_canonical_env(**overrides);
# anything else is structural and may not be overridden.
_CANONICAL_ENV_DEFAULTS: dict[str, Any] = dict(
    # Structural — paper-frozen
    num_regions=NUM_REGIONS,
    eu_region_idx=EU_REGION_IDX,
    mrio_data_root=CANONICAL_MRIO_ROOT,
    mrio_trade=True,
    sector_granularity="emissions-simple",
    num_discrete_action_levels=10,
    diff_reward_mode=True,
    sectoral_welfloss=True,
    cbam_tariff_mode="differential",
    # EU mitigation schedule: ramps 0.30→1.00 over 8 steps (EU Climate Law / net-zero
    # 2050 pathway). Ensures MAC_EU > 0 from t=0 so the differential tariff is
    # non-trivial. [EU Climate Law (EU) 2021/1119; IPCC AR6 mitigation scenarios]
    eu_mitigation_schedule=(
        0.30,
        0.38,
        0.46,
        0.54,
        0.62,
        0.70,
        0.80,
        0.90,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
        1.00,
    ),
    # Sensitivity-eligible — defaults are paper-primary
    dest_alloc_persistence=0.55,
    dest_alloc_baseline_decay=0.0,  # MUST stay 0.0 in headline runs (audit Gate 3)
    welfare_loss_per_unit_tariff=5.0,
    delta_max=3.0,
    cbam_lambda_init=1.0,
    # Action window: max discrete-level change per timestep for mitigation and
    # savings. 0 = unconstrained (smoothness enforced by transition_cost_coef).
    # Previously AW=2, but TC is strictly superior: economic penalty vs mechanical
    # mask. Validated via C-litmus 3/3 PASS at TC=10, AW=0, free savings (May 19).
    action_window_size=0,
    # Grubb et al. (1995) DIAM Eq. 2 transitional abatement cost.
    # Quadratic penalty on rate of mitigation change: TC = c_B × ((μ_t − μ_{t-1})/Δt)².
    # Replaces the mechanical action_window_size mask as the canonical smoothness
    # mechanism. Validated: C-litmus PASS 3/3 at TC=10, AW=0 (May 19 2026).
    # TC sweep (May 19) shows no degradation of conditioning gap at any TC∈[0,10].
    transition_cost_coef=10.0,
    # RCPO additive reward — required for the canonical claim family
    reward_mode="additive_cbam",
)

# Parameters that experiments are allowed to override (for sensitivity studies).
# Any kwarg passed to make_canonical_env() must be in this set OR be one of the
# experiment-specific knobs in _EXPERIMENT_LEVEL_OVERRIDES.
SENSITIVITY_PARAMS: frozenset[str] = frozenset(
    {
        "dest_alloc_persistence",
        "dest_alloc_baseline_decay",
        "welfare_loss_per_unit_tariff",
        "delta_max",
        "cbam_lambda_init",
        "action_window_size",
        "transition_cost_coef",  # Grubb (1995) Eq. 2 transitional cost coefficient
        "transition_cost_asymmetric",  # asymmetric TC (only penalise increases)
        "cbam_cost_normalize_by_output",  # divide cbam_cost by Y_r before λ penalty
    }
)

# Parameters that legitimately vary across experiments (not sensitivity arms).
# These are the policy-design / claim-defining knobs.
_EXPERIMENT_LEVEL_OVERRIDES: frozenset[str] = frozenset(
    {
        "cbam_tariff_rate",  # flat-τ diagnostic ladder
        "cbam_tariff_mode",  # "differential" vs "flat" for tariff-mode comparisons
        "cbam_randomize",  # litmus conditioning (train with random tariff obs)
        "cbam_tariff_rates",  # range for cbam_randomize (e.g. (0.0, 1.0))
        "revenue_share",  # Phase 2B revenue recycling
        "transfer_pool_multiplier",  # Experiment C: external finance amplifier (NCQG scale)
        "transfer_mode",  # "consumption" vs "abatement"
        "transfer_allocation",  # allocation rule
        "eu_mitigation_schedule",  # required for non-trivial differential τ_eff
        "zero_abatement_cost",  # litmus L2
        "no_mitigation",  # litmus L1
        "fixed_savings_rate",  # litmus / free-savings ablation
        "use_trade_friction",  # Phase C-Friction: MRIO-derived iceberg cost penalty
        "sector_granularity",  # "emissions-simple" vs "emissions-sectoral" for sectoral comparisons
    }
)


# ── Canonical PPO / training kwargs ────────────────────────────────────────

# Learning-rate schedule (restored 2026-08-04).  jaxnasium 0.0.23 defaulted
# `anneal_learning_rate=True`, so pre-restructure runs annealed 3e-4 -> 0; the
# 0.1.0 port spelled that as `learning_rate_end=None`, which means CONSTANT
# (Schedule.__call__ returns `start` when `end is None`).  Training therefore
# never settled and the C-litmus tests sampled a still-moving policy, which is
# where the seed-to-seed noise came from.  `learning_rate_end=0.0` restores the
# anneal.
#
# On total_timesteps: 0.0.23 also computed the schedule's transition_steps as
# num_iterations * num_epochs while the optimizer steps once per *minibatch*,
# i.e. 4x too few, so optax clamped the LR to 0 after 20_000 updates — a
# quarter of a 2M-step run.  500_000 steps here gives
# 625 iters * 8 epochs * 4 minibatches = 20_000 optimizer steps, so the LR now
# traverses 3e-4 -> 0 over exactly the same number of updates the old runs
# learned over, without the 1.5M frozen-policy steps that followed.  (Not bit
# identical: those extra steps still refined the observation normalizer.)
CANONICAL_TRAIN_KWARGS: dict[str, Any] = dict(
    total_timesteps=500_000,
    num_steps=100,
    num_envs=int(_os.environ.get("CBAM_NUM_ENVS", 8)),
    learning_rate_start=3e-4,
    learning_rate_end=0.0,
    num_minibatches=4,
    num_epochs=8,
    ent_coef_start=0.01,
    ent_coef_end=None,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    clip_coef=0.2,
    clip_coef_vf=0.5,
    vf_coef=0.5,
    normalize_observations=True,
    normalize_rewards=True,
    log_interval=50,
)

# Evaluation defaults — same across all headline experiments so per-region
# metrics are comparable.
NUM_EVAL_EPISODES: int = 8
EVAL_LAST_T: int = 5  # average over last 5 env steps of each eval episode


# ── Factory ────────────────────────────────────────────────────────────────


def make_canonical_env(
    *,
    for_training: bool = True,
    **overrides: Any,
) -> Any:
    """Build the canonical RiceMRIO env, optionally with whitelisted overrides.

    Parameters
    ----------
    for_training : bool
        If True, also wrap in ``LogWrapper``.  Eval envs still get
        ``StackActionSpaceWrapper`` so action spaces match training.
    **overrides
        Either a sensitivity-eligible kwarg (see SENSITIVITY_PARAMS) or an
        experiment-level kwarg (see _EXPERIMENT_LEVEL_OVERRIDES).  Passing any
        other kwarg raises ValueError — this enforces the canonical-setup
        discipline of audit §6.1.

    Returns
    -------
    Wrapped RiceMRIO
        ``StackActionSpaceWrapper`` always; ``LogWrapper`` when training.
    """
    bad = set(overrides) - SENSITIVITY_PARAMS - _EXPERIMENT_LEVEL_OVERRIDES
    if bad:
        raise ValueError(
            f"make_canonical_env: refusing to override structural parameters "
            f"{sorted(bad)}. Allowed sensitivity params: "
            f"{sorted(SENSITIVITY_PARAMS)}; allowed experiment-level overrides: "
            f"{sorted(_EXPERIMENT_LEVEL_OVERRIDES)}."
        )

    kwargs = dict(_CANONICAL_ENV_DEFAULTS)
    kwargs.update(overrides)

    region_params = load_region_yamls(NUM_REGIONS, yaml_dir=CANONICAL_YAML_DIR)
    env = RiceMRIO(
        region_params=region_params,
        log_info_fn=rcpo_cbam_log_info_fn,
        **kwargs,
    )
    return wrap_rice_env(env, for_training=for_training)


def canonical_env_kwargs() -> dict[str, Any]:
    """Return a copy of the canonical env defaults (for saving to PKL provenance)."""
    return dict(_CANONICAL_ENV_DEFAULTS)


def canonical_train_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return canonical PPO kwargs, optionally overriding e.g. ``total_timesteps``."""
    kwargs = dict(CANONICAL_TRAIN_KWARGS)
    kwargs.update(overrides)
    return kwargs


__all__ = [
    "NUM_REGIONS",
    "EU_REGION_IDX",
    "ROW_REGION_IDX",
    "REGION_NAMES",
    "NON_EU_IDXS",
    "NON_EU_EXPORTER_IDXS",
    "CANONICAL_SEEDS",
    "CANONICAL_YAML_DIR",
    "CANONICAL_MRIO_ROOT",
    "CANONICAL_TRAIN_KWARGS",
    "NUM_EVAL_EPISODES",
    "EVAL_LAST_T",
    "SENSITIVITY_PARAMS",
    "make_canonical_env",
    "canonical_env_kwargs",
    "canonical_train_kwargs",
]
