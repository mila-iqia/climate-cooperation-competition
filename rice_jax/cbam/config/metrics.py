"""metrics.py

Frozen metric semantics for CBAM-RICE headline experiments (audit §6.7).

Each function consumes raw eval-rollout arrays and returns a named scalar or
per-region array. Experiment scripts must NOT compute these inline — they must
call into this module. This is what makes claim-bucket discipline (audit §6.2)
enforceable.

Array conventions (matching `run_single_episode` + `full_state_info_log_fn`):

  trade_flows : np.ndarray
      Shape (n_eval_episodes, n_steps, NR_from, NR_to, NS)
      Sector 0 = dirty, sector 1 = clean (sector_granularity="emissions-simple")
      Units: output (post-aggregation). Diagonal is intra-region "trade".

  mitigation  : np.ndarray
      Shape (n_eval_episodes, n_steps, NR)
      Per-region mitigation rate ∈ [0, 1].

  utility     : np.ndarray
      Shape (n_eval_episodes, n_steps, NR)

Default time aggregation: mean over the last `last_t` env steps of each episode
(EVAL_LAST_T = 5), then mean over episodes.

Default region aggregation for "non-EU exporters": excludes RoW (idx 0) and
EU (idx 3). This is the canonical headline aggregation; it can be overridden
via the `exporter_idxs` argument for sensitivity/appendix figures.

Each function's docstring declares which claim bucket it belongs to:
  - mechanism: diversion / mitigation / crowd-out
  - policy-design: revenue recycling, transfer mode, allocation rule
  - robustness: derived comparisons across conditions
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from cbam.config.canonical_config import (
    EU_REGION_IDX,
    EVAL_LAST_T,
    NON_EU_EXPORTER_IDXS,
)

_EPS = 1e-10


# ── Trade-flow metrics (mechanism bucket) ──────────────────────────────────


def eu_dirty_export_share(
    trade_flows: np.ndarray,
    *,
    eu_region_idx: int = EU_REGION_IDX,
    exporter_idxs: Sequence[int] = NON_EU_EXPORTER_IDXS,
    last_t: int = EVAL_LAST_T,
) -> float:
    """Mean share of dirty-sector exports going to the EU, averaged over
    `exporter_idxs` and the last `last_t` env steps.

    Claim bucket: mechanism.
    Defines "diversion": lower share => more diversion away from EU.

    Pre-condition: trade_flows shape (n_ep, T, NR, NR, NS), sector 0 = dirty.
    """
    tf = trade_flows[:, -last_t:]  # (n_ep, t, NR, NR, NS)
    dirty_to_eu = tf[:, :, :, eu_region_idx, 0]  # (n_ep, t, NR)
    dirty_total = tf[:, :, :, :, 0].sum(-1)  # (n_ep, t, NR)
    ratio = dirty_to_eu[:, :, list(exporter_idxs)] / (
        dirty_total[:, :, list(exporter_idxs)] + _EPS
    )
    return float(ratio.mean())


def per_region_eu_dirty_export_share(
    trade_flows: np.ndarray,
    *,
    eu_region_idx: int = EU_REGION_IDX,
    last_t: int = EVAL_LAST_T,
) -> dict[int, float]:
    """Per-region mean dirty-export share going to EU. Returns ALL non-EU
    regions (including RoW); the display layer is responsible for excluding
    RoW from headline figures.

    Claim bucket: mechanism (per-region decomposition).
    """
    tf = trade_flows[:, -last_t:]
    out: dict[int, float] = {}
    NR = trade_flows.shape[2]
    for r in range(NR):
        if r == eu_region_idx:
            continue
        eu_d = tf[:, :, r, eu_region_idx, 0]
        all_d = tf[:, :, r, :, 0].sum(-1)
        out[r] = float((eu_d / (all_d + _EPS)).mean())
    return out


# ── Mitigation metrics (mechanism + policy-design buckets) ─────────────────


def mean_mitigation_rate(
    mitigation: np.ndarray,
    *,
    region_idxs: Sequence[int] = NON_EU_EXPORTER_IDXS,
    last_t: int = EVAL_LAST_T,
) -> float:
    """Mean mitigation rate across `region_idxs`, averaged over the last
    `last_t` env steps and all eval episodes.

    Claim bucket: mechanism (when comparing pinned vs open scenarios) /
    policy-design (when comparing revenue-recycling arms).

    Default excludes RoW (idx 0) and EU (idx 3) — the canonical headline
    "non-EU exporters" aggregation.
    """
    return float(mitigation[:, -last_t:, list(region_idxs)].mean())


def per_region_mitigation_rate(
    mitigation: np.ndarray,
    *,
    last_t: int = EVAL_LAST_T,
) -> dict[int, float]:
    """Per-region mean mitigation rate over the last `last_t` steps.
    Returns all regions; display layer applies exclusion.

    Claim bucket: mechanism (per-region decomposition).
    """
    NR = mitigation.shape[-1]
    return {r: float(mitigation[:, -last_t:, r].mean()) for r in range(NR)}


# ── Crowd-out (mechanism / policy-design) ──────────────────────────────────


def crowd_out_gap(
    mu_pinned_exports: float,
    mu_both_channels_open: float,
) -> float:
    """Direct crowd-out gap: how much mitigation is lost when the diversion
    channel is opened, holding everything else fixed.

        crowd_out_gap = μ_pinned_exports − μ_both_channels_open

    Claim bucket: mechanism (audit §6.3). Positive value => diversion does
    crowd out mitigation. Used by Experiment A (direct crowd-out attenuation).
    """
    return float(mu_pinned_exports - mu_both_channels_open)


def crowd_out_attenuation(
    gap_no_transfer: float,
    gap_with_transfer: float,
) -> float:
    """Reduction in crowd-out gap attributable to redistribution.

        attenuation = gap_no_transfer − gap_with_transfer

    Claim bucket: policy-design (audit §6.3, claim unlocked).
    Positive value => transfers attenuate crowd-out.
    """
    return float(gap_no_transfer - gap_with_transfer)


# ── Transfer effectiveness (policy-design) ─────────────────────────────────


def transfer_effectiveness(
    mitigation_with_transfer: np.ndarray,
    mitigation_no_transfer: np.ndarray,
    *,
    region_idxs: Sequence[int] = NON_EU_EXPORTER_IDXS,
    last_t: int = EVAL_LAST_T,
) -> float:
    """Difference in mean mitigation rate (non-EU exporters) between a
    transfer arm and the no-transfer baseline.

        effectiveness = μ_with_transfer − μ_no_transfer

    Claim bucket: policy-design. Positive => transfers raise mitigation.
    """
    mu_with = mean_mitigation_rate(
        mitigation_with_transfer,
        region_idxs=region_idxs,
        last_t=last_t,
    )
    mu_without = mean_mitigation_rate(
        mitigation_no_transfer,
        region_idxs=region_idxs,
        last_t=last_t,
    )
    return float(mu_with - mu_without)


# ── Cross-seed aggregation helpers ─────────────────────────────────────────


def seed_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarise a metric across seeds.

    Returns mean, min (worst-case for positive-good metrics), max, std.
    Use `worst_seed` semantics when reporting policy-design claims (audit §6.9).
    """
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std(ddof=0)),
        "n": int(arr.shape[0]),
    }


__all__ = [
    "eu_dirty_export_share",
    "per_region_eu_dirty_export_share",
    "mean_mitigation_rate",
    "per_region_mitigation_rate",
    "crowd_out_gap",
    "crowd_out_attenuation",
    "transfer_effectiveness",
    "seed_summary",
]
