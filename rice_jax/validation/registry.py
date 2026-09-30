"""registry.py

Lightweight experiment registry (audit §6.8).

Each headline experiment declares one ExperimentEntry describing:
  - the question it answers,
  - its claim bucket (mechanism / conditioning / policy-design / robustness),
  - the primary metric (must be a function name in metrics.py),
  - the pass criterion (human-readable),
  - the script implementing it,
  - the seed set used,
  - interpretation limits / caveats.

This file is the table-of-contents of the paper. Adding a new headline
experiment without a registry entry is a process violation.

The registry intentionally does NOT execute anything. It is a static manifest
that experiment scripts and the report consume to enforce provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from validation.canonical_config import CANONICAL_SEEDS


ClaimBucket = Literal["mechanism", "conditioning", "policy-design", "robustness"]


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id:        str
    question:             str
    claim_bucket:         ClaimBucket
    primary_metric:       str                      # must exist in metrics.py
    pass_criterion:       str                      # human-readable
    script:               str                      # path relative to repo root
    seeds:                tuple[int, ...]
    interpretation_limits: str


REGISTRY: dict[str, ExperimentEntry] = {

    # ── Experiment A — Direct crowd-out attenuation test (audit §6.3) ──────
    "A_crowd_out_redist": ExperimentEntry(
        experiment_id   = "A_crowd_out_redist",
        question        = (
            "Does redistribution reduce the mitigation lost to diversion "
            "when both margins are open?"
        ),
        claim_bucket    = "policy-design",
        primary_metric  = "crowd_out_attenuation",
        pass_criterion  = (
            "crowd_out_attenuation > 0 (worst seed) when comparing "
            "revenue_share=1.0 to revenue_share=0.0 under differential CBAM, "
            "and >= 2× larger for transfer_mode='abatement' vs 'consumption'."
        ),
        script          = "rice_jax/validation/cbam_experiment_A_crowdout.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Crowd-out is measured at the env's end-of-horizon; transient "
            "behavior may differ. Pinned-exports arm forces δ=0, which is "
            "an extreme; intermediate δ values are not tested here."
        ),
    ),

    # ── Experiment B — Tariff calibration / τ-ladder (audit §6.4) ──────────
    # Flat τ reframed as diagnostic, NOT a parallel baseline.
    "B_tariff_ladder": ExperimentEntry(
        experiment_id   = "B_tariff_ladder",
        question        = (
            "Does the qualitative crowd-out mechanism survive across flat "
            "τ ∈ {0.05, 0.15, 0.30}, and how does it relate to differential CBAM?"
        ),
        claim_bucket    = "robustness",
        primary_metric  = "crowd_out_gap",
        pass_criterion  = (
            "crowd_out_gap is positive (mean across seeds) at every τ ≥ 0.15, "
            "and differential CBAM lies within the convex hull of the flat ladder."
        ),
        script          = "rice_jax/validation/cbam_experiment_B_tau_ladder.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Flat τ is a mechanism diagnostic, not a policy regime. The "
            "policy-relevant comparison is the differential point."
        ),
    ),

    # ── Experiment C — Multi-seed litmus freeze (audit §6.5) ───────────────
    "C_litmus_multiseed": ExperimentEntry(
        experiment_id   = "C_litmus_multiseed",
        question        = (
            "Which litmus conclusions (L1–L4) are seed-robust on the canonical "
            "9-region differential setup?"
        ),
        claim_bucket    = "mechanism",
        primary_metric  = "eu_dirty_export_share",
        pass_criterion  = (
            "Each of L1–L4 passes its predefined inequality on at least "
            "ceil(N_seeds/2) seeds; min-seed value also satisfies the inequality "
            "for any claim promoted to the headline."
        ),
        script          = "rice_jax/validation/cbam_experiment_C_litmus.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Litmus tests use action masking to isolate channels; they are "
            "mechanism existence proofs, not equilibrium analyses."
        ),
    ),

    # ── Experiment D — Allocation-rule objective split (audit §5.4) ────────
    "D_allocation_split": ExperimentEntry(
        experiment_id   = "D_allocation_split",
        question        = (
            "Which allocation rule wins on which explicit objective "
            "(least diversion / highest mitigation / strongest crowd-out attenuation)?"
        ),
        claim_bucket    = "policy-design",
        primary_metric  = "crowd_out_attenuation",
        pass_criterion  = (
            "Per-objective winner table; only claim a single 'best rule' if "
            "one rule dominates across all three objectives."
        ),
        script          = "rice_jax/validation/cbam_experiment_D_alloc.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Allocation rules are evaluated under fixed revenue_share=1.0 and "
            "transfer_mode='abatement'; rule × mode interactions are not "
            "explored in this experiment."
        ),
    ),

    # ── Experiment F — Sensitivity & uncertainty (audit §6.9) ──────────────
    # Tier 1: differentiable local Jacobian screen. Tier 2: targeted retrain.
    "F_sensitivity_tier1": ExperimentEntry(
        experiment_id   = "F_sensitivity_tier1",
        question        = (
            "Which continuous env parameters have the largest local influence "
            "on crowd_out_gap and eu_dirty_export_share, at the canonical "
            "trained-policy operating point?"
        ),
        claim_bucket    = "robustness",
        primary_metric  = "crowd_out_gap",
        pass_criterion  = (
            "Tornado plot ranks parameters by |∂metric/∂θ|; top 2–3 are "
            "promoted to Tier 2 retraining sweeps."
        ),
        script          = "rice_jax/validation/cbam_experiment_F_jacobian.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Local-only: does not capture the policy-response term "
            "∂π(θ)/∂θ. Tier 2 retraining is required for any claim about "
            "'the policy is robust to θ'."
        ),
    ),

    "F_sensitivity_tier2": ExperimentEntry(
        experiment_id   = "F_sensitivity_tier2",
        question        = (
            "For the parameters surfaced by Tier 1, does the headline crowd-out "
            "claim survive retraining at low/high values?"
        ),
        claim_bucket    = "robustness",
        primary_metric  = "crowd_out_gap",
        pass_criterion  = (
            "Headline-direction crowd_out_gap retains sign on every "
            "(parameter, value, seed) cell; magnitudes reported with seed band "
            "and worst-seed value."
        ),
        script          = "rice_jax/validation/cbam_experiment_F_retrain.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Limited to 2–3 parameters surfaced by Tier 1; not a full "
            "global sensitivity analysis."
        ),
    ),

    # ── Experiment E — Transfer pool amplifier sweep (Phase 2B, 2026-05-21) ─
    # Addresses scale-mismatch finding: raw CBAM revenue (≈0.04/exporter) is
    # ~40× too small to shift agent incentives (ΔU≈0.001 vs. diversion
    # λ·cost≈0.04). Sweep finds M* where redistribution first attenuates
    # crowd-out. Models external climate finance (NCQG/GCF top-up of CBAM pool)
    # following Fischer & Fox (2012) and Böhringer, Fischer & Rosendahl (2010).
    "E_amplifier": ExperimentEntry(
        experiment_id   = "E_amplifier",
        question        = (
            "At what external transfer multiplier (× CBAM revenue) does "
            "redistribution become sufficient to suppress trade diversion and "
            "make mitigation dominant over diversion?"
        ),
        claim_bucket    = "policy-design",
        primary_metric  = "crowd_out_attenuation",
        pass_criterion  = (
            "There exists M* ∈ {1, 2, 5, 10, 20, 50} such that "
            "crowd_out_attenuation > 0 on the worst seed; attenuation is "
            "monotonically non-decreasing in M from M=1 to M*. "
            "Post-hoc FD1–FD8 scorecard must show no active failure modes."
        ),
        script          = "rice_jax/validation/cbam_experiment_C_amplifier.py",
        seeds           = CANONICAL_SEEDS,
        interpretation_limits = (
            "Transfer mode fixed to 'abatement', allocation fixed to 'effort' "
            "(winning combination from Experiment A). Pool amplification models "
            "external climate finance (NCQG/GCF top-up) but does not represent "
            "how that finance is raised or its macroeconomic cost. Diversion "
            "remains frictionless — M* is therefore an upper bound on required "
            "subsidy relative to a model with diversion costs."
        ),
    ),
}


def get(experiment_id: str) -> ExperimentEntry:
    """Lookup helper; raises KeyError with a clear message if missing."""
    if experiment_id not in REGISTRY:
        raise KeyError(
            f"Experiment '{experiment_id}' not in registry. "
            f"Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[experiment_id]


def claim_bucket_of(experiment_id: str) -> ClaimBucket:
    return get(experiment_id).claim_bucket


__all__ = [
    "ExperimentEntry",
    "ClaimBucket",
    "REGISTRY",
    "get",
    "claim_bucket_of",
]
