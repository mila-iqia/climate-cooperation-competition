"""Experiment registry stub — extend as headline experiments are restored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentEntry:
    experiment_id: str
    question: str
    claim_bucket: str
    primary_metric: str
    pass_criterion: str
    script: str
    seeds: tuple[int, ...]


REGISTRY: dict[str, ExperimentEntry] = {
    "C_litmus_multiseed": ExperimentEntry(
        experiment_id="C_litmus_multiseed",
        question="Which litmus conclusions (M/C tests) are seed-robust?",
        claim_bucket="mechanism",
        primary_metric="majority_pass per test",
        pass_criterion="Each test passes on >= ceil(N_seeds/2) seeds",
        script="cbam/drivers/cbam_experiment_C_litmus.py",
        seeds=(0, 1, 2),
    ),
    # Transfer pool amplifier sweep (Phase 2B Tier 2, external finance).
    # Ported from rice_jax/validation/registry.py "E_amplifier" (2026-05-21).
    "E_amplifier": ExperimentEntry(
        experiment_id="E_amplifier",
        question=(
            "At what external transfer multiplier (x CBAM revenue) does "
            "redistribution become sufficient to suppress trade diversion and "
            "make mitigation dominant over diversion?"
        ),
        claim_bucket="policy-design",
        primary_metric="crowd_out_attenuation",
        pass_criterion=(
            "There exists M* in {1, 2, 5, 10, 20, 50} such that "
            "crowd_out_attenuation > 0 on the worst seed; attenuation is "
            "monotonically non-decreasing in M from M=1 to M*."
        ),
        script="cbam/drivers/cbam_experiment_C_amplifier.py",
        seeds=(0, 1, 2),
    ),
    "C_mac_ceiling": ExperimentEntry(
        experiment_id="C_mac_ceiling",
        question=(
            "How large is the mitigation gap caused by realistic abatement "
            "cost under differential CBAM with both channels open?"
        ),
        claim_bucket="policy-design",
        primary_metric="free_minus_costly_mitigation_gap",
        pass_criterion="Descriptive scoping run; report the mitigation gap and financing proxy.",
        script="cbam/drivers/cbam_experiment_C_mac_ceiling.py",
        seeds=(0,),
    ),
}
