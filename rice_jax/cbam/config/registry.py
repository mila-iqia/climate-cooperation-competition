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
}
