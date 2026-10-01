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
    "themis_price_sweep": ExperimentEntry(
        experiment_id="themis_price_sweep",
        question="Does a Themis carbon-payment price ladder raise mitigation "
        "and who joins/pays at each price? (Rasmussen 2025 concept note)",
        claim_bucket="mechanism",
        primary_metric="mean non-EU mitigation rate vs price; membership rate",
        pass_criterion="mean mu monotone non-decreasing in p; cost-neutrality "
        "residual |sum payments| < 1e-4 $T at every price",
        script="cbam/drivers/cbam_experiment_themis_sweep.py",
        seeds=(42,),
    ),
}
