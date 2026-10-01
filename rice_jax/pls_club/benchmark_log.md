# PLS benchmark log

## 2026-10-01 — benchmark scope

Requested comparison: benchmark the probabilistic shield against the original
`BasicClubTariffAmbition` (BCTA) climate-competition implementation.

Scope:

- Use the non-mediator tariff-ambition environments only.
- Compare `PLSClubTariffAmbition` with `BasicClubTariffAmbition`.
- Keep PPO settings, region count, seeds, evaluation episodes, and action
  discretization aligned between conditions.
- Verify that PLS leaves below-floor mitigation actions reachable, while the
  original BCTA hard mask makes those actions unreachable.
- Record mitigation, floor shortfall, realized defection frequency, emissions,
  tariffs, and reward.

The existing comparison driver is
`pls_club/experiments/run_tariff_ambition_comparison.py`. The PLS condition
uses `ShieldedCategoricalLayer`; the BCTA condition uses the stock categorical
layer and its hard mitigation mask. Neither condition uses `RiceClubMediator`.

## 2026-10-01 — smoke benchmark

Command:

```bash
uv run python pls_club/experiments/run_tariff_ambition_comparison.py \
  --timesteps 2000 --num-regions 3 --eval-episodes 2 --seed 0
```

Results are from the final third of climate steps:

| condition | mean mitigation | mean floor shortfall | defection frequency | mean reward |
|---|---:|---:|---:|---:|
| original BCTA hard mask | 0.694 | 0.0000 | 0.0000 | -0.0382 |
| PLS shield | 0.522 | 0.1528 | 0.3333 | 0.0201 |

The short run confirms the intended mechanism: the hard-mask BCTA condition
produced no below-floor actions, while PLS produced below-floor actions and
nonzero realized defection. This is a smoke benchmark, not a statistical
claim; longer matched-seed runs are needed for performance conclusions.

Artifacts:
`pls_club/plots/pls_tariff_ambition_comparison_0.{pkl,csv,png}`.

## 2026-10-01 — three shield modes

Added `pls_club/experiments/run_shield_mode_comparison.py` to compare the
null, constant, and graded PLS modes on the same non-mediator BCTA environment.

Command:

```bash
uv run python pls_club/experiments/run_shield_mode_comparison.py \
  --timesteps 2000 --num-regions 3 --eval-episodes 2 --seed 0
```

Smoke results:

| condition | mean mitigation | mean floor shortfall | defection frequency | mean reward |
|---|---:|---:|---:|---:|
| null/unshielded | 0.367 | 0.2889 | 0.6111 | 0.0674 |
| constant | 0.533 | 0.1444 | 0.3611 | -0.0645 |
| graded | 0.686 | 0.0194 | 0.1667 | -0.0333 |

These are smoke-test results only; the training run is too short for
performance conclusions. Artifacts:
`pls_club/plots/pls_shield_mode_comparison_0.{pkl,csv,png}`.

## 2026-10-01 — collective trajectory report

Added `pls_club/experiments/run_collective_curves.py`, which collects the
requested trajectories across four conditions:

- `No Club`: plain core `Rice`
- `Club`: original `BasicClubTariffAmbition` hard-mask condition
- `Shield 1`: constant PLS weights
- `Shield 2`: graded PLS weights

The collective figure reports global temperature, selected mitigation action,
realized mitigation, savings, utility, output, defection, floor shortfall, and
evaluation time per step. The CSV also includes training time per timestep.

Smoke command:

```bash
uv run python pls_club/experiments/run_collective_curves.py \
  --timesteps 1000 --num-regions 3 --eval-episodes 1 --seed 0
```

Artifacts:
`pls_club/plots/pls_collective_curves_0.{pkl,csv,png}`.

## 2026-10-01 — 1M-step comparison launched

The initial five-condition comparison was launched with 1,000,000 PPO
timesteps per condition, 7 regions, 10 evaluation episodes, and seed 0. It
was stopped after `No Club` completed and `Club` reached its first checkpoint,
in order to remove the `No Shield` condition before restarting.

```bash
uv run python pls_club/experiments/run_collective_curves.py \
  --timesteps 1000000 --num-regions 7 --eval-episodes 10 --seed 0 \
  --progress-every 50000 \
  --progress-log pls_club/plots/pls_collective_curves_1m_progress.log
```

Progress is flushed every 50,000 timesteps. The run currently has completed
the first checkpoint for `No Club` at timestep 51,350 after 594 seconds,
including JAX compilation. The live progress log is
`pls_club/plots/pls_collective_curves_1m_progress.log`.

The collective runner is now configured for four conditions only:
`No Club`, `Club`, `Shield 1`, and `Shield 2`.
