# PLS Climate Club

A simple Probabilistic Logic Shield (PLS) climate club for the core JAX RICE-N environment. This package does not use MRIO.

The implementation follows Yang et al. (2023), *Safe Reinforcement Learning via Probabilistic Logic Shields*. For a club member, mitigation actions are reweighted according to their safety weight:

```text
pi_plus(a | s) proportional to P(safe | a, s) * pi(a | s)
```

The shield replaces the hard mitigation action mask. It does not make defection impossible: actions below the club mitigation floor retain nonzero probability. Defection is sanctioned economically through a tariff imposed by complying club members, following the climate-club mechanism in Nordhaus (2015).

## Variants

### Naive club

`PLSNaiveClubRice` uses fixed club terms and lets each region choose whether to join at every step.

- `club_min_rate`: required mitigation rate for members.
- `club_tariff`: tariff imposed by complying members on non-members and defectors.
- `club_join`: region action with `0 = stay out` and `1 = join`.
- `shield_strength`: downweight applied to below-floor mitigation actions.
- `shield_mode`: `constant` or `graded`.

### Mediator club

`PLSClubMediator` extends the existing mediator protocol. The mediator proposes the minimum mitigation rate and tariff, and regions vote to join during the negotiation cycle.

The mediator variant keeps the proposal and voting stages but removes the hard mitigation floor. Members are shielded softly, and members that defect during the climate step are tariff-sanctioned.

### Tariff-ambition club

`PLSClubTariffAmbition` is `BasicClubTariffAmbition` with the hard floor mask on the mitigation head replaced by the PLS shield. The accepted-proposal floor and the inherited tariff-ambition mask (minimum tariffs keyed on realized mitigation shortfall) are unchanged, so below-floor mitigation remains possible and is tariff-sanctioned.

Compare it against the hard-mask original:

```bash
conda run -n rice-jax python pls_club/experiments/run_tariff_ambition_comparison.py \
  --timesteps 200000 --num-regions 7 --eval-episodes 10 --seed 0
```

This trains both conditions with identical PPO settings and writes `pls_club/plots/pls_tariff_ambition_comparison_<seed>.{pkl,csv,png}` comparing mitigation, floor shortfall, defection frequency, emissions, tariffs, and reward over the last third of climate steps.

## Run training

From the repository root:

```bash
cd rice_jax
conda run -n rice-jax python pls_club/drivers/train_pls_club.py \
  --variant naive --timesteps 200000 --num-regions 7 --seed 0
```

Run the mediator variant with:

```bash
conda run -n rice-jax python pls_club/drivers/train_pls_club.py \
  --variant mediator --timesteps 200000 --num-regions 7 --seed 0
```

Optional shield arguments:

```bash
--shield-strength 0.8
--shield-mode constant
```

Use `--shield-mode graded` to make the safety weight vary continuously with the mitigation shortfall. Training logs are written to `pls_club/training_logs/`.

## Run evaluation

The shield-ablation experiment trains an unshielded control and a PLS policy,
then evaluates both on logged episodes. This isolates the behavioral effect of
the probabilistic shield while keeping the club variant and environment size
fixed.

```bash
conda run -n rice-jax python pls_club/experiments/run_shield_ablation.py \
  --variant naive --timesteps 200000 --num-regions 7 \
  --eval-episodes 10 --seed 0
```

Use `--variant mediator` for the mediator club. The experiment writes a pickle
artifact to `pls_club/plots/`. Run the posthoc analysis without retraining:

```bash
conda run -n rice-jax python pls_club/posthoc/analyze_shield_ablation.py \
  pls_club/plots/pls_club_shield_ablation_naive_0.pkl
```

Posthoc output includes `summary.csv`, `scorecard.md`, `comparison.png`, and a
climate-step `trajectories.png` showing membership, defection, member versus
non-member mitigation, emissions, cumulative emissions, and mediator terms.
Metrics are computed over the final third of each evaluation episode and cover
membership, member defection, mitigation, global emissions, import tariffs,
and mean reward. For the mediator variant, metrics use climate steps only;
propose/evaluate steps repeat stale economic state and are dropped.

Expected qualitative result: the shield lowers the member defection rate and
raises member mitigation relative to the unshielded control at comparable
membership, without increasing emissions.

## Run tests

From `rice_jax/`:

```bash
conda run -n rice-jax python -m pytest pls_club/tests/test_pls_club.py -v
```

The tests cover:

- equivalence with standard action masking for binary weights;
- exact PLS policy renormalization;
- the null shield and non-member behavior;
- naive and mediator null conditions;
- defector tariff sanctions.

## Package layout

```text
pls_club/
  env.py                         # Naive and mediator club environments
  shield.py                      # Safety weights and shielded categorical layer
  config.py                      # Environment and PPO defaults
  drivers/train_pls_club.py     # Training entry point
  experiments/run_shield_ablation.py  # Control vs PLS evaluation experiment
  posthoc/analyze_shield_ablation.py # CSV, scorecard, and plot generation
  tests/test_pls_club.py        # Canonical sanity checks
```

## Important design choice

Membership used for a climate step is the membership that the shield acted on. A member can therefore sample a below-floor mitigation action and defect. The next state records the defector and complying members impose the tariff on that defector and on regions outside the complying set.

The current implementation uses the shielded policy gradient only. The additional safety-loss term from Yang et al. (2023, Eq. 7) is not included.
