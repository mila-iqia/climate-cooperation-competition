# Club Mediator (two-layer architecture on JAX RICE-N)

A mediator agent defines a climate club; regions decide only whether to join.
Built on the base `Rice` env ([../rice_jax/core/env.py](../rice_jax/core/env.py)) —
**not** the MRIO/CBAM variant.

## Mechanism

Each negotiation cycle (`current_timestep % 3`) reuses the base 3-stage layout:

| Stage | Who acts | Action |
|-------|----------|--------|
| 1 — propose | mediator | `club_min_rate` (minimum mitigation for membership), `club_tariff` (tariff on non-members) |
| 2 — evaluate | regions | `club_join` ∈ {reject, accept} |
| 0 — climate/economy | regions | standard RICE actions (savings, mitigation, trade) |

Regions have **no propose actions** (the base bilateral `proposal_ask`/`proposal_promise`
protocol is replaced). Membership is re-proposed and re-voted every cycle.

Consequences:

- **Members**: mitigation rate floored at `club_min_mitigation` via the existing
  action-mask machinery (`minimum_mitigation_rate_all_regions`).
- **Non-members**: every member's import tariff on them is floored at the club
  rate — Nordhaus (2015, AER 105(4)) uniform penalty tariff. The effective tariff
  matrix is also written to `state["import_tariffs"]` so the welfare-loss channel
  (`calc_welfloss_multiplier`) actually bites (base `Rice` leaves it at zero).

## Mediator agent

The mediator joins the agent dict (key `"mediator"`) with the **same action/obs
structure as regions** — required by `process_actions` and the shared-policy PPO.
Role-irrelevant action slots are pinned to a single value via action masks, and a
`role_flag` observation feature lets the shared policy differentiate roles.

Reward modes (`mediator_reward_mode`):

- `"emissions"` (default): `-global_emissions * mediator_reward_scale`. A
  worst-case-baseline offset would be action-independent, so it is omitted.
- `"members"`: mean club membership.

`fixed_club_params=(min_mitigation, tariff)` bypasses the mediator policy — used
for the sanity experiments below.

## Files

| File | Purpose |
|------|---------|
| [env.py](env.py) | `RiceClubMediator(Rice)` — all club/mediator logic |
| [config.py](config.py) | `make_club_env()` factory, PPO defaults, club log fn |
| [drivers/train_club.py](drivers/train_club.py) | Train regions + learned mediator (shared policy) |
| [drivers/sanity_fixed_mediator.py](drivers/sanity_fixed_mediator.py) | Fixed-mediator sanity experiments |
| [tests/test_club_env.py](tests/test_club_env.py) | Fixed-action unit tests incl. canonical null |

## Usage (from `rice_jax/`)

```bash
conda activate rice-jax

# unit tests (canonical null, masks, tariffs, rewards, PPO smoke)
pytest club/tests/test_club_env.py -v

# learned mediator
python club/drivers/train_club.py --timesteps 200000 --mediator-reward emissions

# sanity: low-bar club (μ≥0.1, τ=0.9) should fill up; high-bar (μ≥0.9, τ=0) should not
python club/drivers/sanity_fixed_mediator.py --timesteps 200000
```

`JAXNASIUM_MULTI_AGENT_BATCH_SIZE=false` is set automatically by the drivers/tests.
Standalone training logs are written to `club/training_logs/`, and sanity summaries
are written to `club/plots/`. The `CBAM_EXPERIMENT_DIR` environment variable still
redirects these outputs into a managed experiment directory when set.

## Canonical null condition

`fixed_club_params=(0.0, 0.0)` + all regions rejecting reproduces the base
`Rice(negotiation_on=False)` climate-stage trajectory exactly
(`test_null_club_matches_base_rice`, checked with `allclose`).

## Implementation notes

- `"mediator"` sorts before `"region-XX"`, so the mediator is row 0 in the
  stacked per-action arrays; climate steps slice it off.
- Stage overrides must `state.copy()` before mutating: the base `step_env`
  `lax.switch` branches share the closed-over state dict, and in-place mutation
  leaks tracers in eager mode.
- No `StackActionSpaceWrapper`: the action dict mixes `Discrete(2)` and
  `Discrete(L)`; drivers wrap with `jym.LogWrapper` only.
