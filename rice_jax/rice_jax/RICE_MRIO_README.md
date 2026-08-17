# `RiceMRIO` — MRIO/CBAM RICE Environment

A reference guide for researchers working on the AAAI CBAM submission.

`RiceMRIO` ([`mrio/env.py`](mrio/env.py)) is a JAX/Equinox subclass of the base
`Rice` environment ([`core/env.py`](core/env.py)) that adds **EORA26 multi-region
input–output (MRIO) trade structure** and a **Carbon Border Adjustment Mechanism
(CBAM)** on top of the standard RICE climate–economy model. It is the
single environment behind every CBAM experiment in this repo.

**Research question.** What combination of tariff design, technology transfer, and
cash transfer makes decarbonisation the dominant strategy for emission-heavy
exporters, rather than diverting dirty exports away from the EU?

> **Read these alongside this file**
> - [`MRIO_RICE_DESIGN.md`](MRIO_RICE_DESIGN.md) — conceptual model, MRIO data, research motivation
> - [`EXTENDING.md`](EXTENDING.md) — how the action/observation/state machinery works in this JymKit fork
> - [`../notes/CBAM_ROADMAP.md`](../notes/CBAM_ROADMAP.md) — phase sequence, entry states, exit criteria
> - `.github/instructions/rice-mrio-implementation.instructions.md` — patterns for adding new mechanisms
> - `.github/instructions/cbam-experiment-workflow.instructions.md` — patterns for running experiments

---

## Contents

1. [Quick start](#1-quick-start)
2. [Conda environment](#2-conda-environment)
3. [Two phases: 1B vs 2A](#3-two-phases-1b-vs-2a)
4. [All constructor arguments](#4-all-constructor-arguments)
5. [String-valued options enumerated](#5-string-valued-options-enumerated)
6. [Region indexing (the #1 footgun)](#6-region-indexing-the-1-footgun)
7. [Action space](#7-action-space)
8. [Observation space](#8-observation-space)
9. [State keys added by RiceMRIO](#9-state-keys-added-by-ricemrio)
10. [Step pipeline order](#10-step-pipeline-order)
11. [The canonical config (use this)](#11-the-canonical-config-use-this)
12. [Training](#12-training)
13. [Evaluation & metrics](#13-evaluation--metrics)
14. [Running experiments end-to-end](#14-running-experiments-end-to-end)
15. [Key validation & experiment scripts](#15-key-validation--experiment-scripts)
16. [Headline experiment registry](#16-headline-experiment-registry)
17. [Data files](#17-data-files)
18. [Gotchas & footguns](#18-gotchas--footguns)

---

## 1. Quick start

The **recommended** path for any headline experiment is the canonical factory —
it freezes the paper setup and refuses to let you silently change structural
parameters (see [§11](#11-the-canonical-config-use-this)):

```python
from validation.canonical_config import make_canonical_env

env = make_canonical_env(for_training=False)          # paper-frozen 9-region setup
env = make_canonical_env(for_training=False, revenue_share=1.0)   # one knob varied
```

For a **raw** construction (e.g. quick tests, non-canonical region counts):

```python
from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls

YAML_DIR  = "../cbam_yamls/setup_vuln_9"     # relative to rice_jax/
MRIO_ROOT = "../csv_asset"

rp = load_region_yamls(9, yaml_dir=YAML_DIR)  # -> SimpleNamespace of region params

env = RiceMRIO(
    region_params      = rp,
    num_regions        = 9,
    mrio_data_root     = MRIO_ROOT,
    mrio_trade         = True,
    eu_region_idx      = 3,                   # ALWAYS set explicitly — default 0 is wrong
    sector_granularity = "emissions-simple",
    cbam_tariff_mode   = "differential",
)

import jax
key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
```

`RiceMRIO` is a frozen `eqx.Module`. To make a modified copy, **never** mutate a
field — use `dataclasses.replace`:

```python
from dataclasses import replace
env_rs1 = replace(env, revenue_share=1.0)
```

---

## 2. Conda environment

| Env | Python | Use for |
|-----|--------|---------|
| `rice-jax` | 3.11 | **all** JAX / jaxnasium / rice_jax work |
| `gaia` | 3.14 | no JAX — do **not** use for training |

```bash
cd rice_jax
conda activate rice-jax
pip install -e .          # editable install (first time only)
```

Always run experiment scripts from the `rice_jax/` directory so the relative
`../csv_asset` and `../cbam_yamls` paths resolve.

---

## 3. Two phases: 1B vs 2A

`RiceMRIO` has two behavioural modes, switched by `mrio_trade`:

- **Phase 1B** (`mrio_trade=False`, default): bit-identical to base `Rice`.
  Production is disaggregated across EORA26 sectors using static 2016 shares and
  reaggregated exactly. `production_by_sector` is computed but not consumed.
  Trade still uses the parent's `import_bid` / `export_limit` mechanism.
- **Phase 2A** (`mrio_trade=True`): the CBAM model. The bid/limit actions are
  removed and replaced by `export_reallocation`; CBAM tariffs penalise EU-bound
  dirty exports; new state keys `trade_flows`, `cbam_revenue`, `cbam_cost_all_regions`
  appear. **Every CBAM experiment uses `mrio_trade=True`.**

The rest of this document assumes Phase 2A unless noted.

---

## 4. All constructor arguments

All config fields are `eqx.field(static=True)` (compile-time constants). Dynamic
quantities live in the `state` dict, not as fields. Defaults shown are
backward-compatible (mostly inert).

### 4.1 Inherited from `Rice`

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `region_params` | `SimpleNamespace` (required) | Region economic params from `load_region_yamls(...)`. |
| `num_regions` | `int = 3` | 3, 7, or 20 natively; **9** supported via `yaml_dir` override. |
| `diff_reward_mode` | `bool = True` | Reward is Δutility per step rather than absolute utility. |
| `relative_reward_mode` | `bool = False` | Reward relative to other regions. |
| `num_discrete_action_levels` | `int = 10` | Discretisation `D` of each action dimension. |
| `action_window_size` | `int = 0` | Max discrete-level change per step for mitigation/savings. `0` = unconstrained. Canonical uses `0` + transition cost instead. |
| `disable_trading` | `bool = False` | Base-Rice trade toggle (unused in Phase 2A). |
| `negotiation_on` | `bool = False` | 3-stage propose→evaluate→climate cycle (used by club scenarios). |
| `dmg_function` | `"base" \| "updated"` | RICE damage function form. |
| `temperature_calibration` | `"base" \| "FaIR" \| "DFaIR"` | Temperature module. |
| `carbon_model` | `"base" \| "FaIR" \| "DFaIR" \| "AR5"` | Carbon cycle module. |
| `apply_welfloss` / `apply_welfgain` | `bool = True` | Enable welfare-loss / welfare-gain multipliers. |
| `init_capital_multiplier` | `float = 10.0` | Initial capital scaling. |
| `balance_interest_rate` | `float = 0.1` | Trade balance interest rate. |
| `consumption_substitution_rate` | `float = 0.5` | Armington substitution. |
| `preference_for_domestic` | `float = 0.5` | Home bias in consumption. |
| `log_info_fn` | `Callable = empty_info_log_fn` | Per-step info logger. Set to `rcpo_cbam_log_info_fn` for RCPO. |
| `init_gamma` | `float = 0.99` | Discount factor. |

### 4.2 Phase 1B — MRIO sectoral structure

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `mrio_data_root` | `str = "csv_asset"` | Path to the `csv_asset` dir holding `mrio/aggregated/eora_agg_{N}/` and `CountryClass_{N}.csv`. |
| `sector_granularity` | `str = "full"` | How 26 EORA sectors are collapsed before building trade arrays. See [§5](#5-string-valued-options-enumerated). |

> If the MRIO data directory is missing, `__post_init__` returns early so the env
> can still be constructed lightweight in unit tests (trade arrays stay `None`).

### 4.3 Phase 2A — MRIO trade

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `mrio_trade` | `bool = False` | **Master switch** for Phase 2A behaviour. |
| `eu_region_idx` | `int = 0` | 0-based index of the EU region. **Default 0 is wrong for 7/9 regions — always set explicitly.** |
| `delta_max` | `float = 3.0` | Tanh/logit squashing bound for `export_reallocation` adjustments. `0` pins exports to baseline. |
| `dest_alloc_persistence` | `float = 0.0` | AR(1) weight ρ blending 2016 baseline with previous realised allocation as the logit anchor. `0` = always anchor to 2016; `1` = fully adaptive. Canonical `0.55`. (Roberts & Tybout 1997; Eaton & Kortum 2002.) |
| `dest_alloc_baseline_decay` | `float = 0.0` | Decay exponent for the 2016 anchor weight `(1-ρ)^(decay·t)`. **Keep `0.0` in headline runs** — non-zero makes the anchor vanish by mid-episode and fakes diversion. |
| `cbam_intensity_target_mean` | `float = 0.1` | Rescales raw EORA emissions intensities to this mean so the CBAM signal is non-negligible while preserving cross-sector heterogeneity. |

### 4.4 CBAM tariff

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `cbam_tariff_rate` | `float = 0.0` | Flat CBAM rate (fraction). Used in `"flat"` mode; acts as on/off gate in `"differential"` mode when randomised. |
| `cbam_tariff_mode` | `str = "flat"` | `"flat"` (uniform τ) or `"differential"` (per-region MAC-gap τ_eff). See [§5](#5-string-valued-options-enumerated). |
| `cbam_randomize` | `bool = False` | Sample the per-episode rate from `cbam_tariff_rates` and expose it in the obs — lets one policy learn a τ-conditioned response. |
| `cbam_tariff_rates` | `tuple = (0.0,)` | Pool sampled from when `cbam_randomize=True`, e.g. `(0.0, 0.8)`. |
| `mu_floor_differential` | `float = 0.01` | μ clamp avoiding the MAC singularity at μ→0 (θ₂>1) in differential mode. |
| `welfare_loss_per_unit_tariff` | `float = 0.4` | Welfare-loss amplifier α (Nordhaus 2015). Canonical `5.0` for detectable PPO signal. |
| `sectoral_welfloss` | `bool = False` | If `True`, welfare loss is computed per sector weighted by emissions intensity → gives `export_reallocation` a per-sector gradient (selective diversion). |

### 4.5 Reward channel

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `reward_mode` | `str = "welfloss"` | `"welfloss"` (multiplicative penalty on utility) or `"additive_cbam"` (RCPO: r̂ = ΔU − λ·cost). See [§5](#5-string-valued-options-enumerated). |
| `cbam_lambda_init` | `float = 0.0` | Initial Lagrange multiplier λ in `state["cbam_lambda"]`. Set `>0` for a fixed calibrated penalty that survives episode resets (needed on small-EU-share setups). |

### 4.6 Phase 2B — revenue recycling

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `revenue_share` | `float = 0.0` | Fraction of the CBAM pool redistributed to exporters. `0` = no transfer; `1` = full redistribution. |
| `transfer_mode` | `str = "consumption"` | `"consumption"` (lump-sum cash) or `"abatement"` (earmarked, capped abatement subsidy). See [§5](#5-string-valued-options-enumerated). |
| `transfer_allocation` | `str = "burden"` | How the pool is split: `"burden" \| "effort" \| "equal" \| "vulnerability" \| "hybrid"`. See [§5](#5-string-valued-options-enumerated). |
| `transfer_pool_multiplier` | `float = 1.0` | External-finance amplifier on the pool (NCQG/GCF top-up). `1.0` = pool equals collected CBAM revenue. |

### 4.7 Mitigation / abatement levers

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `eu_mitigation_schedule` | `tuple \| None = None` | Per-step prescribed EU μ pathway (ramps EU's MAC up so differential τ is non-trivial from t=0). Other regions learn freely. Values past the horizon clamp to the last entry. |
| `zero_abatement_cost` | `bool = False` | Zero out abatement costs (free abatement) — isolates the CBAM→mitigation incentive (litmus L2). |
| `no_mitigation` | `bool = False` | Remove `mitigation_rate` from the action space and pin μ=0 (BAU litmus). |
| `fixed_savings_rate` | `bool = False` | Remove `savings_rate` from the action space and pin it to 0.2 (≈Nordhaus optimal). **Keep `False` in headline runs** — fixing savings kills C2 conditioning. |
| `transition_cost_coef` | `float = 0.0` | Grubb et al. (1995) DIAM transitional cost: `TC = coef·((μ_t−μ_{t-1})/Δt)²` as a GDP fraction. Canonical `10.0` (replaces `action_window_size` as the smoothness mechanism). |
| `transition_cost_asymmetric` | `bool = False` | If `True`, only penalise μ *increases* (downward reversals free). |

### 4.8 Regional damages

| Arg | Type / default | Meaning |
|-----|----------------|---------|
| `use_regional_damage_coeff` | `bool = False` | Use per-region damage coefficients instead of uniform yaml `xa_2`. Raises at init if `regional_damage_coeff` is `None`. |
| `regional_damage_coeff` | `np.ndarray = None` | Shape `(num_regions,)`, °C⁻². Per-region quadratic temperature sensitivity. `[LITERATURE NEEDED]` — Ricke et al. 2018 / Hänsel et al. 2020. |

---

## 5. String-valued options enumerated

### `sector_granularity`
Controls how the 26 EORA sectors are aggregated for the `export_reallocation`
action and trade arrays.

| Value | Sectors | Notes |
|-------|---------|-------|
| `"full"` | 26 | All EORA sectors unchanged (default). |
| `"cbam-specific"` | 4 | 3 CBAM sectors separate + "non-CBAM". |
| `"simple"` | 2 | "CBAM" (3 covered sectors) vs "non-CBAM". |
| `"emissions-specific"` | 8 | 7 high-emission sectors separate + "non-CBAM". |
| **`"emissions-simple"`** | **2** | **dirty (idx 0) vs clean (idx 1). Standard for all current experiments.** |

CBAM-covered: Petroleum/Chemical/Non-Metallic Minerals, Metal Products, Electricity/Gas/Water.
Dirty set additionally includes: Other Mfg, Transport Equipment, Construction, Mining & Quarrying.

### `cbam_tariff_mode`

| Value | Formula | Notes |
|-------|---------|-------|
| `"flat"` | uniform `cbam_tariff_rate` | Backward-compatible. Literature-grounded range τ∈{0.05…0.25} (Böhringer et al. 2010; Martin et al. 2014); τ=0.80 is the original empirical diagnostic. |
| **`"differential"`** | `τ_eff[r] = max(0, MAC_EU − MAC_r) / MAC_EU` | Implements the actual CBAM (Reg. 2023/956 Art. 5–7): tariff on the carbon-price differential. As μ_r → μ_EU, τ_eff → 0. Self-incentivising mitigation channel. **Canonical.** |

MAC uses the RICE backstop curve (Nordhaus 2017 DICE-2016R eq. 9):
`MAC_r(μ,t) = p_b_r·(1−δ_pb_r)^(t−1)·μ^(θ₂_r−1)`.
**Null test:** when all regions share EU's μ, `τ_eff[r]=0` → `cbam_cost=0`.

### `reward_mode`

| Value | Reward | Notes |
|-------|--------|-------|
| `"welfloss"` | `Δ(U × welfloss)` | Multiplicative penalty. Needs manual α (`welfare_loss_per_unit_tariff`) calibration. |
| **`"additive_cbam"`** | `ΔU − λ·cbam_cost` | RCPO (Tessler et al. 2019). λ auto-tunes via `RCPOMonitoredPPO`. Requires `log_info_fn=rcpo_cbam_log_info_fn`. **Canonical.** |

### `transfer_mode`

| Value | Effect | Key property |
|-------|--------|--------------|
| `"consumption"` | Transfer added to consumption as lump-sum cash. | Erodes both diversion and mitigation incentives at rs=1 (Böhringer et al. 2010 §4 perverse recycling). |
| `"abatement"` | Transfer earmarked to offset abatement cost, capped at actual abatement spending; excess forfeited. | Diversion incentive unchanged, mitigation made cheaper → asymmetric. (Fischer & Springborn 2011 §3–4.) |

### `transfer_allocation`
All rules zero out the EU region after normalisation; EU never self-transfers.

| Value | Weight per exporter r | Rationale |
|-------|----------------------|-----------|
| `"burden"` | raw CBAM cost `c_r` | EU CBAM Regulation default; rewards staying dirty (moral hazard). |
| `"effort"` | mitigation rate `μ_r` | Rewards abatement effort; breaks dirty-equilibrium trap. (Angelsen et al. 2017 REDD+; Fischer & Springborn 2011.) |
| `"equal"` | `1/(NR−1)` | Pure income effect; null condition decoupling amount from behaviour. |
| `"vulnerability"` | `c_r / Y_r` | Favours small open economies with high exposure-per-GDP. (GCF/NCQG MVI 2024.) |
| `"hybrid"` | `μ_r · (c_r / Y_r)` | Effort × vulnerability. `[LITERATURE NEEDED]`; closest anchor Böhringer et al. 2010 §5. |

---

## 6. Region indexing (the #1 footgun)

`eu_region_idx` defaults to `0`, which is **wrong** for every multi-region setup.
Always pass it explicitly.

| Setup | `num_regions` | `eu_region_idx` | yaml dir |
|-------|---------------|-----------------|----------|
| 3-region | 3 | **1** | `cbam_yamls/setup_3` |
| 7-region | 7 | **5** | `cbam_yamls/setup_7` |
| **9-region (vulnerability, canonical)** | **9** | **3** | `cbam_yamls/setup_vuln_9` |

### 9-region "vulnerability" ordering

| idx | Region | Role |
|-----|--------|------|
| 0 | Rest of World | Passive catch-all — exclude from CBAM plots |
| 1 | Russia + Turkey + Eurasia | High σ, fossil |
| 2 | MENA (Gulf + N. Africa) | Oil exporters |
| **3** | **EU & Western Europe** | `eu_region_idx=3` — never receives transfer |
| 4 | SSA Metals & Mining | CBAM-vulnerability focus |
| 5 | Americas | Mixed |
| 6 | SE Asia & Pacific | Manufacturing |
| 7 | China | Single-country |
| 8 | India | Single-country |

Headline aggregates exclude **RoW (0)** and **EU (3)**: `NON_EU_EXPORTER_IDXS = (1,2,4,5,6,7,8)`.

### 7-region ordering

| idx | 0 | 1 | 2 | 3 | 4 | **5** | 6 |
|-----|---|---|---|---|---|---|---|
| Region | SSA | South Asia | N. America | MENA | LATAM | **Europe & C. Asia (EU)** | E. Asia & Pacific |

---

## 7. Action space

Phase 2A action space, per agent (`mrio_trade=True`):

```python
{
  "export_reallocation": MultiDiscrete([D] * (NS * NR)),   # logit ADJUSTMENT δ
  "savings_rate":        Discrete(D),    # omitted if fixed_savings_rate=True
  "mitigation_rate":     Discrete(D),    # omitted if no_mitigation=True
}
```

- `D = num_discrete_action_levels` (10), `NS = num_sectors`, `NR = num_regions`.
- After `process_actions`, discrete levels are mapped to `[0,1]`; the midpoint
  `D//2` → δ=0 → **MRIO 2016 baseline flows**.
- `export_reallocation` is a **logit adjustment over the baseline destination
  shares**, *not* an absolute share. To recover realised shares, read
  `state["trade_flows"]` (shape `(NR, NR, NS)` = `[from, to, sector]`).

**Example layout** (emissions-simple, 7 regions): 112 dims = 7 agents × 16 dims.
Per agent: `export_reallocation[14]` = sector0[7 dests] + sector1[7 dests],
`mitigation_rate[1]`, `savings_rate[1]`.

`action_window_size` masks gradual change for mitigation/savings only;
`export_reallocation` is intentionally excluded. The minimum-mitigation-rate
floor is also applied as a mask.

---

## 8. Observation space

With `mrio_trade=True`, `generate_observation` returns a compact trade-focused
obs per agent (falls back to the parent's climate-heavy obs when off):

| Key | Shape | Content |
|-----|-------|---------|
| `activity_timestep` | scalar | Episode step |
| `gross_output` | scalar | Own Y |
| `utility` | scalar | Own welfare |
| `trade_flows` | `(NR, NS)` | Own outgoing flows |
| `dest_alloc` | `(NS, NR)` | Own current allocation anchor |
| `cbam_revenue` | `(NR,)` | Who pays how much CBAM (public) |
| `cbam_tariff_rate` | scalar | Active per-episode rate |
| `cbam_cost` | scalar | Own CBAM cost |
| `cbam_lambda` | scalar | Current Lagrange multiplier |
| `revenue_share` | scalar | Fraction redistributed |
| `transfer_received` | scalar | Transfer this step |

Nested dicts are auto-concatenated into a single per-agent vector downstream.

---

## 9. State keys added by `RiceMRIO`

Beyond the base-Rice state, Phase 2A adds (initialised in `_get_initial_state`):

| Key | Shape | Meaning |
|-----|-------|---------|
| `production_by_sector` | `(NR, NS)` | Sector-disaggregated output |
| `trade_flows` | `(NR, NR, NS)` | Realised bilateral flows `[from, to, sector]` |
| `cbam_revenue` | `(NR,)` | CBAM collected (non-zero only in EU row) |
| `cbam_cost_all_regions` | `(NR,)` | Raw CBAM cost per exporter |
| `cbam_lambda` | scalar | RCPO Lagrange multiplier (mutated by the trainer) |
| `transfer_received` | `(NR,)` | Phase 2B transfer per region |
| `dest_alloc_current` | `(NR, NS, NR)` | AR(1) logit anchor carried across steps |
| `cbam_tariff_rate` | scalar | Active per-episode tariff |

---

## 10. Step pipeline order

`step_climate_and_economy` (Phase 2A) runs in this order — insert new mechanisms
at the correct stage:

1. Capture `prev_mitigation` (for transition cost), unpack `export_reallocation`.
2. Inject zeroed legacy trade + hardcoded fixed actions; apply `eu_mitigation_schedule`.
3. `super().step_climate_and_economy(...)` → climate, production, investment.
4. Optional `zero_abatement_cost` compensation; optional transitional cost (Grubb 1995).
5. Disaggregate production into sectors.
6. `_compute_trade_flows` → `trade_flows`, updated `dest_alloc` (AR(1) anchor).
7. `_compute_cbam` → tariff matrix, revenue, raw cost (+ `_postprocess_cbam` club hook).
8. Recompute consumptions with MRIO gross imports.
9. **Revenue transfer** (Phase 2B): `pool → burden_share → transfer_received → mode branch`.
10. Welfare-loss multiplier (sectoral or aggregate).
11. Recompute utilities; apply CBAM penalty per `reward_mode`.
12. Pack updated state.

---

## 11. The canonical config (use this)

[`cbam/config/canonical_config.py`](../cbam/config/canonical_config.py) is the
**single source of truth** for the paper-primary setup. Headline experiments must
import from it rather than redefining kwargs inline.

```python
from validation.canonical_config import (
    make_canonical_env, CANONICAL_SEEDS, CANONICAL_TRAIN_KWARGS,
    EU_REGION_IDX, NON_EU_EXPORTER_IDXS, REGION_NAMES,
)

env = make_canonical_env()                              # training (LogWrapper-wrapped)
env = make_canonical_env(for_training=False)            # raw env for eval
env = make_canonical_env(dest_alloc_persistence=0.30)   # sensitivity arm
env = make_canonical_env(num_regions=7)                 # RAISES — structural param
```

`make_canonical_env(**overrides)` only accepts:
- **Sensitivity params** (continuous, legitimate to vary): `dest_alloc_persistence`,
  `dest_alloc_baseline_decay`, `welfare_loss_per_unit_tariff`, `delta_max`,
  `cbam_lambda_init`, `action_window_size`, `transition_cost_coef`,
  `transition_cost_asymmetric`.
- **Experiment-level knobs** (claim-defining): `cbam_tariff_rate`, `cbam_tariff_mode`,
  `cbam_randomize`, `cbam_tariff_rates`, `revenue_share`, `transfer_pool_multiplier`,
  `transfer_mode`, `transfer_allocation`, `eu_mitigation_schedule`,
  `zero_abatement_cost`, `no_mitigation`, `fixed_savings_rate`.

Anything else raises — this enforces the frozen structural setup.

**Paper-frozen defaults** (as of the latest roadmap update): 9 regions,
`eu_region_idx=3`, `sector_granularity="emissions-simple"`, `diff_reward_mode=True`,
`sectoral_welfloss=True`, `cbam_tariff_mode="differential"`,
`dest_alloc_persistence=0.55`, `dest_alloc_baseline_decay=0.0`,
`welfare_loss_per_unit_tariff=5.0`, `delta_max=3.0`, `cbam_lambda_init=1.0`,
`action_window_size=0`, `transition_cost_coef=10.0`, `reward_mode="additive_cbam"`,
and the ramped `eu_mitigation_schedule` (0.30→1.00 over 8 steps).

**Seed protocol:** `CANONICAL_SEEDS = (0, 1, 2)`. A headline run using a different
seed set is not headline-comparable.

---

## 12. Training

Use jaxnasium ``PPO`` with the log helpers in
[`rice_jax/training/`](../rice_jax/training/), or ``RCPOMonitoredPPO`` when
training under the CBAM cost constraint:

- **`PPO`** (jaxnasium) — stock trainer; per-step ``actions`` / ``rewards`` come
  from ``log_info_fn``, and ``make_csv_log_fn`` / ``make_print_log_fn`` average
  them. Episode returns come from ``LogWrapper``.
- **`RCPOMonitoredPPO`** — PPO subclass with the RCPO λ update (Tessler et al.
  2019) for the CBAM cost constraint. Additionally logs `cbam_lambda`,
  `mean_cbam_cost`. Requires `reward_mode="additive_cbam"` and
  `log_info_fn=rcpo_cbam_log_info_fn` on the env. Hyperparams:
  `rcpo_eta_lambda` (default `5e-7`), `rcpo_alpha_target` (default `0.01`).

```python
import jax
from rice_jax.training import (
    RCPOMonitoredPPO, make_combined_log_fn, make_csv_log_fn,
    make_print_log_fn, rcpo_cbam_log_info_fn,
)
from validation.canonical_config import make_canonical_env, CANONICAL_TRAIN_KWARGS

env = make_canonical_env()   # already has log_info_fn=rcpo_cbam_log_info_fn

ppo = RCPOMonitoredPPO(
    **CANONICAL_TRAIN_KWARGS,
    log_fn=make_combined_log_fn([make_print_log_fn(),
                                 make_csv_log_fn("training_logs/run.csv")]),
)
agent, metrics = ppo.train(jax.random.PRNGKey(0), env)   # returns (PPOAgent, metrics)
```

> `train()` returns a new PPO object. **Do not discard the return value** —
> `ppo.train(...)` without assignment silently trains a throwaway.

CSV columns to watch: `ep_return_mean` (rising → improving), `cbam_lambda`
(rising → penalising cost), `mean_cbam_cost` (falling as λ rises),
`action_mean`/`action_var` (near 0 = at baseline / collapsed).

---

## 13. Evaluation & metrics

After training, roll out evaluation episodes and read `state["trade_flows"]`.

```python
from _experiment_util import run_single_episode   # rice_jax/_experiment_util.py
info = run_single_episode(key, env, agent)         # stacked per-step info dict
```

The **primary exit-criterion metric is EU dirty export share**:

```python
EU_IDX = 3
dirty_to_eu    = trade_flows[:, EU_IDX, 0]            # exports to EU, dirty sector
total_dirty    = trade_flows[:, :, 0].sum(axis=1)     # total dirty exports
eu_dirty_share = dirty_to_eu / (total_dirty + 1e-8)
```

Frozen metric functions live in
[`cbam/config/metrics.py`](../cbam/config/metrics.py) — experiment scripts must call
these rather than computing inline:

| Function | Returns |
|----------|---------|
| `eu_dirty_export_share` | Scalar EU dirty share (aggregated) |
| `per_region_eu_dirty_export_share` | `(NR,)` per-region share |
| `mean_mitigation_rate` | Scalar mean μ |
| `per_region_mitigation_rate` | `(NR,)` per-region μ |
| `crowd_out_gap` | μ_pinned − μ_open (diversion crowd-out) |
| `crowd_out_attenuation` | gap(rs=0) − gap(rs=1) (redistribution effect) |
| `transfer_effectiveness` | Transfer impact on diversion |
| `seed_summary` | mean/std/min across seeds |

Default time aggregation: mean over the last `EVAL_LAST_T=5` steps, then mean over
`NUM_EVAL_EPISODES=8` episodes.

**Exit criteria** (see workflow instructions for thresholds):
- **E1** diversion: `eu_dirty_share` drops >15% rel. for CBAM-exposed regions.
- **E2** mitigation: mean non-EU μ (last 5 steps) > 0.15.
- **E3** RICE sanity: mean μ (all regions) ≈ 0.6–0.7.
- **E4** transfer: `eu_dirty_share[rs=1] > eu_dirty_share[rs=0]` for ≥1 region.

---

## 14. Running experiments end-to-end

There are two ways to run an experiment.

### Option A — direct (quick, flat outputs)

Run the script yourself from `rice_jax/`. Outputs land in the flat `plots/` and
`training_logs/` directories.

```bash
conda activate rice-jax
cd rice_jax
python cbam/drivers/cbam_experiment_A_crowdout.py --timesteps 2000000 --seeds 0,1,2
```

### Option B — managed (`run_cbam_experiment.py`, self-contained folder) ← recommended

[`run_cbam_experiment.py`](../run_cbam_experiment.py) wraps a script in a timestamped,
self-contained experiment folder and can chain the matching post-hoc analysis.

```bash
# train only:
python run_cbam_experiment.py cbam/drivers/cbam_experiment_A_crowdout.py \
    --depth train --timesteps 2000000

# train + post-hoc scorecard:
python run_cbam_experiment.py cbam/drivers/cbam_experiment_A_crowdout.py \
    --depth full --timesteps 2000000 --save-agents

# re-run only the post-hoc on an existing folder:
python run_cbam_experiment.py --posthoc-only cbam/experiment_results/cbam_experiment_A_crowdout_20260515_120000

# resume after Ctrl-C (picks up the latest *_ckpt_*.pkl):
python run_cbam_experiment.py --resume cbam/experiment_results/cbam_experiment_A_crowdout_20260515_120000
```

**Folder layout** created under `cbam/experiment_results/<script_stem>_<YYYYMMDD_HHMMSS>/`:

```
cbam/experiment_results/cbam_experiment_A_crowdout_20260515_120000/
├── config.json      frozen args + metadata (script, seeds, timesteps, num_envs)
├── plots/           summary PNGs + results/checkpoint PKLs
├── logs/            training CSVs (one per condition/seed)
└── posthoc/         scorecard.md + diagnostic PNGs   (depth=full only)
```

**Depth levels** (`--depth`):

| Depth | Does |
|-------|------|
| `train` | Run the script; PKLs + CSVs land in the folder. The script emits its own summary plot. |
| `visualize` | Same as `train` (reserved for a future replot-only pass). |
| `full` | `train` + the matching post-hoc (scorecard, per-region decomposition, convergence overlays). With `--save-agents`, also Layer-2 introspection. |

**Runner flags:** `--depth`, `--num-envs N` (sets `CBAM_NUM_ENVS`, default 8),
`--save-agents` (bundle trained agents into the PKL → enables Layer-2
introspection), `--introspect`, `--posthoc-only <run_dir>`, `--resume <run_dir>`,
`--experiments-dir`. **Any unrecognised flag is forwarded verbatim to the
experiment script** (e.g. `--timesteps`, `--seeds`, experiment-specific knobs);
use `--` to separate runner flags from script flags if they collide.

**How redirection works:** the runner sets two environment variables the script
reads via [`_experiment_util.py`](../_experiment_util.py):
- `CBAM_EXPERIMENT_DIR` → `get_output_dir()` returns `<run_dir>/plots`, `get_log_dir()` returns `<run_dir>/logs`.
- `CBAM_NUM_ENVS` → number of parallel envs.

When unset (direct runs), those helpers fall back to flat `plots/` and `training_logs/`.

### Anatomy of an experiment script

Every headline script follows the same skeleton so `run_cbam_experiment.py`, the
registry, and the post-hocs can all consume it. **Copy an existing script** (e.g.
`cbam_experiment_A_crowdout.py`) rather than starting from scratch.

```python
import matplotlib; matplotlib.use("Agg")           # 1. BEFORE any JAX import
import os as _os, sys as _sys, pickle
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from validation.canonical_config import make_canonical_env, CANONICAL_SEEDS, CANONICAL_TRAIN_KWARGS
from validation.metrics import crowd_out_attenuation, seed_summary
from _experiment_util import get_output_dir, get_log_dir, run_single_episode
from rice_jax.training import RCPOMonitoredPPO, make_csv_log_fn

OUTPUT_DIR = get_output_dir("plots")               # 2. honours CBAM_EXPERIMENT_DIR
LOG_DIR    = get_log_dir("training_logs")

def train_and_eval(cfg, seed):                     # 3. one cell = (condition, seed)
    env = make_canonical_env(**cfg)                #    vary ONLY whitelisted knobs
    ppo = RCPOMonitoredPPO(**CANONICAL_TRAIN_KWARGS,
                           log_fn=make_csv_log_fn(f"{LOG_DIR}/..._{seed}.csv"))
    agent, metrics = ppo.train(key, env)                    #    returns (PPOAgent, metrics)
    raw = make_canonical_env(**cfg, for_training=False)
    info = run_single_episode(eval_key, raw, ppo)  #    trained ppo is the agent
    return {...per-region arrays for ALL regions...}

def main():
    # 4. argparse: --timesteps, --seeds, --replot <pkl>, --resume <ckpt>, + knobs
    if args.replot:                                # 5. skip training, load pkl, replot
        bundle = pickle.load(open(args.replot, "rb")); make_figure(bundle); return
    cells  = [train_and_eval(cfg, s) for cfg in CONDITIONS for s in seeds]
    bundle = {"cells": cells, "env_kwargs": canonical_env_kwargs(), ...}
    pickle.dump(bundle, open(f"{OUTPUT_DIR}/<stem>_{ts}.pkl", "wb"))  # 6. results PKL
    make_figure(bundle)                            # 7. summary PNG → OUTPUT_DIR
```

**Standard argparse flags** (keep these names — the runner and post-hocs rely on them):

| Flag | Purpose |
|------|---------|
| `--timesteps N` | Total training steps per cell |
| `--seeds 0,1,2` / `--seed N` | Seed set (default `CANONICAL_SEEDS`) |
| `--replot <pkl>` | Skip training; regenerate the figure from a saved bundle |
| `--resume <ckpt>` | Skip completed cells; continue from a `*_ckpt_*.pkl` |
| `--save-agents` | Include trained agents in the PKL (enables Layer-2 introspection) |
| `--tests`, `--shares`, `--env-override`, `--free-savings`/`--fixed-savings` | Experiment-specific knobs |

**Rules:**
- Build envs only through `make_canonical_env(**overrides)` — never inline kwargs.
- Store **all `num_regions` values** per metric in the bundle (not pre-excluded),
  so plots can be regenerated with different RoW/EU exclusions without retraining.
- Write periodic `*_ckpt_*.pkl` checkpoints so long runs are `--resume`-able.
- Add an `ExperimentEntry` to [`cbam/config/registry.py`](../cbam/config/registry.py)
  and update [`../notes/CBAM_ROADMAP.md`](../notes/CBAM_ROADMAP.md).

### Results (the PKL bundle)

The results PKL is the experiment's durable artifact. Convention:
- **Naming:** `<script_stem>_<timestamp>.pkl` in `plots/` (or `<run_dir>/plots/`).
- **Structure:** a dict (or list of dicts) keyed by condition (e.g. `revenue_share`,
  `allocation`), each holding per-region metric arrays over **all** regions, plus
  provenance (`canonical_env_kwargs()`, `canonical_train_kwargs()`, seeds).
- **Checkpoints:** `<prefix>_ckpt_<timestamp>.pkl` — partial state for `--resume`.
- Training CSVs (`logs/`) carry the convergence traces (`ep_return_mean`,
  `cbam_lambda`, `mean_cbam_cost`, …).

### Post-hoc analysis

Post-hoc scripts (`cbam_posthoc_*.py`) **read a results PKL and never retrain**.
They emit a report-ready `scorecard.md` plus diagnostic PNGs into `posthoc/`.

- **Layer 1** (no JAX, any 3.11+ env): pass/fail table, per-region decomposition,
  cross-condition ranking, training-curve overlays, markdown scorecard.
- **Layer 2** (needs JAX + `--save-agents` PKLs): policy Jacobian heatmaps,
  first-layer weight norms, and counterfactual obs-perturbation (zero the CBAM
  obs dims and compare actions).

`run_cbam_experiment.py --depth full` auto-routes each experiment to its post-hoc:

| Experiment script | Post-hoc script | Focus |
|-------------------|-----------------|-------|
| `cbam_litmus_mechanism`, `cbam_litmus_conditioning` | `cbam_posthoc_litmus.py` | Litmus scorecard + introspection |
| `cbam_experiment_2b_tier1`, `cbam_experiment_2b_alloc` | `cbam_posthoc_2b.py` | Phase 2B scorecard + ranking |
| `cbam_experiment_A_crowdout` | `cbam_posthoc_A_crowdout.py` | Crowd-out failure-mode diagnosis (H1–H5) |
| `cbam_experiment_C_litmus` | `cbam_posthoc_C_litmus.py` | Multi-seed pass-margin + regional Jacobian |
| `cbam_experiment_C_amplifier` | `cbam_posthoc_C_amplifier.py` | Amplifier failure modes (FD1–FD8) |

Run a post-hoc standalone (PKL-flag names vary: `--pkl` for A/C/amplifier;
`--tier1-pkl`/`--alloc-pkl` for 2B; `--mech-pkl`/`--cond-pkl` for litmus):

```bash
python cbam/posthoc/cbam_posthoc_A_crowdout.py \
    --pkl cbam/experiment_results/cbam_experiment_A_crowdout_*/plots/cbam_A_crowdout_*.pkl \
    --out-dir cbam/experiment_results/cbam_experiment_A_crowdout_*/posthoc
```

---

## 15. Key validation & experiment scripts

Run from `rice_jax/` in the `rice-jax` env. Outputs land in `plots/` and
`training_logs/` (or under `CBAM_EXPERIMENT_DIR` if set).

### Null tests & feature validators (`validate_*.py`)
Each validates one mechanism, ideally with a `jnp.allclose` null condition.

| Script | Validates |
|--------|-----------|
| `validate_emissions_simple.py` | CBAM vs no-CBAM under `emissions-simple` granularity |
| `validate_sectoral_welfloss.py` | Selective dirty-sector diversion from `sectoral_welfloss` |
| `validate_persistence.py` | AR(1) `dest_alloc_persistence` dynamics (perturb-then-relax, no training) |
| `validate_trade_momentum.py` | CBAM × ρ grid effect on EU export share |
| `validate_mitigation_incentive.py` | CBAM→mitigation channel (free vs costly abatement) |
| `validate_rcpo_additive.py` | welfloss vs additive RCPO reward modes |
| `validate_regional_damages.py` | Per-region damage coefficients (B≡A null, C differs) |
| `validate_cbam_gradient.py` | CBAM gradient/incentive signals (V1–V6) |
| `validate_training_conditions.py` | PPO config sweep for mid-episode share artefacts |
| `validate_mrio_clubs.py` | Club scenarios end-to-end + null conditions |

### Litmus & experiment drivers
| Script | Purpose |
|--------|---------|
| `litmus_test.py` | Minimal single-channel litmus (3-region) |
| `cbam_experiment_litmus_diff.py` | 9-region litmus with differential CBAM (L1–L4) |
| `cbam_experiment_9.py` | Full 9-region CBAM experiment (E1–E4) |
| `cbam_experiment_tau_modes.py` | Flat-τ sweep vs differential mode |
| `cbam_experiment_2b_tier1.py` | Phase 2B fixed-fraction `revenue_share` ablation |
| `cbam_experiment_2b_alloc.py` | Phase 2B allocation-rule comparison |
| `cbam_experiment_A_crowdout.py` | Experiment A — crowd-out attenuation (pinned vs open) |
| `cbam_experiment_C_litmus.py` | Multi-seed litmus freeze (M1–M4, C1–C3) |
| `cbam_experiment_C_amplifier.py` | Transfer-pool amplifier sweep |
| `cbam_experiment_2e_clubs.py` | Endogenous CBAM-club design comparison (B1–B3) |
| `cbam_convergence_multiseed.py` | Multi-seed convergence (ctrl vs differential CBAM) |
| `convergence_study.py` | 7-region convergence with `--cbam-rate`, `--plot-only <csv>` |
| `smoke_test_9region.py` | Fast 9-region construction/shape smoke test |

### Tests (`rice_jax/tests/`)
`test_rice_mrio.py` (Phase 1B parity + Phase 2A), `test_mrio_clubs.py` (club scenarios).
Run with `pytest` from `rice_jax/`.

---

## 16. Headline experiment registry

[`cbam/config/registry.py`](../cbam/config/registry.py) is the paper's table of
contents — every headline experiment has an `ExperimentEntry` (question, claim
bucket, primary metric, pass criterion, script, seeds, interpretation limits).
Adding a headline experiment without a registry entry is a process violation.

| ID | Claim bucket | Question (abridged) |
|----|--------------|---------------------|
| `A_crowd_out_redist` | policy-design | Does redistribution reduce mitigation lost to diversion? |
| `B_tariff_ladder` | robustness | Does crowd-out survive across flat τ ∈ {0.05, 0.15, 0.30}? |
| `C_litmus_multiseed` | mechanism | Which litmus conclusions (L1–L4) are seed-robust? |
| `D_allocation_split` | policy-design | Which allocation rule wins on which objective? |
| `E_amplifier` | policy-design | At what transfer multiplier does redistribution suppress diversion? |
| `F_sensitivity_tier1/2` | robustness | Which params matter most; does the claim survive retraining? |

Claim buckets: `mechanism`, `conditioning`, `policy-design`, `robustness`.

---

## 17. Data files

Loaded at `__post_init__` from `mrio_data_root` (`csv_asset/`):

| File | Used for |
|------|----------|
| `mrio/aggregated/eora_agg_{N}/x.txt` | Sector output shares σ_{r,s} |
| `mrio/aggregated/eora_agg_{N}/Z.parquet`, `Y.parquet` | Bilateral trade shares + export fractions |
| `mrio/aggregated/eora_agg_{N}/Q_S.parquet` | CO₂ emissions intensities |
| `CountryClass_{N}.csv` | RICE-index ↔ MRIO-region label mapping |

Region economic yamls (loaded separately via `load_region_yamls`):
`rice_jax/rice_jax/region_yamls/` (native 3/7/20) or `cbam_yamls/setup_*` (CBAM
setups including the 9-region `setup_vuln_9`).

Source: EORA26, 2016, basic prices (Lenzen et al. 2013, *J. Industrial Ecology*).
Aggregated with `csv_asset/aggregate_local_mrio.py`. See
[`../../cbam_yamls/CBAM_DATA_PIPELINE.md`](../../cbam_yamls/CBAM_DATA_PIPELINE.md).

---

## 18. Gotchas & footguns

1. **`eu_region_idx` default `0` is wrong.** Pass it explicitly: 3-region→1,
   7-region→5, 9-region→3. ([§6](#6-region-indexing-the-1-footgun))
2. **`dest_alloc_baseline_decay` must be `0.0` in headline runs.** Non-zero makes
   the 2016 anchor decay to nothing by mid-episode and fakes diversion.
3. **`agent, metrics = ppo.train(...)`** — `train()` returns ``(PPOAgent, metrics)``; keep the agent for eval. Default metrics are per-iteration mean episode returns (learning curve).
4. **`fixed_savings_rate=False` for headline runs** — fixing savings kills the C2
   conditioning result regardless of transition cost.
5. **`replace`, never mutate.** `RiceMRIO` is a frozen pytree; use
   `dataclasses.replace(env, field=val)`.
6. **JAX ↔ matplotlib backend clash (macOS).** Put `import matplotlib;
   matplotlib.use("Agg")` at the **top** of every plotting script *before* any
   JAX import, and lazy-import JAX only inside training code paths — otherwise
   `savefig` produces blank PNGs.
7. **`export_reallocation` is a logit adjustment, not a share.** Read realised
   shares from `state["trade_flows"]`.
8. **Use `make_canonical_env`** for anything headline — inline kwargs drift from
   the frozen setup and are not comparable.
9. **String dispatch is fine** (`if self.transfer_allocation == ...`) because the
   fields are `static=True`; JAX traces a separate branch per value. But no Python
   branching on *dynamic* arrays inside `lax.scan` — use `jnp.where` / `lax.cond`.

---

*Every new mechanism needs a literature citation (or an explicit
`[LITERATURE NEEDED: ...]` tag) and a canonical null condition. See the
implementation instructions and `MRIO_RICE_DESIGN.md` before extending the env.*
