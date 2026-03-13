# Design: `_rice_mrio.py` — Phase 1B MRIO Integration

## Overview

This document describes the approach for `_rice_mrio.py`, a subclass of the JAX
`Rice` environment that introduces **disaggregated sector-level production** based
on EORA26 MRIO data, while **reaggregating back to a scalar per region** before
any downstream economic calculations. The goal is to build and validate the MRIO
scaffolding with zero behavioral change, so the full suite of RICE tests and rollouts
remain identical to the baseline.

This corresponds to **Phase 1B** in `DISAGGREGATED_RICE_DESIGN.md`.

---

## Conceptual Model

### Standard RICE production (unchanged)

At each timestep the original RICE computes a **single aggregate output** per region
via a Cobb-Douglas production function:

$$Y_r = A_r \cdot K_r^{\gamma} \cdot \left(\frac{L_r}{1000}\right)^{1-\gamma}$$

where $A_r$ is total factor productivity, $K_r$ capital, $L_r$ labour.

### Phase 1B addition: distribute then reaggregate

We **interpose** a disaggregation-reaggregation step that is mathematically
transparent:

1. **Compute** $Y_r$ exactly as before (Cobb-Douglas).
2. **Distribute** $Y_r$ across EORA26 sectors using static MRIO output shares:

$$y_{r,s} = \sigma_{r,s} \cdot Y_r, \quad \sigma_{r,s} = \frac{x_{r,s}^{\text{MRIO}}}{\sum_{s'} x_{r,s'}^{\text{MRIO}}}$$

   where $x_{r,s}^{\text{MRIO}}$ is the total sectoral output from the 2016 EORA26 table
   (the latest downloaded year), aggregated to the matching region grouping (3/7/20).

3. **Reaggregate** by summing sectors back to scalar:

$$\hat{Y}_r = \sum_s y_{r,s} = Y_r \quad \text{(exact, by construction of } \sigma \text{)}$$

4. Pass $\hat{Y}_r$ (equivalently $Y_r$) forward to `calc_gross_outputs`. No downstream
   calculation changes.

The new **`production_by_sector`** state key (shape `(N_regions, N_sectors)`) is
stored in state but not read by any downstream method in Phase 1B.

---

## Data

### Source
- EORA26, 2016 (latest year downloaded), basic prices
- Aggregated using `csv_asset/aggregate_local_mrio.py` to match 3/7/20 region groupings
- Output: `csv_asset/mrio/aggregated/eora_agg_{N}/Z.parquet`, `x.txt`, `Q_S.parquet`

### Files used by `_rice_mrio.py`
| File | Content | Used for |
|------|---------|----------|
| `x.txt` | Total output per (region, sector) | Sector output shares $\sigma_{r,s}$ |
| `Q_S.parquet` | Emissions satellite accounts per sector | Sector CO₂ intensities (stored, not yet used) |
| `VA_S.parquet` | Value added per sector | Future TFP anchoring |

### Region matching
EORA regions in the aggregated tables use the `RI` labels from `CountryClass_N.csv`
(e.g. `"Europe & Central Asia"` for the 7-region case). The mapping between RICE's
integer region indices (0..N-1) and MRIO region labels is built at environment
initialization from the same CSV, keyed on `RIG` (the integer group index).

---

## Class Design

```python
class RiceMRIO(Rice):
    """
    Phase 1B: RICE with MRIO-based sectoral disaggregation.

    Production is distributed across EORA26 sectors using static 2016 output
    shares, then reaggregated to a scalar per region. The rest of the environment
    is identical to the base Rice class.
    """

    # Static fields (eqx.field static=True so JAX treats them as compile-time constants)
    sector_output_shares: chex.Array  # shape (N_regions, N_sectors) — static
    sector_names: tuple               # length N_sectors — for logging
    mrio_region_names: tuple          # length N_regions — for logging
    num_sectors: int                  # e.g. 26 for EORA26
```

### Initialization

```python
def __init__(self, mrio_aggregated_dir: str, num_regions: int, **kwargs):
    """
    Parameters
    ----------
    mrio_aggregated_dir : str
        Path to the aggregated MRIO folder, e.g.
        'csv_asset/mrio/aggregated/eora_agg_20'
    """
    # 1. Load x.txt (total output vector)
    x = load_eora_x(mrio_aggregated_dir)  # pd.Series with (region, sector) MultiIndex

    # 2. Pivot to (N_regions, N_sectors) DataFrame
    x_matrix = x.unstack(level='sector')  # rows=regions, cols=sectors

    # 3. Compute row-normalised shares σ_{r,s}
    shares = x_matrix.div(x_matrix.sum(axis=1), axis=0).fillna(0.0)

    # 4. Store ordering so we can assert MRIO region order == RICE region order
    mrio_region_names = shares.index.tolist()   # ordered list of region labels
    sector_names      = shares.columns.tolist() # ordered list of sector names

    # 5. Convert to JAX array (float32)
    sector_output_shares = jnp.array(shares.values, dtype=jnp.float32)

    super().__init__(**kwargs)
    object.__setattr__(self, 'sector_output_shares', sector_output_shares)
    object.__setattr__(self, 'sector_names', tuple(sector_names))
    object.__setattr__(self, 'mrio_region_names', tuple(mrio_region_names))
    object.__setattr__(self, 'num_sectors', len(sector_names))
```

### State additions

`_get_initial_state` is overridden to add:

```python
state["production_by_sector"] = jnp.zeros(
    (self.num_regions, self.num_sectors), dtype=jnp.float32
)
```

Everything else in state is unchanged.

### Production override

Only `calc_productions` is overridden:

```python
def calc_productions(self, state: dict) -> chex.Array:
    # 1. Standard RICE aggregate production (unchanged formula)
    Y = (
        state["production_factor_all_regions"]
        * jnp.power(state["capital_all_regions"], self.region_params.xgamma)
        * jnp.power(state["labor_all_regions"] / 1000, 1 - self.region_params.xgamma)
    )  # shape (N_regions,)

    # 2. Disaggregate: y_{r,s} = σ_{r,s} * Y_r
    #    σ: (N_regions, N_sectors),  Y[:, None]: (N_regions, 1) → broadcast
    production_by_sector = self.sector_output_shares * Y[:, None]
    # shape: (N_regions, N_sectors)

    # 3. Store disaggregated production in state (side-effect via object mutation
    #    is not JAX-friendly; instead return and update in step_climate_and_economy)
    #    ← handled by storing in returned state dict (see below)

    # 4. Reaggregate: Ŷ_r = Σ_s y_{r,s}  ==  Y_r  (by construction)
    Y_check = production_by_sector.sum(axis=1)  # shape (N_regions,)

    return Y_check, production_by_sector
```

`step_climate_and_economy` is overridden minimally to:
1. Unpack the tuple returned by `calc_productions`
2. Add `production_by_sector` to the state update dict
3. Otherwise call `super().step_climate_and_economy()`

```python
def step_climate_and_economy(self, state, actions):
    # Intercept calc_productions, inject production_by_sector into state
    state = super().step_climate_and_economy(state, actions)
    # ← or override the single method and call parent for everything else
    return state
```

> **Implementation note**: Because `calc_productions` is called inside
> `step_climate_and_economy`, the cleanest approach is to override
> `step_climate_and_economy` in `RiceMRIO`, call the parent method as normal,
> and then *also* compute and store `production_by_sector` using the same
> `sector_output_shares` and the already-updated `production_all_regions`
> from state. This avoids touching the parent's internal call chain.

---

## Override Strategy (minimal diff from base class)

```
Rice (unchanged)
 └── RiceMRIO
       ├── __init__            — load MRIO shares, call super().__init__
       ├── _get_initial_state  — add production_by_sector key, call super()
       ├── calc_productions    — SAME formula; additionally compute sector breakdown
       └── step_climate_and_economy
                               — call super(); append production_by_sector to state
```

All other methods (`calc_damages`, `calc_gross_outputs`, `calc_investments`,
`calc_gross_imports`, `calc_trade_sanctions`, `calc_utilities`, etc.) are
**inherited unchanged**.

---

## JAX Compatibility

| Concern | Solution |
|---------|----------|
| `sector_output_shares` must be static | `eqx.field(static=True)` on the array |
| No Python control flow in `calc_productions` | Pure JAX ops: `*`, `[:, None]`, `.sum(axis=1)` |
| `state` is a plain dict (not a pytree class) | `production_by_sector` added as new key in the returned state dict |
| `jit`/`vmap` compatibility | No data-dependent branching; safe |

---

## Validation Plan

To confirm Phase 1B introduces **zero numerical change**:

1. **Scalar equality**: run the same seed rollout for N steps with `Rice` and
   `RiceMRIO`; assert `jnp.allclose(production_all_regions_rice, production_all_regions_mrio)` at every timestep.

2. **Sector sum check**: assert `production_by_sector.sum(axis=1) ≈ production_all_regions` at every timestep.

3. **Share sanity**: assert `sector_output_shares.sum(axis=1) ≈ 1.0` for all regions.

4. **Downstream parity**: assert `gross_output`, `consumption`, `utility`, `global_temperature` are identical between `Rice` and `RiceMRIO`.

These assertions go in a dedicated `test_rice_mrio.py` unit test file.

---

## File Layout

```
rice_jax/rice_jax/
├── _rice.py              (unchanged)
├── _rice_mrio.py         (new: Phase 1B)
└── MRIO_RICE_DESIGN.md   (this file)

csv_asset/mrio/aggregated/
├── eora_agg_3/
├── eora_agg_7/
└── eora_agg_20/
```

---

## Known Limitation: Growth Distribution and Static Sector Shares

### The problem

In Phase 1B, sector output shares $\sigma_{r,s}$ are **static** — fixed at 2016
MRIO values for the entire simulation. This means:

- Capital accumulation from savings is aggregated into $K_r$ and drives growth
  in $Y_r$ via Cobb-Douglas, **but that growth is then split across sectors in
  the same 2016 proportions forever**.
- The savings rate allocates to aggregate capital; there is no concept of saving
  into a specific sector.
- A region that invests heavily in clean energy (for example) would not see its
  energy sector share grow relative to heavy industry.

This is **acceptable for Phase 1B** because the reaggregation is exact and the
purpose is solely to validate the disaggregation scaffolding. The economic
behaviour is bit-identical to base `Rice`.

### Phase 2 resolution: dynamic sector shares via MRIO capital coefficients

When disaggregated trade is introduced (Phase 2), sector shares should become
**dynamic** using the following approach:

1. **Sector-specific capital stocks** $K_{r,s}$, initialised from MRIO value-added
   shares (row from `VA_S.parquet`):

   $$K_{r,s}^{(0)} = \phi_{r,s} \cdot K_r^{(0)}, \quad
     \phi_{r,s} = \frac{\text{VA}_{r,s}}{\sum_{s'} \text{VA}_{r,s'}}$$

2. **Savings allocation rule** — one of three options (ordered by agent control):

   | Option | Mechanism | Agent control |
   |--------|-----------|---------------|
   | A. Fixed VA shares | Invest $s \cdot Y_r$ into sectors proportional to $\phi_{r,s}$ (static) | None |
   | B. Dynamic Leontief | Use MRIO A-matrix to determine required capital per sector given production targets | None |
   | C. MARL action | Agent outputs a sector-allocation vector $\alpha_{r,s}$ alongside savings rate | Full |

   Option A is the natural Phase 2 starting point; Option C is reserved for
   Phase 3+ when sector-level policy levers are introduced.

3. **Sector Cobb-Douglas**: each sector has its own capital and labour share,
   producing a sector output $y_{r,s}$ that no longer needs to sum to the
   prior-step aggregate:

   $$y_{r,s} = A_{r,s} \cdot K_{r,s}^{\gamma_s} \cdot L_{r,s}^{1-\gamma_s}$$

   Labour allocation across sectors follows MRIO employment shares (also from
   value-added table).

4. The **reaggregation collapses** entirely in Phase 2+: $Y_r = \sum_s y_{r,s}$
   now differs from the base RICE value and becomes the emergent aggregate output
   driven by sectoral dynamics.

### Summary of phase progression for production

| Phase | Sector shares | Capital | Savings allocation |
|-------|--------------|---------|--------------------|
| Base Rice | N/A (aggregate) | Aggregate $K_r$ | Aggregate savings rate |
| 1B (this file) | Static 2016 MRIO $\sigma_{r,s}$ | Aggregate $K_r$ | Aggregate — split by $\sigma$ post-hoc |
| 2 | Dynamic via VA shares $\phi_{r,s}$ | Sector $K_{r,s}$ | Invest by fixed sector allocation |
| 3+ | Dynamic, agent-controlled | Sector $K_{r,s}$ | MARL allocation vector $\alpha_{r,s}$ |

---

## What this does NOT do (reserved for later phases)

- Does **not** change the Cobb-Douglas production function
- Does **not** use MRIO technical coefficients (A matrix / Leontief) for production
- Does **not** disaggregate trade, investments, or consumption
- Does **not** alter action space or observations
- Does **not** use MRIO carbon intensities to replace RICE carbon intensities
- Does **not** compute embodied emissions in trade
- Does **not** allocate savings/capital at the sector level (static shares only)

All of the above are Phase 2+ work, building on Phase 1B.

---

## Research Motivation: CBAM, Consumption Leakage, and Strategic Trade Diversion

This section documents the research agenda that Phase 2+ is designed to serve. Understanding the *why* informs every design decision about what to model endogenously, what to calibrate from MRIO, and what to leave as a heuristic.

### The core research question

The EU's Carbon Border Adjustment Mechanism (CBAM) taxes the embedded carbon in
imports of covered goods (steel, cement, aluminium, fertilisers, electricity,
hydrogen). The stated goal is to prevent *carbon leakage* — the relocation of
production to less-regulated jurisdictions. However, there is a second, less
studied leakage channel: **consumption leakage**.

An emission-heavy exporter facing CBAM does not need to move its factories. It
can simply redirect its exports to non-EU markets. Global emissions are
unchanged; the goods are now consumed somewhere without a carbon price. The EU
achieves its own emission accounting target while the atmosphere does not benefit.

**The central strategic question**: given that trade diversion is an option, what
combination of CBAM tariff structure, technology transfer, cash transfers, and
market access guarantees is required to make decarbonization the dominant strategy
for a given exporting country? And which countries are most vulnerable to being
*de facto* exempted from the CBAM pressure via this channel?

---

### The strategic choice being modelled

An emission-heavy exporter facing CBAM faces (at minimum) three responses:

| Option | Mechanism | Effect on global emissions |
|--------|-----------|---------------------------|
| **Decarbonize** | Invest in abatement or clean technology | Reduces emissions |
| **Redirect exports** | Shift sales from EU to non-EU markets | No change |
| **Mix** | Partial decarbonization + partial diversion | Partial reduction |

The key insight is that options 2 and 3 are **strategically rational** if (a) the
EU export share is not dominant, or (b) adjustment costs of trade diversion are
low. This distinguishes consumption leakage from production leakage: the factory
stays; only the invoice changes destination.

**Conceptually, why this matters**: existing IAMs and CGE models typically focus
on production-side responses (does output shift to China?). The consumption-side
response is often assumed away by the model structure. An explicit export portfolio
action for agents makes this channel visible and quantifiable.

---

### Why MRIO is the right empirical grounding

MRIO tables give us, directly and without further modelling, three things needed
for this research:

1. **Bilateral export shares by sector**: what fraction of region $r$'s output in
   sector $s$ is consumed in each destination market. Countries with high EU
   export concentration in CBAM-covered sectors face a real CBAM constraint.
   Countries already exporting primarily to China, the US, or within regional
   blocs have an easy diversion exit.

2. **Emissions intensity per sector-region** (from satellite accounts `Q_S`):
   the embedded carbon at stake — i.e. how much CO₂ are we actually trying to
   influence per unit of trade redirected.

3. **Alternative market depth**: from the final demand columns of the MRIO table,
   we can read off the size of non-EU absorbers for each sector. A large
   alternative market means diversion is cheap. A thin one means diversion has
   adjustment costs because prices drop when more supply enters.

This lets us build a **vulnerability map** — entirely from data, before any
model is run — that identifies which countries are most exposed to CBAM pressure
and which have credible diversion threats. This empirical step is a standalone
contribution independent of the MARL model.

**Technically required**: extract from the aggregated MRIO tables
(`Z.parquet`, `Q_S.parquet`, final demand columns) the following matrices at
initialisation:

```python
# All shape (num_regions, num_regions, num_sectors) or (num_regions, num_sectors)
eu_export_share_by_sector   # fraction of r's output in s going to EU
alt_market_absorption       # non-EU demand capacity per sector (from final demand)
emissions_intensity         # tCO2 per unit output, shape (num_regions, num_sectors)
bilateral_trade_shares      # T_{r→r', s} / x_{r,s}: baseline MRIO trade shares
```

These are **static calibration arrays** loaded at environment init, analogous to
`sector_output_shares` in Phase 1B. They do not need to be differentiable
themselves — they are the data against which the model is calibrated.

---

### The MARL game structure

The research question maps cleanly to the following multi-agent game:

#### Agents

| Agent | Identity | Purpose |
|-------|----------|---------|
| **EU** | Single agent | Sets CBAM rate per sector; decides how to recycle CBAM revenue |
| **Emission-heavy exporters** | One per region | Chooses: carbon tax rate, mitigation investment, export portfolio reallocation |
| **Rest of World** | One agent (or reduced-form) | Absorbs redirected exports; sets a shadow price that rises with excess supply |

Making "rest of world" agentic — rather than a passive absorber — makes the
diversion more realistic: as emission-heavy exporters flood non-EU markets, the
RoW agent responds with its own pricing or access controls, endogenously
increasing the adjustment cost of diversion over time.

#### Agent observations (what agents *see*)

Agents should observe only aggregates visible to real policy-makers:
- Own emissions, output, welfare
- Bilateral trade balances (not full flow matrix)
- Carbon price differential with trading partners
- CBAM tariff rates they face
- Transfer payments received

**Conceptually, why limit observations**: if agents observe the full sectoral
trade flow tensor, the observation space explodes and the policy is not
interpretable. Limiting to aggregate signals forces agents to learn policies that
are implementable by real governments.

#### Agent actions (the critical design split)

The key design decision is to separate what agents control from what is
handled by calibrated heuristics:

**MARL-controlled (policy decisions):**
- `carbon_tax_rate`: scalar or per-sector carbon price (the domestic decarbonization lever)
- `mitigation_rate`: investment in abatement (reduces emissions intensity over time)
- `export_reallocation`: fraction of CBAM-covered sectoral output redirected to non-EU markets — **this is the novel action that makes consumption leakage visible**
- `cbam_rate` (EU only): tariff per sector per trading partner
- `revenue_transfer` (EU only): fraction of CBAM revenue recycled as climate finance to exporting regions

**Heuristic/calibrated (not agent-controlled):**
- Bilateral trade flow routing within the non-EU market (gravity model or fixed MRIO shares)
- Input-output requirements (domestic intermediate demand follows Leontief from MRIO)
- Labour and capital allocation across sectors (MRIO VA shares, Option A from Phase 2 above)

**Conceptually, why this split**: real governments do not micromanage which
specific factory ships to which specific buyer. They set prices and taxes; the
trade routing follows. Modelling routing as a heuristic is realistic and
preserves a tractable action space. The `export_reallocation` scalar (or low-dim
vector over CBAM sectors) is the minimal action that makes the strategic choice
visible without requiring agents to solve a logistics problem.

---

### Why a full CGE is not required

A Computable General Equilibrium model would give exact re-equilibration of all
markets after a policy shock — but it requires solving a fixed-point system
inside every simulation step, which is computationally brutal and difficult to
differentiate through.

This research does **not** require full CGE because:

1. **Timescale mismatch**: RICE operates on 5-year timesteps. Trade structure is
   sticky on those timescales. The MRIO matrix evolves slowly along an SSP-consistent
   trend; within-episode deviations from that trend are the policy signal.

2. **Reduced-form leakage elasticities suffice**: for the consumption leakage
   channel, what matters is how much production shifts destination, not the
   exact price at which all markets clear. A leakage elasticity
   $\epsilon_{r,s}$ (change in EU-directed export share per unit of CBAM cost)
   can be calibrated from MRIO cross-sections and embedded as a static
   parameter.

3. **Differentiable calibration replaces structural solving**: the JAX model is
   used for gradient-based calibration of the response parameters against
   observed MRIO data. This gives the empirical grounding of a CGE without the
   runtime overhead.

4. **The policy question is marginal, not structural**: we are asking "what
   transfer level tips the incentive?", not "what are the new equilibrium wages
   in sector $s$?". The former requires only that welfare responses are
   directionally correct, not exact.

---

### The CBAM revenue recycling mechanism

CBAM is projected to generate significant EU revenue. There is active policy
debate about using it for climate finance in developing countries. In the model,
this becomes the EU agent's `revenue_transfer` action.

**Conceptually**: the transfer changes the payoff structure of the exporting
agent's decarbonization choice. Without transfers, a high-abatement-cost region
always prefers diversion. With sufficient transfers (or technology provision that
reduces abatement cost), decarbonization becomes dominant. The model reveals:

- The **minimum transfer threshold** per region that flips the strategic equilibrium
- Whether **technology transfer** (reducing $\epsilon_{r,s}$, the abatement cost) is
  more efficient than cash (same effect at lower cost when capital access is the
  binding constraint)
- How the **diversion threat** interacts with transfer design — a credible diversion
  option gives exporting regions bargaining power, which should increase optimal
  transfer levels

**Technically required**: add a scalar `revenue_transfer` action to the EU agent,
mapped to a transfer payment into the welfare/consumption of target regions. Wire
it to CBAM revenue so the EU faces a budget constraint: total transfers ≤ collected
CBAM tariff revenue. This is a straightforward addition to the reward function and
state update.

---

### Differentiability as a research tool

The JAX model earns its keep not just for MARL training but for **gradient-based
policy analysis**. Once the model is calibrated, we can:

1. **Gradient-optimize the transfer/tariff schedule** directly using `jax.grad`,
   rather than grid-searching over scenario combinations.

2. **Calibrate leakage elasticities** by differentiating the model's CBAM revenue
   predictions against observed MRIO trade flows.

3. **Sensitivity analysis**: compute $\partial \text{welfare}_r / \partial
   \text{cbam\_rate}_s$ across all regions simultaneously via `jax.jacfwd`, to
   map who wins and loses from each tariff design choice.

This is the unique contribution of building on the JAX RICE-N codebase rather
than a static CGE.

---

### Phase roadmap summary from the research perspective

| Phase | What it enables |
|-------|----------------|
| **1B** (current) | MRIO sectoral structure loaded; `production_by_sector` state exists; bit-identical to base RICE |
| **2A** | Activate `trade_flows` tensor; add `export_reallocation` agent action; wire CBAM tariff to embedded carbon from `Q_S` |
| **2B** | Add `revenue_transfer` EU action; budget-constrain to CBAM revenue; measure decarbonisation vs. diversion equilibria |
| **2C** | Dynamic sector shares (VA-weighted capital); leakage elasticities calibrated from MRIO cross-section |
| **3** | Rest-of-World agent; technology transfer action; gradient-optimise transfer schedules; vulnerability mapping from MRIO |
