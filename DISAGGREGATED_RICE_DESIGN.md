# Extending RICE with Disaggregated Production and Trade (CBAM Expansion)

This document outlines the approach to create a new base class for the RICE model that incorporates disaggregated production and trade of heterogeneous goods, based on Multi-Regional Input-Output (MRIO) data. The goal is to model the expansion of the Carbon Border Adjustment Mechanism (CBAM) while maintaining computational efficiency in JAX.

## Overview

The current JAX RICE implementation uses aggregated trade where regions bid for imports and set tariffs on total imports from other regions. The new implementation will:

1. **Disaggregate production**: Each region produces a vector of heterogeneous goods instead of a single aggregate output
2. **Disaggregate trade**: Trade occurs for specific goods between regions, with separate tariffs and quotas per good
3. **MRIO integration**: Use Multi-Regional Input-Output tables to determine production requirements and trade patterns
4. **CBAM modeling**: Incorporate carbon pricing on traded goods based on their carbon intensity
5. **Efficient allocation**: Use heuristics or trained models to allocate trade efficiently in JAX

## Key Design Decisions

### 1. State Space Modifications

#### Current State (Aggregated)
```python
state = {
    "gross_output_all_regions": jnp.zeros(self.num_regions),  # Single scalar per region
    "import_bids_all_regions": jnp.zeros((self.num_regions, self.num_regions)),  # Region-to-region
    "import_tariffs_all_regions": jnp.zeros((self.num_regions, self.num_regions)),  # Region-to-region
    # ... other states
}
```

#### New State (Disaggregated)
```python
state = {
    "production_by_sector": jnp.zeros((self.num_regions, self.num_sectors)),  # Sectors × Regions
    "trade_flows": jnp.zeros((self.num_regions, self.num_regions, self.num_sectors)),  # Region × Region × Sector
    "sector_tariffs": jnp.zeros((self.num_regions, self.num_regions, self.num_sectors)),  # Region × Region × Sector
    "cbam_prices": jnp.zeros((self.num_regions, self.num_sectors)),  # Carbon prices per sector per importing region
    # ... other states
}
```

### 2. Action Space Modifications

#### Current Actions (Aggregated)
- `import_bid`: Vector of bids to each region (N_REGIONS)
- `import_tariff`: Vector of tariffs on imports from each region (N_REGIONS)
- `export_limit`: Scalar limit on total exports

#### New Actions (Disaggregated)
To avoid exploding the action space, we selectively control only key actions via MARL:

**MARL-Controlled Actions:**
- `mitigation_rate`: Sector-specific mitigation rates (N_SECTORS)
- `savings_rate`: Regional savings rate
- `cbam_intensity`: CBAM enforcement level per sector (scalar or low-dim)

**Heuristic/Automated Actions:**
- `sector_trade_allocations`: Determined by MRIO-trained model or heuristic
- `sector_tariffs`: Set based on CBAM policy rules
- `export_limits`: Set per sector based on production capacity

### 3. MRIO Data Integration

#### Data Structure
```python
class MRIOData:
    def __init__(self, mrio_tables):
        self.num_sectors = mrio_tables.shape[0] // num_regions
        self.intermediate_inputs = mrio_tables  # (N_REGIONS * N_SECTORS) × (N_REGIONS * N_SECTORS)
        self.value_added = value_added_vector  # (N_REGIONS * N_SECTORS)
        self.final_demand = final_demand_matrix  # (N_REGIONS * N_SECTORS) × N_REGIONS
        self.carbon_intensities = carbon_intensity_vector  # (N_REGIONS * N_SECTORS)
```

#### Production Function
Instead of Cobb-Douglas with aggregate capital/labor:
```python
def calc_sector_productions(self, state, actions):
    # Leontief-style production with intermediate inputs
    required_inputs = self.mrio_data.intermediate_inputs @ state["production_by_sector"].flatten()
    required_inputs = required_inputs.reshape((self.num_regions, self.num_sectors))

    # Add value-added requirements (capital, labor with climate damages)
    value_added_required = self.calc_value_added_requirements(state, actions)

    # Production constrained by inputs and value-added
    productions = jnp.minimum(
        required_inputs / self.input_coefficients,
        value_added_required / self.value_added_coefficients
    )

    return productions
```

## Implementation Strategy

### 1. Base Class Structure

```python
class DisaggregatedRice(Rice):
    """RICE with disaggregated production and trade based on MRIO data."""

    num_sectors: int = 10  # Number of sectors from MRIO
    mrio_data: MRIOData = None

    # CBAM parameters
    cbam_enabled: bool = True
    cbam_carbon_price: float = 50.0  # $/tCO2

    def __init__(self, mrio_data_path: str, **kwargs):
        super().__init__(**kwargs)
        self.mrio_data = self.load_mrio_data(mrio_data_path)
        self.num_sectors = self.mrio_data.num_sectors
```

### 2. Trade Allocation Strategies

#### Option A: Heuristic Allocation
```python
def allocate_trade_heuristic(self, state, actions):
    """Allocate trade using MRIO-based heuristics."""
    # Use historical trade shares from MRIO
    base_trade_shares = self.mrio_data.trade_shares  # (region × region × sector)

    # Adjust for economic conditions
    economic_adjustment = self.calc_economic_adjustment(state)

    # Apply CBAM adjustments
    cbam_adjustment = self.calc_cbam_adjustment(state, actions)

    trade_allocations = base_trade_shares * economic_adjustment * cbam_adjustment

    return trade_allocations
```

#### Option B: Trained ML Model
```python
class TradeAllocationModel(eqx.Module):
    layers: list

    def __call__(self, economic_indicators, cbam_signals):
        # Neural network that predicts trade allocations
        x = jnp.concatenate([economic_indicators, cbam_signals])
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return x  # Trade allocation matrix

def allocate_trade_ml(self, state, actions):
    """Allocate trade using trained ML model."""
    economic_indicators = self.extract_economic_indicators(state)
    cbam_signals = self.extract_cbam_signals(actions)

    trade_allocations = self.trade_model(economic_indicators, cbam_signals)

    return trade_allocations
```

### 3. Comprehensive Approaches to Disaggregated Trade

#### **Approach 1: Fixed Proportional Shares (Baseline)**
**Mechanism**: Use historical MRIO trade shares; adjust by scaling factor from agent actions
- **Pros**: Simple, fast, computationally efficient
- **Cons**: Rigid, doesn't capture dynamic shifts in comparative advantage
- **Implementation**:
  ```python
  def allocate_trade_fixed_shares(self, state, actions):
      # Historical trade share matrix (region × region × sector)
      base_shares = self.mrio_data.historical_trade_shares
      
      # Scale by regional import_bid / export_limit actions (aggregated to single value per region)
      import_demand = actions["import_bid"].mean()  # Average across sectors
      export_capacity = actions["export_limit"].mean()
      
      # Apply scale factors
      trade_flows = base_shares * import_demand[None, :, :] * export_capacity[:, None, :]
      
      return trade_flows
  ```
- **Use case**: Phase 2 baseline to validate disaggregation works correctly

#### **Approach 2: Gravity Model**
**Mechanism**: Trade flows determined by GDP, distance, and policy factors (gravity equation)
- **Formula**: $T_{ij,s} = \frac{Y_i \cdot E_j \cdot A_{ij,s}}{GDP_{world}} \cdot e^{-\beta \cdot tariff_{ij,s}}$
- **Pros**: Economically grounded, well-established in trade literature
- **Cons**: Requires calibration, less responsive to CBAM policies
- **Implementation**:
  ```python
  def allocate_trade_gravity(self, state, actions):
      # Extract economic scales
      production = state["production_by_sector"]  # (region × sector)
      
      # Gravity matrix for each sector
      exporter_scale = production.sum(axis=1)  # Exporter size
      importer_scale = production.sum(axis=1)[:, None]  # Importer size
      
      # Distance/preference matrix from MRIO
      bilateral = self.mrio_data.bilateral_preference  # (region × region × sector)
      
      # Tariff resistance
      tariffs = self.calc_sector_tariffs(state, actions)
      tariff_resistance = jnp.exp(-0.5 * tariffs)
      
      # Gravity equation
      trade_potential = exporter_scale[None, :, None] * importer_scale[:, None, :] * bilateral
      trade_flows = trade_potential * tariff_resistance
      
      return trade_flows
  ```
- **Use case**: Phase 3 to add policy responsiveness

#### **Approach 3: Supply-Demand Equilibration (Auction-based)**
**Mechanism**: Iterative equilibration similar to current RICE, but sector-by-sector
- **Agent control**: Import bids and tariffs per sector (or aggregated)
- **Process**:
  1. Each region declares import demand per sector
  2. Each region sets tariffs per sector-pair
  3. Excess demand/supply drives virtual price adjustments
  4. Converge to equilibrium in N iterations
- **Pros**: Captures supply constraints, agent control on trade volume/prices, economically realistic
- **Cons**: More computation, requires convergence check, potential instability
- **Implementation**:
  ```python
  def allocate_trade_equilibration(self, state, actions, num_iterations=5):
      # Initialize virtual prices at 1.0
      virtual_prices = jnp.ones((self.num_regions, self.num_sectors))
      
      for iteration in range(num_iterations):
          # Calculate supply at current prices (with tariff resistance)
          tariffs = self.calc_sector_tariffs(state, actions)
          tariff_resistance = 1.0 - tariffs  # Price adjustment
          
          # Export supply: production minus domestic consumption
          domestic_consumption = state["domestic_final_demand"]  # (region × sector)
          exportable = state["production_by_sector"] - domestic_consumption
          export_supply = jnp.maximum(0, exportable[:, None, :]) * virtual_prices[None, :, :]
          
          # Import demand: import bids adjusted by prices
          import_demand = actions["import_bid"][:, :, None] / virtual_prices[:, None, :]
          
          # Trade flows: minimum of supply and demand
          trade_flows_iter = jnp.minimum(export_supply, import_demand)
          
          # Calculate excess demand
          total_supply = trade_flows_iter.sum(axis=0)
          total_demand = import_demand.sum(axis=1)
          excess_demand = total_demand - total_supply
          
          # Update prices based on excess demand (Walrasian tâtonnement)
          adjustment_factor = 1.0 + 0.1 * excess_demand / (total_supply + 1e-8)
          virtual_prices = virtual_prices * adjustment_factor
      
      return trade_flows_iter
  ```
- **Use case**: Phase 2/3 for realistic trade equilibrium

#### **Approach 4: Input-Output Matching (Supply Chain)**
**Mechanism**: Allocate based on intermediate input requirements from MRIO table
- **Concept**: Some sectors need specific inputs from other sectors; trade is determined by input-output requirements
- **Pros**: Structurally consistent with MRIO, captures sectoral linkages
- **Cons**: Less flexible for agent control, requires careful mapping
- **Implementation**:
  ```python
  def allocate_trade_io_matching(self, state, actions):
      # MRIO intermediate input coefficients
      input_coeff = self.mrio_data.intermediate_input_coeff  # (N_REG×N_SEC × N_REG×N_SEC)
      
      # Required inputs for each region-sector
      production = state["production_by_sector"].flatten()
      required_inputs = input_coeff @ production  # (N_REG×N_SEC)
      required_inputs = required_inputs.reshape((self.num_regions, self.num_sectors))
      
      # Domestic production available
      domestic_available = state["production_by_sector"]
      
      # Trade needed: deficit in each region-sector
      trade_deficit = jnp.maximum(0, required_inputs - domestic_available)
      
      # Allocate trade from surplus regions
      surplus = jnp.maximum(0, domestic_available - required_inputs)
      
      # Normalize and allocate
      total_deficit_per_sector = trade_deficit.sum(axis=0) + 1e-8
      allocation_share = jnp.where(
          total_deficit_per_sector > 0,
          trade_deficit / total_deficit_per_sector,
          0
      )
      
      trade_flows = allocation_share * surplus[:, None, :]
      
      return trade_flows
  ```
- **Use case**: Phase 1B validation, realistic sector interdependencies

#### **Approach 5: Learned Trade Model (Neural Network)**
**Mechanism**: Train NN to predict sector-level trade allocations from state and actions
- **Input features**: Production levels, tariffs, CBAM prices, historical flows, economic indicators
- **Output**: Trade flow matrix (region × region × sector) or allocation weights
- **Pros**: Maximum flexibility, can capture complex patterns, reactive to all policy changes
- **Cons**: Requires offline training data, potential for out-of-distribution issues
- **Architecture**:
  ```python
  class SectorTradeAllocationModel(eqx.Module):
      embedding_layer: eqx.nn.Linear
      hidden_layers: list  # Multiple transformer or attention layers
      output_layer: eqx.nn.Linear
      
      def __call__(self, state_features, policy_actions, sector_id):
          x = jnp.concatenate([state_features, policy_actions])
          x = jax.nn.relu(self.embedding_layer(x))
          
          # Process through transformer or attention mechanism
          for layer in self.hidden_layers:
              x = layer(x)
          
          # Output trade allocation for this sector
          allocation = jax.nn.softmax(self.output_layer(x))
          
          return allocation
  ```
- **Training approach**:
  ```python
  def train_trade_model_offline(self, trajectories, epochs=1000):
      # trajectories: list of (state, actions, resulting_trade_flows)
      
      for epoch in range(epochs):
          for state, actions, true_flows in trajectories:
              features = self.extract_features(state, actions)
              predicted_flows = self.trade_model(features)
              
              # Loss: MSE between predicted and actual flows
              loss = jnp.mean((predicted_flows - true_flows) ** 2)
              
              # Gradient descent update
              gradients = jax.grad(loss)(self.trade_model.parameters)
              self.trade_model.parameters = optax.apply(gradients)
  ```
- **Use case**: Phase 4/5, sophisticated policy exploration

#### **Approach 6: Hybrid Heuristic + Agent Control**
**Mechanism**: Base allocation on heuristics, override with agent sector-specific bids/tariffs
- **Structure**: 
  1. Use gravity or IO-matching for baseline allocation
  2. Agents can adjust sector-specific import demand / export limits
  3. Agents set sector-level tariffs (for CBAM)
  4. Flows adjusted by agent actions via scaling/reweighting
- **Pros**: Balances realism with agent control, computationally efficient
- **Cons**: Requires careful design of agent control levers
- **Implementation**:
  ```python
  def allocate_trade_hybrid(self, state, actions):
      # Baseline from gravity
      base_flows = self.allocate_trade_gravity(state, actions)
      
      # Allow agents to adjust specific sector pairs
      if "sector_import_bid" in actions:  # (region × sector)
          # Adjust import flows per sector
          sector_scaling = actions["sector_import_bid"][:, None, :]  # (region × 1 × sector)
          base_flows = base_flows * sector_scaling
      
      if "sector_export_limit" in actions:  # (region × sector)
          # Constrain export per sector
          export_cap = actions["sector_export_limit"][:, None, :]  # (region × 1 × sector)
          base_flows = jnp.where(
              base_flows > export_cap,
              export_cap,
              base_flows
          )
      
      return base_flows
  ```
- **Use case**: Phase 3, balance agent control with realism

### 4. Comparison Matrix

| Approach | Complexity | Speed | Agent Control | Realism | Scalability |
|----------|-----------|-------|---------------|---------|-------------|
| Fixed Shares | Low | Very High | Low | Low | High |
| Gravity Model | Medium | High | Medium | High | High |
| Equilibration | High | Medium | High | Very High | Medium |
| IO Matching | Medium | High | Low | Very High | Medium |
| Neural Network | Very High | Medium | Medium | Very High | High |
| Hybrid | Medium | High | High | High | High |

### Recommended Progression

1. **Phase 1B**: Fixed Shares (baseline sanity check)
2. **Phase 2**: IO Matching (structurally consistent)
3. **Phase 3**: Gravity Model (add policy responsiveness)
4. **Phase 3+**: Add sector-level agent controls → Hybrid approach
5. **Phase 4**: Collect data, train Neural Network as alternative
6. **Phase 5**: Compare NN vs Hybrid for policy analysis

### 4. Action Space Design

```python
@property
def action_space(self) -> dict[str, jym.Space]:
    N_REGIONS = self.num_regions
    N_SECTORS = self.num_sectors
    N_DISCRETIZATION = self.num_discrete_action_levels

    actions = {
        # MARL-controlled: sector-specific mitigation
        "mitigation_rate": MultiDiscrete([N_DISCRETIZATION] * N_SECTORS),
        "savings_rate": Discrete(N_DISCRETIZATION),
        "cbam_enforcement": Discrete(N_DISCRETIZATION),  # Overall CBAM level
    }

    return {i_to_agent_str(i): actions for i in range(N_REGIONS)}
```

### 4. State Initialization

```python
def _get_initial_state(self, key):
    state = super()._get_initial_state(key)

    # Add disaggregated states
    state.update({
        "production_by_sector": self.mrio_data.initial_production.reshape((self.num_regions, self.num_sectors)),
        "trade_flows": jnp.zeros((self.num_regions, self.num_regions, self.num_sectors)),
        "sector_tariffs": jnp.zeros((self.num_regions, self.num_regions, self.num_sectors)),
        "cbam_prices": jnp.full((self.num_regions, self.num_sectors), self.cbam_carbon_price),
        "carbon_embodied_in_trade": jnp.zeros((self.num_regions, self.num_regions, self.num_sectors)),
    })

    return state
```

### 5. Step Function Modifications

```python
def step_climate_and_economy(self, state, actions):
    # Calculate sector productions
    productions = self.calc_sector_productions(state, actions)

    # Allocate trade (heuristic or ML-based)
    trade_flows = self.allocate_trade(state, actions)

    # Calculate CBAM costs
    cbam_costs = self.calc_cbam_costs(trade_flows, state)

    # Calculate effective imports/exports per sector
    net_imports = self.calc_net_sector_imports(trade_flows, state)

    # Update economy with disaggregated flows
    gross_outputs = self.calc_gross_outputs_from_sectors(productions, net_imports)

    # Continue with standard RICE economy calculations
    # ... (investments, consumption, etc.)

    return state
```

## CBAM Implementation

### Carbon Accounting
```python
def calc_cbam_costs(self, trade_flows, state):
    """Calculate CBAM costs for imported goods."""
    # Carbon intensity of each sector in exporting region
    export_carbon_intensity = state["carbon_intensities"].reshape((self.num_regions, self.num_sectors))

    # Embodied carbon in trade flows
    embodied_carbon = trade_flows * export_carbon_intensity[:, None, :]

    # CBAM price per sector in importing region
    cbam_prices = state["cbam_prices"]

    # Total CBAM costs
    cbam_costs = embodied_carbon * cbam_prices[None, :, :]

    return cbam_costs.sum(axis=(0, 2))  # Per importing region
```

### Tariff Structure
```python
def calc_sector_tariffs(self, state, actions):
    """Set tariffs based on CBAM and trade policy."""
    base_tariffs = jnp.zeros((self.num_regions, self.num_regions, self.num_sectors))

    if self.cbam_enabled:
        # CBAM tariffs based on carbon price differential
        carbon_price_diff = state["cbam_prices"][:, None, :] - state["domestic_carbon_prices"][None, :, :]
        cbam_tariffs = jnp.maximum(0, carbon_price_diff) * self.cbam_enforcement_level

        base_tariffs += cbam_tariffs

    return base_tariffs
```

## Computational Efficiency Considerations

### 1. Vectorization
- Use JAX's `vmap` for sector-level operations
- Pre-compute MRIO matrices as JAX arrays
- Avoid Python loops in favor of array operations

### 2. Trade Allocation Optimization
- Pre-train ML model offline, load as JAX module
- Use efficient heuristics for real-time allocation
- Cache MRIO computations where possible

### 3. Memory Management
- Use appropriate dtypes (float32 vs float64)
- Leverage JAX's memory-efficient operations
- Consider model parallelism for large MRIO tables

## Training and Validation

### Data Requirements
- MRIO tables for all regions (e.g., EXIOBASE, EORA, OECD ICIO)
- Sector-level carbon intensities
- Historical trade data for training allocation models

### Validation Steps
1. **Mass Balance**: Ensure inputs = outputs + value-added for each sector
2. **Trade Balance**: Verify trade flows balance (exports = imports)
3. **Carbon Accounting**: Check embodied carbon calculations
4. **Economic Consistency**: Compare aggregate outputs with original RICE

### Benchmarking
- Compare aggregate economic indicators with baseline RICE
- Validate trade patterns against historical data
- Test CBAM policy impacts on emissions and welfare

## Extension Points

### Scenario Variations
- **CBAM Phases**: Gradual implementation with different coverage
- **Sector Exemptions**: Exclude certain sectors from CBAM
- **Border Adjustments**: Different rules for different trading partners
- **Carbon Pricing Integration**: Link to domestic ETS prices

### Advanced Features
- **Dynamic MRIO**: Update input-output coefficients over time
- **Supply Chain Effects**: Multi-step embodied emissions
- **Consumption-Based Accounting**: Track consumption-based emissions
- **Trade Policy Interactions**: Model retaliatory tariffs

## Implementation Roadmap

1. **Phase 1**: Basic disaggregated production without trade
   - Implement sector-level production with MRIO Leontief structure
   - Reaggregate production back to single scalar for economy calculations
   - Sanity check: verify economics match baseline RICE with disaggregated/reaggregated production

2. **Phase 1B**: Validate disaggregation/reaggregation mechanism
   - Run same scenarios with original and disaggregated models
   - Compare aggregate outputs, welfare, emissions trajectories
   - Ensure no numerical drift or biases introduced
   - Verify MRIO mass balance and accounting

3. **Phase 2**: Disaggregate trade allocation
   - Add sector-level trade flows to state space
   - Implement heuristic trade allocation (MRIO-based shares)
   - Apply sector-level tariffs and import bids
   - Ensure trade flows balance and are economically consistent

4. **Phase 3**: Integrate CBAM pricing
   - Add carbon accounting on traded goods
   - Implement sector/region-specific CBAM prices
   - Model carbon leakage effects
   - Validate carbon accounting is consistent

5. **Phase 4**: Train ML-based trade allocation
   - Collect offline training data from Phase 2/3
   - Train neural network on MRIO and economic indicators
   - Replace heuristics with learned allocation model
   - Benchmark ML vs. heuristic allocations

6. **Phase 5**: Add advanced scenarios and validation
   - Implement CBAM policy variations
   - Add sector exemptions and phasing
   - Validate against empirical trade data
   - Sensitivity analysis on MRIO parameters

This phased approach with the intermediate sanity check (Phase 1B) ensures that the disaggregation mechanism doesn't introduce unintended artifacts while maintaining the computational efficiency of JAX.</content>
<parameter name="filePath">/Users/pwozny/Repos/phd/climate-cooperation-competition/DISAGGREGATED_RICE_DESIGN.md