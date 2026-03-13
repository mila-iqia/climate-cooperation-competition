"""
RiceMRIO — Phase 1B: JAX RICE with MRIO-based sectoral production disaggregation.

The standard Cobb-Douglas production is distribued across EORA26 sectors using
static 2016 output shares, then reaggregated to a scalar per region.  The result
is numerically bit-identical to the base Rice class; the new `production_by_sector`
state key (shape: num_regions × num_sectors) is stored for inspection and future
Phase 2 use.

See rice_jax/rice_jax/MRIO_RICE_DESIGN.md for the full design rationale.
"""

from __future__ import annotations

import os
from typing import Any

import chex
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ._rice import Rice


# ---------------------------------------------------------------------------
# Helper: load and build sector output shares from aggregated EORA x.txt
# ---------------------------------------------------------------------------

def load_sector_shares(
    mrio_aggregated_dir: str,
    num_regions: int,
    country_class_csv: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a (num_regions, num_sectors) sector output share matrix from the
    aggregated EORA26 x.txt file.

    The share matrix σ_{r,s} satisfies Σ_s σ_{r,s} = 1 for all regions r,
    so disaggregating then reaggregating production is an exact identity.

    Parameters
    ----------
    mrio_aggregated_dir:
        Folder produced by aggregate_local_mrio.py, e.g.
        'csv_asset/mrio/aggregated/eora_agg_20'.
    num_regions:
        Number of RICE regions (3, 7, or 20).
    country_class_csv:
        Path to the matching CountryClass CSV, e.g.
        'csv_asset/CountryClass_20.csv'.

    Returns
    -------
    shares : np.ndarray, float32, shape (num_regions, num_sectors)
        Row-normalised sector output shares σ_{r,s}.
    sector_names : list[str]
        Ordered EORA26 sector names (26 entries).
    rice_to_mrio : list[str]
        For each RICE region index i (0-based), the MRIO region label used to
        look up its shares.  Two RICE regions may share the same MRIO label
        when multiple RIG groups were merged in the CountryClass CSV.
    """
    # --- 1. Load x.txt (total output per region-sector) ---
    x_path = os.path.join(mrio_aggregated_dir, "x.txt")
    x_raw = pd.read_csv(x_path, sep="\t", index_col=[0, 1])
    x_raw.index.names = ["region", "sector"]
    x_raw.columns = ["output"]

    # Pivot to DataFrame: rows = MRIO regions, cols = sectors
    x_matrix = x_raw["output"].unstack(level="sector").fillna(0.0)
    # Row-normalise  → shares summing to 1 per region
    row_sums = x_matrix.sum(axis=1)
    shares_df = x_matrix.div(row_sums, axis=0).fillna(0.0)

    # --- 2. Build RIG → MRIO-region-label mapping from CountryClass CSV ---
    class_df = pd.read_csv(country_class_csv)
    class_df["Code"] = class_df["Code"].str.strip()
    class_df["RI"] = class_df["RI"].str.strip()

    # First RI label per RIG integer (1-indexed, matching RICE 1-indexed yams)
    rig_to_label: dict[int, str] = (
        class_df.dropna(subset=["RIG"])
        .groupby("RIG")["RI"]
        .first()
        .sort_index()
        .to_dict()
    )
    # Convert integer keys to int (pandas may produce np.int64)
    rig_to_label = {int(k): v for k, v in rig_to_label.items()}

    mrio_rows = set(shares_df.index.tolist())
    rest_of_world = "Rest of World"

    rice_to_mrio: list[str] = []
    for rig_idx in range(1, num_regions + 1):
        label = rig_to_label.get(rig_idx, rest_of_world)
        if label not in mrio_rows:
            label = rest_of_world
        rice_to_mrio.append(label)

    # --- 3. Assemble (num_regions, num_sectors) shares array in RICE order ---
    sector_names: list[str] = shares_df.columns.tolist()
    num_sectors = len(sector_names)
    shares_array = np.zeros((num_regions, num_sectors), dtype=np.float32)

    for rice_idx, mrio_label in enumerate(rice_to_mrio):
        if mrio_label in shares_df.index:
            shares_array[rice_idx] = shares_df.loc[mrio_label].values.astype(
                np.float32
            )
        else:
            # Fallback: uniform shares (should not normally occur)
            shares_array[rice_idx] = np.full(num_sectors, 1.0 / num_sectors)

    return shares_array, sector_names, rice_to_mrio


# ---------------------------------------------------------------------------
# RiceMRIO subclass
# ---------------------------------------------------------------------------


class RiceMRIO(Rice):
    """
    Phase 1B: Rice with MRIO-based sectoral production disaggregation.

    Uses static 2016 EORA26 output shares to split each region's Cobb-Douglas
    output across 26 EORA26 sectors, then sums back to the scalar expected by
    all downstream calculations.  The round-trip is exact (Σσ = 1), so
    economic behaviour is bit-identical to the base Rice class.

    New state key added
    -------------------
    ``production_by_sector`` : Array shape (num_regions, num_sectors)
        Sectoral breakdown of aggregate production at each timestep.

    Parameters
    ----------
    mrio_data_root : str
        Path to the ``csv_asset`` directory that contains both the aggregated
        MRIO sub-tree and the CountryClass CSVs.  Defaults to ``'csv_asset'``
        (relative to the working directory, i.e. ``rice_jax/``).
        From ``num_regions`` the class automatically resolves:
          - ``{mrio_data_root}/mrio/aggregated/eora_agg_{num_regions}/``
          - ``{mrio_data_root}/CountryClass_{num_regions}.csv``

    All other fields are inherited from ``Rice`` and passed as keyword args.

    Example
    -------
    >>> from rice_jax._rice_mrio import RiceMRIO
    >>> from rice_jax.utils import load_region_yamls
    >>> region_params = load_region_yamls(20)
    >>> env = RiceMRIO(
    ...     num_regions=20,
    ...     region_params=region_params,
    ... )
    """

    # Single root; both sub-paths are derived in __post_init__ from num_regions.
    mrio_data_root: str = eqx.field(static=True, default="csv_asset")

    # Populated in __post_init__ from the above paths.
    sector_output_shares: np.ndarray = eqx.field(static=True, default=None)
    sector_names: tuple = eqx.field(static=True, default=())
    mrio_region_labels: tuple = eqx.field(static=True, default=())
    num_sectors: int = eqx.field(static=True, default=0)

    def __post_init__(self) -> None:
        # Call Rice's __post_init__ first (handles baseline_rewards when
        # relative_reward_mode=True).
        super().__post_init__()

        mrio_aggregated_dir = os.path.join(
            self.mrio_data_root, "mrio", "aggregated", f"eora_agg_{self.num_regions}"
        )
        country_class_csv = os.path.join(
            self.mrio_data_root, f"CountryClass_{self.num_regions}.csv"
        )

        if not os.path.isdir(mrio_aggregated_dir):
            return  # data not present — allow lightweight construction in tests

        shares, sector_names, rice_to_mrio = load_sector_shares(
            mrio_aggregated_dir=mrio_aggregated_dir,
            num_regions=self.num_regions,
            country_class_csv=country_class_csv,
        )

        # Equinox modules are immutable; object.__setattr__ is the approved way
        # to set fields after construction (used in Rice itself for baseline_rewards).
        object.__setattr__(self, "sector_output_shares", shares)
        object.__setattr__(self, "sector_names", tuple(sector_names))
        object.__setattr__(self, "mrio_region_labels", tuple(rice_to_mrio))
        object.__setattr__(self, "num_sectors", len(sector_names))

    # ------------------------------------------------------------------
    # State initialisation: add production_by_sector key
    # ------------------------------------------------------------------

    def _get_initial_state(self, key: chex.PRNGKey) -> dict:
        state = super()._get_initial_state(key)
        state["production_by_sector"] = jnp.zeros(
            (self.num_regions, self.num_sectors), dtype=jnp.float32
        )
        return state

    # ------------------------------------------------------------------
    # Step: disaggregate after parent step, store in state
    # ------------------------------------------------------------------

    def step_climate_and_economy(
        self, state: dict[str, Any], actions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Identical to Rice.step_climate_and_economy, but additionally computes
        and stores ``production_by_sector`` in the returned state dict.

        The disaggregation is applied post-hoc to the aggregate
        ``production_all_regions`` already computed by the parent, so there is
        no change to any other state variable.
        """
        # 1. Run parent step — all economics, climate, trade unchanged
        state = super().step_climate_and_economy(state, actions)

        # 2. Disaggregate aggregate production into sectors
        #    Y: (num_regions,)    σ: (num_regions, num_sectors)
        #    y_{r,s} = σ_{r,s} * Y_r
        Y = state["production_all_regions"]  # (num_regions,)
        shares = jnp.array(self.sector_output_shares)  # (num_regions, num_sectors)
        production_by_sector = shares * Y[:, None]  # (num_regions, num_sectors)

        # 3. Reaggregation sanity: Σ_s y_{r,s} == Y_r (exact by construction)
        # Not asserted at runtime (JAX jit incompatible) but verifiable in tests.

        state = state.copy()
        state["production_by_sector"] = production_by_sector
        return state

    # ------------------------------------------------------------------
    # Convenience: validation helper (run outside jit)
    # ------------------------------------------------------------------

    def validate_shares(self) -> dict[str, Any]:
        """
        Returns a dict of validation diagnostics.  Call outside jit/vmap.

        Checks:
        - Each row of sector_output_shares sums to 1.0 (within float32 tol).
        - No NaN or Inf values in shares.
        - Length of sector_names matches num_sectors.
        - Length of mrio_region_labels matches num_regions.
        """
        shares = self.sector_output_shares  # numpy array
        row_sums = shares.sum(axis=1)
        return {
            "row_sums": row_sums,
            "all_rows_sum_to_one": np.allclose(row_sums, 1.0, atol=1e-5),
            "no_nan": not np.isnan(shares).any(),
            "no_inf": not np.isinf(shares).any(),
            "sector_names_count_ok": len(self.sector_names) == self.num_sectors,
            "region_labels_count_ok": len(self.mrio_region_labels) == self.num_regions,
            "sector_names": self.sector_names,
            "mrio_region_labels": self.mrio_region_labels,
        }
