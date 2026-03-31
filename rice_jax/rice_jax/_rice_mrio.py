"""
RiceMRIO — MRIO-based variant of the JAX RICE environment.

Phase 1B (existing):
    Sectoral production disaggregation.  The standard Cobb-Douglas output is
    split across EORA26 sectors using static 2016 shares, then reaggregated;
    the round-trip is exact and behaviour is bit-identical to base Rice.

Phase 2A (new):
    MRIO-based bilateral trade flows replace the bid/limit mechanism.
    Each agent controls ``export_reallocation`` — a logit-adjustment over the
    MRIO baseline destination shares — while the total export volume per
    (region, sector) remains fixed at the 2016 MRIO level.  A CBAM tariff
    (parameter, not yet an action) penalises EU-destined exports proportional
    to their sectoral emissions intensity, modelled via the existing Rice
    welfare-loss multiplier mechanism.

    New state keys: ``trade_flows`` (NR × NR × NS), ``cbam_revenue`` (NR,).
    Removed actions:  ``import_bid``, ``export_limit``, ``import_tariff``.
    Added action:     ``export_reallocation`` (MultiDiscrete, NS × NR dims).

See CBAM_ROADMAP.md and rice_jax/rice_jax/MRIO_RICE_DESIGN.md for details.
"""

from __future__ import annotations

import os
from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxnasium import Discrete, MultiDiscrete

from ._rice import Rice
from .utils import i_to_agent_str


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
# Phase 2A helpers: bilateral trade shares and emissions intensity
# ---------------------------------------------------------------------------

def _load_x_series(mrio_aggregated_dir: str) -> pd.Series:
    """Load total output as a (region, sector)-indexed Series."""
    x_raw = pd.read_csv(
        os.path.join(mrio_aggregated_dir, "x.txt"), sep="\t", index_col=[0, 1]
    )
    x_raw.index.names = ["region", "sector"]
    x_raw.columns = ["output"]
    return x_raw["output"]


def load_bilateral_trade_shares(
    mrio_aggregated_dir: str,
    rice_to_mrio: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build bilateral export-share arrays from Z (intermediate) + Y (final demand).

    Parameters
    ----------
    mrio_aggregated_dir:
        Folder with Z.parquet, Y.parquet, x.txt.
    rice_to_mrio:
        Ordered list of MRIO region labels, one per RICE region (0-indexed).
        Produced by ``load_sector_shares``.

    Returns
    -------
    dest_alloc_baseline : float32 array, shape (num_rice_regions, num_sectors, num_rice_regions)
        ``dest_alloc_baseline[r, s, d]`` = baseline destination share for
        exporter ``r``, sector ``s``, destination ``d`` (d==r → 0).
        Sums to 1 over the ``d`` axis after masking the diagonal.
    total_export_frac : float32 array, shape (num_rice_regions, num_sectors)
        ``total_export_frac[r, s]`` = fraction of region ``r``'s sector ``s``
        total output that is exported (fixed at MRIO 2016 level).
    """
    num_rice = len(rice_to_mrio)

    Z = pd.read_parquet(os.path.join(mrio_aggregated_dir, "Z.parquet"))
    Y = pd.read_parquet(os.path.join(mrio_aggregated_dir, "Y.parquet"))
    x = _load_x_series(mrio_aggregated_dir)

    sectors = Z.index.get_level_values("sector").unique().tolist()
    num_sectors = len(sectors)
    sec_idx = {s: i for i, s in enumerate(sectors)}

    # Aggregate Z columns over destination sectors → (from_r, from_s) × to_r
    Z_to_r = Z.T.groupby(level="region").sum().T  # (from×sector, to_region)
    # Aggregate Y columns over demand categories → (from_r, from_s) × to_r
    Y_to_r = Y.T.groupby(level=0).sum().T         # same shape

    T = Z_to_r.add(Y_to_r, fill_value=0.0)  # total bilateral trade (incl. domestic)

    mrio_rows_available = set(T.index.get_level_values("region").unique())
    mrio_cols_available = set(T.columns.tolist())

    # Normalise by output → bilateral export shares
    x_aligned = x.reindex(T.index).fillna(1e-10)
    T_shares = T.div(x_aligned, axis=0)  # (from_region, from_sector) × (to_region)

    # Map to RICE-region bilateral matrix
    bilateral = np.zeros((num_rice, num_rice, num_sectors), dtype=np.float64)
    for from_r_idx, from_r_label in enumerate(rice_to_mrio):
        if from_r_label not in mrio_rows_available:
            continue
        try:
            t_from = T_shares.xs(from_r_label, level="region")
        except KeyError:
            continue
        t_from = t_from.reindex(sectors).fillna(0.0)  # (sectors, to_regions)

        for to_r_idx, to_r_label in enumerate(rice_to_mrio):
            if from_r_idx == to_r_idx:
                continue  # no self-export
            if to_r_label not in mrio_cols_available:
                continue
            bilateral[from_r_idx, to_r_idx, :] = t_from[to_r_label].values

    # total_export_frac[r, s] = sum_{r'≠r} bilateral[r, r', s]
    total_export_frac = bilateral.sum(axis=1).astype(np.float32)  # (NR, NS)

    # dest_alloc_baseline[r, s, d] = bilateral[r, d, s] / total_export_frac[r, s]
    bilateral_rsd = bilateral.transpose(0, 2, 1)  # (NR, NS, NR) [from_r, s, to_r]
    safe_total = np.maximum(total_export_frac[:, :, np.newaxis], 1e-10)
    dest_alloc_baseline = (bilateral_rsd / safe_total).astype(np.float32)
    # Ensure self-diagonal is exactly zero
    for r in range(num_rice):
        dest_alloc_baseline[r, :, r] = 0.0

    return dest_alloc_baseline, total_export_frac


def load_emissions_intensity(
    mrio_aggregated_dir: str,
    rice_to_mrio: list[str],
) -> np.ndarray:
    """
    Build a (num_rice_regions, num_sectors) emissions-intensity matrix
    from the Q_S satellite account (CO2-equivalent, tCO2 per unit output).

    Rows with outer index ``'I-GHG-CO2 emissions'`` are summed over all
    sub-categories per (region, sector) column, then divided by total output.
    """
    num_rice = len(rice_to_mrio)

    Q = pd.read_parquet(os.path.join(mrio_aggregated_dir, "Q_S.parquet"))
    x = _load_x_series(mrio_aggregated_dir)

    sectors = x.index.get_level_values("sector").unique().tolist()
    num_sectors = len(sectors)
    sec_idx = {s: i for i, s in enumerate(sectors)}

    # Sum CO2 rows over all sub-categories
    co2_mask = Q.index.get_level_values(0) == "I-GHG-CO2 emissions"
    if not co2_mask.any():
        # Fallback: zero intensities
        return np.zeros((num_rice, num_sectors), dtype=np.float32)

    co2_total = Q.loc[co2_mask].sum(axis=0)  # Series indexed by (region, sector)

    # Normalise by output
    x_safe = x.reindex(co2_total.index).fillna(1e-10)
    intensity_series = co2_total / x_safe

    # Map to RICE regions
    mrio_rows_available = set(intensity_series.index.get_level_values("region").unique())
    intensity = np.zeros((num_rice, num_sectors), dtype=np.float32)
    for r_idx, r_label in enumerate(rice_to_mrio):
        if r_label not in mrio_rows_available:
            continue
        try:
            row = intensity_series.xs(r_label, level="region")  # (sectors,)
        except KeyError:
            continue
        for s_name, s_i in sec_idx.items():
            if s_name in row.index:
                intensity[r_idx, s_i] = float(row.loc[s_name])

    return intensity




# ---------------------------------------------------------------------------
# Sector aggregation for CBAM analysis
# ---------------------------------------------------------------------------

# CBAM-covered sectors in EORA26 (exact names as they appear in x.txt after
# region aggregation by aggregate_local_mrio.py).
_CBAM_SECTORS: frozenset[str] = frozenset(
    {
        "Petroleum, Chemical and Non-Metallic Mineral Products",  # chemicals, fertilisers, cement
        "Metal Products",  # steel, aluminium
        "Electricity, Gas and Water",  # electricity
    }
)

# Additional high-emissions sectors that are not formally CBAM-covered but
# have high emissions intensity in the EORA data (see vulnerability mapping).
_HIGH_EMISSIONS_SECTORS: frozenset[str] = frozenset(
    {
        "Other Manufacturing",
        "Transport Equipment",
        "Construction",
        "Mining and Quarrying",
    }
)


def _build_sector_groups(
    sector_names: list[str],
    granularity: str,
) -> list[tuple[str, list[int]]]:
    """
    Return ordered sector group definitions for the requested granularity.

    Parameters
    ----------
    sector_names : ordered list of raw sector name strings (length NS).
    granularity  : ``"full"`` | ``"cbam-specific"`` | ``"simple"``
                   | ``"emissions-specific"`` | ``"emissions-simple"``.

    Returns
    -------
    groups : list of (group_label, [0-based sector indices])
    """
    if granularity == "full":
        return [(s, [i]) for i, s in enumerate(sector_names)]

    cbam_indices = [i for i, s in enumerate(sector_names) if s in _CBAM_SECTORS]
    non_cbam_indices = [i for i, s in enumerate(sector_names) if s not in _CBAM_SECTORS]

    # Broader dirty set: CBAM + high-emissions non-CBAM sectors
    dirty_set = _CBAM_SECTORS | _HIGH_EMISSIONS_SECTORS
    dirty_indices = [i for i, s in enumerate(sector_names) if s in dirty_set]
    clean_indices = [i for i, s in enumerate(sector_names) if s not in dirty_set]

    if granularity == "simple":
        return [("CBAM", cbam_indices), ("non-CBAM", non_cbam_indices)]

    if granularity == "cbam-specific":
        groups = [(s, [i]) for i, s in enumerate(sector_names) if s in _CBAM_SECTORS]
        groups.append(("non-CBAM", non_cbam_indices))
        return groups

    if granularity == "emissions-simple":
        return [("CBAM", dirty_indices), ("non-CBAM", clean_indices)]

    if granularity == "emissions-specific":
        groups = [(s, [i]) for i, s in enumerate(sector_names) if s in dirty_set]
        groups.append(("non-CBAM", clean_indices))
        return groups

    raise ValueError(
        f"Unknown sector_granularity: {granularity!r}. "
        "Choose 'full', 'cbam-specific', 'simple', "
        "'emissions-specific', or 'emissions-simple'."
    )


def _aggregate_sector_arrays(
    groups: list[tuple[str, list[int]]],
    shares: np.ndarray,           # (NR, NS_old)
    total_export_frac: np.ndarray,  # (NR, NS_old)
    dest_alloc_baseline: np.ndarray | None,  # (NR, NS_old, NR) or None
    emissions_intensity: np.ndarray | None,  # (NR, NS_old) or None
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    """
    Collapse sector arrays from NS_old sectors into len(groups) coarser groups.

    Aggregation rules
    -----------------
    * ``sector_output_shares``  : sum over sectors in the group.
    * ``total_export_frac``     : weighted average using sector shares as weights.
    * ``dest_alloc_baseline``   : weighted average using export volume
                                  (shares × total_export_frac) as weights.
    * ``emissions_intensity``   : weighted average using export volume as weights.

    These rules ensure the downstream welfloss formula is invariant to the
    aggregation: for any group g,
        Σ_{s∈g} X_{r,s}^EU · σ_{r,s} ≡ X_{r,g}^EU · σ_agg_{r,g}
    (up to the approximation that all sectors in a group share the same
    destination mix, which is exact for the "full" pass-through case).
    """
    NR = shares.shape[0]
    NS_new = len(groups)

    new_names = [name for name, _ in groups]
    new_shares = np.zeros((NR, NS_new), dtype=np.float32)
    new_tef = np.zeros((NR, NS_new), dtype=np.float32)
    new_dab = (
        np.zeros((NR, NS_new, NR), dtype=np.float32)
        if dest_alloc_baseline is not None
        else None
    )
    new_int = (
        np.zeros((NR, NS_new), dtype=np.float32)
        if emissions_intensity is not None
        else None
    )

    for g_idx, (_, s_indices) in enumerate(groups):
        if not s_indices:
            continue
        s_idx = np.array(s_indices)

        # Sector output shares: simple sum
        new_shares[:, g_idx] = shares[:, s_idx].sum(axis=1)

        # Export fraction: weighted average by output share
        w = shares[:, s_idx]  # (NR, n_s)
        w_sum = w.sum(axis=1)  # (NR,)
        safe_w = np.where(w_sum > 0, w_sum, 1.0)
        new_tef[:, g_idx] = (w * total_export_frac[:, s_idx]).sum(axis=1) / safe_w

        # Export volume weight: proportional to absolute export = share * tef
        vol = shares[:, s_idx] * total_export_frac[:, s_idx]  # (NR, n_s)
        vol_sum = vol.sum(axis=1)  # (NR,)
        safe_vol = np.where(vol_sum > 0, vol_sum, 1.0)

        if dest_alloc_baseline is not None:
            # dab: (NR, NS_old, NR) → weighted avg over s_idx, result → (:, g_idx, :)
            dab_sub = dest_alloc_baseline[:, s_idx, :]  # (NR, n_s, NR)
            vol_exp = vol[:, :, np.newaxis]  # (NR, n_s, 1)
            new_dab[:, g_idx, :] = (
                (vol_exp * dab_sub).sum(axis=1) / safe_vol[:, np.newaxis]
            )

        if emissions_intensity is not None:
            new_int[:, g_idx] = (
                (vol * emissions_intensity[:, s_idx]).sum(axis=1) / safe_vol
            )

    return new_names, new_shares, new_tef, new_dab, new_int


# ---------------------------------------------------------------------------


class RiceMRIO(Rice):
    """
    Phase 1B + 2A: Rice with MRIO-based sectoral disaggregation and trade.

    Phase 1B behaviour (mrio_trade=False, default):
        Bit-identical to base Rice.  ``production_by_sector`` is computed and
        stored but not consumed; all trade uses the parent's import-bid/export-
        limit mechanism unchanged.

    Phase 2A behaviour (mrio_trade=True):
        • ``import_bid``, ``export_limit``, ``import_tariff`` removed from
          action space.
        • ``export_reallocation`` added: logit-adjustments over the MRIO 2016
          baseline destination shares.  Zero action = MRIO baseline flows.
        • CBAM tariff (``cbam_tariff_rate``, a scalar parameter) penalises
          EU-destined exports via the Rice welfare-loss multiplier mechanism.
        • New state keys: ``trade_flows`` (NR×NR×NS), ``cbam_revenue`` (NR,).

    Parameters
    ----------
    mrio_data_root : str
        Path to the ``csv_asset`` directory.  Defaults to ``'csv_asset'``
        (relative to the working directory, i.e. ``rice_jax/``).
    mrio_trade : bool
        Enable Phase 2A MRIO trade.  Requires bilateral MRIO data.
    cbam_tariff_rate : float
        CBAM tariff (fraction, e.g. 0.1 = 10%).  Applied as a welfare-loss
        multiplier to exporters sending goods to ``eu_region_idx``.
    eu_region_idx : int
        0-based RICE region index of the EU region (default 0).
    delta_max : float
        Tanh squashing bound for export_reallocation logit-adjustments.
    """

    # ------------------------------------------------------------------ Phase 1B
    mrio_data_root: str = eqx.field(static=True, default="csv_asset")

    sector_output_shares: np.ndarray = eqx.field(static=True, default=None)
    sector_names: tuple = eqx.field(static=True, default=())
    mrio_region_labels: tuple = eqx.field(static=True, default=())
    num_sectors: int = eqx.field(static=True, default=0)

    # ------------------------------------------------------------------ Phase 2A
    mrio_trade: bool = eqx.field(static=True, default=False)
    cbam_tariff_rate: float = eqx.field(static=True, default=0.0)
    # When True, cbam_tariff_rate is randomized per episode from
    # cbam_tariff_rates (sampled uniformly).  The active rate is stored
    # in state["cbam_tariff_rate"] and appended to each agent's obs,
    # letting a single policy learn a CBAM-conditioned response.
    cbam_randomize: bool = eqx.field(static=True, default=False)
    cbam_tariff_rates: tuple = eqx.field(static=True, default=(0.0,))
    eu_region_idx: int = eqx.field(static=True, default=0)
    delta_max: float = eqx.field(static=True, default=3.0)
    # Target mean for intensity normalisation (rescales real EORA values so
    # the CBAM signal is non-negligible while preserving cross-sector heterogeneity).
    cbam_intensity_target_mean: float = eqx.field(static=True, default=0.1)

    # Trade-flow persistence: AR(1) weight ρ ∈ [0,1] that blends the 2016
    # MRIO baseline with the previous period's realised allocation as the
    # logit anchor.  ρ=0 → always anchor to 2016 (current behaviour).
    # ρ=1 → fully adaptive (previous allocation becomes the new baseline).
    # Literature support: Roberts & Tybout (1997), Eaton & Kortum (2002).
    dest_alloc_persistence: float = eqx.field(static=True, default=0.0)

    # Baseline-weight decay exponent for the AR(1) anchor.  Controls how
    # quickly the 2016 baseline influence diminishes over time.
    #   0.0 (default) = no decay: constant weight (1-ρ) every step (original).
    #   1.0           = standard exponential decay: weight = (1-ρ)^t.
    #   > 1.0         = faster-than-exponential decay.
    #   ∈ (0, 1)      = slower-than-exponential decay.
    # When active (> 0), the anchor formula becomes:
    #   w_t = max(1-ρ, ε)^(decay·t)
    #   anchor = w_t · baseline + (1 - w_t) · prev_alloc
    dest_alloc_baseline_decay: float = eqx.field(static=True, default=0.0)

    # Sectoral welfloss: when True, the welfare-loss multiplier is computed
    # sector-by-sector using actual EU-bound flows per sector weighted by that
    # sector's emissions intensity, then summed to a scalar per region:
    #   welfloss[r] = 1 - Σ_s (X_{r,s}^EU * σ_{r,s} * τ * α) / Y_r
    # This gives export_reallocation a per-sector reward gradient — dirty-
    # sector EU exports are penalised more than clean ones, so agents learn
    # to divert selectively rather than in aggregate.
    # When False (default), the aggregate Approach B formula is used.
    sectoral_welfloss: bool = eqx.field(static=True, default=False)

    # Action-space reduction flags.  When True the corresponding action is
    # removed from the action space and a hardcoded value is injected instead,
    # leaving export_reallocation as the sole driver of differentiated reward.
    # fixed_savings_rate: savings_rate fixed to 0.2 (≈ Nordhaus optimal).
    # no_mitigation:      mitigation_rate fixed to 0.0 (BAU baseline).
    fixed_savings_rate: bool = eqx.field(static=True, default=False)
    no_mitigation: bool = eqx.field(static=True, default=False)

    # Sector granularity for the export_reallocation action space.
    # Controls how the 26 EORA sectors are aggregated before building the
    # trade arrays, reducing the action-space dimensionality.
    #
    #   "full"              — all 26 EORA sectors unchanged (default).
    #   "cbam-specific"     — 3 CBAM sectors separate + "non-CBAM" (4 total).
    #   "simple"            — 2 sectors: "CBAM" and "non-CBAM".
    #   "emissions-specific" — 7 high-emission sectors separate (3 CBAM +
    #                          Other Mfg, Transport Equip, Construction,
    #                          Mining) + "non-CBAM" bucket (8 total).
    #   "emissions-simple"   — 2 sectors: "CBAM" (all 7 dirty) and "non-CBAM".
    #
    # Only affects the export_reallocation action dimension and the underlying
    # trade arrays.  Production shares etc. are aggregated consistently.
    sector_granularity: str = eqx.field(static=True, default="full")

    # Nordhaus (2015) welfare-loss amplifier α.  Default 0.4 reflects the
    # empirical deadweight-loss estimate.  Increase for mechanism-validation
    # experiments where the realistic penalty is too small for PPO to detect.
    welfare_loss_per_unit_tariff: float = eqx.field(static=True, default=0.4)

    # Populated in __post_init__ when mrio_trade=True
    dest_alloc_baseline: np.ndarray = eqx.field(static=True, default=None)
    total_export_frac: np.ndarray = eqx.field(static=True, default=None)
    emissions_intensity: np.ndarray = eqx.field(static=True, default=None)

    # ------------------------------------------------------------------ init

    def __post_init__(self) -> None:
        super().__post_init__()

        mrio_aggregated_dir = os.path.join(
            self.mrio_data_root, "mrio", "aggregated", f"eora_agg_{self.num_regions}"
        )
        country_class_csv = os.path.join(
            self.mrio_data_root, f"CountryClass_{self.num_regions}.csv"
        )

        if not os.path.isdir(mrio_aggregated_dir):
            return  # data not present — allow lightweight construction in tests

        # --- Phase 1B: sector output shares ---
        shares, sector_names, rice_to_mrio = load_sector_shares(
            mrio_aggregated_dir=mrio_aggregated_dir,
            num_regions=self.num_regions,
            country_class_csv=country_class_csv,
        )

        # --- Phase 2A: bilateral trade data (load at full 26-sector granularity) ---
        dest_alloc: np.ndarray | None = None
        total_export_frac: np.ndarray | None = None
        intensity: np.ndarray | None = None
        if self.mrio_trade:
            dest_alloc, total_export_frac = load_bilateral_trade_shares(
                mrio_aggregated_dir=mrio_aggregated_dir,
                rice_to_mrio=list(rice_to_mrio),
            )
            intensity = load_emissions_intensity(
                mrio_aggregated_dir=mrio_aggregated_dir,
                rice_to_mrio=list(rice_to_mrio),
            )
            # Normalise to target mean so CBAM signal is non-negligible while
            # preserving cross-sector heterogeneity (Approach A).
            intensity_mean = float(intensity.mean())
            if intensity_mean > 0:
                intensity = intensity / intensity_mean * self.cbam_intensity_target_mean

        # --- Sector aggregation (if granularity != "full") ---
        if self.sector_granularity != "full":
            groups = _build_sector_groups(list(sector_names), self.sector_granularity)
            new_names, shares, new_tef, new_dab, new_int = _aggregate_sector_arrays(
                groups=groups,
                shares=shares,
                total_export_frac=(
                    total_export_frac
                    if total_export_frac is not None
                    else np.zeros((self.num_regions, len(sector_names)), dtype=np.float32)
                ),
                dest_alloc_baseline=dest_alloc,
                emissions_intensity=intensity,
            )
            sector_names = new_names
            if self.mrio_trade:
                total_export_frac = new_tef
                dest_alloc = new_dab
                intensity = new_int

        object.__setattr__(self, "sector_output_shares", shares)
        object.__setattr__(self, "sector_names", tuple(sector_names))
        object.__setattr__(self, "mrio_region_labels", tuple(rice_to_mrio))
        object.__setattr__(self, "num_sectors", len(sector_names))

        if self.mrio_trade:
            object.__setattr__(self, "dest_alloc_baseline", dest_alloc)
            object.__setattr__(self, "total_export_frac", total_export_frac)
            object.__setattr__(self, "emissions_intensity", intensity)

    # ------------------------------------------------------------------ state

    def _get_initial_state(self, key: chex.PRNGKey) -> dict:
        state = super()._get_initial_state(key)
        if self.num_sectors > 0:
            state["production_by_sector"] = jnp.zeros(
                (self.num_regions, self.num_sectors), dtype=jnp.float32
            )
        if self.mrio_trade:
            state["trade_flows"] = jnp.zeros(
                (self.num_regions, self.num_regions, self.num_sectors),
                dtype=jnp.float32,
            )
            state["cbam_revenue"] = jnp.zeros(self.num_regions, dtype=jnp.float32)
            # Current destination-allocation logit anchor (NR, NS, NR).
            # Initialised to the 2016 MRIO baseline; updated each step when
            # dest_alloc_persistence > 0.
            state["dest_alloc_current"] = jnp.array(
                self.dest_alloc_baseline, dtype=jnp.float32
            ) if self.dest_alloc_baseline is not None else jnp.zeros(
                (self.num_regions, self.num_sectors, self.num_regions), dtype=jnp.float32
            )
        # Store per-episode CBAM tariff rate in state.  When cbam_randomize
        # is on, sample uniformly from the configured set; otherwise use the
        # static value so existing behaviour is preserved.
        if self.cbam_randomize:
            rates = jnp.array(self.cbam_tariff_rates)
            idx = jax.random.randint(key, (), 0, len(self.cbam_tariff_rates))
            state["cbam_tariff_rate"] = rates[idx]
        else:
            state["cbam_tariff_rate"] = jnp.float32(self.cbam_tariff_rate)
        return state

    # ------------------------------------------------------------------ observations

    def generate_observation(self, state: dict[str, Any]) -> dict[str, Any]:
        """Trade-focused observation when ``mrio_trade=True``.

        Replaces the parent's climate-heavy obs with a compact set of
        features that are directly relevant to the export-reallocation
        decision:

        Per agent (private):
          - activity_timestep          (1,)
          - gross_output               (1,)   own Y
          - utility                    (1,)   own welfare signal
          - trade_flows[agent]         (NR, NS)  own outgoing flows
          - dest_alloc_current[agent]  (NS, NR)  current allocation anchor

        Shared (public):
          - cbam_revenue               (NR,)  who pays how much CBAM

        Falls back to parent obs when mrio_trade is off.
        """
        if not self.mrio_trade:
            return super().generate_observation(state)

        obs = {}
        for agent_id in range(self.num_regions):
            obs[i_to_agent_str(agent_id)] = {
                "activity_timestep": state["activity_timestep"],
                "gross_output": state["gross_output_all_regions"][agent_id],
                "utility": state["utility_all_regions"][agent_id],
                "trade_flows": state["trade_flows"][agent_id],           # (NR, NS)
                "dest_alloc": state["dest_alloc_current"][agent_id],     # (NS, NR)
                "cbam_revenue": state["cbam_revenue"],                   # (NR,)
                "cbam_tariff_rate": state["cbam_tariff_rate"],            # scalar
            }
        return obs

    # ------------------------------------------------------------------ action space / masks (Phase 2A only)

    @property
    def action_space(self) -> dict:
        if not self.mrio_trade:
            return super().action_space

        N = self.num_regions
        D = self.num_discrete_action_levels
        NS = self.num_sectors
        # Logit-adjustment vector over (sector × destination) pairs.
        # Action midpoint (D//2) → δ=0 → MRIO baseline flows.
        actions: dict = {"export_reallocation": MultiDiscrete([D] * (NS * N))}
        if not self.fixed_savings_rate:
            actions["savings_rate"] = Discrete(D)
        if not self.no_mitigation:
            actions["mitigation_rate"] = Discrete(D)
        return {i_to_agent_str(i): actions for i in range(N)}

    def generate_action_masks(self, state: dict) -> dict:
        if not self.mrio_trade:
            return super().generate_action_masks(state)

        D = self.num_discrete_action_levels
        NS = self.num_sectors
        N = self.num_regions

        mask = {}
        for agent_id in range(N):
            astr = i_to_agent_str(agent_id)
            agent_mask: dict = {"export_reallocation": np.ones((NS * N, D), dtype=np.float32)}
            if not self.fixed_savings_rate:
                agent_mask["savings_rate"] = np.ones(D, dtype=np.float32)
            if not self.no_mitigation:
                agent_mask["mitigation_rate"] = np.ones(D, dtype=np.float32)
            mask[astr] = agent_mask

        # Apply minimum mitigation rate masking (same logic as Rice)
        if not self.no_mitigation:
            min_mit = state["minimum_mitigation_rate_all_regions"]
            for agent_id in range(N):
                min_rate = min_mit[agent_id] * D
                mask[i_to_agent_str(agent_id)]["mitigation_rate"] = (
                    jnp.arange(D) >= min_rate
                )

        return mask

    # ------------------------------------------------------------------ Phase 2A core

    def _compute_trade_flows(
        self,
        production_by_sector: chex.Array,
        export_reallocation: chex.Array,
        prev_dest_alloc: chex.Array,
        timestep: chex.Array,
    ) -> tuple[chex.Array, chex.Array]:
        """
        Compute bilateral sector-level trade flows.

        Parameters
        ----------
        production_by_sector : (NR, NS)
        export_reallocation : (NR, NS*NR)  values in [0, 1] after process_actions
        prev_dest_alloc : (NR, NS, NR)  previous period's realised allocation
            (used only when dest_alloc_persistence > 0)
        timestep : scalar  current activity_timestep (1-based after increment)

        Returns
        -------
        trade_flows : (NR, NR, NS)  [from_r, to_r, sector]
        dest_alloc  : (NR, NS, NR)  realised destination shares this period
        """
        # Remap discrete [0,1] actions to logit adjustments [-delta_max, +delta_max].
        delta = (export_reallocation * 2.0 - 1.0) * self.delta_max
        delta = delta.reshape(self.num_regions, self.num_sectors, self.num_regions)

        # AR(1) logit anchor: blend 2016 baseline with previous realised allocation.
        #   anchor = (1-ρ)·baseline + ρ·prev_alloc   (in probability space)
        # then take log for the logit.  This implements trade-flow persistence
        # (Roberts & Tybout 1997; Eaton & Kortum 2002).
        rho = self.dest_alloc_persistence
        dest_baseline = jnp.array(self.dest_alloc_baseline)  # (NR, NS, NR)
        decay = self.dest_alloc_baseline_decay
        if decay > 0.0:
            # Decaying baseline influence: weight = (1-ρ)^(decay·t)
            # At t=1 with decay=1.0 this equals the constant formula;
            # for t>1 the baseline contribution shrinks exponentially.
            w_t = jnp.power(jnp.maximum(1.0 - rho, 1e-10), decay * timestep)
            anchor = w_t * dest_baseline + (1.0 - w_t) * prev_dest_alloc
        else:
            anchor = (1.0 - rho) * dest_baseline + rho * prev_dest_alloc  # (NR, NS, NR)

        # Mask self-destination via very negative logit
        self_mask = jnp.eye(self.num_regions)[:, jnp.newaxis, :]  # (NR, 1, NR)
        log_anchor = jnp.where(
            self_mask > 0,
            -1e9,
            jnp.log(anchor + 1e-10),
        )
        adjusted = log_anchor + delta  # (NR, NS, NR)
        dest_alloc = jax.nn.softmax(adjusted, axis=2)  # (NR, NS, NR)

        # Scale by fixed baseline export fraction
        total_export_frac = jnp.array(self.total_export_frac)  # (NR, NS)
        export_volume = production_by_sector * total_export_frac  # (NR, NS)

        trade_flows_rsd = export_volume[:, :, jnp.newaxis] * dest_alloc  # (NR, NS, NR)
        trade_flows = trade_flows_rsd.transpose(0, 2, 1)  # (NR, NR, NS) [from, to, s]
        return trade_flows, dest_alloc

    def _compute_cbam(
        self,
        trade_flows: chex.Array,
        gross_imports_mrio: chex.Array,
        cbam_tariff_rate: chex.Array | None = None,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """
        Compute CBAM-related quantities.

        Returns
        -------
        cbam_tariff_matrix : (NR, NR)
            Effective tariff rate per (importer, exporter) pair;
            non-zero only in EU row.
        cbam_revenue : (NR,)
            Revenue collected by EU from each exporter's goods.
        """
        # What each region r exports to EU, by sector
        eu_exports_by_sector = trade_flows[:, self.eu_region_idx, :]  # (NR, NS)

        # Use state-based rate when provided, else fall back to static field
        rate = cbam_tariff_rate if cbam_tariff_rate is not None else self.cbam_tariff_rate

        # CBAM cost imposed on exporter r
        cbam_cost = (
            eu_exports_by_sector
            * jnp.array(self.emissions_intensity)
            * rate
        ).sum(axis=1)  # (NR,)

        # Effective tariff rate per exporter: cbam_cost[r] / gross_imports[EU, r]
        eu_gross = gross_imports_mrio[self.eu_region_idx] + 1e-8  # (NR,)
        effective_rate = jnp.clip(cbam_cost / eu_gross, 0.0, 1.0)  # (NR,)

        cbam_tariff_matrix = jnp.zeros((self.num_regions, self.num_regions))
        cbam_tariff_matrix = cbam_tariff_matrix.at[self.eu_region_idx].set(
            effective_rate
        )

        cbam_revenue = jnp.zeros(self.num_regions).at[self.eu_region_idx].set(
            cbam_cost.sum()
        )

        return cbam_tariff_matrix, cbam_revenue

    # ------------------------------------------------------------------ step

    def step_climate_and_economy(
        self, state: dict[str, Any], actions: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Phase 1B: passes through to parent + stores production_by_sector.
        Phase 2A: replaces trade with MRIO flows; applies CBAM welfare penalty.
        """
        if not self.mrio_trade:
            # ---- Phase 1B (unchanged behaviour) ----
            state = super().step_climate_and_economy(state, actions)
            if self.sector_output_shares is not None:
                Y = state["production_all_regions"]
                shares = jnp.array(self.sector_output_shares)
                state = state.copy()
                state["production_by_sector"] = shares * Y[:, jnp.newaxis]
            return state

        # ---- Phase 2A ----
        # 1. Extract MRIO action; inject zeroed legacy trade for parent compatibility.
        export_reallocation = actions["export_reallocation"]  # (NR, NS*NR) in [0,1]
        parent_actions = dict(actions)
        parent_actions["export_limit"] = jnp.zeros(self.num_regions)
        parent_actions["import_bid"] = jnp.zeros(
            (self.num_regions, self.num_regions)
        )
        parent_actions["import_tariff"] = jnp.zeros(
            (self.num_regions, self.num_regions)
        )
        # Inject hardcoded values for actions removed from the action space.
        if self.fixed_savings_rate:
            parent_actions["savings_rate"] = jnp.full(self.num_regions, 0.2)
        if self.no_mitigation:
            parent_actions["mitigation_rate"] = jnp.zeros(self.num_regions)

        # 2. Run parent with zeroed trade → correct climate/production/investment;
        #    consumption/utilities will be re-computed below.
        state = super().step_climate_and_economy(state, parent_actions)

        # 3. Disaggregate production into sectors
        Y = state["production_all_regions"]  # (NR,)
        shares = jnp.array(self.sector_output_shares)
        production_by_sector = shares * Y[:, jnp.newaxis]  # (NR, NS)

        # 4. Compute MRIO-based bilateral trade flows (AR(1) adaptive baseline)
        prev_dest_alloc = state["dest_alloc_current"]  # (NR, NS, NR)
        trade_flows, dest_alloc_new = self._compute_trade_flows(
            production_by_sector, export_reallocation, prev_dest_alloc,
            state["activity_timestep"],
        )
        # gross_imports_mrio[to_r, from_r] = sum_s trade_flows[from_r, to_r, s]
        gross_imports_mrio = trade_flows.sum(axis=2).T  # (NR, NR) [to, from]

        # 5. Compute CBAM: effective tariff matrix + revenue
        active_rate = state["cbam_tariff_rate"]
        cbam_tariff_matrix, cbam_revenue = self._compute_cbam(
            trade_flows, gross_imports_mrio, cbam_tariff_rate=active_rate
        )

        # 6. Recompute consumptions with MRIO gross imports (no tariff on quantity)
        gross_outputs = state["gross_output_all_regions"]
        investments = state["investment_all_regions"]
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports_mrio, gross_imports_mrio
        )

        # 7. Welfare-loss multiplier.
        #
        # Approach B (sectoral_welfloss=False, default):
        #   welfloss[r] = 1 - (EU_imports_from_r / Y_r) * eff_rate[r] * α
        #   Aggregate formula; export_reallocation has no per-sector gradient
        #   — all sectors in a region face the same blended effective rate.
        #
        # Sectoral welfloss (sectoral_welfloss=True):
        #   welfloss[r] = 1 - Σ_s (X_{r,s}^EU * σ_{r,s} * τ * α) / Y_r
        #   Each EU-bound sector flow is weighted by its own emissions intensity
        #   σ_{r,s}.  Dirty sectors cost more than clean ones, giving
        #   export_reallocation a genuine per-sector reward gradient.
        if self.sectoral_welfloss and self.emissions_intensity is not None:
            # eu_exports_by_sector: (NR, NS) — from_r × sector, destined for EU
            eu_exports_by_sector = trade_flows[:, self.eu_region_idx, :]  # (NR, NS)
            intensity = jnp.array(self.emissions_intensity)               # (NR, NS)
            sectoral_loss = (
                eu_exports_by_sector
                * intensity
                * active_rate
                * self.welfare_loss_per_unit_tariff
            )  # (NR, NS)
            welfloss = jnp.clip(
                1.0 - sectoral_loss.sum(axis=1) / (gross_outputs + 1e-8),
                0.0,
                1.0,
            )  # (NR,)
        else:
            eu_imports_from_r = gross_imports_mrio[self.eu_region_idx]  # (NR,)
            eu_volume_ratio = eu_imports_from_r / (gross_outputs + 1e-8)  # (NR,)
            welfloss = jnp.clip(
                1.0
                - eu_volume_ratio
                * cbam_tariff_matrix.sum(axis=0)  # effective rate per exporter
                * self.welfare_loss_per_unit_tariff,
                0.0,
                1.0,
            )

        # 8. Recompute utilities with MRIO consumptions
        utilities = self.calc_utilities(state, consumptions)
        utility_times_welfloss = utilities * welfloss

        # 9. Update state
        state = state.copy()
        state.update(
            {
                "production_by_sector": production_by_sector,
                "trade_flows": trade_flows,
                "cbam_revenue": cbam_revenue,
                # Overwrite consumption / utility with MRIO values
                "aggregate_consumption": consumptions,
                "utility_all_regions": utilities,
                "utility_times_welfloss_all_regions": utility_times_welfloss,
                # Store MRIO imports for logging compatibility
                "imports_minus_tariffs": gross_imports_mrio,
                "import_bids_all_regions": gross_imports_mrio,
                "normalized_import_bids_all_regions": gross_imports_mrio,
                "import_tariffs": cbam_tariff_matrix,
                # Updated logit anchor for next step's AR(1) blend
                "dest_alloc_current": dest_alloc_new,
            }
        )
        return state

    # ------------------------------------------------------------------ validation

    def validate_shares(self) -> dict[str, Any]:
        """
        Validation diagnostics for Phase 1B + 2A static arrays.
        Safe to call outside jit/vmap.
        """
        result: dict[str, Any] = {}

        # Phase 1B
        if self.sector_output_shares is not None:
            shares = self.sector_output_shares
            row_sums = shares.sum(axis=1)
            result.update(
                {
                    "sector_output_row_sums": row_sums,
                    "sector_shares_sum_to_one": np.allclose(row_sums, 1.0, atol=1e-5),
                    "sector_shares_no_nan": not np.isnan(shares).any(),
                    "sector_names": self.sector_names,
                    "mrio_region_labels": self.mrio_region_labels,
                }
            )

        # Phase 2A
        if self.dest_alloc_baseline is not None:
            dab = self.dest_alloc_baseline  # (NR, NS, NR)
            tef = self.total_export_frac  # (NR, NS)
            result.update(
                {
                    "dest_alloc_no_nan": not np.isnan(dab).any(),
                    "total_export_frac_in_0_1": bool(
                        np.all(tef >= 0) and np.all(tef <= 1.0)
                    ),
                    "emissions_intensity_no_nan": not np.isnan(
                        self.emissions_intensity
                    ).any(),
                }
            )

        return result

