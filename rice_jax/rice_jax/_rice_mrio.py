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
import warnings
from typing import Any

# RiceMRIO stores large numpy arrays (dest_alloc_baseline, sector_output_shares,
# total_export_frac, emissions_intensity) in eqx.field(static=True) fields.
# This is intentional: they are compile-time constants accessed inside JIT via
# explicit jnp.array(...) conversions.  Equinox's is_array check does not
# distinguish numpy from JAX arrays, so it warns incorrectly here.
warnings.filterwarnings(
    "ignore",
    message="A JAX array is being set as static",
    category=UserWarning,
)

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

    # Prescribed EU mitigation pathway.  Tuple of floats in [0,1], one per
    # episode timestep (0-indexed by activity_timestep before parent increments).
    # Overrides EU's mitigation action each step; all other agents learn freely.
    # Values beyond the episode horizon are clamped to the last entry.
    # Gives MAC_EU > 0 from step 1 → nonzero τ_eff from the first gradient.
    # [LITERATURE NEEDED: EU ETS trajectory reference]
    eu_mitigation_schedule: tuple | None = eqx.field(static=True, default=None)

    # When True, abatement costs are zeroed out (abatement_cost = 0 for all
    # regions).  Used for the Phase 2B motivating experiment: with free
    # abatement, agents should learn to fully mitigate under CBAM because
    # reducing μ → lower embedded emissions → lower CBAM cost at no expense.
    # This is the canonical null condition that isolates the CBAM-mitigation
    # incentive channel before adding realistic abatement costs in Phase 2B.
    zero_abatement_cost: bool = eqx.field(static=True, default=False)

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

    # Reward mode: controls how the CBAM tariff penalty enters the reward.
    #   "welfloss" (default) — multiplicative welfare-loss multiplier on utility:
    #       r_t = Δ(U_t × welfloss_t).  Requires manual α calibration.
    #   "additive_cbam" — additive penalty following RCPO (Tessler et al. 2019):
    #       r_t = ΔU_t − λ · CBAM_cost_t.  λ is a Lagrange multiplier
    #       that auto-tunes on a slower timescale.  CBAM_cost is the raw
    #       export-weighted tariff penalty (not divided by Y_r), making
    #       the signal size-invariant across regions.
    reward_mode: str = eqx.field(static=True, default="welfloss")

    # Initial (and post-reset) value of the Lagrange multiplier λ stored in
    # state["cbam_lambda"].  Default 0.0 preserves existing RCPO behaviour
    # (λ starts at 0 and is grown by RCPOMonitoredPPO).  Set to a positive
    # value when using a *fixed* calibrated penalty instead of auto-tuned RCPO
    # (e.g. cbam_lambda_init=1.0 for 9-region diversion experiments where the
    # EU trade share is too small for RCPO to accumulate λ across episode
    # boundaries).
    cbam_lambda_init: float = eqx.field(static=True, default=0.0)

    # Phase 2B Tier 1: fraction of CBAM revenue pool redistributed to exporters.
    # 0.0 (default) = no transfer (current behaviour); 1.0 = full redistribution.
    # Transfer to each non-EU exporter r is proportional to r's CBAM burden:
    #   transfer[r] = revenue_share * pool * cbam_cost[r] / (pool + ε)
    # Added to exporter consumption before utility is computed, so the subsidy
    # reduces the effective CBAM burden without changing the welfloss multiplier.
    # EU does not self-transfer; it retains (1 - revenue_share) implicitly.
    # Reference: Phase 2B design plan, Tier 1 ablation grid.
    revenue_share: float = eqx.field(static=True, default=0.0)

    # Controls how CBAM revenue transfers are applied to recipient exporters.
    #
    #   "consumption" (default, Mode A): transfer added directly to consumption
    #       as a lump-sum cash payment.  Net effective CBAM cost = (1-rs)·c_r.
    #       At rs=1 both the diversion and mitigation incentives are fully eroded
    #       — the Böhringer, Fischer & Rosendahl (2010 §4) perverse recycling result.
    #
    #   "abatement" (Mode B): transfer is earmarked to offset the abatement-cost
    #       deduction already applied inside calc_gross_outputs.  The subsidy is
    #       capped at the region's actual abatement spending; any excess reverts to
    #       free consumption (same as Mode A).  Gross output and investment are
    #       updated consistently before consumptions are recomputed.
    #
    #       Key asymmetry vs Mode A: diversion incentive is unchanged (full CBAM
    #       cost still penalises EU-bound dirty exports) but mitigation becomes
    #       cheaper (abatement cost partially covered), so Mode B preserves the
    #       mitigation channel even at rs=1 while still penalising diversion.
    #       Grounded in: Fischer & Springborn (2011) "Emissions Targets and the
    #       Real Business Cycle: Intensity Targets Versus Caps or Taxes",
    #       J. Environmental Economics and Management 62(3), §3–4 — earmarked
    #       green R&D/technology transfer lowers marginal abatement cost over
    #       time vs. lump-sum cash which protects households but does not
    #       trigger structural industrial transformation.  Chiroleu-Assouline &
    #       Fodha (2014) confirm conditionality on "intermediate outputs"
    #       (e.g., specific technology installation) outperforms simple
    #       results-based transfers for industrial sectors.
    transfer_mode: str = eqx.field(static=True, default="consumption")

    # Controls how the CBAM revenue pool is split across recipient exporters.
    # All rules zero-out the EU region after allocation.
    #
    #   "burden" (default): proportional to each exporter's raw CBAM cost c_r.
    #       Matches the EU CBAM Regulation intent; rewards staying dirty
    #       (Böhringer, Fischer & Rosendahl 2010 §4 perverse recycling).
    #
    #   "effort": proportional to each exporter's current mitigation rate μ_r.
    #       Directly rewards abatement effort — breaks the dirty-equilibrium
    #       trap for large low-μ exporters (China).
    #       Grounded in: Angelsen et al. (2017) "REDD+ as Result-based Aid:
    #       General Lessons and Bilateral Agreements of Norway" — performance-
    #       conditional payments shift firm optimisation from cost-recovery to
    #       innovation-incentive.  Fischer & Springborn (2011) §4: intensity-
    #       based rebating is preferred over output-based when the emissions
    #       price is below the social cost of carbon.  Nordhaus (2015) climate
    #       clubs AEA P&P — transfer conditionality is the "carrot" to the
    #       trade-sanction "stick" for expanding de facto carbon pricing.
    #       Risk: bilateral moral hazard (Chiroleu-Assouline & Fodha 2014) —
    #       recipient manipulates μ measurement baseline.
    #
    #   "equal": uniform split across all non-EU exporters (1/(NR-1)).
    #       Fully decouples transfer from behaviour; pure income effect.
    #       Useful as a null condition: if "equal" produces the same response
    #       as "burden", the incentive channel doesn't matter — only the amount.
    #
    #   "vulnerability": proportional to CBAM cost normalised by gross output
    #       (c_r / Y_r).  Favours small open economies with high CBAM exposure
    #       relative to their size (SSA, India over China/RoW).
    #       Grounded in: GCF/UNFCCC NCQG (2024) Multidimensional Vulnerability
    #       Index (MVI) — supplements GNI per capita with structural exposure
    #       (geographic isolation, fiscal fragility, natural disaster risk).
    #       CEEW India CBAM report (2024): c_r/Y_r correctly identifies MSMEs
    #       in iron/steel as most exposed relative to economic size.  Limitation:
    #       ADB (2024) CGE modelling shows the rule fails for large emitters
    #       (China) because it diverts transfers away from the "scale effect"
    #       of Chinese industrial abatement — confirmed by our simulation.
    #
    #   "hybrid": effort × vulnerability weight, w_r = μ_r × (c_r / Y_r).
    #       Combines performance conditionality (effort) with equity targeting
    #       (vulnerability).  Proposed by Gemini literature synthesis (2026) as
    #       the mechanism most consistent with both REDD+ RBA literature and
    #       NCQG equity criteria.  Empirically: should give SSA (high c_r/Y_r)
    #       a large pool *conditional* on raising μ, and give China (low c_r/Y_r)
    #       an effort incentive without the full equal-split windfall.
    #       [LITERATURE NEEDED: peer-reviewed hybrid effort×vulnerability rule
    #       for CBAM specifically — emerging as of 2026; closest anchor is
    #       Böhringer et al. (2010) §5 combined OBA+intensity rebate analysis]
    transfer_allocation: str = eqx.field(static=True, default="burden")

    # ── CBAM tariff mode ──────────────────────────────────────────────────────
    # Controls how the effective CBAM tariff rate is computed each step.
    #
    #   "flat" (default): scalar cbam_tariff_rate applied uniformly to all
    #       exporters.  Backward-compatible with all existing experiments.
    #       Calibration note: τ=0.80 (original) has no literature anchor.
    #       Literature-grounded range: τ ∈ {0.05, 0.10, 0.15, 0.25}, derived
    #       from Böhringer, Fischer & Rosendahl (2010) Table 2: effective ad-
    #       valorem equivalent of €65-100/tCO₂ EU ETS price on iron/steel
    #       (8-22%), cement (15-25%), and aluminium (10-18%).  Weighted median
    #       across CBAM sectors ≈ 0.15.  Comparable: Martin, de Preux & Wagner
    #       (2014) JIE §4 Table 3: 10-25% effective rate for UK CCL-covered
    #       sectors.  Recommended default for new experiments: 0.15.
    #
    #   "differential": per-region effective tariff τ_eff[r] =
    #       max(0, MAC_EU(μ_EU) − MAC_r(μ_r)) / MAC_EU(μ_EU)
    #       where MAC_r is the RICE marginal abatement cost of region r at its
    #       current mitigation rate.  Implements the actual CBAM mechanism
    #       (EU CBAM Reg. 2023/956 Art. 5–7): tariff is charged only on the
    #       carbon price differential between EU ETS and the exporter's
    #       implicit carbon price.  As exporter μ_r rises toward EU μ_EU,
    #       τ_eff → 0, restoring full EU market access.  This creates the
    #       self-incentivising mitigation channel that the "flat" mode lacks.
    #
    #       RICE MAC formula (Nordhaus 2017, DICE-2016R eq. 9):
    #         MAC_r(μ_r, t) = p_b_r · (1-δ_pb_r)^(t-1) · μ_r^(θ₂_r - 1)
    #       where all parameters are region-specific from xp_b, xdelta_pb,
    #       xtheta_2; and μ is the current mitigation rate from state.
    #       The MAC is dimensionless in RICE (fraction of Y per unit μ change
    #       per unit carbon intensity); normalised to [0,1] by EU's own MAC
    #       so τ_eff is always in [0, 1].
    #
    #       Canonical null test: when all regions have the same mitigation rate
    #       as EU, τ_eff[r] = 0 for all r → cbam_cost = 0.
    #
    #       Singularity guard: MAC is undefined at μ=0 when θ₂>1 (MACs → ∞).
    #       Clamp μ_safe = max(μ, mu_floor_differential) before computing MAC.
    cbam_tariff_mode: str = eqx.field(static=True, default="flat")

    # Floor mitigation rate for differential mode MAC computation.
    # Prevents divide-by-zero singularity when μ_r = 0 (MAC → ∞ for θ₂ > 1).
    # Default 0.01 corresponds to ≈1% abatement — consistent with Nordhaus
    # (2017) BAU scenario minimal mitigation (0-5% range in early periods).
    mu_floor_differential: float = eqx.field(static=True, default=0.01)

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
            state["cbam_cost_all_regions"] = jnp.zeros(
                self.num_regions, dtype=jnp.float32
            )
            # Lagrange multiplier for additive_cbam reward mode (RCPO).
            # Lives in state so the training loop can mutate it without
            # reconstructing the (frozen) equinox environment module.
            # Initialised to cbam_lambda_init (default 0.0); positive values
            # implement a fixed calibrated penalty that persists across episode
            # resets — preventing the λ-zeroing bug in RCPO on small EU-share
            # setups where per-episode resets prevent λ accumulation.
            state["cbam_lambda"] = jnp.float32(self.cbam_lambda_init)
            # Phase 2B: transfer received by each region this step.
            state["transfer_received"] = jnp.zeros(self.num_regions, dtype=jnp.float32)
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
                "cbam_cost": state["cbam_cost_all_regions"][agent_id],    # scalar — own CBAM cost
                "cbam_lambda": state["cbam_lambda"],                     # scalar — current Lagrange multiplier
                "revenue_share": jnp.float32(self.revenue_share),         # scalar — fraction redistributed
                "transfer_received": state["transfer_received"][agent_id], # scalar — transfer received this step
            }
        return obs

    # ------------------------------------------------------------------ rewards (RCPO override)

    def generate_rewards(
        self, new_state: dict, old_state: dict
    ) -> dict[str, float]:
        """Reward with optional additive CBAM penalty (RCPO).

        reward_mode="welfloss" (default):
            Delegates to parent: r_t = Δ(U × welfloss).

        reward_mode="additive_cbam":
            r_t = ΔU_t − λ · CBAM_cost_t
            where CBAM_cost is the raw export-weighted tariff penalty
            (not divided by Y_r) and λ is a Lagrange multiplier stored
            in state["cbam_lambda"], updated by the training loop.

        Reference: Tessler et al. (2019), "Reward Constrained Policy
        Optimization", ICLR 2019, §4.2 Eq. 10.
        """
        if self.reward_mode != "additive_cbam":
            return super().generate_rewards(new_state, old_state)

        # ΔU (pure utility change, no welfloss)
        reward = new_state["utility_all_regions"]
        if self.diff_reward_mode:
            reward = reward - old_state["utility_all_regions"]

        # Subtract λ · CBAM_cost
        lam = new_state["cbam_lambda"]
        cbam_cost = new_state["cbam_cost_all_regions"]  # (NR,)
        reward = reward - lam * cbam_cost

        return {i_to_agent_str(i): reward[i] for i in range(self.num_regions)}

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
        mitigation_rates: chex.Array | None = None,
        activity_timestep: chex.Array | None = None,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        """
        Compute CBAM-related quantities.

        Parameters
        ----------
        mitigation_rates : (NR,) optional
            Current-step mitigation rates from state["mitigation_rates_all_regions"].
            When provided, the effective embedded carbon intensity is scaled by
            (1 - μ_r), so mitigation reduces CBAM burden proportionally across
            all sectors of a region.  This follows EU CBAM Regulation 2023/956
            Art. 7: tariff is charged on *actual* embedded emissions, not on a
            frozen intensity baseline.
            When None (default), the static 2016 EORA intensity is used —
            backward-compatible with all existing experiments.
        activity_timestep : scalar, optional
            Required for cbam_tariff_mode="differential" to compute time-
            varying MAC via the RICE backstop price decay factor.

        Returns
        -------
        cbam_tariff_matrix : (NR, NR)
            Effective tariff rate per (importer, exporter) pair;
            non-zero only in EU row.
        cbam_revenue : (NR,)
            Revenue collected by EU from each exporter's goods.
        cbam_cost : (NR,)
            Raw CBAM cost borne by each exporting region.
        """
        # What each region r exports to EU, by sector
        eu_exports_by_sector = trade_flows[:, self.eu_region_idx, :]  # (NR, NS)

        # ── Tariff rate (scalar "flat" or per-region (NR,) "differential") ──
        if self.cbam_tariff_mode == "differential" and mitigation_rates is not None:
            # Per-region effective tariff rate: τ_eff[r] = max(0, MAC_EU - MAC_r) / MAC_EU
            # RICE MAC formula (Nordhaus 2017 DICE-2016R eq. 9):
            #   MAC_r(μ, t) = p_b_r · (1-δ_pb_r)^(t-1) · μ^(θ₂_r - 1)
            # All parameters are region-specific; MAC is dimensionless in RICE.
            # Normalised by EU's own MAC so τ_eff is always in [0, 1].
            p_b     = jnp.array(self.region_params.xp_b, dtype=jnp.float32)        # (NR,)
            delta_pb = jnp.array(self.region_params.xdelta_pb, dtype=jnp.float32)  # (NR,)
            theta2  = jnp.array(self.region_params.xtheta_2, dtype=jnp.float32)    # (NR,)
            t = activity_timestep if activity_timestep is not None else 1.0
            decay   = jnp.power(jnp.maximum(1.0 - delta_pb, 0.0), t - 1.0)       # (NR,)

            # Clamp μ to avoid singularity (MAC → ∞ when θ₂ > 1, μ → 0)
            mu_safe = jnp.maximum(mitigation_rates, self.mu_floor_differential)    # (NR,)
            mac     = p_b * decay * jnp.power(mu_safe, theta2 - 1.0)              # (NR,)

            mac_eu  = mac[self.eu_region_idx]                                       # scalar
            # τ_eff[r] = (MAC_EU - MAC_r) / MAC_EU, clamped to [0, 1]
            # EU's own rate is 0 by construction (MAC_EU - MAC_EU = 0)
            safe_eu = jnp.maximum(mac_eu, 1e-8)
            rate_per_region = jnp.clip((mac_eu - mac) / safe_eu, 0.0, 1.0)        # (NR,)
            rate_per_region = rate_per_region.at[self.eu_region_idx].set(0.0)
            # When cbam_randomize is active, cbam_tariff_rate from state acts as a
            # binary gate (0.0 = CBAM off, 1.0 = full differential).  Without
            # randomize the gate is always 1 (backward-compatible).
            scale = (
                jnp.clip(cbam_tariff_rate, 0.0, 1.0)
                if (self.cbam_randomize and cbam_tariff_rate is not None)
                else 1.0
            )
            # Broadcast to (NR, NS) for the cost calculation below
            rate = rate_per_region[:, None] * scale                                 # (NR, 1)
        else:
            # "flat" mode: scalar rate (backward-compatible)
            # Use state-based rate when provided, else fall back to static field
            flat = cbam_tariff_rate if cbam_tariff_rate is not None else self.cbam_tariff_rate
            rate = flat  # scalar — broadcasts to (NR, NS) naturally

        # Effective embedded carbon intensity per (region, sector).
        # Base: static 2016 EORA σ_{r,s} captures cross-sector heterogeneity.
        # Dynamic: scale by (1-μ_r) so that mitigation reduces CBAM burden.
        # Uniform μ across sectors is the correct Phase 2A/2B assumption —
        # sector-specific abatement requires sector-specific capital (Phase 2C).
        # Reference: EU CBAM Reg. 2023/956 Art. 7 — tariff on actual emissions.
        effective_intensity = jnp.array(self.emissions_intensity)  # (NR, NS)
        if mitigation_rates is not None and self.cbam_tariff_mode != "differential":
            # In differential mode, μ enters via MAC; don't double-count via intensity
            abatement_factor = jnp.clip(1.0 - mitigation_rates, 0.0, 1.0)  # (NR,)
            effective_intensity = effective_intensity * abatement_factor[:, None]

        # CBAM cost imposed on exporter r
        cbam_cost = (
            eu_exports_by_sector
            * effective_intensity
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

        return cbam_tariff_matrix, cbam_revenue, cbam_cost

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

        # Prescribed EU pathway: fix EU's mitigation to schedule value.
        # activity_timestep is 0-based here (parent increments inside super()).
        if self.eu_mitigation_schedule is not None:
            schedule = jnp.array(self.eu_mitigation_schedule, dtype=jnp.float32)
            t_idx = jnp.clip(
                jnp.int32(state["activity_timestep"]), 0, schedule.shape[0] - 1
            )
            parent_actions["mitigation_rate"] = (
                parent_actions["mitigation_rate"]
                .at[self.eu_region_idx].set(schedule[t_idx])
            )

        # 2. Run parent with zeroed trade → correct climate/production/investment;
        #    consumption/utilities will be re-computed below.
        state = super().step_climate_and_economy(state, parent_actions)

        # Zero out abatement costs if requested (Phase 2B motivating experiment).
        # The parent already deducted abatement_cost from gross_output inside
        # calc_gross_outputs, so we compensate by scaling gross_output back up.
        # Specifically: gross_output = damages * (1 - abatement_cost) * production
        # → with abatement_cost=0: gross_output = damages * production.
        if self.zero_abatement_cost:
            abatement_cost = state["abatement_cost_all_regions"]  # (NR,)
            denom = jnp.maximum(1.0 - abatement_cost, 1e-8)
            # Reverse the (1 - abatement_cost) factor the parent applied to both
            # gross_output and investment (investment = savings_rate * gross_output,
            # so it was also computed from the penalised output).
            state = state.copy()
            state["gross_output_all_regions"]  = state["gross_output_all_regions"] / denom
            state["investment_all_regions"]    = state["investment_all_regions"] / denom
            state["abatement_cost_all_regions"] = jnp.zeros_like(abatement_cost)

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

        # 5. Compute CBAM: effective tariff matrix + revenue + raw cost.
        # Pass current mitigation rates so that μ > 0 reduces embedded emissions
        # and therefore the CBAM burden (EU CBAM Reg. 2023/956 Art. 7).
        # mitigation_rates_all_regions was updated by the parent step above.
        active_rate = state["cbam_tariff_rate"]
        mit_rates = (
            None if self.no_mitigation
            else state["mitigation_rates_all_regions"]
        )
        cbam_tariff_matrix, cbam_revenue, cbam_cost_raw = self._compute_cbam(
            trade_flows, gross_imports_mrio,
            cbam_tariff_rate=active_rate,
            mitigation_rates=mit_rates,
            activity_timestep=state["activity_timestep"],
        )

        # 6. Recompute consumptions with MRIO gross imports (no tariff on quantity)
        gross_outputs = state["gross_output_all_regions"]
        investments = state["investment_all_regions"]
        consumptions = self.calc_consumptions(
            gross_outputs, investments, gross_imports_mrio, gross_imports_mrio
        )

        # 6b. Phase 2B: revenue transfer to exporters.
        # Redistribute `revenue_share` fraction of the total CBAM pool back to
        # exporters.  EU is the collector and does not self-transfer.
        # Allocation rule is controlled by self.transfer_allocation.
        pool = cbam_cost_raw.sum()  # scalar — total revenue collected by EU

        # ── Allocation rule ────────────────────────────────────────────────
        if self.transfer_allocation == "effort":
            # Proportional to current mitigation rate μ_r.
            # Directly rewards abatement effort; breaks dirty-equilibrium trap
            # for large low-μ exporters.  EU share zeroed after normalisation.
            mu = state["mitigation_rates_all_regions"]                 # (NR,)
            mu_ex = mu.at[self.eu_region_idx].set(0.0)
            burden_share = mu_ex / (mu_ex.sum() + 1e-8)
        elif self.transfer_allocation == "equal":
            # Uniform split across all non-EU exporters.
            # Pure income effect; decouples transfer amount from behaviour.
            mask = jnp.ones(self.num_regions).at[self.eu_region_idx].set(0.0)
            burden_share = mask / (mask.sum() + 1e-8)
        elif self.transfer_allocation == "vulnerability":
            # Proportional to CBAM cost / gross output (c_r / Y_r).
            # Favours small open economies with high CBAM exposure relative
            # to their economic size (SSA, India over China).
            # Grounded in GCF/NCQG MVI criteria (UNFCCC 2024).
            vul = cbam_cost_raw / (gross_outputs + 1e-8)               # (NR,)
            vul = vul.at[self.eu_region_idx].set(0.0)
            burden_share = vul / (vul.sum() + 1e-8)
        elif self.transfer_allocation == "hybrid":
            # w_r = μ_r × (c_r / Y_r) — effort × vulnerability.
            # Performance-conditional (effort) combined with equity targeting
            # (vulnerability/exposure).  Proposed in Gemini literature synthesis
            # (2026); closest anchor is Böhringer et al. (2010) §5 combined
            # OBA+intensity rebate.  EU share zeroed after normalisation.
            mu  = state["mitigation_rates_all_regions"]                # (NR,)
            vul = cbam_cost_raw / (gross_outputs + 1e-8)               # (NR,)
            w   = mu * vul
            w   = w.at[self.eu_region_idx].set(0.0)
            burden_share = w / (w.sum() + 1e-8)
        else:
            # "burden" (default): proportional to raw CBAM cost c_r.
            # Böhringer, Fischer & Rosendahl (2010) §4: mimics OBA for foreign
            # producers, offsetting trade barrier but risking moral hazard.
            burden_share = cbam_cost_raw / (pool + 1e-8)               # (NR,)

        transfer_received = self.revenue_share * pool * burden_share   # (NR,)
        transfer_received = transfer_received.at[self.eu_region_idx].set(0.0)

        if self.transfer_mode == "abatement":
            # Mode B — earmarked abatement subsidy, no consumption spillover.
            # The transfer reduces the effective abatement-cost deduction that
            # the parent's calc_gross_outputs already applied:
            #   gross_output = damage * (1 - abatement_cost) * production
            # We reverse part of that (1-abatement_cost) factor, capped at
            # actual abatement spending.  Any excess is forfeited — it does NOT
            # enter the exporter's consumption.  This ensures the only channel
            # through which the transfer raises welfare is by making mitigation
            # cheaper, not by providing a general income transfer.
            # Key property: diversion incentive (full CBAM cost) is unchanged;
            # mitigation cost is reduced → pure asymmetric incentive structure.
            abatement_cost_frac = state["abatement_cost_all_regions"]  # (NR,)
            denom = jnp.maximum(1.0 - abatement_cost_frac, 1e-8)
            gross_output_pre_abatement = gross_outputs / denom        # (NR,)
            abatement_spending = abatement_cost_frac * gross_output_pre_abatement  # (NR,)
            subsidy = jnp.minimum(transfer_received, abatement_spending)  # (NR,)
            # Implied savings rate (investment / gross_output) keeps capital
            # accumulation consistent with the updated gross_output.
            savings_rate_implied = investments / jnp.maximum(gross_outputs, 1e-8)  # (NR,)
            gross_outputs_new = gross_outputs + subsidy
            investments_new = investments + savings_rate_implied * subsidy
            consumptions = self.calc_consumptions(
                gross_outputs_new, investments_new,
                gross_imports_mrio, gross_imports_mrio,
            )
        else:
            # Mode A — free consumption (default).
            # Full transfer added to consumption as a lump-sum cash payment.
            # Net effective CBAM cost = (1 - revenue_share) * cbam_cost[r].
            consumptions = consumptions + transfer_received

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

        # 9. Apply CBAM penalty according to reward_mode.
        #    "welfloss":       utility_times_welfloss = U × welfloss  (status quo)
        #    "additive_cbam":  utility_times_welfloss = U  (penalty applied in
        #                      generate_rewards via λ · cbam_cost_raw)
        if self.reward_mode == "additive_cbam":
            utility_times_welfloss = utilities  # no multiplicative penalty
        else:
            utility_times_welfloss = utilities * welfloss

        # 10. Update state
        state = state.copy()
        state.update(
            {
                "production_by_sector": production_by_sector,
                "trade_flows": trade_flows,
                "cbam_revenue": cbam_revenue,
                "cbam_cost_all_regions": cbam_cost_raw,
                "transfer_received": transfer_received,
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

