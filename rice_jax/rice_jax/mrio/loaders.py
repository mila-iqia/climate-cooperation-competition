"""MRIO data loaders and sector aggregation helpers."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

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


