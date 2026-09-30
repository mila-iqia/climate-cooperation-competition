"""validate_trade_friction.py
Phase C-Friction validation: friction-off vs friction-on comparison.

Uses fixed agents (no training) — midpoint actions for the null condition
and asymmetric diversion actions to exercise the friction cost mechanism.

Four conditions
---------------
  A  no_friction | midpoint   — canonical null baseline (friction_cost == 0 everywhere)
  B  no_friction | diversion  — free diversion, no economic penalty
  C  friction_on | midpoint   — confirms null condition holds even with matrix loaded
  D  friction_on | diversion  — friction active, measures GDP / utility impact

Figures (saved to plots/)
--------------------------
  Fig 1  GDP impact panel (NR sub-plots)
          • B vs D — gross_output over episode; D should be ≤ B
          • inset bar: cumulative GDP loss D – B
  Fig 2  Friction cost panel (NR sub-plots)
          • trade_friction_cost for all 4 conditions
          • A and C should be identically zero (canonical null)
  Fig 3  Allocation deviation panel (NR sub-plots)
          • mean absolute deviation |dest_alloc – baseline| summed over (s,j)
          • midpoint conditions → 0; diversion → positive
  Fig 4  EU export share panel (NR sub-plots)
          • fraction of each region's total exports that go to EU
          • both diversion conditions; friction may cause secondary shift
  Fig 5  Summary bar chart
          • per-region mean friction_cost / gross_output (%), condition D only

Usage (from rice_jax/ directory):
    conda run -n rice-jax python validation/validate_trade_friction.py

Output: validation/plots/trade_friction_validation_<TIMESTAMP>.png (Figs 1-5 on one canvas)
"""

import matplotlib
matplotlib.use("Agg")                        # must precede any JAX import

import os
import sys
import copy
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from rice_jax import RiceMRIO
from rice_jax.utils import load_region_yamls, i_to_agent_str

# ── Configuration ─────────────────────────────────────────────────────────────

NUM_REGIONS   = 7
EU_REGION_IDX = 5           # Europe & Central Asia in 7-region; NEVER 0
SEED          = 42
OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
DPI           = 150

# Iceberg cost for synthetic matrix when no calibrated .npy exists.
# 2.0 means re-routing exports to a new destination costs an amount equal
# to the export volume × deviation × τ.  This is intentionally large so the
# friction effect is clearly visible in the plots.
TAU_SYNTHETIC = 2.0

MRIO_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "csv_asset",
)

# Base env kwargs (shared across all conditions)
_BASE_KWARGS = dict(
    num_regions         = NUM_REGIONS,
    mrio_data_root      = MRIO_DATA_ROOT,
    mrio_trade          = True,
    cbam_tariff_rate    = 0.0,          # CBAM off — isolates friction effect
    cbam_randomize      = False,
    eu_region_idx       = EU_REGION_IDX,
    dest_alloc_persistence    = 0.0,    # anchor always = 2016 baseline; cleaner null
    dest_alloc_baseline_decay = 0.0,
    diff_reward_mode    = True,
    num_discrete_action_levels = 10,
    sectoral_welfloss   = False,
    fixed_savings_rate  = True,         # remove savings degree of freedom
    no_mitigation       = True,         # remove mitigation degree of freedom
    sector_granularity  = "emissions-simple",
    welfare_loss_per_unit_tariff = 5.0,
    use_trade_friction  = False,        # overridden below for friction-on envs
)

# ── Region labels for plots ────────────────────────────────────────────────────

_REGION_LABELS_7 = [
    "Sub-Sah. Africa",
    "South Asia",
    "North America",
    "MENA",
    "Latin Am. & Carib.",
    "EU & Ctrl Asia",
    "East Asia & Pac.",
]

# ── Build environments ─────────────────────────────────────────────────────────

def _load_friction_matrix(env_base: RiceMRIO) -> np.ndarray:
    """Load calibrated friction matrix if available, else use synthetic uniform τ."""
    NR = env_base.num_regions
    NS = env_base.num_sectors
    granularity = env_base.sector_granularity

    calibrated = os.path.join(
        MRIO_DATA_ROOT,
        f"trade_friction_{NR}r_{granularity}.npy",
    )
    if os.path.isfile(calibrated):
        tau = np.load(calibrated).astype(np.float32)
        print(f"[friction] Loaded calibrated matrix from {os.path.basename(calibrated)}: "
              f"mean τ={tau.mean():.3f}, max τ={tau.max():.3f}")
    else:
        tau = np.full((NR, NR, NS), TAU_SYNTHETIC, dtype=np.float32)
        for r in range(NR):
            tau[r, r, :] = 1.0
        print(f"[friction] No calibrated file found — using synthetic τ={TAU_SYNTHETIC} "
              f"(off-diagonal), 1.0 (diagonal).")
    return tau


def build_envs(region_params):
    """Return (env_no_friction, env_friction) pair."""
    env_no_friction = RiceMRIO(region_params=region_params, **_BASE_KWARGS)

    tau = _load_friction_matrix(env_no_friction)

    # Build friction env: construct without file loading, then inject matrix.
    env_friction = copy.copy(env_no_friction)
    object.__setattr__(env_friction, "use_trade_friction", True)
    object.__setattr__(env_friction, "trade_friction_matrix", tau)

    return env_no_friction, env_friction


# ── Fixed-action agents ────────────────────────────────────────────────────────

def _midpoint_action(env: RiceMRIO) -> dict:
    """All discrete actions at D//2 → δ=0 for every (sector, destination) slot.

    Uses env.sample_action to pick up the correct action structure (respects
    fixed_savings_rate / no_mitigation flags), then overwrites every value with
    the midpoint level D//2.
    """
    D   = env.num_discrete_action_levels
    mid = D // 2
    base = env.sample_action(jax.random.PRNGKey(0))
    return {
        agent: {k: jnp.full_like(v, mid) for k, v in agent_acts.items()}
        for agent, agent_acts in base.items()
    }


def _diversion_action(env: RiceMRIO) -> dict:
    """Asymmetric export_reallocation: alternating (D-1) and 0 across destination dims.

    A uniform action cancels inside softmax (constant shift) → zero deviation.
    Alternating high/low creates differential logits → softmax redistributes →
    non-trivial deviation from baseline → non-zero friction cost.

    All other actions (savings_rate / mitigation_rate if present) stay at D//2.
    """
    D  = env.num_discrete_action_levels
    NR = env.num_regions
    NS = env.num_sectors
    n  = NS * NR
    alt = jnp.array(
        [(D - 1) if i % 2 == 0 else 0 for i in range(n)], dtype=jnp.int32
    )
    mid = D // 2
    base = env.sample_action(jax.random.PRNGKey(0))
    result = {}
    for agent, agent_acts in base.items():
        result[agent] = {
            k: (alt if k == "export_reallocation" else jnp.full_like(v, mid))
            for k, v in agent_acts.items()
        }
    return result


# ── Rollout ────────────────────────────────────────────────────────────────────

def _rollout(env: RiceMRIO, action_dict: dict, seed: int = SEED) -> dict[str, np.ndarray]:
    """
    Run one full episode using a fixed action.

    Returns a dict of time-series arrays, each of shape (T, ...) where T is
    the episode length.  Values are extracted directly from the state dict
    (no LogWrapper needed) and converted to float32 numpy.
    """
    key = jax.random.PRNGKey(seed)
    obs, state = env.reset(key)
    T = int(env.episode_length)

    processed = env.process_actions(action_dict, state)

    records: dict[str, list] = {
        "gross_output":       [],   # (NR,) per step
        "trade_friction_cost": [],  # (NR,) per step
        "cbam_revenue":       [],   # (NR,) per step
        "utility":            [],   # (NR,) per step
        "dest_alloc_dev":     [],   # (NR,) mean |alloc - baseline| per step
        "eu_export_share":    [],   # (NR,) per step
    }

    for _ in range(T):
        state = env.step_climate_and_economy(state, processed)

        Y    = np.array(state["gross_output_all_regions"],   dtype=np.float32)  # (NR,)
        fc   = np.array(state["trade_friction_cost"],        dtype=np.float32)  # (NR,)
        rev  = np.array(state["cbam_revenue"],               dtype=np.float32)  # (NR,)
        util = np.array(state["utility_times_welfloss_all_regions"],
                        dtype=np.float32)                                         # (NR,)

        # Allocation deviation: mean absolute deviation from 2016 baseline
        # dest_alloc_current (NR, NS, NR); baseline (NR, NS, NR)
        alloc   = np.array(state["dest_alloc_current"],  dtype=np.float32)  # (NR,NS,NR)
        baseline = np.array(env.dest_alloc_baseline,     dtype=np.float32)  # (NR,NS,NR)
        dev = np.abs(alloc - baseline).mean(axis=(1, 2))  # (NR,) — mean over (s,j)

        # EU export share: trade_flows[r, EU, :].sum() / trade_flows[r, :, :].sum()
        tf = np.array(state["trade_flows"], dtype=np.float32)  # (NR,NR,NS)
        eu_out   = tf[:, EU_REGION_IDX, :].sum(axis=1)  # (NR,)
        total_out = tf.sum(axis=(1, 2)) + 1e-10           # (NR,)
        eu_share  = eu_out / total_out                     # (NR,)

        records["gross_output"].append(Y)
        records["trade_friction_cost"].append(fc)
        records["cbam_revenue"].append(rev)
        records["utility"].append(util)
        records["dest_alloc_dev"].append(dev)
        records["eu_export_share"].append(eu_share)

    return {k: np.stack(v, axis=0) for k, v in records.items()}  # (T, NR) each


# ── Plotting ───────────────────────────────────────────────────────────────────

_CONDITION_LABELS = {
    "A": "No friction | midpoint (null)",
    "B": "No friction | diversion",
    "C": "Friction on | midpoint (null)",
    "D": "Friction on | diversion",
}
_COLORS = {"A": "#888888", "B": "#2b7cbb", "C": "#e0ae1e", "D": "#e05c2a"}
_LS     = {"A": "--",      "B": "-",       "C": "--",      "D": "-"}
_LW     = {"A": 1.0,       "B": 1.8,       "C": 1.0,       "D": 1.8}


def _region_labels(n: int) -> list[str]:
    if n == len(_REGION_LABELS_7):
        return _REGION_LABELS_7
    return [f"Region {i}" for i in range(n)]


def make_figure(
    data: dict[str, dict[str, np.ndarray]],
    NR: int,
    tau_label: str,
) -> plt.Figure:
    """
    Create the 5-section validation figure.

    Parameters
    ----------
    data : {"A": rollout_dict, "B": ..., "C": ..., "D": ...}
    NR   : number of regions
    tau_label : description of the friction matrix used
    """
    region_names = _region_labels(NR)
    T = data["A"]["gross_output"].shape[0]
    steps = np.arange(1, T + 1)

    # Identify non-EU regions for plotting (EU row still shown, just lighter)
    non_eu = [r for r in range(NR) if r != EU_REGION_IDX]

    ncols = NR
    nrows = 6   # gross_output, friction_cost, deviation, eu_share, utility, summary
    fig = plt.figure(figsize=(3 * ncols, 3.0 * nrows), constrained_layout=True)
    fig.suptitle(
        f"Phase C-Friction Validation — friction-off vs friction-on\n"
        f"Fixed agents (no training) · {NR} regions · {tau_label}",
        fontsize=11, fontweight="bold",
    )
    outer = gridspec.GridSpec(nrows, ncols, figure=fig)

    # ── Row labels ────────────────────────────────────────────────────────────
    row_titles = [
        "Gross output  (no-CBAM, fixed savings)",
        "Trade friction cost",
        "Mean dest-alloc deviation from 2016 baseline",
        "EU export share",
        "Utility × welfare-loss multiplier",
        "Summary: mean friction cost as % GDP  (condition D)",
    ]

    # ── Rows 0-4: per-region timeseries ───────────────────────────────────────
    metrics = [
        "gross_output",
        "trade_friction_cost",
        "dest_alloc_dev",
        "eu_export_share",
        "utility",
    ]
    ylabels = [
        "Gross output\n($T USD / yr)",
        "Friction cost\n($T USD / yr)",
        "Mean |Δ alloc|",
        "EU export share",
        "Utility × welfloss",
    ]

    for row, (metric, ylabel) in enumerate(zip(metrics, ylabels)):
        for col in range(ncols):
            r = col     # region index = column
            ax = fig.add_subplot(outer[row, col])

            for cond in ["A", "B", "C", "D"]:
                y = data[cond][metric][:, r]
                ax.plot(
                    steps, y,
                    color=_COLORS[cond], ls=_LS[cond], lw=_LW[cond],
                    label=_CONDITION_LABELS[cond] if col == 0 else None,
                    alpha=0.85,
                )

            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)

            # Annotate null condition on friction_cost plot
            if metric == "trade_friction_cost":
                a_max = data["A"]["trade_friction_cost"][:, r].max()
                c_max = data["C"]["trade_friction_cost"][:, r].max()
                if a_max < 1e-7 and c_max < 1e-7:
                    ax.text(
                        0.98, 0.95, "null ✓",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=6, color="green",
                    )

            if row == 0:
                ax.set_title(region_names[r], fontsize=8, fontweight="bold",
                             color="#c0392b" if r == EU_REGION_IDX else "black")

            if row == len(metrics) - 1:
                ax.set_xlabel("Episode step", fontsize=6)

    # ── Row 5: summary bar chart ───────────────────────────────────────────────
    mean_friction_pct = (
        data["D"]["trade_friction_cost"].mean(axis=0)
        / (data["D"]["gross_output"].mean(axis=0) + 1e-10)
        * 100.0
    )   # (NR,)
    mean_gdp_loss_pct = (
        (data["B"]["gross_output"] - data["D"]["gross_output"]).mean(axis=0)
        / (data["B"]["gross_output"].mean(axis=0) + 1e-10)
        * 100.0
    )   # (NR,)

    ax_sum = fig.add_subplot(outer[5, :])   # full-width summary
    x = np.arange(NR)
    w = 0.35
    bars_fc = ax_sum.bar(x - w/2, mean_friction_pct, width=w,
                         color=_COLORS["D"], label="Mean friction cost / GDP (%)", alpha=0.8)
    bars_gl = ax_sum.bar(x + w/2, mean_gdp_loss_pct, width=w,
                         color=_COLORS["B"], label="Mean GDP loss friction-on vs off (%)",
                         alpha=0.8)
    ax_sum.axhline(0, color="black", lw=0.5)
    ax_sum.set_xticks(x)
    ax_sum.set_xticklabels(region_names, rotation=20, ha="right", fontsize=7)
    ax_sum.set_ylabel("% of GDP", fontsize=8)
    ax_sum.set_title(row_titles[5], fontsize=8)
    ax_sum.legend(fontsize=7, loc="upper right")
    ax_sum.tick_params(labelsize=7)

    # Add value annotations on bars
    for bar in bars_fc:
        h = bar.get_height()
        if abs(h) > 0.001:
            ax_sum.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                        f"{h:.2f}%", ha="center", va="bottom", fontsize=5)

    # ── Global legend ─────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color=_COLORS[c], ls=_LS[c], lw=_LW[c],
                   label=_CONDITION_LABELS[c])
        for c in ["A", "B", "C", "D"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    return fig


def make_null_verification_figure(
    data: dict[str, dict[str, np.ndarray]],
    NR: int,
) -> plt.Figure:
    """
    One-panel diagnostic confirming canonical null condition:
    trade_friction_cost == 0 at all timesteps under midpoint actions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle(
        "Canonical null condition: midpoint actions → deviation = 0 → friction cost = 0",
        fontsize=10, fontweight="bold",
    )
    steps = np.arange(1, data["A"]["trade_friction_cost"].shape[0] + 1)

    for ax, cond, title in zip(
        axes,
        ["A", "C"],
        ["Condition A  (no_friction | midpoint)", "Condition C  (friction_on | midpoint)"],
    ):
        max_cost = data[cond]["trade_friction_cost"].max(axis=1)  # (T,) max over regions
        ax.plot(steps, max_cost, color="black", lw=1.5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Episode step", fontsize=8)
        ax.set_ylabel("max(friction_cost) over regions", fontsize=8)
        ax.tick_params(labelsize=7)
        # Reference zero line
        ax.axhline(0, color="red", ls="--", lw=0.8, label="expected = 0")
        ax.legend(fontsize=7)
        # Annotate pass / fail
        threshold = 1e-5
        passed = float(max_cost.max()) < threshold
        label = f"{'PASS ✓' if passed else 'FAIL ✗'}  max={max_cost.max():.2e}"
        ax.text(0.98, 0.95, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="green" if passed else "red", fontweight="bold")

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("[validate_trade_friction] Loading region params …")
    region_params = load_region_yamls(NUM_REGIONS)

    print("[validate_trade_friction] Building environments …")
    env_no, env_fr = build_envs(region_params)

    print(f"  env_no_friction: NR={env_no.num_regions}, NS={env_no.num_sectors}, "
          f"use_trade_friction={env_no.use_trade_friction}")
    print(f"  env_friction:    NR={env_fr.num_regions}, NS={env_fr.num_sectors}, "
          f"use_trade_friction={env_fr.use_trade_friction}, "
          f"tau_mean={env_fr.trade_friction_matrix.mean():.3f}")

    act_mid = _midpoint_action(env_no)
    act_div = _diversion_action(env_no)

    # Determine tau label for figure title
    calibrated_path = os.path.join(
        MRIO_DATA_ROOT,
        f"trade_friction_{NUM_REGIONS}r_{env_no.sector_granularity}.npy",
    )
    tau_label = (
        f"calibrated τ from {os.path.basename(calibrated_path)}"
        if os.path.isfile(calibrated_path)
        else f"synthetic τ={TAU_SYNTHETIC} (calibration not yet run)"
    )

    print("[validate_trade_friction] Running rollouts …")
    print("  A: no_friction | midpoint  …", end=" ", flush=True)
    data_A = _rollout(env_no, act_mid)
    print(f"done  (friction_cost_max={data_A['trade_friction_cost'].max():.2e})")

    print("  B: no_friction | diversion …", end=" ", flush=True)
    data_B = _rollout(env_no, act_div)
    print(f"done  (friction_cost_max={data_B['trade_friction_cost'].max():.2e})")

    print("  C: friction_on | midpoint  …", end=" ", flush=True)
    data_C = _rollout(env_fr, act_mid)
    print(f"done  (friction_cost_max={data_C['trade_friction_cost'].max():.2e})")

    print("  D: friction_on | diversion …", end=" ", flush=True)
    data_D = _rollout(env_fr, act_div)
    print(f"done  (friction_cost_max={data_D['trade_friction_cost'].max():.2e})")

    data = {"A": data_A, "B": data_B, "C": data_C, "D": data_D}

    # ── Print text summary ────────────────────────────────────────────────────
    region_names = _region_labels(NUM_REGIONS)
    print("\n── Friction summary (condition D — friction_on | diversion) ──")
    print(f"{'Region':<24}  {'mean friction cost':>18}  {'mean GDP loss %':>15}  {'null A OK':>9}  {'null C OK':>9}")
    for r in range(NUM_REGIONS):
        mean_fc = data_D["trade_friction_cost"][:, r].mean()
        mean_Y_nf = data_B["gross_output"][:, r].mean()
        mean_Y_fr = data_D["gross_output"][:, r].mean()
        gdp_loss_pct = (mean_Y_nf - mean_Y_fr) / (mean_Y_nf + 1e-10) * 100
        null_A = data_A["trade_friction_cost"][:, r].max() < 1e-5
        null_C = data_C["trade_friction_cost"][:, r].max() < 1e-5
        print(
            f"  {region_names[r]:<22}  {mean_fc:>18.4e}  {gdp_loss_pct:>14.3f}%"
            f"  {'✓' if null_A else '✗':>9}  {'✓' if null_C else '✗':>9}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n[validate_trade_friction] Generating main figure …")
    fig_main = make_figure(data, NUM_REGIONS, tau_label)
    path_main = os.path.join(OUTPUT_DIR, f"trade_friction_validation_{timestamp}.png")
    fig_main.savefig(path_main, dpi=DPI, bbox_inches="tight")
    plt.close(fig_main)
    print(f"  Saved → {path_main}")

    print("[validate_trade_friction] Generating null-verification figure …")
    fig_null = make_null_verification_figure(data, NUM_REGIONS)
    path_null = os.path.join(OUTPUT_DIR, f"trade_friction_null_{timestamp}.png")
    fig_null.savefig(path_null, dpi=DPI, bbox_inches="tight")
    plt.close(fig_null)
    print(f"  Saved → {path_null}")

    print("\n[validate_trade_friction] Done.")


if __name__ == "__main__":
    main()
