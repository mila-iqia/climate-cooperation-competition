"""plot_transition_cost.py

Explanatory figure for the Grubb (1995) transition-cost mechanism.

Compares four environments on two preset μ trajectories:

  baseline   AW=0, TC=0     — reference (no constraint)
  tc_soft    AW=0, TC=10    — soft economic penalty (Grubb 1995 Eq. 2)
  tc_hard    AW=0, TC=50    — stiff economic penalty
  aw2        AW=2, TC=0     — mechanical action-window block (mask only)

Figure layout (2 rows × 3 cols + 1 summary bar row):
  Row 0  JUMP   μ over time | normalised GO | per-step TC fraction
  Row 1  GRADUAL μ over time | normalised GO | per-step TC fraction
  Row 2  Cumulative GO loss  |  AW mask schematic

Run (from repo root):
  conda run -n rice-jax python rice_jax/validation/plot_transition_cost.py

Output: rice_jax/validation/plots/transition_cost_<timestamp>.png

Literature:
  Grubb, Chapuis & Ha Duong (1995) Energy Policy 23(4-5):417-432, §3, Eq. 2
  Grubb, Wieners & Yang (2021) WIREs Climate Change 12:e698, Eq. 2
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import textwrap
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls

# ── Config ────────────────────────────────────────────────────────────────

KEY      = jax.random.PRNGKey(0)
NR       = 7
NS       = 2
EU_IDX   = 5
N_STEPS  = 10
SAVINGS  = 0.22
DT       = 5.0   # xDelta years

_THIS     = _os.path.dirname(_os.path.abspath(__file__))
MRIO_ROOT = _os.path.abspath(_os.path.join(_THIS, "..", "..", "csv_asset"))
OUT_DIR   = _os.path.join(_THIS, "plots")

ENV_STYLES = {
    "baseline": dict(color="#444444", ls="--",  lw=1.4, label="Baseline (AW=0, TC=0)"),
    "tc_soft":  dict(color="#1f77b4", ls="-",   lw=2.2, label="TC-soft  (AW=0, c_B=10)"),
    "tc_hard":  dict(color="#d62728", ls="-",   lw=2.2, label="TC-hard  (AW=0, c_B=50)"),
    "aw2":      dict(color="#ff7f0e", ls=(0,(5,3)), lw=1.8, label="AW=2     (mask, c_B=0)"),
}

TRAJ_LABELS = {
    "jump":    "JUMP  (0 → 80% in 1 step)",
    "gradual": "GRADUAL  (+10 %pt/step)",
}


# ── Environment factory ───────────────────────────────────────────────────

def _make_env(*, action_window_size: int = 0, transition_cost_coef: float = 0.0
              ) -> RiceMRIO:
    region_params = load_region_yamls(NR)
    return RiceMRIO(
        num_regions                = NR,
        region_params              = region_params,
        mrio_data_root             = MRIO_ROOT,
        mrio_trade                 = True,
        eu_region_idx              = EU_IDX,
        cbam_tariff_rate           = 0.0,
        cbam_randomize             = False,
        dest_alloc_persistence     = 0.55,
        dest_alloc_baseline_decay  = 1.0,
        diff_reward_mode           = True,
        num_discrete_action_levels = 10,
        sectoral_welfloss          = True,
        sector_granularity         = "emissions-simple",
        welfare_loss_per_unit_tariff = 5.0,
        action_window_size         = action_window_size,
        transition_cost_coef       = transition_cost_coef,
    )


def _make_actions(mu: float | np.ndarray) -> dict:
    mu_arr = (jnp.full((NR,), float(mu)) if np.isscalar(mu)
              else jnp.array(mu, dtype=jnp.float32))
    return {
        "mitigation_rate"   : mu_arr,
        "savings_rate"      : jnp.full((NR,), SAVINGS),
        "export_reallocation": jnp.zeros((NR, NS * NR)),
        "export_limit"      : jnp.zeros((NR,)),
        "import_bid"        : jnp.zeros((NR, NR)),
        "import_tariff"     : jnp.zeros((NR, NR)),
    }


def _rollout(env: RiceMRIO, mu_sched: list[float]) -> dict[str, np.ndarray]:
    _, state = env.reset(KEY)
    rows = {k: [] for k in ["mu", "gross_output", "abatement_cost"]}
    for mu in mu_sched[:N_STEPS]:
        state = env.step_climate_and_economy(state, _make_actions(mu))
        rows["mu"].append(float(np.array(state["mitigation_rates_all_regions"]).mean()))
        rows["gross_output"].append(float(np.array(state["gross_output_all_regions"]).sum()))
        rows["abatement_cost"].append(float(np.array(state["abatement_cost_all_regions"]).mean()))
    return {k: np.array(v) for k, v in rows.items()}


def _analytic_tc(mu_sched: list[float], coef: float) -> np.ndarray:
    """Analytic TC fraction per step for a uniform-μ schedule."""
    mus  = np.array(mu_sched[:N_STEPS])
    prev = np.concatenate([[0.0], mus[:-1]])
    return coef * ((mus - prev) / DT) ** 2


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building environments …")
    envs = {
        "baseline": _make_env(action_window_size=0, transition_cost_coef=0.0),
        "tc_soft":  _make_env(action_window_size=0, transition_cost_coef=10.0),
        "tc_hard":  _make_env(action_window_size=0, transition_cost_coef=50.0),
        "aw2":      _make_env(action_window_size=2, transition_cost_coef=0.0),
    }

    mu_jump    = [0.80] + [0.80] * (N_STEPS - 1)
    mu_gradual = [min(0.10 * (t + 1), 0.80) for t in range(N_STEPS)]
    trajectories = {"jump": mu_jump, "gradual": mu_gradual}

    print("Running rollouts …")
    results: dict[str, dict[str, dict]] = {}
    for traj, sched in trajectories.items():
        results[traj] = {name: _rollout(env, sched) for name, env in envs.items()}
    print("  OK — building figure …")

    steps = np.arange(N_STEPS)
    years = 2020 + steps * 5

    # ── Figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 11))
    fig.patch.set_facecolor("#fafafa")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[3, 3, 2.2],
        hspace=0.55,
        top=0.86, bottom=0.05,
    )
    row0_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.35)
    row1_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.35)
    bot_gs  = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2], wspace=0.45)

    axes_top    = [fig.add_subplot(row0_gs[j]) for j in range(3)]
    axes_bot    = [fig.add_subplot(row1_gs[j]) for j in range(3)]
    ax_bar      = fig.add_subplot(bot_gs[0])
    ax_mask     = fig.add_subplot(bot_gs[1])

    fig.suptitle(
        "Grubb (1995) Transition-Cost mechanism vs Action-Window constraint\n"
        "Deterministic rollouts — 7-region RICE-MRIO, no CBAM, preset μ schedules",
        fontsize=12, fontweight="bold", y=0.995,
    )

    def _annotate_row(ax0, traj_name):
        """Add trajectory label to leftmost axis."""
        ax0.text(
            -0.22, 0.5, TRAJ_LABELS[traj_name],
            transform=ax0.transAxes, fontsize=9, fontweight="bold",
            rotation=90, va="center", ha="center",
            color="#333333",
        )

    # ── Helper: plot one panel ────────────────────────────────────────────

    def _plot_mu(ax, traj: str, show_xlabel: bool = True):
        sched = trajectories[traj][:N_STEPS]
        for name, sty in ENV_STYLES.items():
            r = results[traj][name]
            ax.plot(years, r["mu"], **sty, marker="o", ms=4)
        ax.set_ylim(-0.02, 1.0)
        ax.set_ylabel("Mean μ (mitigation rate)", fontsize=8)
        ax.set_title("Mitigation rate μ", fontsize=9, pad=4)
        ax.grid(alpha=0.25)
        if show_xlabel:
            ax.set_xlabel("Year", fontsize=8)

    def _plot_go(ax, traj: str, show_xlabel: bool = True):
        base = results[traj]["baseline"]["gross_output"]
        for name, sty in ENV_STYLES.items():
            r = results[traj][name]
            rel = 100.0 * (r["gross_output"] - base) / np.maximum(base, 1e-8)
            ax.plot(years, rel, **sty, marker="o", ms=4)
        ax.axhline(0, color="#888888", lw=0.8, ls="--")
        ax.set_ylabel("GO change vs baseline (%)", fontsize=8)
        ax.set_title("Gross output (relative to baseline)", fontsize=9, pad=4)
        ax.grid(alpha=0.25)
        if show_xlabel:
            ax.set_xlabel("Year", fontsize=8)

    def _plot_tc(ax, traj: str, show_xlabel: bool = True):
        """Abatement cost (mean across regions), split into enduring + TC."""
        base_ac  = results[traj]["baseline"]["abatement_cost"]
        soft_ac  = results[traj]["tc_soft"]["abatement_cost"]
        hard_ac  = results[traj]["tc_hard"]["abatement_cost"]
        tc_soft_emp = np.maximum(soft_ac - base_ac, 0)
        tc_hard_emp = np.maximum(hard_ac - base_ac, 0)

        # Analytic TC
        sched = trajectories[traj]
        tc_soft_ana = _analytic_tc(sched, 10.0)
        tc_hard_ana = _analytic_tc(sched, 50.0)

        ax.bar(years - 1.2, base_ac,    width=2.2, color="#cccccc",
               label="Enduring cost (baseline)", zorder=2)
        ax.bar(years - 1.2, tc_soft_emp, width=2.2, bottom=base_ac,
               color="#1f77b4", alpha=0.7, label="TC empirical (c_B=10)", zorder=2)
        ax.bar(years + 1.2, base_ac,    width=2.2, color="#cccccc",
               label="_nolegend_", zorder=2)
        ax.bar(years + 1.2, tc_hard_emp, width=2.2, bottom=base_ac,
               color="#d62728", alpha=0.7, label="TC empirical (c_B=50)", zorder=2)
        ax.plot(years, tc_soft_ana + base_ac, color="#1f77b4",
                ls="--", lw=1.3, label="TC analytic (c_B=10)")
        ax.plot(years, tc_hard_ana + base_ac, color="#d62728",
                ls="--", lw=1.3, label="TC analytic (c_B=50)")

        ax.set_ylabel("Mean abatement cost fraction", fontsize=8)
        ax.set_title("Abatement cost: enduring + transitional", fontsize=9, pad=4)
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8)
        ax.grid(alpha=0.25, axis="y")
        if show_xlabel:
            ax.set_xlabel("Year", fontsize=8)

    # ── JUMP row ─────────────────────────────────────────────────────────
    _annotate_row(axes_top[0], "jump")
    _plot_mu(axes_top[0], "jump", show_xlabel=False)
    _plot_go(axes_top[1], "jump", show_xlabel=False)
    _plot_tc(axes_top[2], "jump", show_xlabel=False)
    for ax in axes_top:
        ax.tick_params(labelsize=7.5)

    # ── GRADUAL row ───────────────────────────────────────────────────────
    _annotate_row(axes_bot[0], "gradual")
    _plot_mu(axes_bot[0], "gradual")
    _plot_go(axes_bot[1], "gradual")
    _plot_tc(axes_bot[2], "gradual")
    for ax in axes_bot:
        ax.tick_params(labelsize=7.5)

    # ── Legend strip (top row, shared) ───────────────────────────────────
    handles = [mpatches.Patch(**{k: v for k, v in sty.items()
                                 if k in ("color", "label")},
                              linewidth=sty.get("lw", 1.5))
               for sty in ENV_STYLES.values()]
    fig.legend(
        handles=handles, loc="upper center", ncol=4,
        fontsize=8.5, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.915),
    )

    # ── Summary bar chart ─────────────────────────────────────────────────
    ax = ax_bar
    traj_names = ["jump", "gradual"]
    env_names  = ["baseline", "tc_soft", "tc_hard", "aw2"]
    x_base     = np.arange(len(traj_names))
    bar_w      = 0.18
    offsets    = np.linspace(-0.27, 0.27, len(env_names))
    for i, (name, sty) in enumerate(ENV_STYLES.items()):
        losses = []
        for traj in traj_names:
            base_total = results[traj]["baseline"]["gross_output"].sum()
            env_total  = results[traj][name]["gross_output"].sum()
            losses.append(100.0 * (base_total - env_total) / max(base_total, 1e-8))
        ax.bar(x_base + offsets[i], losses, width=bar_w,
               color=sty["color"], alpha=0.85, label=sty["label"], zorder=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_base)
    ax.set_xticklabels([TRAJ_LABELS[t] for t in traj_names], fontsize=8)
    ax.set_ylabel("Cumulative GO loss vs baseline (%)", fontsize=8)
    ax.set_title("Cumulative gross-output cost over 10 steps", fontsize=9)
    ax.legend(fontsize=7, framealpha=0.85, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(labelsize=7.5)
    # Annotate bars with values
    for i, name in enumerate(env_names):
        for j, traj in enumerate(traj_names):
            base_total = results[traj]["baseline"]["gross_output"].sum()
            env_total  = results[traj][name]["gross_output"].sum()
            loss = 100.0 * (base_total - env_total) / max(base_total, 1e-8)
            if abs(loss) > 0.01:
                ax.text(
                    j + offsets[i], loss + (0.04 if loss >= 0 else -0.12),
                    f"{loss:.1f}%", ha="center", va="bottom", fontsize=6.5, rotation=90,
                )

    # ── Action-window mask schematic ──────────────────────────────────────
    ax = ax_mask
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_xlabel("Discrete action level (0 = 0%,  9 = 90%)", fontsize=8)
    ax.set_ylabel("Accessible from current level →", fontsize=8)
    ax.set_title("AW=2 action-mask  vs  TC: which levels are reachable?", fontsize=9)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.tick_params(labelsize=7)

    # Show reachability from each starting level for AW=2
    LEVELS = 10
    AW = 2
    for cur in range(LEVELS):
        for tgt in range(LEVELS):
            allowed = abs(tgt - cur) <= AW
            c = "#b3d9f7" if allowed else "#f5c6c6"
            rect = plt.Rectangle((tgt - 0.5, cur - 0.5), 1, 1,
                                  color=c, zorder=1)
            ax.add_patch(rect)
            if tgt == cur:
                rect2 = plt.Rectangle((tgt - 0.5, cur - 0.5), 1, 1,
                                       color="#aaaaaa", zorder=2, alpha=0.4)
                ax.add_patch(rect2)

    # Highlight μ=0.80 (level 8) target
    ax.axvline(7.5, color="#d62728", lw=1.5, ls="--", alpha=0.7,
               label="μ=0.80 target (level 8)")
    ax.axhline(-0.5, color="none")  # dummy for spacing

    # Annotation boxes
    ax.text(
        8.0, -0.1, "μ=0.80\ntarget", color="#d62728", fontsize=7,
        ha="center", va="top",
    )
    blue_patch  = mpatches.Patch(color="#b3d9f7", label="Accessible (AW=2 allows)")
    red_patch   = mpatches.Patch(color="#f5c6c6", label="Blocked (AW=2 forbids)")
    gray_patch  = mpatches.Patch(color="#aaaaaa", alpha=0.7, label="Current level (diagonal)")
    ax.legend(handles=[blue_patch, red_patch, gray_patch],
              fontsize=7, loc="lower right", framealpha=0.9)

    # Add TC annotation text
    ax.text(
        4.5, 9.8,
        "TC: all levels accessible but rapid Δμ incurs a GDP penalty",
        ha="center", va="bottom", fontsize=7.5, style="italic",
        color="#1f77b4",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f4fd", alpha=0.9),
    )

    # ── Axis spines cleanup ───────────────────────────────────────────────
    for row_axes in (axes_top, axes_bot):
        for ax in row_axes:
            ax.spines[["top", "right"]].set_visible(False)
    for ax in (ax_bar, ax_mask):
        ax.spines[["top", "right"]].set_visible(False)

    # ── Row labels ────────────────────────────────────────────────────────
    for col_ax, label in zip(
        [axes_top[0], axes_bot[0]],
        ["JUMP\n(0→80% instant)", "GRADUAL\n(+10%/step)"],
    ):
        col_ax.text(
            -0.28, 0.5, label, transform=col_ax.transAxes,
            fontsize=8.5, fontweight="bold", rotation=90,
            va="center", ha="center", color="#222222",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#eeeeee", alpha=0.7),
        )

    # ── Footer ────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Grubb et al. (1995) DIAM Eq. 2: TC = c_B × ((μ_t − μ_{t−1}) / Δt)²  "
        "applied as fraction of gross output.  c_B=10 (soft), c_B=50 (hard).  "
        "AW=2 is a training-time policy mask — no physics cost, prevents policy "
        "expressing large conditioning gaps (breaks C-test).",
        ha="center", fontsize=7, color="#555555", style="italic",
    )

    # ── Save ─────────────────────────────────────────────────────────────
    _os.makedirs(OUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _os.path.join(OUT_DIR, f"transition_cost_{ts}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
