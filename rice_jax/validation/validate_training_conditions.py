"""validate_training_conditions.py

Systematically tests 5 PPO training configurations to resolve the
mid-episode EU export-share bump observed in MENA (steps 10-15).

For each config trains CBAM (τ=0.80) and no-CBAM (τ=0) conditions,
saves a 3×2 diagnostic plot, records key metrics, then auto-generates
TRAINING_CONDITIONS_FINDINGS.md with a comparison table.

Metrics collected per config:
  bump_mag      — mean CBAM-dirty EU share [steps 8-12] minus [steps 0-3],
                  MENA region.  Positive = bump exists.
  separation    — mean no-CBAM dirty minus CBAM dirty at end [17-19], MENA.
                  Positive = CBAM successfully diverts.
  welfloss      — mean CBAM welfloss across full episode, MENA.
  rew_gap       — mean no-CBAM reward minus CBAM reward at end, MENA.

Usage:
    python validate_training_conditions.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))



from __future__ import annotations

import os
import textwrap
import traceback
from dataclasses import replace
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxnasium.algorithms import PPO

import jaxnasium as jym

from _experiment_util import run_single_episode
from rice_jax import RiceMRIO
from rice_jax.utils import full_state_info_log_fn, load_region_yamls

# ── Fixed parameters (shared across all configs) ──────────────────────────────

NUM_REGIONS: int = 7
MRIO_DATA_ROOT: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csv_asset")
EU_REGION_IDX: int = 0
DEST_ALLOC_PERSISTENCE: float = 0.55
DEST_ALLOC_BASELINE_DECAY: float = 1.0
CBAM_RATE: float = 0.80
FOCUS_REGIONS: list[int] = [1, 3]   # South Asia(1), MENA(3)
TOTAL_TIMESTEPS: int = 500_000
NUM_STEPS: int = 100
NUM_EVAL_EPISODES: int = 8
SEED: int = 42
OUTPUT_DIR: str = "plots/training_conditions"
DPI: int = 130

# ── Base PPO kwargs (each config overrides selected fields) ───────────────────

BASE_PPO: dict = dict(
    total_timesteps=TOTAL_TIMESTEPS,
    learning_rate=2.5e-4,
    num_steps=NUM_STEPS,
    num_envs=4,
    num_minibatches=4,
    num_epochs=4,
    ent_coef=2.0,           # ← baseline (current default)
    anneal_ent_coef=0.05,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=1.0,
    clip_coef=0.2,
    clip_coef_vf=0.5,
    vf_coef=0.5,
    normalize_observations=True,
    normalize_rewards=True,
    log_function="tqdm",
)

# ── 5 experimental configurations ─────────────────────────────────────────────

CONFIGS: list[dict] = [
    {
        "name": "C1_low_ent",
        "label": "Low entropy (ent=0.05, anneal→0)",
        "hypothesis": "ent_coef=2.0 dominates the PPO loss, causing erratic "
                      "exploration and mid-episode drift. Dropping to 0.05 "
                      "(standard PPO default) should stabilise the policy.",
        "overrides": dict(ent_coef=0.05, anneal_ent_coef=0.0),
    },
    {
        "name": "C2_more_envs",
        "label": "More envs (ent=0.05, num_envs=8)",
        "hypothesis": "Doubling rollout environments increases the diversity of "
                      "experience per update, reducing policy variance and the "
                      "correlated mid-episode spikes.",
        "overrides": dict(ent_coef=0.05, anneal_ent_coef=0.0, num_envs=8),
    },
    {
        "name": "C3_more_epochs",
        "label": "More epochs (ent=0.05, epochs=8)",
        "hypothesis": "More SGD passes per batch extracts more signal from each "
                      "rollout, accelerating convergence without extra env steps.",
        "overrides": dict(ent_coef=0.05, anneal_ent_coef=0.0, num_epochs=8),
    },
    {
        "name": "C4_high_lr",
        "label": "Higher LR (ent=0.05, lr=5e-4)",
        "hypothesis": "A larger learning rate drives the policy to converge "
                      "faster within 500k steps, potentially settling the "
                      "mid-episode allocation before the bump arises.",
        "overrides": dict(ent_coef=0.05, anneal_ent_coef=0.0, learning_rate=5e-4),
    },
    {
        "name": "C5_combined",
        "label": "Combined (ent=0.01, envs=8, epochs=8, lr=3e-4)",
        "hypothesis": "Stack all improvements: very low entropy, more diverse "
                      "rollouts, more gradient steps, moderate LR increase.",
        "overrides": dict(
            ent_coef=0.01, anneal_ent_coef=0.0,
            num_envs=8, num_epochs=8, learning_rate=3e-4,
        ),
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_env(cbam_tariff_rate: float, region_params) -> jym.LogWrapper:
    env = RiceMRIO(
        region_params=region_params,
        num_regions=NUM_REGIONS,
        mrio_data_root=MRIO_DATA_ROOT,
        mrio_trade=True,
        dest_alloc_persistence=DEST_ALLOC_PERSISTENCE,
        dest_alloc_baseline_decay=DEST_ALLOC_BASELINE_DECAY,
        cbam_tariff_rate=cbam_tariff_rate,
        eu_region_idx=EU_REGION_IDX,
        diff_reward_mode=True,
        num_discrete_action_levels=10,
        sectoral_welfloss=True,
        fixed_savings_rate=True,
        no_mitigation=True,
        sector_granularity="emissions-simple",
        welfare_loss_per_unit_tariff=50.0,
    )
    return jym.LogWrapper(env)


def _train_and_collect(
    cbam_tariff_rate: float,
    region_params,
    seed: jax.Array,
    ppo_kwargs: dict,
) -> dict[str, np.ndarray]:
    wrapped_env = _build_env(cbam_tariff_rate, region_params)
    lbl = f"τ={cbam_tariff_rate:.2f}" if cbam_tariff_rate > 0 else "no-CBAM"

    agent = PPO(**ppo_kwargs)
    print(f"    [{lbl}] Training {ppo_kwargs['total_timesteps']:,} steps...", flush=True)
    agent = agent.train(seed, wrapped_env)

    eval_env = replace(wrapped_env._env, log_info_fn=full_state_info_log_fn)
    all_flows, all_uwl, all_util = [], [], []

    for ep_id in range(NUM_EVAL_EPISODES):
        ep_key = jax.random.fold_in(seed, 10_000 + ep_id)
        logs = run_single_episode(ep_key, eval_env, agent)
        all_flows.append(np.array(logs["trade_flows"]))  # (T, NR, NR, NS)
        uwl = np.stack(
            [np.array(logs["utility_times_welfloss_all_regions"][r]) for r in range(NUM_REGIONS)],
            axis=-1,
        )
        util = np.stack(
            [np.array(logs["utility_all_regions"][r]) for r in range(NUM_REGIONS)],
            axis=-1,
        )
        all_uwl.append(uwl)
        all_util.append(util)

    return {
        "trade_flows":      np.stack(all_flows, axis=0),   # (E, T, NR, NR, NS)
        "utility_welfloss": np.stack(all_uwl,   axis=0),   # (E, T, NR)
        "utility":          np.stack(all_util,  axis=0),   # (E, T, NR)
    }


# ── Metrics ───────────────────────────────────────────────────────────────────


def _compute_metrics(no_cbam: dict, cbam: dict, r_idx: int) -> dict[str, float]:
    """Scalar metrics for one region."""
    flows_c = cbam["trade_flows"]          # (E, T, NR, NR, NS)
    flows_n = no_cbam["trade_flows"]

    def _eu_dirty_share(flows):
        eu_dirty = flows[:, :, r_idx, EU_REGION_IDX, 0]          # (E, T)
        tot_dirty = flows[:, :, r_idx, :, 0].sum(axis=-1)         # (E, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
        return np.nanmean(share, axis=0)  # (T,)

    share_c = _eu_dirty_share(flows_c)
    share_n = _eu_dirty_share(flows_n)
    T = share_c.shape[0]

    early  = slice(0, min(4, T))
    mid    = slice(min(8, T - 1), min(13, T))
    late   = slice(max(0, T - 3), T)

    bump_mag = float(np.nanmean(share_c[mid]) - np.nanmean(share_c[early]))
    separation = float(np.nanmean(share_n[late]) - np.nanmean(share_c[late]))

    uwl_c = cbam["utility_welfloss"][:, :, r_idx]      # (E, T)
    util_c = cbam["utility"][:, :, r_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        wl = np.where(np.abs(util_c) > 1e-12, uwl_c / util_c, 1.0)
    mean_welfloss = float(np.nanmean(wl))

    uwl_n = no_cbam["utility_welfloss"][:, :, r_idx]
    rew_gap = float(np.nanmean(uwl_n[:, late]) - np.nanmean(uwl_c[:, late]))

    return dict(
        bump_mag=round(bump_mag, 4),
        separation=round(separation, 4),
        mean_welfloss=round(mean_welfloss, 4),
        rew_gap=round(rew_gap, 4),
    )


# ── Plotting ──────────────────────────────────────────────────────────────────


def _make_figure(
    no_cbam: dict,
    cbam: dict,
    region_labels: list[str],
    config_label: str,
    ppo_kwargs: dict,
) -> plt.Figure:
    n_cols = len(FOCUS_REGIONS)
    T = no_cbam["trade_flows"].shape[1]
    ts = np.arange(T)

    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 11), constrained_layout=True)
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    for col_i, r_idx in enumerate(FOCUS_REGIONS):
        rlbl = region_labels[r_idx] if r_idx < len(region_labels) else f"R{r_idx}"

        ax_share = axes[0, col_i]
        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            flows = data["trade_flows"]
            eu_dirty = flows[:, :, r_idx, EU_REGION_IDX, 0]
            eu_clean = flows[:, :, r_idx, EU_REGION_IDX, 1]
            tot_dirty = flows[:, :, r_idx, :, 0].sum(axis=-1)
            tot_clean = flows[:, :, r_idx, :, 1].sum(axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                sd = np.where(tot_dirty > 1e-12, eu_dirty / tot_dirty, np.nan)
                sc = np.where(tot_clean > 1e-12, eu_clean / tot_clean, np.nan)
            for share, ls, lbl in [(sd, "-", f"{cond_label} dirty"), (sc, "--", f"{cond_label} clean")]:
                m = np.nanmean(share, axis=0)
                s = np.nanstd(share, axis=0)
                ax_share.plot(ts, m, color=color, ls=ls, lw=2 if ls == "-" else 1.5, label=lbl)
                ax_share.fill_between(ts, m - s, m + s, alpha=0.10, color=color)
        ax_share.set_title(rlbl, fontsize=12, fontweight="bold")
        ax_share.set_ylabel("EU export share")
        ax_share.set_ylim(bottom=0)
        ax_share.legend(fontsize=7, loc="best")
        ax_share.grid(alpha=0.2)

        ax_rew = axes[1, col_i]
        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            uwl = data["utility_welfloss"][:, :, r_idx]
            m = np.nanmean(uwl, axis=0)
            s = np.nanstd(uwl, axis=0)
            ax_rew.plot(ts, m, color=color, lw=2, label=cond_label)
            ax_rew.fill_between(ts, m - s, m + s, alpha=0.13, color=color)
        ax_rew.set_title(rlbl, fontsize=12, fontweight="bold")
        ax_rew.set_ylabel("Reward (utility × welfloss)")
        ax_rew.legend(fontsize=7)
        ax_rew.grid(alpha=0.2)

        ax_wl = axes[2, col_i]
        for cond_label, data, color in [
            ("no-CBAM", no_cbam, "#5577cc"),
            ("CBAM", cbam, "#e05c2a"),
        ]:
            uwl = data["utility_welfloss"][:, :, r_idx]
            util = data["utility"][:, :, r_idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                wl = np.where(np.abs(util) > 1e-12, uwl / util, 1.0)
            m = np.nanmean(wl, axis=0)
            s = np.nanstd(wl, axis=0)
            ax_wl.plot(ts, m, color=color, lw=2, label=cond_label)
            ax_wl.fill_between(ts, m - s, m + s, alpha=0.13, color=color)
        ax_wl.set_title(rlbl, fontsize=12, fontweight="bold")
        ax_wl.set_ylabel("Welfloss multiplier")
        ax_wl.set_xlabel("Episode step")
        ax_wl.legend(fontsize=7)
        ax_wl.grid(alpha=0.2)

    for ri, lbl in enumerate([
        "EU Export Share\n(solid=dirty, dashed=clean)",
        "Reward (utility × welfloss)",
        "Welfloss multiplier",
    ]):
        axes[ri, 0].annotate(
            lbl, xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=8, rotation=90, va="center", ha="right", color="dimgray",
        )

    ent = ppo_kwargs.get("ent_coef", "?")
    lr = ppo_kwargs.get("learning_rate", "?")
    envs = ppo_kwargs.get("num_envs", "?")
    epochs = ppo_kwargs.get("num_epochs", "?")
    fig.suptitle(
        f"{config_label}\n"
        f"ent_coef={ent}  lr={lr}  num_envs={envs}  num_epochs={epochs}  "
        f"steps={ppo_kwargs['total_timesteps']:,}  eval_eps={NUM_EVAL_EPISODES}\n"
        f"ρ={DEST_ALLOC_PERSISTENCE}  decay={DEST_ALLOC_BASELINE_DECAY}  "
        f"τ={CBAM_RATE:.0%}  welfare_loss_per_unit_tariff=50",
        fontsize=10, fontweight="bold", y=1.02,
    )
    return fig


# ── Markdown report ───────────────────────────────────────────────────────────

_NOTE_BUMP_POS   = "> 0.02"
_NOTE_BUMP_WEAK  = "0–0.02"
_NOTE_BUMP_GONE  = "≤ 0"

def _bump_verdict(v: float) -> str:
    if v > 0.02:  return "❌ bump"
    if v > 0.00:  return "⚠ weak"
    return "✅ gone"

def _sep_verdict(v: float) -> str:
    if v > 0.03:  return "✅ diverts"
    if v > 0.01:  return "⚠ partial"
    return "❌ none"


def _write_markdown(
    results: list[dict],  # list of {cfg, metrics_SA, metrics_MENA}
    out_path: str,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Training Conditions Experiment — Findings",
        "",
        f"*Generated: {ts}*",
        "",
        "## Setup",
        "",
        f"- **Environment**: RiceMRIO 7-region, `emissions-simple` (CBAM-dirty / non-CBAM clean)",
        f"- **Fixed params**: ρ={DEST_ALLOC_PERSISTENCE}, decay={DEST_ALLOC_BASELINE_DECAY}, "
        f"τ={CBAM_RATE:.0%}, `welfare_loss_per_unit_tariff=50`, "
        f"`fixed_savings_rate=True`, `no_mitigation=True`",
        f"- **Steps**: {TOTAL_TIMESTEPS:,} per condition (×2 = CBAM + no-CBAM)",
        f"- **Eval episodes**: {NUM_EVAL_EPISODES}",
        f"- **Focus regions**: South Asia (idx 1), MENA (idx 3)",
        "",
        "## Metric definitions",
        "",
        "| Metric | Definition |",
        "|--------|-----------|",
        "| `bump_mag` | Mean CBAM-dirty EU share at steps 8–12 minus steps 0–3 (MENA). Positive = bump exists. |",
        "| `separation` | Mean (no-CBAM dirty − CBAM dirty) at steps 17–19. Positive = CBAM learns to divert. |",
        "| `mean_welfloss` | Mean CBAM welfare-loss multiplier (< 1 = CBAM penalty applied). |",
        "| `rew_gap` | Mean (no-CBAM reward − CBAM reward) at steps 17–19. Positive = CBAM penalises reward. |",
        "",
        "## Results — MENA (primary region of concern)",
        "",
        "| Config | bump_mag | verdict | separation | verdict | welfloss | rew_gap |",
        "|--------|----------|---------|-----------|---------|----------|---------|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['cfg']['label']} | ERROR | — | — | — | — | — |")
            continue
        m = r["metrics_MENA"]
        lines.append(
            f"| {r['cfg']['label']} "
            f"| {m['bump_mag']:+.4f} | {_bump_verdict(m['bump_mag'])} "
            f"| {m['separation']:+.4f} | {_sep_verdict(m['separation'])} "
            f"| {m['mean_welfloss']:.4f} "
            f"| {m['rew_gap']:+.4f} |"
        )

    lines += [
        "",
        "## Results — South Asia",
        "",
        "| Config | bump_mag | separation | welfloss |",
        "|--------|----------|-----------|---------|",
    ]
    for r in results:
        if r.get("error"):
            lines.append(f"| {r['cfg']['label']} | ERROR | — | — |")
            continue
        m = r["metrics_SA"]
        lines.append(
            f"| {r['cfg']['label']} "
            f"| {m['bump_mag']:+.4f} "
            f"| {m['separation']:+.4f} "
            f"| {m['mean_welfloss']:.4f} |"
        )

    lines += ["", "## Per-config notes", ""]
    for r in results:
        cfg = r["cfg"]
        lines.append(f"### {cfg['label']}")
        lines.append("")
        lines.append(f"**Hypothesis**: {cfg['hypothesis']}")
        lines.append("")
        overrides_str = ", ".join(f"`{k}={v}`" for k, v in cfg["overrides"].items())
        lines.append(f"**PPO overrides over base**: {overrides_str}")
        lines.append("")
        if r.get("error"):
            lines.append(f"**ERROR**: ```\n{r['error']}\n```")
        else:
            mM = r["metrics_MENA"]
            mS = r["metrics_SA"]
            lines.append(
                f"MENA bump_mag={mM['bump_mag']:+.4f} ({_bump_verdict(mM['bump_mag'])}), "
                f"separation={mM['separation']:+.4f} ({_sep_verdict(mM['separation'])})"
            )
            lines.append(
                f"South Asia bump_mag={mS['bump_mag']:+.4f}, "
                f"separation={mS['separation']:+.4f}"
            )
        lines.append("")
        lines.append(f"![plot]({cfg['name']}.png)")
        lines.append("")

    # Overall recommendation
    valid = [r for r in results if not r.get("error")]
    if valid:
        best = min(valid, key=lambda r: r["metrics_MENA"]["bump_mag"])
        best_sep = max(valid, key=lambda r: r["metrics_MENA"]["separation"])
        lines += [
            "## Recommendation",
            "",
            f"**Lowest bump**: {best['cfg']['label']} "
            f"(bump_mag={best['metrics_MENA']['bump_mag']:+.4f})",
            f"**Best diversion**: {best_sep['cfg']['label']} "
            f"(separation={best_sep['metrics_MENA']['separation']:+.4f})",
            "",
            "**Suggested next step**: Use the lowest-bump config as the new baseline "
            "in `validate_emissions_simple.py` and extend to 1M–2M steps to confirm stability.",
            "",
        ]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nMarkdown saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    seed = jax.random.PRNGKey(SEED)
    region_params = load_region_yamls(NUM_REGIONS)

    sample_env = _build_env(0.0, region_params)._env
    region_labels = list(sample_env.mrio_region_labels)
    focus_names = [region_labels[r] for r in FOCUS_REGIONS]

    print("=" * 70)
    print("  Training Conditions Experiment")
    print(f"  Focus regions: {focus_names}")
    print(f"  {len(CONFIGS)} configs × 2 conditions × {TOTAL_TIMESTEPS:,} steps")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results: list[dict] = []

    for cfg_i, cfg in enumerate(CONFIGS):
        print(f"\n[{cfg_i+1}/{len(CONFIGS)}] {cfg['label']}")
        ppo_kwargs = {**BASE_PPO, **cfg["overrides"]}

        try:
            seed_nc = jax.random.fold_in(seed, cfg_i * 100 + 0)
            seed_cb = jax.random.fold_in(seed, cfg_i * 100 + 1)
            no_cbam = _train_and_collect(0.0,      region_params, seed_nc, ppo_kwargs)
            cbam    = _train_and_collect(CBAM_RATE, region_params, seed_cb, ppo_kwargs)

            metrics_MENA = _compute_metrics(no_cbam, cbam, r_idx=3)
            metrics_SA   = _compute_metrics(no_cbam, cbam, r_idx=1)

            print(f"  MENA   bump={metrics_MENA['bump_mag']:+.4f}  "
                  f"sep={metrics_MENA['separation']:+.4f}  "
                  f"welfloss={metrics_MENA['mean_welfloss']:.4f}")
            print(f"  S.Asia bump={metrics_SA['bump_mag']:+.4f}  "
                  f"sep={metrics_SA['separation']:+.4f}  "
                  f"welfloss={metrics_SA['mean_welfloss']:.4f}")

            fig = _make_figure(no_cbam, cbam, region_labels, cfg["label"], ppo_kwargs)
            plot_path = os.path.join(OUTPUT_DIR, f"{cfg['name']}.png")
            fig.savefig(plot_path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  Plot saved: {plot_path}")

            results.append(dict(
                cfg=cfg,
                metrics_MENA=metrics_MENA,
                metrics_SA=metrics_SA,
            ))

        except Exception as e:
            print(f"  ERROR in config {cfg['name']}: {e}")
            traceback.print_exc()
            results.append(dict(cfg=cfg, error=str(e)))

    md_path = os.path.join(OUTPUT_DIR, "TRAINING_CONDITIONS_FINDINGS.md")
    _write_markdown(results, md_path)


if __name__ == "__main__":
    main()
