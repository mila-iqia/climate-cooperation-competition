"""team_brief.py

Minimal team-brief generator.

Produces a single Markdown file with:
  - Definitions of C-litmus tests C1, C2, C3 and the actual measured values
    from cbam_experiment_C_litmus_20260519_101422_free_savings_t10_no_m_allpass
  - A list of deviations from the RICE-N baseline

Plots are copied from the source experiment directory into the output dir so
the .md is self-contained.

Usage (from rice_jax/):
    conda activate rice-jax        # only needed if you want to re-extract metrics
    python validation/team_brief.py
"""
from __future__ import annotations

import os
import pickle
import shutil
from datetime import datetime


# ── Paths ─────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_RICEJAX    = os.path.join(_REPO_ROOT, "rice_jax")

SOURCE_EXPERIMENT = os.path.join(
    _RICEJAX, "experiments",
    "cbam_experiment_C_litmus_20260519_101422_free_savings_t10_no_m_allpass",
)
SOURCE_PKL = os.path.join(
    SOURCE_EXPERIMENT, "plots",
    "cbam_C_litmus_3seeds_2M_20260519_101424.pkl",
)
SOURCE_MAIN_PLOT = os.path.join(
    SOURCE_EXPERIMENT, "plots",
    "cbam_C_litmus_3seeds_2M_20260519_101424.png",
)
SOURCE_REGIONAL_PLOT = os.path.join(
    SOURCE_EXPERIMENT, "posthoc",
    "posthoc_C_regional_20260519_124636.png",
)


# ── Pull numeric results from the pickle ──────────────────────────────────────

def _load_results() -> dict:
    """Extract per-seed scalars for C1/C2/C3."""
    with open(SOURCE_PKL, "rb") as fh:
        d = pickle.load(fh)

    out = {"c1": {}, "c2": {}, "c3": {}, "summary": d["summary"]}
    for seed, sr in d["all_results"].items():
        for t in ("c1", "c2", "c3"):
            entry  = sr[t]
            passed = entry["passed"]
            data   = entry["data"]
            scalars = {}
            for k, v in data.items():
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    scalars[k] = float(v)
                elif hasattr(v, "item") and getattr(v, "ndim", 1) == 0:
                    try:
                        scalars[k] = float(v.item())
                    except Exception:
                        pass
            out[t][seed] = {"passed": passed, **scalars}
    return out


# ── MD template ───────────────────────────────────────────────────────────────

_MD = """\
# CBAM-RICE · Team Brief

_Generated {timestamp}_

Source experiment: `experiments/cbam_experiment_C_litmus_20260519_101422_free_savings_t10_no_m_allpass`
(3 seeds × 2M timesteps · 9-region vulnerability · differential CBAM · TC=10, AW=0, free savings)

---

## C-litmus tests · definitions and results

Each test asks whether **one trained policy** reacts to the CBAM signal in its
observation. All three pass on all three seeds (verdict: **{verdict}**).

### C1 · Export Conditioning

A single agent trained with `cbam_randomize=True`. The same weights are then
evaluated at τ = 0 (off) and τ = 1 (on). The metric is the relative drop in
each region's EU-directed dirty-export share.

**Pass:** within-policy EU-dirty-share gap > 15% relative.

| Seed | EU dirty share (τ=0) | EU dirty share (τ=1) | Relative gap |
|------|----------------------|----------------------|--------------|
| 0    | {c1_0_off:.4f}       | {c1_0_on:.4f}        | {c1_0_gap:.1%}   |
| 1    | {c1_1_off:.4f}       | {c1_1_on:.4f}        | {c1_1_gap:.1%}   |
| 2    | {c1_2_off:.4f}       | {c1_2_on:.4f}        | {c1_2_gap:.1%}   |

All seeds PASS — the policy diverts dirty exports away from the EU when the
CBAM signal turns on.

### C2 · Mitigation Conditioning Under Pinned Exports

Randomised differential CBAM with the diversion channel shut off
(`delta_max=0`), so mitigation is the only response margin. Two arms:

- **C2a:** `zero_abatement_cost=True` (costless mitigation)
- **C2b:** realistic abatement cost (costly)

Primary metric: μ(τ=1) − μ(τ=0). Graded **strong** if the gap exceeds 0.05.

| Seed | C2a μ(on) | C2a μ(off) | gap a   | C2b μ(on) | C2b μ(off) | gap b   |
|------|-----------|------------|---------|-----------|------------|---------|
| 0    | {c2_0_aon:.3f} | {c2_0_aoff:.3f} | {c2_0_gapa:+.3f} | {c2_0_bon:.3f} | {c2_0_boff:.3f} | {c2_0_gapb:+.3f} |
| 1    | {c2_1_aon:.3f} | {c2_1_aoff:.3f} | {c2_1_gapa:+.3f} | {c2_1_bon:.3f} | {c2_1_boff:.3f} | {c2_1_gapb:+.3f} |
| 2    | {c2_2_aon:.3f} | {c2_2_aoff:.3f} | {c2_2_gapa:+.3f} | {c2_2_bon:.3f} | {c2_2_boff:.3f} | {c2_2_gapb:+.3f} |

All seeds graded **strong** — the policy mitigates more when the CBAM signal
is on, in both the costless and costly arms.

### C3 · Conditioned Crowd-Out

Same costly setting as C2b, but now **both channels open**
(`delta_max=3`, exports and mitigation both free). C2b μ_on is the reference;
C3 μ_on is the mitigation level when diversion is also available.

**Pass:** μ_on (both open) < μ_on (pinned, from C2b).

| Seed | μ_on (both, C3) | μ_on (pinned, C2b) | Crowd-out drop |
|------|-----------------|--------------------|----------------|
| 0    | {c3_0_on:.3f}   | {c2_0_bon:.3f}     | −{c3_0_drop:.3f}  |
| 1    | {c3_1_on:.3f}   | {c2_1_bon:.3f}     | −{c3_1_drop:.3f}  |
| 2    | {c3_2_on:.3f}   | {c2_2_bon:.3f}     | −{c3_2_drop:.3f}  |

All seeds PASS — opening the diversion channel reduces mitigation effort.
The drop is the conditioned crowd-out the policy fix has to close.

### Plots

![C-litmus main figure](cbam_C_litmus_main.png)

**Main figure** · per-test bars, training convergence, and per-region
mitigation under the on/off CBAM signal for each seed.

![C-litmus regional breakdown](posthoc_C_regional.png)

**Regional breakdown** · same metrics resolved across the 9 vulnerability
regions; SSA-Mining, India and SE Asia show the largest crowd-out.

---

## Deviations from RICE-N

The model in `rice_jax/_rice_mrio.py` inherits the RICE-N base
(`rice_jax/_rice.py`, do-not-modify) and adds the following:

### 1 · Action space

- **Export reallocation action** (new) — per-agent logit adjustment δ over the
  2016 MRIO destination baseline, sized (num_sectors × num_regions).
  Bounded by `delta_max`.
- **Mitigation persistence (transition cost)** — Grubb (1995) DIAM Eq. 2,
  quadratic penalty on Δμ between timesteps (`transition_cost_coef`).
  Replaces the mechanical `action_window_size` mask.
- **Free savings** — `fixed_savings_rate=False`; agents choose their own
  savings rate (required for C2 conditioning to survive transition cost).

### 2 · Calibration data

- **EORA26 2016** bilateral trade tables (Lenzen et al. 2013) aggregated into
  the 9-region vulnerability ordering.
  Pipeline in `csv_asset/mrio/aggregated/eora_agg_9/`.
- **Sector aggregation** — 26 EORA sectors collapsed to
  `sector_granularity="emissions-simple"` (dirty / clean) using emissions
  intensity weights.
- **Welfare-loss-per-unit-tariff** — per-region `wl_r` values calibrated
  against He, Zhai & Ma (2022/2025), Chepeliev (2021), IMF WP (2025),
  ACF/LSE (2023). _[Primary-source cross-check pending — CBAM_ROADMAP §Phase-2C.]_

### 3 · Trade overhaul

- New env field `mrio_trade=True` enables the trade module entirely; the
  RICE-N default has no bilateral trade.
- New state keys: `trade_flows` (NR × NR × NS) and `cbam_revenue` (NR,).
- Destination shares follow a persistent geometric blend:
  `share_t = (1−ρ)·logit(action) + ρ·share_{{t−1}}`
  with `dest_alloc_persistence=0.55` (paper-frozen).
- Anchor decay `dest_alloc_baseline_decay=0.0` — the 2016 MRIO anchor does
  not decay in headline runs (audit Gate 3).

### 4 · CBAM implementation

- **Differential tariff** (`cbam_tariff_mode="differential"`):
  τ_eff[r] = max(0, MAC_EU − MAC_r) / MAC_EU. Self-incentivising — as region
  r matches EU carbon pricing, its tariff shrinks to zero.
- **EU mitigation schedule** ramps 0.30 → 1.00 over 8 steps (EU Climate
  Law (EU) 2021/1119) so MAC_EU > 0 from t = 0.
- **Reward mode** `additive_cbam` (RCPO): r̂ = ΔU − λ · c, with λ auto-tuned
  via `RCPOMonitoredPPO` (Tessler et al. 2019).
- **Revenue recycling** — optional transfer pool with `revenue_share`,
  `transfer_mode` (consumption / abatement) and `transfer_allocation` rule
  (effort wins; see Experiment A).

### 5 · Regional specification

- **9-region vulnerability aggregation** (`CountryClass_cbam_vuln_9.csv`)
  replaces the 27-region RICE-N default. Regions ordered by CBAM
  vulnerability: RoW, Russia+Eur., MENA, **EU (idx 3)**, SSA-Mining,
  Americas, SE Asia, China, India.
- Region yamls live in `cbam_yamls/setup_vuln_9/`; loaded via
  `load_region_yamls(9, yaml_dir=...)`.
- `eu_region_idx` is always passed explicitly — the default in
  `_rice_mrio.py` is wrong for non-7-region setups.

### 6 · Regional damages

- New static field `regional_damage_coeff` (`(num_regions,)` float32, default
  `None` → uniform yaml fallback) and a `calc_damages` override that injects
  the per-region xa\\_2 into the Nordhaus quadratic damage function.
- Canonical null test in `tests/test_rice_mrio.py::TestRegionalDamageCoeff`
  verifies bit-identical behaviour vs the uniform fallback when all
  coefficients equal the yaml baseline (0.00236).
- Calibrated coefficients (placeholder, pending cross-check) source from
  Kompas et al. (2018), Hansel et al. (2020), Ricke et al. (2018),
  RICE-2010 — see `validate_regional_damages.py`.

---

_Source code: `rice_jax/validation/team_brief.py`. Plots copied verbatim from
the source experiment (no re-rendering)._
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    res = _load_results()

    ts      = datetime.now().strftime("%Y-%m-%d %H:%M")
    fts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("plots", f"team_brief_{fts}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy source plots so the MD is self-contained
    shutil.copy(SOURCE_MAIN_PLOT,     os.path.join(out_dir, "cbam_C_litmus_main.png"))
    shutil.copy(SOURCE_REGIONAL_PLOT, os.path.join(out_dir, "posthoc_C_regional.png"))

    c1, c2, c3 = res["c1"], res["c2"], res["c3"]
    seeds = sorted(c1.keys())
    assert seeds == [0, 1, 2], f"Expected seeds [0,1,2], got {seeds}"

    verdict = (
        "PASS · 3/3 seeds on all of C1, C2, C3"
        if all(res["summary"][t]["all_pass"] for t in ("c1", "c2", "c3"))
        else "MIXED"
    )

    md = _MD.format(
        timestamp = ts,
        verdict   = verdict,

        # C1
        c1_0_on=c1[0]["share_on"], c1_0_off=c1[0]["share_off"], c1_0_gap=c1[0]["cond_gap"],
        c1_1_on=c1[1]["share_on"], c1_1_off=c1[1]["share_off"], c1_1_gap=c1[1]["cond_gap"],
        c1_2_on=c1[2]["share_on"], c1_2_off=c1[2]["share_off"], c1_2_gap=c1[2]["cond_gap"],

        # C2
        c2_0_aon=c2[0]["mu_a_on"], c2_0_aoff=c2[0]["mu_a_off"], c2_0_gapa=c2[0]["gap_a"],
        c2_0_bon=c2[0]["mu_b_on"], c2_0_boff=c2[0]["mu_b_off"], c2_0_gapb=c2[0]["gap_b"],
        c2_1_aon=c2[1]["mu_a_on"], c2_1_aoff=c2[1]["mu_a_off"], c2_1_gapa=c2[1]["gap_a"],
        c2_1_bon=c2[1]["mu_b_on"], c2_1_boff=c2[1]["mu_b_off"], c2_1_gapb=c2[1]["gap_b"],
        c2_2_aon=c2[2]["mu_a_on"], c2_2_aoff=c2[2]["mu_a_off"], c2_2_gapa=c2[2]["gap_a"],
        c2_2_bon=c2[2]["mu_b_on"], c2_2_boff=c2[2]["mu_b_off"], c2_2_gapb=c2[2]["gap_b"],

        # C3
        c3_0_on=c3[0]["mu_on"], c3_0_drop=(c2[0]["mu_b_on"] - c3[0]["mu_on"]),
        c3_1_on=c3[1]["mu_on"], c3_1_drop=(c2[1]["mu_b_on"] - c3[1]["mu_on"]),
        c3_2_on=c3[2]["mu_on"], c3_2_drop=(c2[2]["mu_b_on"] - c3[2]["mu_on"]),
    )

    md_path = os.path.join(out_dir, "team_brief.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    print(f"Wrote {os.path.abspath(md_path)}")


if __name__ == "__main__":
    main()
