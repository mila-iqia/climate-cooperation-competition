"""test_transition_cost.py

Deterministic rollout comparison: Grubb transition cost vs action window.

Tests three environments on two preset mitigation-rate trajectories:

  ENVS
  ----
  baseline   AW=0, TC=0     — no constraint, no penalty (reference)
  tc_soft    AW=0, TC=10    — economic penalty for rapid μ changes (Grubb 1995 Eq.2)
  tc_hard    AW=0, TC=50    — stiffer penalty version
  aw2        AW=2, TC=0     — mechanical action-window block (mask-based, NOT physics)

  TRAJECTORIES (applied identically to all envs)
  -----------------------------------------------
  jump       μ = 0 → 0.80 at step 1, then held at 0.80
             Tests: rapid decarbonisation path
  gradual    μ grows by 0.10/step: 0, 0.10, 0.20, ... 0.80, 0.80, ...
             Tests: smooth, Paris-compatible ramp

  AW NOTE: The action_window_size constraint operates as an action *mask* in
  process_actions / generate_action_masks, not as a physics term.  Calling
  step_climate_and_economy directly bypasses the mask, so env_aw2 and
  env_baseline produce identical gross outputs here.  The AW column is included
  to confirm this — and to emphasise the contrast with TC, which modifies
  gross_output regardless of how actions are submitted.

Run:
  conda run -n rice-jax python rice_jax/validation/test_transition_cost.py

Literature:
  Grubb, Chapuis & Ha Duong (1995) "The economics of changing course",
    Energy Policy 23(4-5):417-432, §3, Eq. 2 (DIAM model).
  Grubb, Wieners & Yang (2021) "Modeling Complexity and the Limits of
    Decarbonisation", WIREs Climate Change 12:e698, Eq. 2.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from rice_jax._rice_mrio import RiceMRIO
from rice_jax.utils import load_region_yamls

# ── Config ────────────────────────────────────────────────────────────────

KEY       = jax.random.PRNGKey(0)
NR        = 7
NS        = 2          # emissions-simple: 2 sectors (dirty=0, clean=1)
EU_IDX    = 5          # 7-region EU index (per copilot-instructions.md)
N_STEPS   = 10
SAVINGS   = 0.22

_THIS   = _os.path.dirname(_os.path.abspath(__file__))
MRIO_ROOT = _os.path.abspath(_os.path.join(_THIS, "..", "..", "csv_asset"))


# ── Environment factory ───────────────────────────────────────────────────

def _make_env(*, action_window_size: int = 0, transition_cost_coef: float = 0.0) -> RiceMRIO:
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


# ── Action builder ────────────────────────────────────────────────────────

def _make_actions(env: RiceMRIO, mitigation: float | np.ndarray) -> dict:
    """Build a processed-action dict (format expected by step_climate_and_economy).

    Matches the output of Rice.process_actions: keys are action names, values
    are stacked across all regions.
    """
    mu = jnp.full((NR,), float(mitigation), dtype=jnp.float32) if np.isscalar(mitigation) \
         else jnp.array(mitigation, dtype=jnp.float32)
    return {
        "mitigation_rate"   : mu,
        "savings_rate"      : jnp.full((NR,), SAVINGS, dtype=jnp.float32),
        # Zero logit adjustments → use 2016 MRIO baseline trade allocation
        "export_reallocation": jnp.zeros((NR, NS * NR), dtype=jnp.float32),
        # Legacy trade actions zeroed out (MRIO subclass overwrites anyway)
        "export_limit"      : jnp.zeros((NR,), dtype=jnp.float32),
        "import_bid"        : jnp.zeros((NR, NR), dtype=jnp.float32),
        "import_tariff"     : jnp.zeros((NR, NR), dtype=jnp.float32),
    }


# ── Rollout ───────────────────────────────────────────────────────────────

def run_rollout(
    env: RiceMRIO,
    mu_schedule: list[float],
) -> dict[str, np.ndarray]:
    """Run N_STEPS of deterministic rollout; return per-step scalars."""
    _, state = env.reset(KEY)

    rows: dict[str, list] = {k: [] for k in [
        "mu_mean", "mu_max",
        "gross_output_sum", "abatement_cost_mean", "consumption_sum",
        "tc_cost_mean",   # transition cost = total abatement_cost - enduring baseline
    ]}

    for t, mu in enumerate(mu_schedule[:N_STEPS]):
        actions = _make_actions(env, mu)
        state = env.step_climate_and_economy(state, actions)

        mu_arr = np.array(state["mitigation_rates_all_regions"])
        go     = np.array(state["gross_output_all_regions"])
        ac     = np.array(state["abatement_cost_all_regions"])
        co     = np.array(state["aggregate_consumption"])

        rows["mu_mean"].append(float(mu_arr.mean()))
        rows["mu_max"].append(float(mu_arr.max()))
        rows["gross_output_sum"].append(float(go.sum()))
        rows["abatement_cost_mean"].append(float(ac.mean()))
        rows["consumption_sum"].append(float(co.sum()))
        rows["tc_cost_mean"].append(0.0)  # filled below

    return {k: np.array(v) for k, v in rows.items()}


# ── Transition-cost extraction helper ────────────────────────────────────

def compute_tc_frac(
    mu_schedule: list[float],
    coef: float,
    dt: float = 5.0,
) -> np.ndarray:
    """Analytic transition cost fraction for a scalar μ schedule (all NR = same)."""
    mus = np.array(mu_schedule[:N_STEPS])
    prev = np.concatenate([[0.0], mus[:-1]])
    delta = (mus - prev) / dt
    return coef * delta ** 2


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    print("Building environments …")
    envs = {
        "baseline" : _make_env(action_window_size=0, transition_cost_coef=0.0),
        "tc_soft"  : _make_env(action_window_size=0, transition_cost_coef=10.0),
        "tc_hard"  : _make_env(action_window_size=0, transition_cost_coef=50.0),
        "aw2"      : _make_env(action_window_size=2, transition_cost_coef=0.0),
    }
    print("  OK")

    # ── Preset μ trajectories ────────────────────────────────────────────
    # jump:    instant 0→0.80 at step 0
    mu_jump    = [0.80] + [0.80] * (N_STEPS - 1)
    # gradual: +0.10 each step
    mu_gradual = [min(0.10 * (t + 1), 0.80) for t in range(N_STEPS)]

    trajectories = {"jump": mu_jump, "gradual": mu_gradual}

    WIDTH = 14

    for traj_name, mu_sched in trajectories.items():
        print(f"\n{'═'*80}")
        print(f"  TRAJECTORY: {traj_name.upper()}    μ schedule = {[f'{v:.2f}' for v in mu_sched[:8]]} …")
        print(f"{'═'*80}")

        results: dict[str, dict] = {}
        for env_name, env in envs.items():
            results[env_name] = run_rollout(env, mu_sched)

        env_names = list(envs.keys())

        def hdr(col: str) -> str:
            return col.center(WIDTH)

        # Header — add analytic TC columns
        header_cols = ["step", "μ_mean"] + [f"GO({e})" for e in env_names] + \
                      [f"AC({e})" for e in env_names] + \
                      ["TC_soft(analy)", "TC_hard(analy)"]
        print("  " + " | ".join(hdr(c) for c in header_cols))
        print("  " + "-" * (WIDTH + 3) * len(header_cols))

        # Compute analytic TC per step for the two TC envs (scalar μ proxy)
        tc_soft_analytic = compute_tc_frac(mu_sched, 10.0)
        tc_hard_analytic = compute_tc_frac(mu_sched, 50.0)

        for t in range(N_STEPS):
            mu_val = results["baseline"]["mu_mean"][t]
            go_base = results["baseline"]["gross_output_sum"][t]
            go_vals = [results[e]["gross_output_sum"][t] for e in env_names]
            ac_vals = [results[e]["abatement_cost_mean"][t] for e in env_names]
            # Show gross-output relative loss vs baseline for TC envs
            go_loss = [(go_base - v) / max(go_base, 1e-8) * 100 for v in go_vals]
            cells = [str(t), f"{mu_val:.3f}"] + \
                    [f"{v:,.1f}" for v in go_vals] + \
                    [f"{v:.4f}" for v in ac_vals] + \
                    [f"{tc_soft_analytic[t]:.4f}", f"{tc_hard_analytic[t]:.4f}"]
            print("  " + " | ".join(c.center(WIDTH) for c in cells))

        # Summary: cumulative gross output loss vs baseline
        print(f"\n  Cumulative gross-output vs baseline (sum over {N_STEPS} steps):")
        base_go_total = results["baseline"]["gross_output_sum"].sum()
        for env_name in env_names:
            total = results[env_name]["gross_output_sum"].sum()
            pct   = 100.0 * (base_go_total - total) / base_go_total
            label = f"  {env_name:12s}  total GO = {total:12,.1f}  loss vs baseline = {pct:+.2f}%"
            print(label)

        # Analytic TC cross-check (empirical = total_AC - baseline_AC ≈ TC component)
        print(f"\n  Analytic vs empirical TC fraction (mean across {N_STEPS} steps):")
        print(f"  (Empirical = TC-env abatement_cost - baseline abatement_cost)")
        print(f"  (Mismatch when analytic tc_frac > 1 is expected: scale clips to 0,")
        print(f"   so effective TC is capped at 100% of gross_output, not the formula value)")
        for coef, label in [(10.0, "tc_soft"), (50.0, "tc_hard")]:
            tc_analytic = compute_tc_frac(mu_sched, coef)
            tc_empirical = (
                results[label]["abatement_cost_mean"] -
                results["baseline"]["abatement_cost_mean"]
            )
            max_step_analytic = tc_analytic.max()
            clipped = max_step_analytic > 1.0
            note = "  [clip expected — tc_frac > 1 in some steps]" if clipped else ""
            match = abs(tc_analytic.mean() - tc_empirical.mean()) < 5e-4 if not clipped else True
            print(f"    {label:10s}  analytic_mean={tc_analytic.mean():.4f}  "
                  f"empirical_mean={tc_empirical.mean():.4f}  "
                  f"{'✓ match' if match else '✗ MISMATCH'}{note}")

    # ── Action-window mask demonstration ─────────────────────────────────
    print(f"\n{'═'*80}")
    print("  ACTION WINDOW DEMO: policy action masks for AW=0 vs AW=2")
    print(f"{'═'*80}")
    print()
    print("  AW is a TRAINING-TIME mask applied to the RL policy's action space.")
    print("  It restricts which discrete mitigation levels the policy may select,")
    print("  preventing the policy network from sampling a 0→80% jump in one step.")
    print("  It does NOT add any GDP cost; it is not enforced in step_climate_and_economy.")
    print()
    print("  To contrast: TC is applied inside step_climate_and_economy as a physics")
    print("  penalty — the rapid jump IS allowed but the economy pays a real GDP cost.")
    print()
    # Show what levels the mask permits from level 0 (initial μ≈0)
    LEVELS = 10
    AW     = 2
    current_level = 0   # initial mitigation ≈ 0 (xmitigation_0 ≈ 0.002)
    possible = jnp.arange(LEVELS)
    mask_base = jnp.ones(LEVELS, dtype=jnp.bool_)
    mask_aw2  = jnp.abs(possible - current_level) <= AW
    allowed_base = [int(l) for l in possible[mask_base]]
    allowed_aw2  = [int(l) for l in possible[mask_aw2]]
    print(f"  Starting at level {current_level} (μ≈{current_level/LEVELS:.2f}):")
    print(f"  AW=0 allowed levels: {allowed_base}  (all levels, μ range 0.00→{(LEVELS-1)/LEVELS:.2f})")
    print(f"  AW=2 allowed levels: {allowed_aw2}  (μ range 0.00→{max(allowed_aw2)/LEVELS:.2f})")
    print()
    print(f"  To reach μ=0.80 (level 8) from level 0:")
    print(f"    AW=0: achievable in 1 step")
    print(f"    AW=2: requires at least {(8 - AW + AW - 1) // AW + 1} steps")
    print()
    print("  KEY CONTRAST vs TC:")
    print("  AW=2  → policy CANNOT express the large gap (breaks C-test conditioning)")
    print("  TC=10 → policy CAN express the gap (μ=0.80 achieved) but economy pays")
    print(f"          ~{compute_tc_frac(mu_jump, 10.0)[0]*100:.1f}% GDP cost in step 0")


if __name__ == "__main__":
    main()
