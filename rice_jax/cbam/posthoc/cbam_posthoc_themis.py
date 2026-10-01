"""cbam_posthoc_themis.py

Posthoc scorecard for the Themis price sweep (cbam_experiment_themis_sweep.py).
Reads the sweep pkl — no retraining.

Outputs (to --out-dir):
    scorecard.md            summary table + invariant checks
    temperature.png         atmospheric T trajectories per price
    mitigation.png          per-region mean μ vs price
    membership.png          per-region membership rate over time, per price
    payments.png            per-region net payments per price
    equity.png              per-capita consumption dispersion vs price

Usage:
    python cbam/posthoc/cbam_posthoc_themis.py --pkl <sweep.pkl> --out-dir <dir>
"""

import argparse
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COST_NEUTRALITY_TOL = 1e-4  # $T/step — |Σ payments| must stay below this


def _load(pkl_path):
    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


def _colors(prices):
    cmap = plt.get_cmap("viridis")
    return {p: cmap(i / max(len(prices) - 1, 1)) for i, p in enumerate(prices)}


def _region_labels(payload):
    names = payload["region_names"]
    return [names[i] for i in range(payload["num_regions"])]


# ── Plots ───────────────────────────────────────────────────────────────────


def plot_temperature(payload, out_path):
    prices = payload["prices"]
    colors = _colors(prices)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in prices:
        temp = payload["results"][p]["eval"]["temp_atmosphere"]  # (E, T)
        mean, sd = temp.mean(0), temp.std(0)
        ax.plot(mean, color=colors[p], label=f"p={p:g} $/t")
        ax.fill_between(range(len(mean)), mean - sd, mean + sd,
                        color=colors[p], alpha=0.15)
    ax.set_xlabel("step")
    ax.set_ylabel("atmospheric ΔT (°C)")
    ax.set_title("Temperature trajectory vs Themis price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mitigation(payload, out_path):
    prices = payload["prices"]
    labels = _region_labels(payload)
    nr = payload["num_regions"]
    colors = _colors(prices)
    width = 0.8 / len(prices)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, p in enumerate(prices):
        mu = payload["results"][p]["eval"]["mitigation"].mean((0, 1))  # (NR,)
        ax.bar(np.arange(nr) + j * width, mu, width, color=colors[p],
               label=f"p={p:g}")
    ax.set_xticks(np.arange(nr) + 0.4)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean μ")
    ax.set_title("Per-region mitigation rate vs Themis price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_membership(payload, out_path):
    prices = payload["prices"]
    labels = _region_labels(payload)
    fig, axes = plt.subplots(
        1, len(prices), figsize=(4 * len(prices), 4), sharey=True, squeeze=False
    )
    for ax, p in zip(axes[0], prices):
        mem = payload["results"][p]["eval"]["membership"].mean(0)  # (T, NR)
        im = ax.imshow(mem.T, aspect="auto", cmap="Greens", vmin=0, vmax=1)
        ax.set_title(f"p={p:g}", fontsize=9)
        ax.set_xlabel("step")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="membership rate")
    fig.suptitle("Themis membership by region and step")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_payments(payload, out_path):
    prices = payload["prices"]
    labels = _region_labels(payload)
    nr = payload["num_regions"]
    colors = _colors(prices)
    width = 0.8 / len(prices)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, p in enumerate(prices):
        pay = payload["results"][p]["eval"]["payments"].mean((0, 1))  # (NR,)
        ax.bar(np.arange(nr) + j * width, pay, width, color=colors[p],
               label=f"p={p:g}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(np.arange(nr) + 0.4)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("net payment ($T/step)")
    ax.set_title("Per-region mean Themis payment (positive = recipient)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_equity(payload, out_path):
    """Dispersion of per-capita consumption across regions vs price."""
    prices = payload["prices"]
    disp = []
    for p in prices:
        ev = payload["results"][p]["eval"]
        cons_pc = ev["consumption"] / np.maximum(ev["labor"], 1e-8)  # (E, T, NR)
        # coefficient of variation across regions, averaged over episodes/steps
        cv = cons_pc.std(-1) / np.maximum(cons_pc.mean(-1), 1e-8)
        disp.append(cv.mean())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prices, disp, "o-")
    ax.set_xlabel("Themis price ($/tCO2e)")
    ax.set_ylabel("CV of per-capita consumption")
    ax.set_title("Cross-region consumption inequality vs Themis price")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return dict(zip(prices, disp))


# ── Scorecard ───────────────────────────────────────────────────────────────


def write_scorecard(payload, equity, out_path):
    prices = payload["prices"]
    lines = [
        "# Themis Price Sweep — Scorecard",
        "",
        f"- env: `{payload['env_kind']}` | membership: `{payload['membership_mode']}` "
        f"| seed: {payload['seed']} | timesteps: {payload['timesteps']:,}",
        "- Mechanism: Rasmussen (2025) Themis concept note §1 — cost-neutral "
        "per-capita-excess carbon payments among members.",
        "",
        "| p ($/t) | final ΔT (°C) | mean μ | membership | max\\|Σpay\\| ($T) | "
        "neutrality | consumption CV |",
        "|---|---|---|---|---|---|---|",
    ]
    neutrality_all_ok = True
    for p in prices:
        ev = payload["results"][p]["eval"]
        residual = float(np.abs(ev["payments"].sum(-1)).max())
        ok = residual < COST_NEUTRALITY_TOL
        neutrality_all_ok &= ok
        lines.append(
            f"| {p:g} | {ev['temp_atmosphere'].mean(0)[-1]:.3f} "
            f"| {ev['mitigation'].mean():.3f} "
            f"| {ev['membership'].mean():.2f} "
            f"| {residual:.2e} "
            f"| {'PASS' if ok else 'FAIL'} "
            f"| {equity[p]:.3f} |"
        )

    p0, p_hi = prices[0], prices[-1]
    t0 = payload["results"][p0]["eval"]["temp_atmosphere"].mean(0)[-1]
    t_hi = payload["results"][p_hi]["eval"]["temp_atmosphere"].mean(0)[-1]
    mu0 = payload["results"][p0]["eval"]["mitigation"].mean()
    mu_hi = payload["results"][p_hi]["eval"]["mitigation"].mean()
    lines += [
        "",
        "## Headline deltas (highest price vs control)",
        "",
        f"- ΔT change: {t_hi - t0:+.3f} °C (p={p_hi:g} vs p={p0:g})",
        f"- mean μ change: {mu_hi - mu0:+.3f}",
        f"- cost neutrality: {'PASS (all prices)' if neutrality_all_ok else 'FAIL'} "
        f"(tol {COST_NEUTRALITY_TOL:g} $T/step)",
        "",
        "## Files",
        "",
        "- temperature.png, mitigation.png, membership.png, payments.png, equity.png",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  scorecard → {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.pkl), "posthoc")
    os.makedirs(out_dir, exist_ok=True)

    payload = _load(args.pkl)
    plot_temperature(payload, os.path.join(out_dir, "temperature.png"))
    plot_mitigation(payload, os.path.join(out_dir, "mitigation.png"))
    plot_membership(payload, os.path.join(out_dir, "membership.png"))
    plot_payments(payload, os.path.join(out_dir, "payments.png"))
    equity = plot_equity(payload, os.path.join(out_dir, "equity.png"))
    write_scorecard(payload, equity, os.path.join(out_dir, "scorecard.md"))
    print(f"  posthoc outputs → {out_dir}")


if __name__ == "__main__":
    main()
