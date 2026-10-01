"""Posthoc scorecard for a PLS shield ablation artifact.

Usage (from rice_jax/):

    conda run -n rice-jax python pls_club/posthoc/analyze_shield_ablation.py \
        pls_club/plots/pls_club_shield_ablation_naive_0.pkl

Writes ``summary.csv``, ``scorecard.md``, and ``comparison.png`` to a sibling
posthoc directory unless ``--output-dir`` is supplied.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _last_third(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    length = max(1, values.shape[0] // 3)
    return values[-length:]


def _climate_indices(num_steps: int, variant: str) -> np.ndarray:
    """Step indices whose logged info reflects a climate/economy step.

    Info at step index i corresponds to current_timestep = i + 1; the mediator
    variant cycles propose/evaluate/climate, so climate steps satisfy
    (i + 1) % 3 == 0. The naive variant has climate steps only.
    """
    steps = np.arange(num_steps)
    if variant == "mediator":
        return steps[(steps + 1) % 3 == 0]
    return steps


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    total = mask.sum()
    return float((values * mask).sum() / total) if total > 0 else float("nan")


def _iter_leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_leaves(value)
    elif isinstance(tree, (tuple, list)):
        for value in tree:
            yield from _iter_leaves(value)
    else:
        yield np.asarray(tree, dtype=float)


def _region_rewards(rewards: dict, idx: np.ndarray) -> np.ndarray:
    """(steps, regions) reward matrix at the given step indices."""
    region_keys = sorted(k for k in rewards if str(k).startswith("region-"))
    return np.stack(
        [np.asarray(rewards[k], dtype=float)[idx] for k in region_keys], axis=-1
    )


def _metric(info: dict, key: str, default: float = float("nan")) -> np.ndarray:
    value = info.get(key)
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray(value, dtype=float)


def summarize_episode(
    run_metadata: dict, episode: dict, episode_index: int, variant: str
) -> dict[str, object]:
    # keep climate/economy steps only: mediator propose/evaluate steps repeat
    # stale economic state and would dilute every metric
    idx = _climate_indices(len(_metric(episode, "global_emissions")), variant)
    membership = _metric(episode, "club_membership")[idx]
    defectors = _metric(episode, "club_defectors")[idx]
    mitigation = _metric(episode, "mitigation_rates_all_regions")[idx]
    emissions = _metric(episode, "global_emissions")[idx]
    tariffs = _metric(episode, "import_tariffs")[idx]
    tail_membership = _last_third(membership)
    tail_defectors = _last_third(defectors)
    tail_mitigation = _last_third(mitigation)
    tail_emissions = _last_third(emissions)
    tail_tariffs = _last_third(tariffs)

    member_count = tail_membership.sum(axis=-1)
    defect_count = tail_defectors.sum(axis=-1)
    defection_rate = np.divide(
        defect_count,
        member_count,
        out=np.zeros_like(defect_count),
        where=member_count > 0,
    )
    rewards = episode.get("rewards") or {}
    region_rewards = _region_rewards(rewards, idx) if rewards else None
    row = {
        "condition": run_metadata["condition"],
        "seed": run_metadata["seed"],
        "episode": episode_index,
        "shield_strength": run_metadata["shield_strength"],
        "shield_mode": run_metadata["shield_mode"],
        "mean_membership_last_third": float(tail_membership.mean()),
        "final_membership": float(membership[-1].mean()),
        "mean_defector_rate_last_third": float(tail_defectors.mean()),
        "mean_member_defection_rate_last_third": float(defection_rate.mean()),
        "final_defectors": float(defectors[-1].mean()),
        "mean_mitigation_last_third": float(tail_mitigation.mean()),
        "mean_member_mitigation_last_third": _masked_mean(
            tail_mitigation, tail_membership
        ),
        "mean_nonmember_mitigation_last_third": _masked_mean(
            tail_mitigation, 1.0 - tail_membership
        ),
        "mean_global_emissions_last_third": float(tail_emissions.mean()),
        "cumulative_emissions": float(emissions.sum()),
        "mean_import_tariff_last_third": float(tail_tariffs.mean()),
        "mean_reward": (
            float(_last_third(region_rewards).mean())
            if region_rewards is not None
            else float("nan")
        ),
    }
    if "club_min_mitigation" in episode:  # mediator variant only
        row["mean_club_min_mitigation_last_third"] = float(
            _last_third(_metric(episode, "club_min_mitigation")[idx]).mean()
        )
        row["mean_club_tariff_rate_last_third"] = float(
            _last_third(_metric(episode, "club_tariff_rate")[idx]).mean()
        )
        if "mediator" in rewards:
            mediator = np.asarray(rewards["mediator"], dtype=float)[idx]
            row["mean_mediator_reward_last_third"] = float(
                _last_third(mediator).mean()
            )
    return row


def _write_csv(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_scorecard(rows: list[dict[str, object]], artifact: dict, output: Path) -> None:
    conditions = sorted({str(row["condition"]) for row in rows})
    lines = [
        "# PLS Shield Ablation Scorecard",
        "",
        f"Variant: `{artifact['variant']}`  ",
        f"Regions: `{artifact['num_regions']}`  ",
        f"Training timesteps per condition: `{artifact['timesteps']}`",
        "",
        "Metrics are averaged over the final third of each evaluation episode.",
        "",
        "| Condition | Membership | Defector rate | Mitigation | Emissions | Import tariff | Reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        mean = lambda key: np.nanmean([float(row[key]) for row in subset])
        lines.append(
            f"| {condition} | {mean('mean_membership_last_third'):.3f} | "
            f"{mean('mean_member_defection_rate_last_third'):.3f} | "
            f"{mean('mean_mitigation_last_third'):.3f} | "
            f"{mean('mean_global_emissions_last_third'):.3f} | "
            f"{mean('mean_import_tariff_last_third'):.3f} | "
            f"{mean('mean_reward'):.3f} |"
        )
    lines.extend(
        [
            "",
            "The unshielded condition is the training and evaluation control. "
            "The PLS condition downweights below-floor mitigation actions but "
            "does not remove them, so defection remains possible.",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    conditions = sorted({str(row["condition"]) for row in rows})
    metrics = [
        ("mean_membership_last_third", "Membership", (0.0, 1.0)),
        ("mean_member_defection_rate_last_third", "Member defection", (0.0, 1.0)),
        ("mean_mitigation_last_third", "Mitigation", (0.0, 1.0)),
        ("mean_global_emissions_last_third", "Global emissions", None),
        ("mean_import_tariff_last_third", "Mean import tariff", (0.0, 1.0)),
        ("mean_reward", "Mean reward", None),
    ]
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(conditions))
    for axis, (key, label, limits) in zip(axes.flat, metrics):
        means = []
        errors = []
        for condition in conditions:
            values = np.asarray(
                [float(row[key]) for row in rows if row["condition"] == condition]
            )
            means.append(np.nanmean(values))
            errors.append(np.nanstd(values))
        axis.bar(x, means, yerr=errors, capsize=4, color=["#657786", "#d1495b"])
        axis.set_title(label)
        axis.set_xticks(x, conditions)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
    figure.suptitle("PLS shield ablation")
    figure.savefig(output, dpi=160)
    plt.close(figure)


_CONDITION_COLORS = {"unshielded": "#657786", "pls_shield": "#d1495b"}


def _episode_series(episode: dict, variant: str) -> dict[str, np.ndarray]:
    idx = _climate_indices(len(_metric(episode, "global_emissions")), variant)
    membership = _metric(episode, "club_membership")[idx]
    defectors = _metric(episode, "club_defectors")[idx]
    mitigation = _metric(episode, "mitigation_rates_all_regions")[idx]
    emissions = _metric(episode, "global_emissions")[idx]
    member_count = membership.sum(axis=-1)
    series = {
        "membership": membership.mean(axis=-1),
        "defection_rate": np.divide(
            defectors.sum(axis=-1),
            member_count,
            out=np.zeros_like(member_count),
            where=member_count > 0,
        ),
        "member_mitigation": np.array(
            [_masked_mean(mitigation[t], membership[t]) for t in range(len(idx))]
        ),
        "nonmember_mitigation": np.array(
            [_masked_mean(mitigation[t], 1.0 - membership[t]) for t in range(len(idx))]
        ),
        "emissions": emissions,
        "cumulative_emissions": emissions.cumsum(),
    }
    if "club_min_mitigation" in episode:
        series["club_min_mitigation"] = _metric(episode, "club_min_mitigation")[idx]
        series["club_tariff_rate"] = _metric(episode, "club_tariff_rate")[idx]
    return series


def _band(axis, episodes: list[np.ndarray], color: str, label: str, linestyle="-"):
    stacked = np.stack(episodes)
    valid = ~np.isnan(stacked).all(axis=0)
    if not valid.any():
        return
    mean = np.nanmean(stacked[:, valid], axis=0)
    std = np.nanstd(stacked[:, valid], axis=0)
    x = np.flatnonzero(valid)
    axis.plot(x, mean, color=color, linestyle=linestyle, label=label)
    axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


def _plot_trajectories(artifact: dict, output: Path) -> None:
    """Mean +/- std over evaluation episodes, per condition, climate steps only."""
    variant = artifact["variant"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    ax_mem, ax_def, ax_mit, ax_em, ax_cum, ax_extra = axes.flat

    for run in artifact["runs"]:
        condition = str(run["condition"])
        color = _CONDITION_COLORS.get(condition, "#2a9d8f")
        series = [_episode_series(episode, variant) for episode in run["episodes"]]
        pick = lambda key: [s[key] for s in series]
        _band(ax_mem, pick("membership"), color, condition)
        _band(ax_def, pick("defection_rate"), color, condition)
        _band(ax_mit, pick("member_mitigation"), color, f"{condition} members")
        _band(
            ax_mit,
            pick("nonmember_mitigation"),
            color,
            f"{condition} non-members",
            linestyle="--",
        )
        _band(ax_em, pick("emissions"), color, condition)
        _band(ax_cum, pick("cumulative_emissions"), color, condition)
        if "club_min_mitigation" in series[0]:
            _band(ax_extra, pick("club_min_mitigation"), color, f"{condition} floor")
            _band(
                ax_extra,
                pick("club_tariff_rate"),
                color,
                f"{condition} tariff",
                linestyle="--",
            )

    ax_mem.set_title("Club membership fraction")
    ax_mem.set_ylim(-0.05, 1.05)
    ax_def.set_title("Member defection rate")
    ax_def.set_ylim(-0.05, 1.05)
    ax_mit.set_title("Mitigation: members vs non-members")
    ax_mit.set_ylim(-0.05, 1.05)
    ax_em.set_title("Global emissions per step")
    ax_cum.set_title("Cumulative global emissions")
    if variant == "mediator":
        ax_extra.set_title("Mediator club terms")
        ax_extra.set_ylim(-0.05, 1.05)
    else:
        ax_extra.set_title("(fixed club terms — naive variant)")
        ax_extra.set_axis_off()
    for axis in axes.flat:
        if axis.has_data():
            axis.set_xlabel("activity step")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.suptitle(f"PLS shield ablation trajectories — {variant} variant")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a PLS shield ablation")
    parser.add_argument("input", type=Path, help="Pickle emitted by run_shield_ablation.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    with args.input.open("rb") as file:
        artifact = pickle.load(file)
    if artifact.get("experiment") != "pls_club_shield_ablation":
        raise ValueError("Input is not a PLS shield ablation artifact")

    output_dir = args.output_dir or args.input.parent / f"{args.input.stem}_posthoc"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        summarize_episode(run, episode, episode_index, artifact["variant"])
        for run in artifact["runs"]
        for episode_index, episode in enumerate(run["episodes"])
    ]
    _write_csv(rows, output_dir / "summary.csv")
    _write_scorecard(rows, artifact, output_dir / "scorecard.md")
    _plot(rows, output_dir / "comparison.png")
    _plot_trajectories(artifact, output_dir / "trajectories.png")

    print(f"Summary written to {output_dir / 'summary.csv'}")
    print(f"Scorecard written to {output_dir / 'scorecard.md'}")
    print(f"Plot written to {output_dir / 'comparison.png'}")
    print(f"Trajectories written to {output_dir / 'trajectories.png'}")


if __name__ == "__main__":
    main()
