#!/usr/bin/env python3
"""Generate vector convergence plots from the real training logs.

The raw logs are copied from the training repository into this paper directory:

* convergence_raw_iter_losses.csv records per-step training and validation loss.
* convergence_raw_telemetry.csv records eval events and forbidden-language checks.

Duplicate early steps occur after restarts. For each step, this script keeps the
last observed nonzero value and writes a processed CSV used only as a figure
derivative.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW_LOSSES = ROOT / "convergence_raw_iter_losses.csv"
RAW_TELEMETRY = ROOT / "convergence_raw_telemetry.csv"
PROCESSED = ROOT / "convergence_data.csv"


def read_loss_points() -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_by_step: dict[int, float] = {}
    validation_by_step: dict[int, float] = {}

    with RAW_LOSSES.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            if step <= 0:
                continue

            train_loss = float(row["loss"])
            if train_loss > 0.0:
                train_by_step[step] = train_loss

            validation_loss = float(row["val_loss"])
            if validation_loss > 0.0:
                validation_by_step[step] = validation_loss

    return sorted(train_by_step.items()), sorted(validation_by_step.items())


def read_forbidden_points() -> list[tuple[int, float]]:
    forbidden_by_step: dict[int, float] = {}

    with RAW_TELEMETRY.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["event"] != "eval":
                continue
            step = int(row["step"])
            if step <= 0:
                continue
            forbidden_by_step[step] = float(row["avg_forbidden"])

    return sorted(forbidden_by_step.items())


def write_processed_csv(
    train: list[tuple[int, float]],
    validation: list[tuple[int, float]],
    forbidden: list[tuple[int, float]],
) -> None:
    train_by_step = dict(train)
    validation_by_step = dict(validation)
    forbidden_by_step = dict(forbidden)
    all_steps = sorted(set(train_by_step) | set(validation_by_step) | set(forbidden_by_step))

    with PROCESSED.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "train_loss",
                "validation_loss",
                "avg_forbidden",
            ),
        )
        writer.writeheader()
        for step in all_steps:
            writer.writerow(
                {
                    "step": step,
                    "train_loss": train_by_step.get(step, ""),
                    "validation_loss": validation_by_step.get(step, ""),
                    "avg_forbidden": forbidden_by_step.get(step, ""),
                }
            )


def main() -> None:
    train, validation = read_loss_points()
    forbidden = read_forbidden_points()
    write_processed_csv(train, validation, forbidden)

    train_steps, train_raw_values = zip(*train)
    validation_steps, validation_values = zip(*validation)
    forbidden_steps, forbidden_values = zip(*forbidden)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(7.16, 3.0), constrained_layout=True)
    ax.plot(
        train_steps,
        train_raw_values,
        color="#f1b66f",
        linewidth=0.35,
        label="train loss, raw",
    )
    ax.plot(
        validation_steps,
        validation_values,
        color="#1f77b4",
        linewidth=1.6,
        label="validation loss",
    )

    ax.set_title("QLoRA Convergence on Canonical Coding Trajectories")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_xlim(0, max(train_steps))
    ax.set_ylim(0.35, 2.85)
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0)

    ax2 = ax.twinx()
    ax2.plot(
        forbidden_steps,
        forbidden_values,
        color="#2ca02c",
        linewidth=1.2,
        linestyle="--",
        label="forbidden-language markers",
    )
    ax2.set_ylabel("markers")
    ax2.set_ylim(-0.05, 1.0)
    ax2.set_yticks([0, 1])
    ax2.tick_params(axis="y", colors="#2ca02c")
    ax2.spines["right"].set_color("#2ca02c")

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, loc="upper right", frameon=True, framealpha=1.0)

    for suffix in ("pdf", "eps"):
        fig.savefig(ROOT / f"convergence_plot.{suffix}", format=suffix)

    print(f"raw training rows after restart de-duplication: {len(train)}")
    print(f"validation eval rows after restart de-duplication: {len(validation)}")
    print(f"train loss: first={train[0][1]:.6f} at step {train[0][0]}, final_raw={train[-1][1]:.6f} at step {train[-1][0]}")
    print(
        "validation loss: "
        f"first={validation[0][1]:.6f} at step {validation[0][0]}, "
        f"final={validation[-1][1]:.6f} at step {validation[-1][0]}, "
        f"minimum={min(validation, key=lambda item: item[1])[1]:.6f} at step {min(validation, key=lambda item: item[1])[0]}"
    )
    print(f"max forbidden-language marker average: {max(forbidden_values):.6f}")


if __name__ == "__main__":
    main()
