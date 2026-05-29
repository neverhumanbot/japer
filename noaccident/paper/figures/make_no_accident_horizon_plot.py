#!/usr/bin/env python3
"""Generate the No-Accident Horizon vector plot."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent


def saturating_curve(x: np.ndarray, limit: float, rate: float) -> np.ndarray:
    scaled = (1.0 - np.exp(-rate * x)) / (1.0 - np.exp(-rate))
    return 1.0 + (limit - 1.0) * scaled


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.linspace(0.0, 1.0, 400)
    conservative = saturating_curve(x, limit=30.0, rate=3.1)
    aggressive = saturating_curve(x, limit=500.0, rate=2.35)

    fig, ax = plt.subplots(figsize=(7.4, 4.15), constrained_layout=True)

    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(700.0, 0.82)

    band_color = "#e7edf4"
    conservative_color = "#197a54"
    aggressive_color = "#aa3030"
    horizon_color = "#34495e"
    grid_color = "#d7dde5"

    ax.axhspan(30.0, 500.0, facecolor=band_color, edgecolor="none", zorder=0)
    ax.plot(x, conservative, color=conservative_color, linewidth=2.4, zorder=3)
    ax.plot(x, aggressive, color=aggressive_color, linewidth=2.4, zorder=3)
    ax.axhline(100.0, color=horizon_color, linewidth=1.5, linestyle=(0, (5, 4)), zorder=2)

    ax.text(0.18, 52.0, "No-Accident Horizon", color=horizon_color, weight="bold")
    ax.text(0.70, 25.0, "Conservative", color=conservative_color, weight="bold")
    ax.text(0.70, 380.0, "Aggressive", color=aggressive_color, weight="bold")
    ax.text(0.18, 92.0, "100x planning limit", color=horizon_color)
    ax.text(0.025, 1.15, "1x current", color="#263238", weight="bold")

    ax.set_xlabel("Canonical substrate maturity")
    ax.set_ylabel("Improvement multiplier\n(larger reductions lower on chart)")

    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"])
    yticks = [1, 3, 10, 30, 100, 300, 500]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick}x" for tick in yticks])

    ax.grid(axis="y", which="major", color=grid_color, linewidth=0.8)
    ax.grid(axis="x", which="major", color=grid_color, linewidth=0.6)
    ax.tick_params(axis="both", colors="#263238")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#607d8b")
    ax.spines["bottom"].set_color("#607d8b")

    fig.savefig(OUT_DIR / "no_accident_horizon.pdf", dpi=600)
    fig.savefig(OUT_DIR / "no_accident_horizon.eps", dpi=600)


if __name__ == "__main__":
    main()
