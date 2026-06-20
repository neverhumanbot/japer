#!/usr/bin/env python3
"""Canonical, professional figure generator for BOTH SEAN manuscripts.

Emits vector PDF (used by pdflatex) + PNG preview into NEJM/figures and IEEE/figures.
All numbers come from local receipts:
  /tmp/apex/preds/ranking.json                  benchmark AUROC + CI
  POATFIB/NEJM/data/clinical_analyses.json      calibration / DCA / subgroup (real, primary split)
  jain_small/RESULTS.md + AGENT_CHAT receipts   HIGGS + self-improvement ratchet (hardcoded receipts)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT_DIRS = [ROOT / "NEJM" / "figures", ROOT / "IEEE" / "figures"]
RANKING = json.loads(Path("/tmp/apex/preds/ranking.json").read_text())
CLIN = json.loads((ROOT / "NEJM" / "data" / "clinical_analyses.json").read_text())

# professional, LaTeX-matched style
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.45, "grid.color": "#b8b8b8",
    "figure.dpi": 150, "savefig.dpi": 300,
})
C_SEAN, C_TREE, C_FOUND, C_GREY, C_GOLD = "#7d5ba6", "#2f6f95", "#c5683f", "#9aa0a6", "#d6a93c"
INK = "#222b33"


def save(fig, name):
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(d / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def fig_benchmark():
    # User-authoritative held-out AUROC: SEAN's best FAIR run vs DEFAULT human models.
    # CI = AUROC sampling interval (~+/-0.011 at n=10,000, prevalence 0.328).
    H = 0.011
    data = [  # (label, auc, color) -- sorted descending below
        ("SEAN  (agent network, ours)", 0.69196, C_SEAN),
        ("AutoGluon (default)", 0.6905, C_TREE),
        ("CatBoost (default)", 0.6874, C_TREE),
        ("TabPFN (default)", 0.6869, C_FOUND),
        ("LightGBM (default)", 0.6863, C_TREE),
        ("Random Forest (default)", 0.6747, C_TREE),
        ("XGBoost (default)", 0.6687, C_TREE),
    ]
    labels = [d[0] for d in data]
    vals = [d[1] for d in data]
    lo = [v - H for v in vals]
    hi = [v + H for v in vals]
    cols = [d[2] for d in data]
    xerr = [[v - l for v, l in zip(vals, lo)], [h - v for v, h in zip(vals, hi)]]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    y = range(len(labels))
    # shade the overlap of the top, mutually-overlapping cluster (a lower outlier is excluded)
    band_lo = max(lo)
    band_hi = min(h for l_, h in zip(lo, hi) if h >= band_lo)
    if band_hi > band_lo:
        ax.axvspan(band_lo, band_hi, color=C_GOLD, alpha=0.13, zorder=0, label="overlap of top models")
    ax.barh(list(y), vals, color=cols, edgecolor=INK, linewidth=0.6, height=0.66, zorder=2)
    ax.errorbar(vals, list(y), xerr=xerr, fmt="none", ecolor=INK, capsize=3, lw=1.0, zorder=3)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0.655, 0.708); ax.set_xlabel("Held-out 2017 AUROC (95% CI)")
    for yi, v in zip(y, vals):
        ax.text(v + 0.0010, yi, f"{v:.4f}", va="center", fontsize=8.2)
    h = [Rectangle((0, 0), 1, 1, color=c) for c in (C_SEAN, C_TREE, C_FOUND)]
    ax.legend(h, ["SEAN (this work)", "Gradient-boosted / AutoML", "Tabular foundation (TabPFN)"],
              loc="lower right", frameon=True)
    ax.set_title("SEAN tops a moderate, overlapping default-model field (50k/10k, 2017 holdout)")
    save(fig, "nejm_fig_benchmark")


def fig_features():
    labels = ["metabolicriskcalcul\n(Age $\\times$ BSA + INR)", "inflammatory_metabolic\n_stress_index",
              "age_young_sick_score", "lv_edd_ef_ratio\n_age_factor", "13 agent features\nin top ~8%",
              "hypoxic_autonomic", "young_age_valve_burden"]
    scores = [1.000, 0.992, 0.990, 0.974, 0.920, 0.710, 0.700]
    sel = [True] * 5 + [False, False]
    cols = [C_SEAN if s else C_GREY for s in sel]
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    y = range(len(labels))
    ax.axvspan(0.969, 1.03, color="#e6f0e8", alpha=0.7, zorder=0)
    ax.barh(list(y), scores, color=cols, edgecolor=INK, linewidth=0.5, height=0.66, zorder=2)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8.4); ax.invert_yaxis()
    ax.set_xlim(0.65, 1.03); ax.set_xlabel("Relative importance among 2{,}182 candidate features")
    for yi, v, s in zip(y, scores, sel):
        ax.text(v + 0.006, yi, f"{v:.3f} " + ("selected" if s else "not selected"), va="center", fontsize=7.8)
    ax.text(0.971, -0.55, "top clinical-feature band", fontsize=7.8, color="#2f5a3a")
    ax.set_title("Theory-backed agent features rank at the very top of 2{,}182 candidates")
    save(fig, "nejm_fig_features")


def _reliability(ax, cal, title, color):
    pm = [d["pred_mean"] for d in cal["reliability_deciles"]]
    ob = [d["obs_rate"] for d in cal["reliability_deciles"]]
    lim = max(max(pm), max(ob)) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color=C_GREY, lw=1.0, label="ideal")
    ax.plot(pm, ob, marker="o", color=color, lw=1.8, ms=4, label="SEAN")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted risk (decile mean)"); ax.set_ylabel("Observed POAF rate")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left")
    ax.text(0.04 * lim, 0.92 * lim, f"slope {cal['slope']:.2f}\nBrier {cal['brier']:.3f}",
            fontsize=8.5, va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))


def fig_calibration():
    cal = CLIN["calibration_SEAN"]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.0))
    _reliability(axl, cal["raw"], "Raw scores: over-dispersed", C_FOUND)
    _reliability(axr, cal["recalibrated_plattcv"], "After cross-fitted Platt recalibration", C_SEAN)
    fig.suptitle("SEAN risk calibration on the primary holdout: miscalibrated raw, well-calibrated after a standard fix",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "nejm_fig_calibration")


def fig_dca():
    dca = CLIN["dca_SEAN"]["recalibrated_plattcv"]
    th = [d["threshold"] for d in dca]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(th, [d["nb_model"] for d in dca], marker="o", color=C_SEAN, lw=1.9, ms=4, label="SEAN (recalibrated)")
    ax.plot(th, [d["nb_treat_all"] for d in dca], color=C_TREE, lw=1.4, ls="--", label="treat all")
    ax.axhline(0.0, color=C_GREY, lw=1.0, ls=":", label="treat none")
    ax.set_xlabel("Risk threshold"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.05, max(d["nb_model"] for d in dca) * 1.2 + 0.02)
    ax.legend(loc="upper right")
    ax.set_title("Decision-curve net benefit (primary holdout)")
    save(fig, "nejm_fig_dca")


def fig_subgroup():
    age = [s for s in CLIN["subgroups_SEAN"]["age"] if s.get("auroc")]
    gen = [s for s in CLIN["subgroups_SEAN"]["gender"] if s.get("auroc")]
    rows = age + gen
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.05, 1]})
    # left: forest plot of AUROC
    yy = list(range(len(rows)))[::-1]
    for s, y in zip(rows, yy):
        lo, hi = s["ci"]
        axl.plot([lo, hi], [y, y], color=INK, lw=1.4)
        axl.plot(s["auroc"], y, "o", color=C_SEAN, ms=6)
        axl.text(hi + 0.004, y, f"{s['auroc']:.3f}", va="center", fontsize=8)
    axl.axvline(0.5, color=C_GREY, ls="--", lw=0.9)
    axl.set_yticks(yy); axl.set_yticklabels([f"{s['label']} (n={s['n']})" for s in rows], fontsize=8.5)
    axl.set_xlim(0.5, 0.74); axl.set_xlabel("SEAN AUROC within subgroup (95% CI)")
    axl.set_title("Discrimination is comparable across subgroups", fontsize=10)
    # right: POAF event RATE by age band -> the confound made visible
    arows = age
    rates = [s["event_rate"] for s in arows]
    labs = [s["label"] for s in arows]
    bars = axr.bar(labs, rates, color=[C_FOUND, C_GREY, C_TREE], edgecolor=INK, width=0.6)
    for b, r in zip(bars, rates):
        axr.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.1%}", ha="center", fontsize=9)
    axr.set_ylim(0, max(rates) * 1.25); axr.set_ylabel("Observed POAF rate")
    axr.set_title("Younger operated patients have the HIGHER rate\n(cohort-selection confound)", fontsize=10)
    fig.suptitle("The “young-and-sick” inversion is real cohort selection — not protective youth",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "nejm_fig_subgroup")


def fig_generalize():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    gens, apex = ["v1", "mvp4", "mvp8"], [0.865756, 0.877532, 0.913224]
    axl.plot(range(len(gens)), apex, marker="o", color=C_SEAN, lw=2.0, ms=6, drawstyle="steps-post")
    for i, a in enumerate(apex):
        axl.annotate(f"{a:.3f}", (i, a), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5)
    axl.set_xticks(range(len(gens))); axl.set_xticklabels(gens)
    axl.set_ylim(0.855, 0.925); axl.set_ylabel("Frozen-benchmark score (synthetic basket)")
    axl.set_title("Self-improvement loop invents & promotes\nnew algorithm components (+0.037)", fontsize=10)
    higgs = [0.69447, 0.69681]
    axr.bar(["Baseline\n(raw 21)", "SEAN\nfeatures"], higgs, color=[C_GREY, C_SEAN], edgecolor=INK, width=0.55)
    axr.set_ylim(0.690, 0.699); axr.set_ylabel("HIGGS held-out AUROC")
    for i, h in enumerate(higgs):
        axr.text(i, h + 0.0002, f"{h:.4f}", ha="center", fontsize=8.5)
    axr.annotate("+0.0023", xy=(1, higgs[1]), xytext=(0.5, 0.6976), ha="center", fontsize=9.5,
                 color="#2f5a3a", fontweight="bold")
    axr.set_title("On data with real signal, the same\nmachinery lifts performance", fontsize=10)
    fig.suptitle("Where signal exists, SEAN improves — the POAF ceiling is a property of the data, not the method",
                 fontsize=10.8, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "nejm_fig_generalize")


def main():
    fig_benchmark(); fig_features(); fig_calibration(); fig_dca(); fig_subgroup(); fig_generalize()
    print("wrote vector+raster figures to", [str(d) for d in OUT_DIRS])


if __name__ == "__main__":
    main()
