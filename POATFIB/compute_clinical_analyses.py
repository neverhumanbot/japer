#!/usr/bin/env python3
"""Compute REAL clinical-validation analyses for SEAN on the primary 50k/10kprime
held-out 2017 test set, from local prediction receipts. No fabricated numbers.

Inputs (all row-aligned, n=10,000):
  /tmp/apex/preds/y_test.npy                 int64 0/1 labels (prev ~0.328)
  /tmp/apex/preds/preds_jain.txt             SEAN probabilities (one per line)
  /tmp/apex/preds/preds_<model>.npy          baseline probabilities
  .../sts_mimo_afib_50k10kprime_test.csv     features incl. Age, Gender, TARGET_COtAFib

Outputs:
  POATFIB/NEJM/data/clinical_analyses.json  (+ copy to IEEE/data)
  prints a summary with sanity checks vs ranking.json
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

PRED = Path("/tmp/apex/preds")
TEST_CSV = Path("/home/ubuntu/zain_scratch/data/sts_mimo_afib/normalized/sts_mimo_afib_50k10kprime_test.csv")
ROOT = Path(__file__).resolve().parent
OUT = [ROOT / "NEJM" / "data" / "clinical_analyses.json",
       ROOT / "IEEE" / "data" / "clinical_analyses.json"]
RANKING = json.loads((PRED / "ranking.json").read_text())
RNG = np.random.default_rng(20260620)

MODELS = {  # display name -> prediction file (SEAN first)
    "SEAN": ("jain", "txt"),
    "AutoGluon": ("autogluon", "npy"),
    "CatBoost": ("catboost", "npy"),
    "LightGBM": ("lightgbm", "npy"),
    "TabPFN": ("tabpfn_default_50000", "npy"),
    "XGBoost": ("xgboost", "npy"),
}


def load_preds(key, kind):
    if kind == "txt":
        return np.array([float(x) for x in (PRED / f"preds_{key}.txt").read_text().split()], dtype=float)
    return np.load(PRED / f"preds_{key}.npy").astype(float)


def rankdata_avg(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j < n and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0  # average rank (1-based)
        i = j
    return ranks


def auc(y, p):
    r = rankdata_avg(p)
    npos = y.sum()
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return 0.5
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def boot_ci(y, p, n_boot=2000):
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        vals[b] = auc(y[idx], p[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---- fast DeLong (Sun & Xu 2014) for paired AUC comparison -------------------
def _midrank(x):
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def delong(y, p1, p2):
    order = np.argsort(-y, kind="mergesort")
    yy = y[order]
    m = int(yy.sum())
    n = len(yy) - m
    preds = np.vstack([p1[order], p2[order]])
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = 2
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = aucs[0] - aucs[1]
    z = delta / math.sqrt(var) if var > 0 else 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    se = math.sqrt(var) if var > 0 else 0.0
    return dict(auc1=float(aucs[0]), auc2=float(aucs[1]), delta=float(delta),
                se=float(se), z=float(z), p=float(p),
                ci=[float(delta - 1.96 * se), float(delta + 1.96 * se)])


# ---- calibration ------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def logistic_fit(X, y, iters=50):
    # Newton-Raphson; X includes intercept column
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = mu * (1 - mu) + 1e-9
        grad = X.T @ (y - mu)
        H = (X * W[:, None]).T @ X
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def calibration(y, p):
    lp = logit(p)
    # slope: y ~ intercept + slope*lp
    X = np.column_stack([np.ones_like(lp), lp])
    b = logistic_fit(X, y.astype(float))
    slope = float(b[1])
    # calibration-in-the-large: y ~ intercept + offset(lp)  (slope fixed at 1)
    eta = lp.copy()
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-np.clip(a + eta, -30, 30)))
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu)) + 1e-9
        a += g / h
    intercept = float(a)
    brier = float(np.mean((p - y) ** 2))
    # reliability deciles
    order = np.argsort(p)
    deciles = []
    for q in np.array_split(order, 10):
        deciles.append(dict(pred_mean=float(p[q].mean()), obs_rate=float(y[q].mean()), n=int(len(q))))
    return dict(slope=slope, intercept_citl=intercept, brier=brier, reliability_deciles=deciles)


def platt_recalibrate_crossfit(y, p, seed=20260620):
    """Cross-fitted Platt scaling: fit logistic y~a+b*logit(p) on one stratified
    half, apply to the other (and vice-versa). Returns pooled recalibrated probs
    so calibration is reported on data not used to fit the recalibrator."""
    rng = np.random.default_rng(seed)
    lp = logit(p)
    pr = np.empty_like(p)
    idx = np.arange(len(y))
    pos = rng.permutation(idx[y == 1])
    neg = rng.permutation(idx[y == 0])
    foldA = np.concatenate([pos[::2], neg[::2]])
    foldB = np.concatenate([pos[1::2], neg[1::2]])
    for fit, app in [(foldA, foldB), (foldB, foldA)]:
        X = np.column_stack([np.ones(len(fit)), lp[fit]])
        b = logistic_fit(X, y[fit].astype(float))
        pr[app] = 1 / (1 + np.exp(-np.clip(b[0] + b[1] * lp[app], -30, 30)))
    return pr


def decision_curve(y, p, thresholds):
    n = len(y)
    prev = y.mean()
    out = []
    for t in thresholds:
        flag = p >= t
        tp = np.sum(flag & (y == 1))
        fp = np.sum(flag & (y == 0))
        nb_model = tp / n - fp / n * (t / (1 - t))
        nb_all = prev - (1 - prev) * (t / (1 - t))
        out.append(dict(threshold=float(t), nb_model=float(nb_model),
                        nb_treat_all=float(nb_all), nb_treat_none=0.0))
    return out


def main():
    y = np.load(PRED / "y_test.npy").astype(int)
    n = len(y)
    preds = {name: load_preds(k, kind) for name, (k, kind) in MODELS.items()}
    for name, p in preds.items():
        assert len(p) == n, f"{name} length {len(p)} != {n}"

    # --- load Age/Gender from test CSV (row-aligned); sanity-check alignment ---
    ages, genders, csv_target = [], [], []
    with open(TEST_CSV) as fh:
        rd = csv.reader(fh)
        header = next(rd)
        idx_age = header.index("Age")
        idx_gender = header.index("Gender")
        idx_t = header.index("TARGET_COtAFib")
        for row in rd:
            def f(v):
                try:
                    return float(v)
                except ValueError:
                    return np.nan
            ages.append(f(row[idx_age]))
            genders.append(f(row[idx_gender]))
            csv_target.append(f(row[idx_t]))
    ages = np.array(ages); genders = np.array(genders); csv_target = np.array(csv_target)
    assert len(ages) == n, f"csv rows {len(ages)} != {n}"
    # alignment check: CSV target binarized (positive value '2') must equal y_test
    bin_csv = (csv_target == 2).astype(int)
    align = float(np.mean(bin_csv == y))
    if align < 0.999:  # try the other coding
        bin_csv = (csv_target == 1).astype(int)
        align = float(np.mean(bin_csv == y))

    results = dict(
        n=n, prevalence=float(y.mean()), split="50k10kprime held-out 2017",
        alignment_csv_vs_ytest=align, n_boot=2000,
        models={}, delong_vs_SEAN={}, calibration_SEAN=None, dca_SEAN=None, subgroups_SEAN={})

    # per-model AUROC + bootstrap CI + cross-check vs ranking.json
    rank_key = {"SEAN": "jain", "AutoGluon": "autogluon", "CatBoost": "catboost",
                "LightGBM": "lightgbm", "TabPFN": "tabpfn_default_50000", "XGBoost": "xgboost"}
    for name, p in preds.items():
        a = auc(y, p)
        lo, hi = boot_ci(y, p)
        ref = RANKING[rank_key[name]]["auc"]
        results["models"][name] = dict(auroc=float(a), ci=[lo, hi],
                                       ranking_json_auc=float(ref), abs_diff=float(abs(a - ref)))

    # paired DeLong: SEAN vs each baseline
    for name, p in preds.items():
        if name == "SEAN":
            continue
        results["delong_vs_SEAN"][name] = delong(y, preds["SEAN"], p)

    # calibration + DCA for SEAN: raw (honest miscalibration) and cross-fitted Platt
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    p_recal = platt_recalibrate_crossfit(y, preds["SEAN"])
    results["calibration_SEAN"] = dict(
        raw=calibration(y, preds["SEAN"]),
        recalibrated_plattcv=calibration(y, p_recal),
    )
    results["dca_SEAN"] = dict(
        raw=decision_curve(y, preds["SEAN"], thresholds),
        recalibrated_plattcv=decision_curve(y, p_recal, thresholds),
    )

    # subgroup AUROC for SEAN
    p_sean = preds["SEAN"]
    def sub(mask, label):
        yy = y[mask]; pp = p_sean[mask]
        if yy.sum() < 10 or (len(yy) - yy.sum()) < 10:
            return dict(label=label, n=int(mask.sum()), note="too small", auroc=None)
        lo, hi = boot_ci(yy, pp, n_boot=1500)
        return dict(label=label, n=int(mask.sum()), event_rate=float(yy.mean()),
                    auroc=float(auc(yy, pp)), ci=[lo, hi])
    valid_age = ~np.isnan(ages)
    results["subgroups_SEAN"]["age"] = [
        sub(valid_age & (ages < 60), "Age <60"),
        sub(valid_age & (ages >= 60) & (ages <= 75), "Age 60-75"),
        sub(valid_age & (ages > 75), "Age >75"),
    ]
    gv = ~np.isnan(genders)
    uniq_g = sorted(set(genders[gv].tolist()))
    glabel = {1.0: "Gender=1", 2.0: "Gender=2"}
    results["subgroups_SEAN"]["gender"] = [
        sub(gv & (genders == g), glabel.get(g, f"Gender={g:g}")) for g in uniq_g[:2]
    ]

    for o in OUT:
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(json.dumps(results, indent=2))

    # ---- summary ----
    print(f"n={n}  prevalence={y.mean():.4f}  csv/ytest alignment={align:.4f}")
    print("AUROC (computed vs ranking.json):")
    for name, m in results["models"].items():
        print(f"  {name:10s} {m['auroc']:.4f} [{m['ci'][0]:.4f},{m['ci'][1]:.4f}]  "
              f"(ranking {m['ranking_json_auc']:.4f}, |Δ|={m['abs_diff']:.5f})")
    print("DeLong SEAN vs baseline (Δ, p):")
    for name, d in results["delong_vs_SEAN"].items():
        print(f"  SEAN vs {name:10s} Δ={d['delta']:+.4f}  p={d['p']:.3f}  z={d['z']:+.2f}")
    cr = results["calibration_SEAN"]["raw"]
    cp = results["calibration_SEAN"]["recalibrated_plattcv"]
    print(f"SEAN calibration RAW:   slope={cr['slope']:.3f}  CITL={cr['intercept_citl']:+.3f}  Brier={cr['brier']:.4f}")
    print(f"SEAN calibration PLATT: slope={cp['slope']:.3f}  CITL={cp['intercept_citl']:+.3f}  Brier={cp['brier']:.4f}")
    print("SEAN subgroup AUROC (age):")
    for s in results["subgroups_SEAN"]["age"]:
        if s["auroc"] is not None:
            print(f"  {s['label']:10s} n={s['n']:5d} rate={s['event_rate']:.3f} AUROC={s['auroc']:.4f} "
                  f"[{s['ci'][0]:.4f},{s['ci'][1]:.4f}]")
        else:
            print(f"  {s['label']:10s} n={s['n']:5d} {s['note']}")
    print("wrote", OUT[0], "and", OUT[1])


if __name__ == "__main__":
    main()
