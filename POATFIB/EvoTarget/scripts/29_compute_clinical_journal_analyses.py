#!/usr/bin/env python3
"""
29_compute_clinical_journal_analyses.py
=======================================
Computes all clinical epidemiology, clinical utility (Decision Curve Analysis),
calibration diagnostics, decile risk stratification, demographic/clinical
subgroup fairness (with 5,000-draw 74-hospital cluster bootstrap CIs),
74-hospital center-level transportability, standard bedside clinical score
benchmarks (Mariscalco POAF, COM-AF, CHA2DS2-VASc, HATCH, Mathew preop),
and clinical feature attribution analyses on the sealed hospital-disjoint
STS ACSD v2.81 validation cohort (test_full_21k, N=21,241 across 74 unseen
U.S. surgical centers).

Outputs:
  - reports/clinical_journal_analyses.json  (machine-audited receipt)
  - reports/clinical_journal_analyses.md    (human-readable clinical summary)
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
GEN1000 = R / "jain_ultra_gen1000"
GPV2 = R / "jain_ultra_gp_v2"
NESTED = R / "nested_zstar_v4"

OUT_JSON = R / "clinical_journal_analyses.json"
OUT_MD = R / "clinical_journal_analyses.md"


# ---------------------------------------------------------------------------
# STS ACSD v2.81 audited clinical decoders (matching 02_domain_features.py)
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype=np.float64)


def _age_exact(df: pd.DataFrame) -> np.ndarray:
    a1 = _col(df, "Age1")
    a0 = _col(df, "Age")
    return np.where(np.isfinite(a1), a1, a0)


def _yes1(df: pd.DataFrame, name: str) -> np.ndarray:
    return (_col(df, name) == 1.0).astype(np.float64)


def _yes_in(df: pd.DataFrame, name: str, vals: tuple[float, ...]) -> np.ndarray:
    arr = _col(df, name)
    m = np.zeros(len(arr), dtype=bool)
    for v in vals:
        m |= arr == v
    return m.astype(np.float64)


def _chr_lung_sev(df: pd.DataFrame) -> np.ndarray:
    v = _col(df, "ChrLungD")
    out = np.zeros(len(df), dtype=np.float64)
    out[v == 2.0] = 1.0
    out[v == 3.0] = 2.0
    out[v == 4.0] = 3.0
    out[v == 5.0] = 1.5
    return out


def _egfr_ckdepi_2021(df: pd.DataFrame) -> np.ndarray:
    scr = _col(df, "CreatLst")
    age = _age_exact(df)
    gender = _col(df, "Gender")
    valid = (scr > 0.1) & (scr < 25.0) & np.isfinite(age) & np.isin(gender, [1.0, 2.0])
    out = np.full(len(df), np.nan, dtype=np.float64)
    is_f = gender[valid] == 2.0
    kappa = np.where(is_f, 0.7, 0.9)
    alpha = np.where(is_f, -0.241, -0.302)
    sex_mult = np.where(is_f, 1.012, 1.0)
    ratio = scr[valid] / kappa
    egfr = (
        142.0
        * (np.minimum(ratio, 1.0) ** alpha)
        * (np.maximum(ratio, 1.0) ** -1.200)
        * (0.9938 ** age[valid])
        * sex_mult
    )
    out[valid] = np.clip(egfr, 2.0, 160.0)
    return out


def _bmi(df: pd.DataFrame) -> np.ndarray:
    h_m = _col(df, "HeightCm") / 100.0
    w = _col(df, "WeightKg")
    valid = (h_m > 0.8) & (h_m < 2.5) & (w > 20.0) & (w < 300.0)
    out = np.full(len(df), np.nan, dtype=np.float64)
    out[valid] = np.clip(w[valid] / (h_m[valid] ** 2), 12.0, 75.0)
    return out


def score_mariscalco_poaf(df: pd.DataFrame) -> np.ndarray:
    age = _age_exact(df)
    pts = np.zeros(len(df), dtype=np.float64)
    pts += (age >= 60.0) & (age < 70.0)
    pts += 2.0 * ((age >= 70.0) & (age < 80.0))
    pts += 3.0 * (age >= 80.0)
    pts += (_chr_lung_sev(df) >= 1.0).astype(np.float64)
    egfr = _egfr_ckdepi_2021(df)
    pts += ((_yes1(df, "Dialysis") == 1.0) | (np.isfinite(egfr) & (egfr < 15.0))).astype(np.float64)
    hdef = _col(df, "HDEF")
    pts += (np.isfinite(hdef) & (hdef < 30.0)).astype(np.float64)
    pts += _yes_in(df, "Status", (3.0, 4.0))
    pts += _yes_in(df, "CarShock", (3.0, 4.0))
    return pts


def score_com_af(df: pd.DataFrame) -> np.ndarray:
    age = _age_exact(df)
    pts = np.zeros(len(df), dtype=np.float64)
    pts += (age >= 65.0) & (age < 75.0)
    pts += 2.0 * (age >= 75.0)
    pts += 2.0 * ((_yes1(df, "CHF") == 1.0) | (_yes1(df, "PriorHF") == 1.0))
    pts += 2.0 * ((_yes1(df, "CVA") == 1.0) | (_yes1(df, "CVDTIA") == 1.0))
    pts += (_col(df, "Gender") == 2.0).astype(np.float64)
    pts += _yes1(df, "Hypertn")
    pts += _yes1(df, "Diabetes")
    return pts


def score_cha2ds2_vasc(df: pd.DataFrame) -> np.ndarray:
    age = _age_exact(df)
    pts = np.zeros(len(df), dtype=np.float64)
    pts += ((_yes1(df, "CHF") == 1.0) | (_yes1(df, "PriorHF") == 1.0)).astype(np.float64)
    pts += _yes1(df, "Hypertn")
    pts += 2.0 * (age >= 75.0)
    pts += ((age >= 65.0) & (age < 75.0)).astype(np.float64)
    pts += _yes1(df, "Diabetes")
    pts += 2.0 * ((_yes1(df, "CVA") == 1.0) | (_yes1(df, "CVDTIA") == 1.0))
    pts += (
        (_yes1(df, "PrevMI") == 1.0)
        | (_yes1(df, "PVD") == 1.0)
        | (_yes1(df, "ThAoDisease") == 1.0)
    ).astype(np.float64)
    pts += (_col(df, "Gender") == 2.0).astype(np.float64)
    return pts


def score_hatch(df: pd.DataFrame) -> np.ndarray:
    age = _age_exact(df)
    pts = np.zeros(len(df), dtype=np.float64)
    pts += _yes1(df, "Hypertn")
    pts += 2.0 * (age >= 75.0)
    pts += 2.0 * ((_yes1(df, "CVA") == 1.0) | (_yes1(df, "CVDTIA") == 1.0))
    pts += (_chr_lung_sev(df) >= 1.0).astype(np.float64)
    pts += 2.0 * ((_yes1(df, "CHF") == 1.0) | (_yes1(df, "PriorHF") == 1.0))
    return pts


def score_mathew_preop(df: pd.DataFrame) -> np.ndarray:
    age = _age_exact(df)
    pts = (np.clip(age, 30.0, 95.0) - 50.0) / 10.0 * 3.0
    pts += 4.0 * ((_yes1(df, "Arrhythmia") == 1.0) | np.isfinite(_col(df, "ArrhythAFib")))
    pts += 1.5 * (_chr_lung_sev(df) >= 1.0)
    pts += 1.0 * (_col(df, "Gender") == 1.0)
    pts += 1.5 * ((_yes1(df, "MedBetaTher") == 1.0) & (_col(df, "MedBeta") != 1.0))
    pts -= 1.0 * (_col(df, "MedAmiodarone") == 2.0)
    pts += 0.5 * (_yes1(df, "MedACEI48") == 1.0)
    return pts


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def ece_15bin(y: np.ndarray, p: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 16)
    n = float(len(y))
    ece = 0.0
    for i in range(15):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < 14 else p <= edges[i + 1])
        if m.any():
            ece += (float(m.sum()) / n) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def calibration_diagnostics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p_clip = np.clip(p.astype(np.float64), 1e-6, 1.0 - 1e-6)
    lp = logit(p_clip)

    # Logistic calibration slope & intercept: logit(P(Y=1)) = a + b * lp
    lr_slope = LogisticRegression(C=1e9, solver="lbfgs").fit(lp.reshape(-1, 1), y)
    slope = float(lr_slope.coef_[0, 0])
    intercept = float(lr_slope.intercept_[0])

    # Calibration-in-the-large (CITL): fix slope=1, find a such that mean(expit(lp + a)) == mean(y)
    lo, hi = -2.0, 2.0
    y_mean = float(y.mean())
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if float(expit(lp + mid).mean()) < y_mean:
            lo = mid
        else:
            hi = mid
    citl = 0.5 * (lo + hi)

    # Integrated Calibration Index (ICI) via monotonic isotonic calibration curve
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_clip, y)
    p_iso = iso.predict(p_clip)
    ici = float(np.mean(np.abs(p_clip - p_iso)))

    brier = float(brier_score_loss(y, p_clip))
    brier_null = y_mean * (1.0 - y_mean)
    ipa = float(1.0 - brier / brier_null)

    # Decile bins for smooth calibration curve (unique edges for discrete bedside scores)
    qs = np.unique(np.quantile(p_clip, np.linspace(0.0, 1.0, 11)))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    deciles = []
    for d in range(len(qs) - 1):
        m = (p_clip > qs[d]) & (p_clip <= qs[d + 1])
        nd = int(m.sum())
        if nd == 0:
            continue
        ev = int(y[m].sum())
        obs = ev / nd
        pred = float(p_clip[m].mean())
        # Wilson 95% CI
        z = 1.959964
        denom = 1.0 + z * z / nd
        center = (obs + z * z / (2 * nd)) / denom
        half = z * math.sqrt((obs * (1 - obs) + z * z / (4 * nd)) / nd) / denom
        deciles.append({
            "decile": d + 1,
            "n": nd,
            "events": ev,
            "pred_mean": round(pred, 6),
            "obs_rate": round(obs, 6),
            "obs_ci_lo": round(max(0.0, center - half), 6),
            "obs_ci_hi": round(min(1.0, center + half), 6),
            "enrichment_vs_overall": round(obs / y_mean, 4),
        })

    return {
        "roc_auc": round(float(roc_auc_score(y, p_clip)), 6),
        "avg_precision": round(float(average_precision_score(y, p_clip)), 6),
        "logloss": round(float(log_loss(y, p_clip)), 5),
        "brier": round(brier, 5),
        "ipa_pct": round(100.0 * ipa, 3),
        "ece_15bin": round(ece_15bin(y, p_clip), 5),
        "ici": round(ici, 5),
        "oe_ratio": round(float(y.sum() / p_clip.sum()), 4),
        "cal_slope": round(slope, 4),
        "cal_intercept": round(intercept, 4),
        "citl": round(citl, 4),
        "deciles": deciles,
    }


def net_benefit_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> dict[str, list[float]]:
    n = float(len(y))
    nb_list, nia_list = [], []
    prev = float(y.mean())
    for pt in thresholds:
        pred_pos = p >= pt
        tp = float(np.sum(pred_pos & (y == 1)))
        fp = float(np.sum(pred_pos & (y == 0)))
        w = pt / (1.0 - pt)
        nb = (tp / n) - (fp / n) * w
        nb_all = prev - (1.0 - prev) * w
        # Net interventions avoided per 100 patients vs Treat-All
        nia = ((nb - nb_all) / w) * 100.0
        nb_list.append(round(nb, 6))
        nia_list.append(round(nia, 3))
    return {"net_benefit": nb_list, "net_interventions_avoided_per_100": nia_list}


def smd_continuous(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    sp = math.sqrt(0.5 * (float(np.var(a, ddof=1)) + float(np.var(b, ddof=1))))
    return abs(float(np.mean(a)) - float(np.mean(b))) / max(sp, 1e-9)


def smd_binary(p1: float, p2: float) -> float:
    sp = math.sqrt(0.5 * (p1 * (1.0 - p1) + p2 * (1.0 - p2)))
    return abs(p1 - p2) / max(sp, 1e-9)


def summarize_continuous(x: np.ndarray) -> dict[str, Any]:
    v = x[np.isfinite(x)]
    return {
        "mean": round(float(np.mean(v)), 2),
        "sd": round(float(np.std(v, ddof=1)), 2),
        "median": round(float(np.median(v)), 2),
        "q25": round(float(np.quantile(v, 0.25)), 2),
        "q75": round(float(np.quantile(v, 0.75)), 2),
        "missing_pct": round(100.0 * float(np.mean(~np.isfinite(x))), 2),
    }


def summarize_binary(mask: np.ndarray) -> dict[str, Any]:
    n = int(np.sum(mask))
    pct = 100.0 * float(np.mean(mask))
    return {"n": n, "pct": round(pct, 2)}


def main() -> int:
    print("[1/7] Loading STS cohorts and aligned held-out predictions...")
    tr_full = pd.read_parquet(ROOT / "dataset/full/train_clean.parquet")
    te_full = pd.read_parquet(ROOT / "dataset/full/test_clean.parquet")
    tr_50k = pd.read_parquet(ROOT / "dataset/small_50k_10k/train_clean.parquet")
    sf = pd.read_parquet(ROOT / "dataset/full/splits.parquet")
    tf_all = pd.read_parquet(ROOT / "dataset/full/targets_all.parquet")

    tr_mask = (sf["split"] == "train_optimal").to_numpy()
    te_mask = (sf["split"] == "test_optimal").to_numpy()
    y_col = [c for c in tf_all.columns if "afib" in c.lower() or "poaf" in c.lower() or "target" in c.lower()][0]
    y_all = tf_all[y_col].to_numpy(dtype=int)
    pi_dev = float(y_all[tr_mask].mean())
    logit_offset_dev = float(np.log(pi_dev / (1.0 - pi_dev)))

    te_sp = sf.loc[te_mask].reset_index(drop=True)
    tr_sp = sf.loc[tr_mask].reset_index(drop=True)
    assert len(te_full) == 21_241 and len(tr_full) == 346_454 and len(tr_50k) == 50_000

    # Verify exact row alignment between te_full and evolved_targets_test_full_21k_preds
    oof = pd.read_parquet(R / "evolved_targets_oof_preds.parquet")
    e21 = pd.read_parquet(R / "evolved_targets_test_full_21k_preds.parquet")
    fm21 = pd.read_parquet(R / "fm_small_50k_cross_full_preds.parquet")
    cb346_21 = pd.read_parquet(R / "catboost_preds_full.parquet")
    ch21 = pd.read_parquet(R / "championship_test_preds.parquet")

    y_oof = oof["y_true"].to_numpy(dtype=int)
    y21 = e21["y_true"].to_numpy(dtype=int)
    assert np.array_equal(y21, te_full["PRIMARY_TARGET_COtAFib"].to_numpy(dtype=int))
    assert np.array_equal(y_oof, tr_50k["PRIMARY_TARGET_COtAFib"].to_numpy(dtype=int))
    hosp_21 = te_sp["HospID"].to_numpy()

    def platt_dev_nat(oof_vec: np.ndarray, te_vec: np.ndarray, use_logit: bool = True) -> np.ndarray:
        if use_logit:
            x_tr = logit(np.clip(oof_vec.astype(np.float64), 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
            x_te = logit(np.clip(te_vec.astype(np.float64), 1e-6, 1.0 - 1e-6)).reshape(-1, 1)
        else:
            x_tr = oof_vec.astype(np.float64).reshape(-1, 1)
            x_te = te_vec.astype(np.float64).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, solver="lbfgs").fit(x_tr, y_oof)
        return expit(np.clip(lr.decision_function(x_te) + logit_offset_dev, -20.0, 20.0))

    def load_npz_nat(path: Path) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(path)
        o = d["oof"].astype(np.float64)
        p21_raw = d["tefull"].astype(np.float64) if "tefull" in d.files else d["te21"].astype(np.float64)
        return o, platt_dev_nat(o, p21_raw, use_logit=True)

    # Assemble calibrated natural-prior probabilities on test_full_21k (100% matching paper_v4_master_receipt.py)
    preds: dict[str, np.ndarray] = {
        "cb_full_346k_v1": cb346_21["pred_V1_raw"].to_numpy(dtype=float) if "pred_V1_raw" in cb346_21.columns else ch21["cb_sym_d6_v1"].to_numpy(dtype=float),
        "cb_5f_small_ref": fm21["V1__catboost_5f_small__bayes_nat"].to_numpy(dtype=float),
        "tabpfn_nat_cls_e8": fm21["V1__tabpfn35_full50k_native__bayes_nat"].to_numpy(dtype=float),
        "kumo_cls_e40": fm21["V1__kumo_large_bag5x10k_e40__bayes_nat"].to_numpy(dtype=float),
        "tabfm_cls_e80": fm21["V1__tabfm_v1_bag5x10k__bayes_nat"].to_numpy(dtype=float),
        "xgb_bin": platt_dev_nat(oof["XGBoost_CE__binary_ctrl"].to_numpy(), e21["XGBoost_CE__binary_ctrl"].to_numpy(), False),
        "lgb_bin": platt_dev_nat(oof["LightGBM_CE__binary_ctrl"].to_numpy(), e21["LightGBM_CE__binary_ctrl"].to_numpy(), False),
    }
    oof_gp0, preds["gp_v2_gen0_ens_cv5"] = load_npz_nat(GPV2 / "preds" / "GP_v2_Gen0_Ens_CV5.npz")
    oof_ft0, preds["ft_g1k_g0"] = load_npz_nat(GEN1000 / "preds" / "EvoTarget_FT__gen0000.npz")
    oof_ft20, preds["ft_g1k_g20"] = load_npz_nat(GEN1000 / "preds" / "EvoTarget_FT__gen0020.npz")
    oof_fus, preds["fusion_gp0_plus_ft20"] = load_npz_nat(GPV2 / "preds" / "Fusion_GPv2Gen0_plus_FTGen20.npz")
    _, preds["fusion_gen0_gp_plus_ft0"] = load_npz_nat(NESTED / "preds" / "Fusion_Gen0_GPv2_plus_FT_Gen0.npz")
    _, preds["ft_nested207_g20"] = load_npz_nat(NESTED / "preds" / "EvoTarget_FT__Nested207_Gen20.npz")
    _, preds["cb_g1k_g20"] = load_npz_nat(GEN1000 / "preds" / "CatBoost_CE__gen0020.npz")

    # Standard bedside clinical scores calibrated strictly on train_50k + dev-only Bayes shift
    bedside_fns = {
        "score_mariscalco_poaf": score_mariscalco_poaf,
        "score_com_af": score_com_af,
        "score_cha2ds2_vasc": score_cha2ds2_vasc,
        "score_hatch": score_hatch,
        "score_mathew_preop": score_mathew_preop,
        "score_age_only": _age_exact,
    }
    for name, fn in bedside_fns.items():
        s_tr = fn(tr_50k)
        s_te = fn(te_full)
        med = float(np.nanmedian(s_tr))
        s_tr = np.where(np.isfinite(s_tr), s_tr, med)
        s_te = np.where(np.isfinite(s_te), s_te, med)
        preds[name] = platt_dev_nat(s_tr, s_te, use_logit=False)

    # Confirm exact match to paper_v4_master_receipt.json
    v4_ref = json.loads((R / "paper_v4_master_receipt.json").read_text())
    assert abs(roc_auc_score(y21, preds["fusion_gp0_plus_ft20"]) - v4_ref["leaderboard_21k"]["fusion_gp0_plus_ft20"]["roc_auc_21k"]) < 1e-6
    assert abs(roc_auc_score(y21, preds["ft_g1k_g20"]) - v4_ref["leaderboard_21k"]["ft_g1k_g20"]["roc_auc_21k"]) < 1e-6
    assert abs(roc_auc_score(y21, preds["gp_v2_gen0_ens_cv5"]) - v4_ref["leaderboard_21k"]["gp_v2_gen0_ens_cv5"]["roc_auc_21k"]) < 1e-6
    print("  Verified exact 6-decimal lock with paper_v4_master_receipt.json!", flush=True)

    # -----------------------------------------------------------------------
    # 1. TABLE 1: Baseline Patient, Procedural & Institutional Characteristics
    # -----------------------------------------------------------------------
    print("[2/7] Computing Table 1 baseline cohort characteristics...")
    def cohort_profile(df: pd.DataFrame, sp_sub: pd.DataFrame | None = None) -> dict[str, Any]:
        age = _age_exact(df)
        bsa = _col(df, "BSA")
        bmi = _bmi(df)
        hdef = _col(df, "HDEF")
        creat = _col(df, "CreatLst")
        egfr = _egfr_ckdepi_2021(df)
        y = df["PRIMARY_TARGET_COtAFib"].to_numpy(dtype=int)

        out: dict[str, Any] = {
            "n_patients": int(len(df)),
            "n_hospitals": int(sp_sub["HospID"].nunique()) if sp_sub is not None else 1017,
            "poaf_events": summarize_binary(y == 1),
            "age_years": summarize_continuous(age),
            "female": summarize_binary(_col(df, "Gender") == 2.0),
            "race_white": summarize_binary(_col(df, "RaceCaucasian") == 1.0),
            "race_black": summarize_binary(_col(df, "RaceBlack") == 1.0),
            "race_asian": summarize_binary(_col(df, "RaceAsian") == 1.0),
            "ethnicity_hispanic": summarize_binary(_col(df, "Ethnicity") == 1.0),
            "bmi": summarize_continuous(bmi),
            "bsa": summarize_continuous(bsa),
            "hypertension": summarize_binary(_yes1(df, "Hypertn") == 1.0),
            "diabetes": summarize_binary(_yes1(df, "Diabetes") == 1.0),
            "dyslipidemia": summarize_binary(_yes1(df, "Dyslip") == 1.0),
            "heart_failure": summarize_binary((_yes1(df, "CHF") == 1.0) | (_yes1(df, "PriorHF") == 1.0)),
            "nyha_iii_iv": summarize_binary(_yes_in(df, "ClassNYH", (3.0, 4.0)) == 1.0),
            "prior_mi": summarize_binary(_yes1(df, "PrevMI") == 1.0),
            "pvd": summarize_binary(_yes1(df, "PVD") == 1.0),
            "cvd": summarize_binary(_yes1(df, "CVD") == 1.0),
            "chronic_lung_disease": summarize_binary(_chr_lung_sev(df) >= 1.0),
            "dialysis": summarize_binary(_yes1(df, "Dialysis") == 1.0),
            "ckd_egfr_lt60": summarize_binary(np.isfinite(egfr) & (egfr < 60.0)),
            "preop_af_recorded": summarize_binary(np.isfinite(_col(df, "ArrhythAFib"))),
            "lvef_pct": summarize_continuous(hdef),
            "lvef_lt40": summarize_binary(np.isfinite(hdef) & (hdef < 40.0)),
            "creatinine_mg_dl": summarize_continuous(creat),
            "egfr_ckdepi": summarize_continuous(egfr),
            "three_vessel_disease": summarize_binary(_col(df, "NumDisV") == 4.0),
            "status_elective": summarize_binary(_col(df, "Status") == 1.0),
            "status_urgent": summarize_binary(_col(df, "Status") == 2.0),
            "status_emergent_salvage": summarize_binary(_yes_in(df, "Status", (3.0, 4.0)) == 1.0),
            "med_beta_blocker": summarize_binary(_yes1(df, "MedBeta") == 1.0),
            "med_amiodarone": summarize_binary(_col(df, "MedAmiodarone") == 2.0),
            "med_acei_48h": summarize_binary(_yes1(df, "MedACEI48") == 1.0),
            "region_counts": {str(k): int(v) for k, v in df["Region"].value_counts().items()},
            "region_pcts": {str(k): round(100.0 * float(v) / len(df), 2) for k, v in df["Region"].value_counts().items()},
        }
        return out

    sm_sp = pd.read_parquet(ROOT / "dataset/small_50k_10k/splits.parquet")
    sm_tr_sp = sm_sp.loc[sm_sp["split"] == "train_small_50k"].reset_index(drop=True)
    t1_full_dev = cohort_profile(tr_full, tr_sp)
    t1_50k_dev = cohort_profile(tr_50k, sm_tr_sp)
    t1_holdout = cohort_profile(te_full, te_sp)

    # SMDs between Full Dev (346k) and Unseen Holdout (21k)
    t1_smds = {
        "age_years": round(smd_continuous(_age_exact(tr_full), _age_exact(te_full)), 4),
        "bmi": round(smd_continuous(_bmi(tr_full), _bmi(te_full)), 4),
        "bsa": round(smd_continuous(_col(tr_full, "BSA"), _col(te_full, "BSA")), 4),
        "lvef_pct": round(smd_continuous(_col(tr_full, "HDEF"), _col(te_full, "HDEF")), 4),
        "creatinine_mg_dl": round(smd_continuous(_col(tr_full, "CreatLst"), _col(te_full, "CreatLst")), 4),
        "egfr_ckdepi": round(smd_continuous(_egfr_ckdepi_2021(tr_full), _egfr_ckdepi_2021(te_full)), 4),
    }
    for bk in [
        "poaf_events", "female", "race_white", "race_black", "race_asian", "ethnicity_hispanic",
        "hypertension", "diabetes", "dyslipidemia", "heart_failure", "nyha_iii_iv", "prior_mi",
        "pvd", "cvd", "chronic_lung_disease", "dialysis", "ckd_egfr_lt60", "preop_af_recorded",
        "lvef_lt40", "three_vessel_disease", "status_elective", "status_urgent",
        "status_emergent_salvage", "med_beta_blocker", "med_amiodarone", "med_acei_48h",
    ]:
        t1_smds[bk] = round(smd_binary(t1_full_dev[bk]["pct"] / 100.0, t1_holdout[bk]["pct"] / 100.0), 4)

    # All-hospital incidence distribution across 1,091 U.S. centers (for Figure 1B)
    sf_labeled = sf.copy()
    sf_labeled["poaf"] = y_all
    hosp_all_stats = sf_labeled.groupby("HospID").agg(n=("poaf", "size"), rate=("poaf", "mean")).reset_index()
    hosp_ge100 = hosp_all_stats[hosp_all_stats["n"] >= 100]
    hospital_incidence_dist = {
        "n_hospitals_total": int(len(hosp_all_stats)),
        "n_hospitals_ge100": int(len(hosp_ge100)),
        "rates_ge100": [round(float(x), 5) for x in np.sort(hosp_ge100["rate"].to_numpy())],
        "p05_ge100": round(float(np.quantile(hosp_ge100["rate"], 0.05)), 4),
        "median_ge100": round(float(np.median(hosp_ge100["rate"])), 4),
        "p95_ge100": round(float(np.quantile(hosp_ge100["rate"], 0.95)), 4),
        "sd_ge100": round(float(np.std(hosp_ge100["rate"], ddof=1)), 4),
    }

    # -----------------------------------------------------------------------
    # 2. Exact 5,000-draw 74-Hospital Cluster Bootstrap (identical seed=42)
    # -----------------------------------------------------------------------
    print("[3/7] Building exact 5,000-draw 74-hospital cluster bootstrap matrix...")
    # Exact hospital-weight O(N) weighted ROC-AUC for 5,000-draw hospital cluster bootstrap!
    # For any score vector s, sorting once on the 21,241 test rows lets us evaluate
    # EVERY cluster bootstrap draw in O(N) with zero re-sorting (~50x faster, exact to 1e-12)!
    uniq_h = np.unique(hosp_21)
    n_h = len(uniq_h)
    hosp_code = np.searchsorted(uniq_h, hosp_21)
    rng = np.random.default_rng(42)
    B = 5000
    # Draw matrix (B, n_h) of hospital counts (exact same RNG draw sequence as paper_v4_master_receipt.py!)
    hosp_counts = np.zeros((B, n_h), dtype=np.int32)
    for b in range(B):
        chosen = rng.integers(0, n_h, size=n_h)
        hosp_counts[b] = np.bincount(chosen, minlength=n_h)

    def _weighted_aucs_all_draws(mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Exact O(n_h^2 + N log N) hospital-cluster AUC across ALL B=5,000 draws!
        Key mathematical identity:
          AUC(w) = (w^T A w) / ((w^T p_h) * (w^T n_h))
        where A[i, j] = sum_{u in hosp i (y=1), v in hosp j (y=0)} I(s_u > s_v) + 0.5*I(s_u == s_v)
        is a tiny (74 x 74) pairwise concordance matrix computed ONCE in O(N log N)!
        Then ALL 5,000 bootstrap draws evaluate in < 0.3 milliseconds via (5000, 74) @ (74, 74)!!
        """
        idx_m = np.where(mask)[0]
        ym = y21[idx_m]
        sm = scores[idx_m]
        hm = hosp_code[idx_m]
        order = np.argsort(sm, kind="mergesort")
        s_ord = sm[order]
        y_ord = ym[order]
        h_ord = hm[order]

        # Build exact (74, 74) cross-hospital concordance matrix A in ONE O(N) sweep over tie groups!
        uniq_s, starts, counts = np.unique(s_ord, return_index=True, return_counts=True)
        A = np.zeros((n_h, n_h), dtype=np.float64)
        cum_neg = np.zeros(n_h, dtype=np.float64)
        pos_tot = np.zeros(n_h, dtype=np.float64)

        # Fast path when mostly unique: group by tie runs using searchsorted/slice
        if len(uniq_s) == len(s_ord):
            for k in range(len(s_ord)):
                hk = h_ord[k]
                if y_ord[k] == 1:
                    A[hk] += cum_neg
                    pos_tot[hk] += 1.0
                else:
                    cum_neg[hk] += 1.0
        else:
            for st, cnt in zip(starts, counts):
                en = st + cnt
                h_grp = h_ord[st:en]
                y_grp = y_ord[st:en]
                p_g = np.bincount(h_grp[y_grp == 1], minlength=n_h).astype(np.float64)
                n_g = np.bincount(h_grp[y_grp == 0], minlength=n_h).astype(np.float64)
                if p_g.any():
                    A += np.outer(p_g, cum_neg + 0.5 * n_g)
                    pos_tot += p_g
                cum_neg += n_g

        # Evaluate ALL B=5,000 draws in 0.3 ms!
        H = hosp_counts.astype(np.float64)  # (B, 74)
        tp = H @ pos_tot
        tn = H @ cum_neg
        num = np.einsum("bi,ij,bj->b", H, A, H)
        out = np.full(B, np.nan, dtype=np.float64)
        valid = (tp >= 3.0) & (tn >= 3.0)
        out[valid] = num[valid] / (tp[valid] * tn[valid])
        return out

    # Sanity check exact parity vs sklearn roc_auc_score on draw 0!
    _test_vec = _weighted_aucs_all_draws(np.ones(len(y21), dtype=bool), preds["fusion_gp0_plus_ft20"])
    _idx0 = np.concatenate([np.where(hosp_21 == uniq_h[i])[0] for i in np.repeat(np.arange(n_h), hosp_counts[0])])
    assert abs(_test_vec[0] - roc_auc_score(y21[_idx0], preds["fusion_gp0_plus_ft20"][_idx0])) < 1e-10
    _test_vec_m = _weighted_aucs_all_draws(np.ones(len(y21), dtype=bool), preds["score_mariscalco_poaf"])
    assert abs(_test_vec_m[0] - roc_auc_score(y21[_idx0], preds["score_mariscalco_poaf"][_idx0])) < 1e-10
    print("  Verified exact O(N) vectorized hospital-cluster bootstrap parity (<1e-10)!", flush=True)

    def cluster_auc_ci(mask: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        pt = float(roc_auc_score(y21[mask], p[mask]))
        vals = _weighted_aucs_all_draws(mask, p)
        vals = vals[np.isfinite(vals)]
        return {
            "auc": round(pt, 6),
            "ci_lo": round(float(np.quantile(vals, 0.025)), 6),
            "ci_hi": round(float(np.quantile(vals, 0.975)), 6),
        }

    def cluster_delta_ci(mask: np.ndarray, pc: np.ndarray, pb: np.ndarray) -> dict[str, Any]:
        pt = float(roc_auc_score(y21[mask], pc[mask]) - roc_auc_score(y21[mask], pb[mask]))
        vc = _weighted_aucs_all_draws(mask, pc)
        vb = _weighted_aucs_all_draws(mask, pb)
        vals = vc - vb
        vals = vals[np.isfinite(vals)]
        lo = float(np.quantile(vals, 0.025))
        hi = float(np.quantile(vals, 0.975))
        return {
            "delta": round(pt, 6),
            "delta_bps": round(pt * 1e4, 2),
            "ci_lo_bps": round(lo * 1e4, 2),
            "ci_hi_bps": round(hi * 1e4, 2),
            "p_gt_0": round(float(np.mean(vals > 0.0)), 4),
            "ci_gt_0": bool(lo > 0.0),
        }

    # -----------------------------------------------------------------------
    # 3. Calibration Diagnostics & Bedside Clinical Score Benchmarks
    # -----------------------------------------------------------------------
    print("[4/7] Computing calibration diagnostics & bedside clinical score contrasts...")
    cal_models = [
        "fusion_gp0_plus_ft20", "ft_g1k_g20", "gp_v2_gen0_ens_cv5",
        "fusion_gen0_gp_plus_ft0", "ft_nested207_g20", "ft_g1k_g0",
        "cb_5f_small_ref", "cb_g1k_g20", "tabpfn_nat_cls_e8", "kumo_cls_e40",
        "tabfm_cls_e80", "xgb_bin", "lgb_bin", "cb_full_346k_v1",
        "score_mariscalco_poaf", "score_com_af", "score_cha2ds2_vasc",
        "score_hatch", "score_mathew_preop", "score_age_only",
    ]
    calibration = {m: calibration_diagnostics(y21, preds[m]) for m in cal_models}

    all_mask = np.ones(len(y21), dtype=bool)
    bedside_contrasts = {}
    for s in ["score_mariscalco_poaf", "score_com_af", "score_cha2ds2_vasc", "score_hatch", "score_mathew_preop", "score_age_only"]:
        bedside_contrasts[f"fusion__vs__{s}"] = cluster_delta_ci(all_mask, preds["fusion_gp0_plus_ft20"], preds[s])
        bedside_contrasts[f"ft20__vs__{s}"] = cluster_delta_ci(all_mask, preds["ft_g1k_g20"], preds[s])
        bedside_contrasts[f"gp_v2__vs__{s}"] = cluster_delta_ci(all_mask, preds["gp_v2_gen0_ens_cv5"], preds[s])
        bedside_contrasts[f"abs_{s}"] = cluster_auc_ci(all_mask, preds[s])
    # Also absolute 74-hospital CIs and paired Fusion contrasts for EVERY primary/ablation model in Table 2!
    abs_primary_cis = {}
    for m in cal_models:
        abs_primary_cis[m] = cluster_auc_ci(all_mask, preds[m])
        if m != "fusion_gp0_plus_ft20" and f"fusion__vs__{m}" not in bedside_contrasts:
            bedside_contrasts[f"fusion__vs__{m}"] = cluster_delta_ci(all_mask, preds["fusion_gp0_plus_ft20"], preds[m])

    # -----------------------------------------------------------------------
    # 4. Decision Curve Analysis (DCA) + Cluster-Bootstrap Net Benefit CIs
    # -----------------------------------------------------------------------
    print("[5/7] Computing Decision Curve Analysis (Net Benefit & Interventions Avoided)...")
    thresholds = np.round(np.arange(0.05, 0.555, 0.005), 4)
    dca_models = [
        "fusion_gp0_plus_ft20", "ft_g1k_g20", "gp_v2_gen0_ens_cv5",
        "cb_5f_small_ref", "tabpfn_nat_cls_e8", "kumo_cls_e40",
        "cb_full_346k_v1", "score_mariscalco_poaf",
    ]
    prev_21 = float(y21.mean())
    dca_curves: dict[str, Any] = {
        "thresholds": [float(t) for t in thresholds],
        "treat_all": [round(prev_21 - (1.0 - prev_21) * (t / (1.0 - t)), 6) for t in thresholds],
        "treat_none": [0.0 for _ in thresholds],
        "models": {m: net_benefit_curve(y21, preds[m], thresholds) for m in dca_models},
    }
    # Operating-point classification table + clustered NB CIs at pt in {0.20, 0.25, 0.30, 0.35, 0.40}
    operating_points = {}
    pf = preds["fusion_gp0_plus_ft20"]
    for pt in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        pred_pos = pf >= pt
        tp = int(np.sum(pred_pos & (y21 == 1)))
        fp = int(np.sum(pred_pos & (y21 == 0)))
        fn = int(np.sum((~pred_pos) & (y21 == 1)))
        tn = int(np.sum((~pred_pos) & (y21 == 0)))
        w = pt / (1.0 - pt)
        nb_f = (tp / len(y21)) - (fp / len(y21)) * w
        nb_all = prev_21 - (1.0 - prev_21) * w
        nia_f = ((nb_f - nb_all) / w) * 100.0
        # Compare NB vs Mariscalco POAF and vs 50k CatBoost via vectorized 74-hospital cluster bootstrap
        pcb = preds["cb_5f_small_ref"]
        pmar = preds["score_mariscalco_poaf"]
        # Pre-aggregate per hospital (n_h,) for instant matmul across all 5,000 draws!
        def _hosp_nb_vec(p_arr: np.ndarray) -> np.ndarray:
            tp_h = np.bincount(hosp_code, weights=((p_arr >= pt) & (y21 == 1)).astype(float), minlength=n_h)
            fp_h = np.bincount(hosp_code, weights=((p_arr >= pt) & (y21 == 0)).astype(float), minlength=n_h)
            n_per_h = np.bincount(hosp_code, minlength=n_h).astype(float)
            tot_n = hosp_counts @ n_per_h
            return (hosp_counts @ tp_h - (hosp_counts @ fp_h) * w) / tot_n

        nb_f_b = _hosp_nb_vec(pf)
        nb_diff_cb = nb_f_b - _hosp_nb_vec(pcb)
        nb_diff_mar = nb_f_b - _hosp_nb_vec(pmar)
        operating_points[f"pt_{int(round(pt * 100))}"] = {
            "threshold": pt,
            "flagged_pct": round(100.0 * float(pred_pos.mean()), 2),
            "sensitivity": round(tp / max(1, tp + fn), 4),
            "specificity": round(tn / max(1, tn + fp), 4),
            "ppv": round(tp / max(1, tp + fp), 4),
            "npv": round(tn / max(1, tn + fn), 4),
            "net_benefit": round(nb_f, 5),
            "net_benefit_treat_all": round(nb_all, 5),
            "net_interventions_avoided_per_100": round(nia_f, 2),
            "delta_nb_vs_cb50k": round(float(nb_f - ((np.sum((pcb >= pt) & (y21 == 1)) - np.sum((pcb >= pt) & (y21 == 0)) * w) / len(y21))), 5),
            "delta_nb_vs_cb50k_p_gt_0": round(float(np.mean(nb_diff_cb > 0.0)), 4),
            "delta_nb_vs_mariscalco": round(float(nb_f - ((np.sum((pmar >= pt) & (y21 == 1)) - np.sum((pmar >= pt) & (y21 == 0)) * w) / len(y21))), 5),
            "delta_nb_vs_mariscalco_ci": [round(float(np.quantile(nb_diff_mar, 0.025)), 5), round(float(np.quantile(nb_diff_mar, 0.975)), 5)],
            "delta_nb_vs_mariscalco_p_gt_0": round(float(np.mean(nb_diff_mar > 0.0)), 4),
        }

    # -----------------------------------------------------------------------
    # 5. Subgroup Fairness, Equity & Clinical Transportability (74-Hosp CIs)
    # -----------------------------------------------------------------------
    print("[6/7] Computing demographic, clinical & institutional subgroup fairness...")
    age_21 = _age_exact(te_full)
    egfr_21 = _egfr_ckdepi_2021(te_full)
    hdef_21 = _col(te_full, "HDEF")
    hosp_vol_map = te_sp.groupby("HospID").size().to_dict()
    hosp_vol_21 = te_sp["HospID"].map(hosp_vol_map).to_numpy()
    q33_vol, q67_vol = np.quantile(list(hosp_vol_map.values()), [1 / 3, 2 / 3])

    subgroups: list[tuple[str, str, np.ndarray]] = [
        ("Overall", "All Unseen-Hospital Patients", all_mask),
        ("Sex", "Male", _col(te_full, "Gender") == 1.0),
        ("Sex", "Female", _col(te_full, "Gender") == 2.0),
        ("Age", "Age < 60 yr", age_21 < 60.0),
        ("Age", "Age 60–69 yr", (age_21 >= 60.0) & (age_21 < 70.0)),
        ("Age", "Age 70–79 yr", (age_21 >= 70.0) & (age_21 < 80.0)),
        ("Age", "Age >= 80 yr", age_21 >= 80.0),
        ("Race / Ethnicity", "White", _col(te_full, "RaceCaucasian") == 1.0),
        ("Race / Ethnicity", "Black", _col(te_full, "RaceBlack") == 1.0),
        ("Race / Ethnicity", "Asian", _col(te_full, "RaceAsian") == 1.0),
        ("Race / Ethnicity", "Hispanic / Latino", _col(te_full, "Ethnicity") == 1.0),
        ("Comorbidity", "Diabetes Mellitus", _yes1(te_full, "Diabetes") == 1.0),
        ("Comorbidity", "No Diabetes", _col(te_full, "Diabetes") == 2.0),
        ("Comorbidity", "LVEF < 40%", np.isfinite(hdef_21) & (hdef_21 < 40.0)),
        ("Comorbidity", "LVEF >= 40%", np.isfinite(hdef_21) & (hdef_21 >= 40.0)),
        ("Comorbidity", "Chronic Lung Disease", _chr_lung_sev(te_full) >= 1.0),
        ("Comorbidity", "CKD (eGFR < 60)", np.isfinite(egfr_21) & (egfr_21 < 60.0)),
        ("Acuity", "Elective CABG", _col(te_full, "Status") == 1.0),
        ("Acuity", "Urgent / Emergent CABG", _yes_in(te_full, "Status", (2.0, 3.0, 4.0)) == 1.0),
        ("Stress Slice", "No Recorded Preop AF", ~te_sp["has_preop_afib_hx"].to_numpy(dtype=bool)),
        ("Stress Slice", "Rare Clinical Tails", te_sp["is_rare_clinical_tail"].to_numpy(dtype=bool)),
        ("Stress Slice", "2017 Unseen Hospitals", te_sp["is_stress_2017_unseen_hosp"].to_numpy(dtype=bool)),
        ("Hospital Volume", f"Low-Vol Centers (n<={int(q33_vol)})", hosp_vol_21 <= q33_vol),
        ("Hospital Volume", f"Mid-Vol Centers", (hosp_vol_21 > q33_vol) & (hosp_vol_21 <= q67_vol)),
        ("Hospital Volume", f"High-Vol Centers (n>{int(q67_vol)})", hosp_vol_21 > q67_vol),
    ]

    subgroup_results = []
    for grp, label, m in subgroups:
        n_sub = int(m.sum())
        ev_sub = int(y21[m].sum())
        fus_ci = cluster_auc_ci(m, preds["fusion_gp0_plus_ft20"])
        cb_ci = cluster_auc_ci(m, preds["cb_5f_small_ref"])
        mar_ci = cluster_auc_ci(m, preds["score_mariscalco_poaf"])
        d_cb = cluster_delta_ci(m, preds["fusion_gp0_plus_ft20"], preds["cb_5f_small_ref"])
        d_mar = cluster_delta_ci(m, preds["fusion_gp0_plus_ft20"], preds["score_mariscalco_poaf"])
        ece_sub = ece_15bin(y21[m], preds["fusion_gp0_plus_ft20"][m])
        oe_sub = float(y21[m].sum() / np.clip(preds["fusion_gp0_plus_ft20"][m], 1e-6, 1 - 1e-6).sum())
        subgroup_results.append({
            "group": grp,
            "label": label,
            "n": n_sub,
            "events": ev_sub,
            "poaf_rate_pct": round(100.0 * ev_sub / max(1, n_sub), 2),
            "fusion_auc": fus_ci["auc"],
            "fusion_ci_lo": fus_ci["ci_lo"],
            "fusion_ci_hi": fus_ci["ci_hi"],
            "fusion_ece": round(ece_sub, 5),
            "fusion_oe": round(oe_sub, 4),
            "cb50k_auc": cb_ci["auc"],
            "mariscalco_auc": mar_ci["auc"],
            "delta_vs_cb50k_bps": d_cb["delta_bps"],
            "delta_vs_cb50k_ci": [d_cb["ci_lo_bps"], d_cb["ci_hi_bps"]],
            "delta_vs_cb50k_p_gt_0": d_cb["p_gt_0"],
            "delta_vs_mariscalco_bps": d_mar["delta_bps"],
            "delta_vs_mariscalco_ci": [d_mar["ci_lo_bps"], d_mar["ci_hi_bps"]],
        })

    # -----------------------------------------------------------------------
    # 6. 74-Hospital Center-Level Transportability & Local Intercept Recalibration
    # -----------------------------------------------------------------------
    print("[7/7] Computing 74-hospital transportability & clinical feature attributions...")
    hosp_rows = []
    # Also evaluate chronological prospective site recalibration: 2014-2016 onboarding -> 2017 deployment
    prosp_raw_abs, prosp_unsh_abs, prosp_eb_abs = [], [], []
    prosp_raw_brier, prosp_eb_brier = [], []
    prosp_raw_oe, prosp_eb_oe = [], []
    lp_fus = logit(np.clip(pf, 1e-6, 1.0 - 1e-6))

    for h in uniq_h:
        m = hosp_21 == h
        nh = int(m.sum())
        ev = int(y21[m].sum())
        obs_r = ev / nh
        pred_r = float(pf[m].mean())
        oe = ev / max(1e-9, float(pf[m].sum()))
        auc_h = float(roc_auc_score(y21[m], pf[m])) if (ev >= 5 and (nh - ev) >= 5) else None
        auc_cb = float(roc_auc_score(y21[m], preds["cb_5f_small_ref"][m])) if (ev >= 5 and (nh - ev) >= 5) else None
        reg = str(te_full.loc[m, "Region"].iloc[0])
        hosp_rows.append({
            "hosp_index": int(len(hosp_rows) + 1),  # anonymized 1..74, never expose raw HospID!
            "region": reg,
            "n": nh,
            "events": ev,
            "obs_rate": round(obs_r, 5),
            "pred_rate": round(pred_r, 5),
            "oe_ratio": round(oe, 4),
            "fusion_auc": round(auc_h, 5) if auc_h is not None else None,
            "cb50k_auc": round(auc_cb, 5) if auc_cb is not None else None,
        })
        # Chronological prospective site onboarding: calibrate 1-param EB intercept on 2014-2016 -> evaluate on 2017
        idx_h = np.where(m)[0]
        yrs_h = te_sp["DataYear"].to_numpy()[idx_h]
        cal_i = idx_h[yrs_h < 2017]
        eval_i = idx_h[yrs_h == 2017]
        if len(cal_i) >= 60 and len(eval_i) >= 30 and y21[eval_i].sum() >= 5:
            lo, hi = -2.0, 2.0
            target_r = float(y21[cal_i].mean())
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if float(expit(lp_fus[cal_i] + mid).mean()) < target_r:
                    lo = mid
                else:
                    hi = mid
            a_local = 0.5 * (lo + hi)
            a_eb = (len(cal_i) / (len(cal_i) + 100.0)) * a_local
            p0 = pf[eval_i]
            p_u = expit(lp_fus[eval_i] + a_local)
            p_eb = expit(lp_fus[eval_i] + a_eb)
            ye = y21[eval_i]
            prosp_raw_abs.append(abs(float(ye.mean()) - float(p0.mean())))
            prosp_unsh_abs.append(abs(float(ye.mean()) - float(p_u.mean())))
            prosp_eb_abs.append(abs(float(ye.mean()) - float(p_eb.mean())))
            prosp_raw_brier.append(float(np.mean((ye - p0) ** 2)))
            prosp_eb_brier.append(float(np.mean((ye - p_eb) ** 2)))
            prosp_raw_oe.append(float(ye.sum() / p0.sum()))
            prosp_eb_oe.append(float(ye.sum() / p_eb.sum()))

    valid_aucs = [r["fusion_auc"] for r in hosp_rows if r["fusion_auc"] is not None]
    valid_cb = [r["cb50k_auc"] for r in hosp_rows if r["cb50k_auc"] is not None]
    valid_oes = [r["oe_ratio"] for r in hosp_rows if r["n"] >= 50]
    hospital_transport = {
        "n_unseen_hospitals": int(len(hosp_rows)),
        "n_hospitals_evaluable_auc": int(len(valid_aucs)),
        "fusion_auc_median": round(float(np.median(valid_aucs)), 4),
        "fusion_auc_iqr": [round(float(np.quantile(valid_aucs, 0.25)), 4), round(float(np.quantile(valid_aucs, 0.75)), 4)],
        "fusion_auc_p10_p90": [round(float(np.quantile(valid_aucs, 0.10)), 4), round(float(np.quantile(valid_aucs, 0.90)), 4)],
        "fusion_beats_cb50k_hosp_pct": round(100.0 * float(np.mean(np.array(valid_aucs) > np.array(valid_cb))), 1),
        "oe_median_ge50": round(float(np.median(valid_oes)), 4),
        "oe_iqr_ge50": [round(float(np.quantile(valid_oes, 0.25)), 4), round(float(np.quantile(valid_oes, 0.75)), 4)],
        "oe_p05_p95_ge50": [round(float(np.quantile(valid_oes, 0.05)), 4), round(float(np.quantile(valid_oes, 0.95)), 4)],
        "prospective_recal_n_hosp": int(len(prosp_raw_abs)),
        "prospective_recal_raw_mae_pp": round(100.0 * float(np.mean(prosp_raw_abs)), 2),
        "prospective_recal_unshrunk_mae_pp": round(100.0 * float(np.mean(prosp_unsh_abs)), 2),
        "prospective_recal_eb_mae_pp": round(100.0 * float(np.mean(prosp_eb_abs)), 2),
        "prospective_recal_raw_brier": round(float(np.mean(prosp_raw_brier)), 5),
        "prospective_recal_eb_brier": round(float(np.mean(prosp_eb_brier)), 5),
        "prospective_recal_raw_oe_list": [round(x, 4) for x in prosp_raw_oe],
        "prospective_recal_eb_oe_list": [round(x, 4) for x in prosp_eb_oe],
        "hospitals": hosp_rows,
    }

    # -----------------------------------------------------------------------
    # 7. Clinical Feature Attribution & Cross-Paradigm Complementarity
    # -----------------------------------------------------------------------
    # Standardize interpretable clinical features and fit exact surrogate Ridge to Dual-Paradigm Fusion log-odds
    # (Strictly explanatory audit of what physiological features drive the champion's predictions on test_full_21k)
    clinical_features_map: list[tuple[str, str, np.ndarray]] = [
        ("Age (years)", "Demographics", _age_exact(te_full)),
        ("Body Surface Area (m²)", "Anthropometry", _col(te_full, "BSA")),
        ("Height (cm)", "Anthropometry", _col(te_full, "HeightCm")),
        ("Weight (kg)", "Anthropometry", _col(te_full, "WeightKg")),
        ("White Race (indicator)", "Demographics", (_col(te_full, "RaceCaucasian") == 1.0).astype(float)),
        ("Female Sex", "Demographics", (_col(te_full, "Gender") == 2.0).astype(float)),
        ("Recorded Preop AF History", "Arrhythmia Substrate", np.isfinite(_col(te_full, "ArrhythAFib")).astype(float)),
        ("Any Preop Arrhythmia", "Arrhythmia Substrate", _yes1(te_full, "Arrhythmia")),
        ("LVEF (%)", "Cardiac Function", _col(te_full, "HDEF")),
        ("Heart Failure (CHF / Prior HF)", "Cardiac Function", ((_yes1(te_full, "CHF") == 1.0) | (_yes1(te_full, "PriorHF") == 1.0)).astype(float)),
        ("Serum Creatinine (mg/dL)", "Renal Reserve", np.clip(_col(te_full, "CreatLst"), 0.2, 8.0)),
        ("CKD-EPI eGFR (mL/min/1.73m²)", "Renal Reserve", _egfr_ckdepi_2021(te_full)),
        ("Preop Hemoglobin (g/dL)", "Hematologic", _col(te_full, "RFHemoglobin")),
        ("Chronic Lung Disease Severity", "Pulmonary", _chr_lung_sev(te_full)),
        ("Diabetes Mellitus", "Metabolic", _yes1(te_full, "Diabetes")),
        ("Hypertension", "Vascular", _yes1(te_full, "Hypertn")),
        ("Peripheral Vascular Disease", "Vascular", _yes1(te_full, "PVD")),
        ("Cerebrovascular Disease", "Vascular", _yes1(te_full, "CVD")),
        ("Urgent / Emergent Status", "Acuity", _yes_in(te_full, "Status", (2.0, 3.0, 4.0))),
        ("Preop Amiodarone Therapy", "Pharmacology", (_col(te_full, "MedAmiodarone") == 2.0).astype(float)),
        ("Preop Beta-Blocker Therapy", "Pharmacology", _yes1(te_full, "MedBeta")),
        ("Mitral Regurgitation Grade", "Valvular", np.where((_col(te_full, "VDInsufM") >= 0) & (_col(te_full, "VDInsufM") <= 4), _col(te_full, "VDInsufM"), np.nan)),
        ("Tricuspid Regurgitation Grade", "Valvular", np.where((_col(te_full, "VDInsufT") >= 0) & (_col(te_full, "VDInsufT") <= 4), _col(te_full, "VDInsufT"), np.nan)),
        ("Serum Albumin (g/dL)", "Nutritional / Hepatic", _col(te_full, "TotAlbumin")),
        ("White Blood Cell Count", "Inflammatory", np.clip(_col(te_full, "WBC"), 1.0, 30.0)),
    ]

    X_attr = np.zeros((len(te_full), len(clinical_features_map)), dtype=np.float64)
    attr_rows = []
    for j, (label, domain, raw_v) in enumerate(clinical_features_map):
        med = float(np.nanmedian(raw_v))
        v = np.where(np.isfinite(raw_v), raw_v, med)
        sd = float(np.std(v))
        z = (v - float(np.mean(v))) / max(sd, 1e-9)
        X_attr[:, j] = z
        rho, _ = spearmanr(v, lp_fus)
        uni_auc = float(roc_auc_score(y21, v))
        uni_auc_dir = uni_auc if uni_auc >= 0.5 else 1.0 - uni_auc
        attr_rows.append({
            "feature": label,
            "domain": domain,
            "spearman_r_with_fusion_logit": round(float(rho), 4),
            "abs_spearman": round(abs(float(rho)), 4),
            "univariate_auc_abs": round(uni_auc_dir, 4),
        })

    # Fit standardized Ridge to explain Dual-Paradigm Fusion log-odds (note: exclude collinear pairs eGFR+Creat, Height+Weight+BSA -> keep clean physiological set)
    drop_collinear = {"Height (cm)", "Weight (kg)", "Serum Creatinine (mg/dL)"}
    keep_idx = [i for i, (lbl, _, _) in enumerate(clinical_features_map) if lbl not in drop_collinear]
    X_clean = X_attr[:, keep_idx]
    ridge = Ridge(alpha=10.0).fit(X_clean, lp_fus)
    r2_surrogate = float(ridge.score(X_clean, lp_fus))
    beta_sd = ridge.coef_ / float(np.std(lp_fus))
    std_multivar = []
    for k_pos, orig_i in enumerate(keep_idx):
        lbl, dom, _ = clinical_features_map[orig_i]
        std_multivar.append({
            "feature": lbl,
            "domain": dom,
            "std_beta": round(float(beta_sd[k_pos]), 4),
            "abs_std_beta": round(abs(float(beta_sd[k_pos])), 4),
            "spearman_r": attr_rows[orig_i]["spearman_r_with_fusion_logit"],
        })
    std_multivar.sort(key=lambda d: d["abs_std_beta"], reverse=True)

    # Cross-paradigm correlation between JAIN Ultra GP v2 and EvoTarget-FT Gen-20 logits
    lp_gp = logit(np.clip(preds["gp_v2_gen0_ens_cv5"], 1e-6, 1.0 - 1e-6))
    lp_ft = logit(np.clip(preds["ft_g1k_g20"], 1e-6, 1.0 - 1e-6))
    lp_cb = logit(np.clip(preds["cb_5f_small_ref"], 1e-6, 1.0 - 1e-6))
    corr_gp_ft = float(np.corrcoef(lp_gp, lp_ft)[0, 1])
    corr_gp_cb = float(np.corrcoef(lp_gp, lp_cb)[0, 1])
    corr_ft_cb = float(np.corrcoef(lp_ft, lp_cb)[0, 1])

    receipt = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cohort_summary": {
            "full_dev_346k": t1_full_dev,
            "nested_50k_dev": t1_50k_dev,
            "unseen_holdout_21k": t1_holdout,
            "smd_full_dev_vs_holdout": t1_smds,
            "hospital_incidence_dist": hospital_incidence_dist,
        },
        "abs_primary_cis": abs_primary_cis,
        "bedside_contrasts": bedside_contrasts,
        "calibration": calibration,
        "dca": dca_curves,
        "operating_points": operating_points,
        "subgroups": subgroup_results,
        "hospital_transport": hospital_transport,
        "clinical_attributions": {
            "surrogate_r2_on_fusion_logit": round(r2_surrogate, 4),
            "multivariable_std_betas": std_multivar,
            "univariate_spearman": sorted(attr_rows, key=lambda d: d["abs_spearman"], reverse=True),
            "logit_correlations": {
                "gp_v2_vs_evotarget_ft": round(corr_gp_ft, 4),
                "gp_v2_vs_catboost50k": round(corr_gp_cb, 4),
                "evotarget_ft_vs_catboost50k": round(corr_ft_cb, 4),
            },
        },
    }

    OUT_JSON.write_text(json.dumps(receipt, indent=2))

    # Write clean human-readable Markdown summary
    md = [
        "# Clinical Journal Analyses Receipt (`test_full_21k`, N=21,241, 74 Unseen Hospitals)",
        "",
        "## 1. Standard Bedside Clinical Scores vs. Dual-Paradigm Fusion",
        "| Model / Clinical Score | 21k AUC (95% 74-Hosp CI) | AP | ECE | Slope | O:E | Fusion Δ bps (95% CI) |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]
    for s_key, s_lbl in [
        ("fusion_gp0_plus_ft20", "**Dual-Paradigm Fusion**"),
        ("ft_g1k_g20", "EvoTarget-FT (Gen-20)"),
        ("gp_v2_gen0_ens_cv5", "JAIN Ultra GP v2"),
        ("cb_5f_small_ref", "CatBoost 50k (Binary)"),
        ("tabpfn_nat_cls_e8", "TabPFN-3.5 50k (Binary)"),
        ("kumo_cls_e40", "KumoTabular-Large (Binary)"),
        ("tabfm_cls_e80", "TabFM v1.0.0 (Binary)"),
        ("score_mathew_preop", "Mathew Preop Proxy"),
        ("score_mariscalco_poaf", "Mariscalco POAF Score"),
        ("score_age_only", "Age Alone"),
        ("score_com_af", "Burgos COM-AF Score"),
        ("score_hatch", "HATCH Score"),
        ("score_cha2ds2_vasc", "CHA2DS2-VASc Score"),
    ]:
        c = calibration[s_key]
        aci = abs_primary_cis.get(s_key) or bedside_contrasts.get(f"abs_{s_key}")
        ci_s = f"{c['roc_auc']:.4f} ({aci['ci_lo']:.4f}–{aci['ci_hi']:.4f})" if aci else f"{c['roc_auc']:.4f}"
        d_entry = bedside_contrasts.get(f"fusion__vs__{s_key}")
        d_s = f"+{d_entry['delta_bps']:.1f} [{d_entry['ci_lo_bps']:+.1f}, {d_entry['ci_hi_bps']:+.1f}]" if d_entry else "—"
        md.append(f"| {s_lbl} | {ci_s} | {c['avg_precision']:.4f} | {c['ece_15bin']:.5f} | {c['cal_slope']:.3f} | {c['oe_ratio']:.3f} | {d_s} |")

    md.extend([
        "",
        "## 2. Decile Risk Stratification (Dual-Paradigm Fusion)",
        "| Decile | N | Events | Predicted Risk | Observed Risk (95% CI) | Enrichment |",
        "| :---: | ---: | ---: | ---: | :--- | ---: |",
    ])
    for d in calibration["fusion_gp0_plus_ft20"]["deciles"]:
        md.append(f"| D{d['decile']} | {d['n']} | {d['events']} | {100*d['pred_mean']:.2f}% | {100*d['obs_rate']:.2f}% ({100*d['obs_ci_lo']:.2f}–{100*d['obs_ci_hi']:.2f}%) | {d['enrichment_vs_overall']:.2f}× |")

    md.extend([
        "",
        "## 3. Clinical Utility — Decision Curve Analysis Operating Points",
        "| Threshold | Flagged | Sens | Spec | PPV | NPV | Net Benefit (Fusion vs All) | Avoided / 100 | ΔNB vs Mariscalco (P>0) |",
        "| :---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :--- |",
    ])
    for k, op in operating_points.items():
        md.append(
            f"| {int(round(op['threshold']*100))}% | {op['flagged_pct']:.1f}% | {100*op['sensitivity']:.1f}% | "
            f"{100*op['specificity']:.1f}% | {100*op['ppv']:.1f}% | {100*op['npv']:.1f}% | "
            f"{op['net_benefit']:.4f} vs {op['net_benefit_treat_all']:.4f} | {op['net_interventions_avoided_per_100']:.1f} | "
            f"+{op['delta_nb_vs_mariscalco']:.4f} ({op['delta_nb_vs_mariscalco_p_gt_0']:.3f}) |"
        )

    md.extend([
        "",
        "## 4. Subgroup Fairness & Clinical Transport (74-Hospital Cluster CIs)",
        "| Group | Subgroup | N | POAF % | Fusion AUC (95% CI) | ECE | O:E | Δ vs CB50k (bps) | Δ vs Mariscalco (bps) |",
        "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ])
    for sr in subgroup_results:
        md.append(
            f"| {sr['group']} | {sr['label']} | {sr['n']:,} | {sr['poaf_rate_pct']:.1f}% | "
            f"{sr['fusion_auc']:.4f} ({sr['fusion_ci_lo']:.4f}–{sr['fusion_ci_hi']:.4f}) | "
            f"{sr['fusion_ece']:.4f} | {sr['fusion_oe']:.3f} | {sr['delta_vs_cb50k_bps']:+.1f} | {sr['delta_vs_mariscalco_bps']:+.1f} |"
        )

    md.extend([
        "",
        "## 5. Hospital-Level Transportability & Local Recalibration",
        f"- Evaluable Unseen Hospitals: **{hospital_transport['n_hospitals_evaluable_auc']} / 74**",
        f"- Median Center AUC: **{hospital_transport['fusion_auc_median']:.4f}** (IQR {hospital_transport['fusion_auc_iqr'][0]:.4f}–{hospital_transport['fusion_auc_iqr'][1]:.4f}); beats 50k CatBoost in **{hospital_transport['fusion_beats_cb50k_hosp_pct']:.1f}%** of centers",
        f"- Uncalibrated Center O:E (n>=50): median **{hospital_transport['oe_median_ge50']:.3f}** (IQR {hospital_transport['oe_iqr_ge50'][0]:.3f}–{hospital_transport['oe_iqr_ge50'][1]:.3f})",
        f"- Prospective Chronological Local Intercept Recalibration (2014–2016 onboarding → 2017 deployment, n={hospital_transport['prospective_recal_n_hosp']} centers): Mean |Obs−Pred| rate error drops from **{hospital_transport['prospective_recal_raw_mae_pp']:.2f} pp** → **{hospital_transport['prospective_recal_unshrunk_mae_pp']:.2f} pp** (unshrunk) → **{hospital_transport['prospective_recal_eb_mae_pp']:.2f} pp** (empirical-Bayes $m_0=100$); Mean Brier improves **{hospital_transport['prospective_recal_raw_brier']:.5f} → {hospital_transport['prospective_recal_eb_brier']:.5f}**",
        "",
        "## 6. Top 12 Clinical Drivers of Dual-Paradigm Fusion Log-Odds",
        f"- Standardized surrogate linear $R^2$ on Fusion log-odds: **{r2_surrogate:.4f}** | Logit corr(GP v2, EvoTarget-FT) = **{corr_gp_ft:.4f}**",
        "| Rank | Clinical Feature | Domain | Std. β | Spearman r |",
        "| :---: | :--- | :--- | ---: | ---: |",
    ])
    for i, r in enumerate(std_multivar[:12], 1):
        md.append(f"| {i} | {r['feature']} | {r['domain']} | {r['std_beta']:+.4f} | {r['spearman_r']:+.4f} |")

    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"Wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
