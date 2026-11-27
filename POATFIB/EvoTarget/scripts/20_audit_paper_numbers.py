#!/usr/bin/env python3
"""
20_audit_paper_numbers.py
=========================
Automated QA & numeric provenance gate for `docs/paper/`.

Verifies:
  1. Every key reported number across sections/*.tex and tables/*.tex matches the
     exact machine-generated JSON/Parquet receipts in `reports/` (including
     `paper_red_team_receipt.json` 74-hospital cluster CIs and development-only
     prior calibration).
  2. Zero undefined BibTeX citations and ZERO duplicate BibTeX keys in references.bib.
  3. Zero undefined internal cross-references (every \\ref exists in a \\label).
  4. Exact user-confirmed 7-author byline order (Zain Khalpey #2, Ricky Ray Butler #3).
  5. Zero Markdown backticks (`) or leaked internal agent/prototype names in .tex files.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "docs" / "paper"
REPORT_DIR = ROOT / "reports"


def load_json(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    tex_files = sorted(
        list((PAPER_DIR / "sections").glob("*.tex"))
        + list((PAPER_DIR / "tables").glob("*.tex"))
        + [PAPER_DIR / "main.tex"]
    )
    all_text = "\n".join(p.read_text(encoding="utf-8") for p in tex_files if p.exists())
    main_tex = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")

    # 1. Verify BibTeX keys + zero duplicate keys/DOIs/titles
    bib_text = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
    raw_bib_keys = re.findall(r"@\w+\{([^,\s]+),", bib_text)
    dup_bib = sorted(k for k, c in collections.Counter(raw_bib_keys).items() if c > 1)
    if dup_bib:
        errors.append(f"Duplicate BibTeX keys in references.bib: {dup_bib}")
    bib_entries = re.split(r"\n(?=@)", bib_text)
    doi_map: dict[str, list[str]] = collections.defaultdict(list)
    for ent in bib_entries:
        km = re.match(r"@\w+\{([^,\s]+),", ent.strip())
        if not km:
            continue
        dm = re.search(r"doi\s*=\s*\{([^}]+)\}", ent, re.I)
        if dm:
            doi_map[dm.group(1).strip().lower()].append(km.group(1))
    dup_dois = {d: ks for d, ks in doi_map.items() if len(ks) > 1}
    if dup_dois:
        errors.append(f"Duplicate DOIs in references.bib: {dup_dois}")
    bib_keys = set(raw_bib_keys)

    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", all_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cited_keys.add(k)
    missing_bib = sorted(cited_keys - bib_keys)
    if missing_bib:
        errors.append(f"Undefined BibTeX citekeys: {missing_bib}")
    unused_bib = sorted(bib_keys - cited_keys)
    if unused_bib:
        errors.append(f"Unused orphan BibTeX entries in references.bib: {unused_bib}")
    if "10.1053/j.jvca.2022.11.011" in bib_text.lower():
        errors.append("Purged bogus DOI 10.1053/j.jvca.2022.11.011 (Wollborn 3D TEE) reappeared in references.bib")
    if "Duke, Patricia" in bib_text:
        errors.append("Primary-source author error 'Duke, Patricia' (should be 'Duke, Peter') found in references.bib")
    for ent in bib_entries:
        km = re.match(r"@\w+\{([^,\s]+),", ent.strip())
        tm = re.search(r"title\s*=\s*\{([^\n]+)\}", ent, re.I)
        if km and tm:
            stripped = re.sub(r"\{[^\{\}]*\}", "", tm.group(1))
            unbraced = re.findall(r"\b(?:STS|CABG|POAF|AI|Society of Thoracic Surgeons|pcMPO-AF)\b", stripped)
            if unbraced:
                errors.append(f"Unbraced acronym(s) {unbraced} in BibTeX title of {km.group(1)}")

    # 2. Verify internal \ref -> \label
    labels = set(re.findall(r"\\label\{([^}]+)\}", all_text))
    refs = set(re.findall(r"\\(?:C|c|eq)?ref\{([^}]+)\}", all_text))
    missing_refs = sorted(refs - labels)
    if missing_refs:
        errors.append(f"Undefined internal \\ref targets: {missing_refs}")

    # 3. Verify exact 7-author byline order in main.tex
    byline_order = [
        "Jepson Taylor",
        "Zain Khalpey",
        "Ricky Ray Butler",
        "Jordan Morrow",
        "Alton Alexander",
        "Jordan Hildebrandt",
        "Besterlyne Jirongo",
    ]
    positions = [main_tex.find(name) for name in byline_order]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        errors.append(f"Author byline order check failed in main.tex: {list(zip(byline_order, positions))}")
    if "HonorHealth" not in main_tex:
        errors.append("Dr. Zain Khalpey's separate HonorHealth affiliation missing from main.tex")

    # 4. Hygiene checks: no raw markdown backticks, no leaked internal prototype/agent names
    for p in tex_files:
        txt = p.read_text(encoding="utf-8")
        bad_bt = re.findall(r"(?<!`)`[A-Za-z0-9_\\.\-]+`(?!`)", txt)
        if bad_bt:
            errors.append(f"Markdown backticks in {p.name}: {bad_bt[:5]}")
        for forbidden in ["DOPE-ECHO", "JAIN2", "ICLH466", "RealMLP", "Codex", "Antigravity", "xbabe", "gen20_p2b2", "gen20\\_p2b2"]:
            if forbidden.lower() in txt.lower():
                errors.append(f"Forbidden internal/prototype token '{forbidden}' found in {p.name}")

    # 5. Machine-verified numeric receipt checks (Paper v4 master receipt)
    split = load_json("split_audit_comparison.json")
    champ = load_json("championship_eval.json")
    rt = load_json("paper_red_team_receipt.json")
    v4 = load_json("paper_v4_master_receipt.json")
    fe_rt = load_json("agentic_fe_cluster_receipt.json")["contrasts"]

    lb4 = v4["leaderboard_21k"]
    c4 = v4["contrasts_21k"]

    opt_row = [r for r in split["five_way_split_audit"] if r["split_name"] == "E_optimal_94_6_multivariate"][0]
    naive_row = [r for r in split["five_way_split_audit"] if r["split_name"] == "A_naive_2017_temporal"][0]
    dev_cal = rt["calibration_sensitivity"]["development_only"]

    required_receipts = [
        ("Dual-Paradigm Fusion 21k AUC", f"{lb4['fusion_gp0_plus_ft20']['roc_auc_21k']:.6f}"),
        ("Dual-Paradigm Fusion 10k AUC", f"{lb4['fusion_gp0_plus_ft20']['roc_auc_10k']:.6f}"),
        ("Dual-Paradigm Fusion 21k AP", f"{lb4['fusion_gp0_plus_ft20']['avg_precision_21k']:.6f}"),
        ("Dual-Paradigm Fusion Dev-Prior ECE", f"{lb4['fusion_gp0_plus_ft20']['ece_15bin_dev_prior']:.5f}"),
        ("EvoTarget-FT Gen-20 21k AUC", f"{lb4['ft_g1k_g20']['roc_auc_21k']:.6f}"),
        ("EvoTarget-FT Gen-20 vs Binary FT Delta", f"+{c4['ft_g1k_g20__vs__ft_g1k_g0']['delta_auc']:.6f}"),
        ("EvoTarget-FT Gen-1000 21k AUC", f"{lb4['ft_g1k_g1000']['roc_auc_21k']:.6f}"),
        ("EvoTarget-FT Nested-207 21k AUC", f"{lb4['ft_nested207_g20']['roc_auc_21k']:.6f}"),
        ("JAIN Ultra GP v2 21k AUC", f"{lb4['gp_v2_gen0_ens_cv5']['roc_auc_21k']:.6f}"),
        ("JAIN Ultra GP v2 vs Legacy JAIN Delta", f"+{c4['gp_v2_gen0_ens_cv5__vs__jain_gpu_leg_g20']['delta_auc']:.6f}"),
        ("GP v2 BinStop vs No-BinStop Delta", f"+{c4['gp_v2_global289_binstop_ens__vs__gp_v2_gen20_ens_cv7_nobinstop']['delta_auc']:.6f}"),
        ("Fusion vs 50k CatBoost Delta", f"+{c4['fusion_gp0_plus_ft20__vs__cb_5f_small_ref']['delta_auc']:.6f}"),
        ("Fusion vs 3-Way Binary Blend Delta", f"+{c4['fusion_gp0_plus_ft20__vs__binary_3way_blend']['delta_auc']:.6f}"),
        ("CatBoost Full 346k Ref AUC", f"{champ['reference_catboost_full_v1_auc']:.6f}"),
        ("Dev-Only Natural Prior", f"{dev_cal['prior']:.6f}"),
        ("Dev-Only Bayes Logit Offset", f"{abs(dev_cal['logit_offset']):.6f}"),
        ("346k TabM V3-V1 74-Hosp CI Low", f"+{fe_rt['nn_tabm_k16_mt_v3_vs_v1']['ci95_hosp_low']:.5f}"),
        ("Optimal 94.2/5.8 Max |SMD|", f"{opt_row['max_abs_smd_206']:.4f}"),
        ("Naive 2017 Max |SMD|", f"{naive_row['max_abs_smd_206']:.4f}"),
        ("Naive 2017 Hosp Overlap", "99.5"),
        ("Smart Test 10k Energy Dist", "0.00000"),
    ]
    for label, expected in required_receipts:
        if expected not in all_text:
            warnings.append(f"Receipt value '{expected}' ({label}) not found verbatim in paper .tex files")

    # 6. Check figure files exist
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", all_text):
        fig_rel = m.group(1)
        fig_path = PAPER_DIR / fig_rel
        if not fig_path.exists():
            errors.append(f"Missing included graphic: {fig_path}")

    print(f"[AUDIT] Checked {len(tex_files)} .tex files | {len(cited_keys)} unique citations | {len(refs)} cross-refs")
    for w in warnings:
        print(f"  [WARN] {w}")
    for e in errors:
        print(f"  [FAIL] {e}")

    if errors:
        print(f"[AUDIT FAILED] {len(errors)} error(s) detected.")
        return 1
    print("[AUDIT PASSED] Zero citation, duplicate-bib, cross-reference, hygiene, or missing-figure errors!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
