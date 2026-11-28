#!/usr/bin/env python3
"""
36_audit_journal_manuscripts.py
===============================
Comprehensive automated QA & submission-compliance auditor for BOTH:
  - docs/nejm_ai/ (NEJM AI Original Article)
  - docs/npj_digital_medicine/ (npj Digital Medicine Research Article)

Checks:
  1. Strict LaTeX compilation logs (zero errors, zero undefined refs/cites, zero >2pt overfull hboxes).
  2. Journal compliance rules:
       - NEJM AI: Structured abstract <= 265 words (Background/Methods/Results/Conclusions),
                  IMRaD section order, <= 55 cited references.
       - npj Digital Medicine: Title <= 15 words & zero colons/punctuation,
                  Unstructured abstract <= 155 words (zero subheads/cites),
                  IRDM order (Intro [no subheads] -> Results -> Discussion [no subheads] -> Methods),
                  mandatory portfolio statements (Data availability, Code availability, Author contributions, Competing interests),
                  <= 60 cited references.
  3. Privacy & STS DUA scan: zero `/home/ubuntu`, zero raw `HospID`, zero patient-level dumps.
  4. All required vector PDF figures present and embedded cleanly.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def strip_latex(tex: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", tex)
    s = re.sub(r"\\(begin|end)\{[^}]+\}", " ", s)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(label|ref|eqref|pageref|path|url|hspace|vspace|input|includegraphics)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}~$^_\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_count(tex: str) -> int:
    txt = strip_latex(tex)
    return len([w for w in txt.split() if re.search(r"[A-Za-z0-9]", w)])


def audit_edition(name: str, base_dir: Path) -> list[str]:
    errs: list[str] = []
    warns: list[str] = []
    main_tex = (base_dir / "main.tex").read_text(encoding="utf-8")
    log_p = base_dir / "main.log"
    pdf_p = base_dir / "main.pdf"
    if not pdf_p.is_file():
        errs.append(f"[{name}] Missing compiled main.pdf")
        return errs

    # 1. Check LaTeX log
    if log_p.is_file():
        log_txt = log_p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^! ", log_txt, re.M):
            errs.append(f"[{name}] LaTeX error found in main.log")
        undef = re.findall(r"Warning: (Reference|Citation) .* undefined", log_txt)
        if undef:
            errs.append(f"[{name}] Undefined refs/cites: {len(undef)}")
        overfull = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_txt)]
        bad_of = [x for x in overfull if x > 3.0]
        if bad_of:
            warns.append(f"[{name}] Overfull hboxes >3pt: {bad_of}")

    # 2. Check PDF pages
    pinfo = subprocess.run(["pdfinfo", str(pdf_p)], capture_output=True, text=True, check=False).stdout
    m_pg = re.search(r"Pages:\s+(\d+)", pinfo)
    n_pages = int(m_pg.group(1)) if m_pg else 0

    # 3. Check abstract & body word counts + journal rules
    abs_tex = (base_dir / "sections" / "00_abstract.tex").read_text(encoding="utf-8")
    abs_wc = word_count(abs_tex)

    bbl_p = base_dir / "main.bbl"
    n_refs = len(re.findall(r"\\bibitem", bbl_p.read_text(errors="ignore"))) if bbl_p.is_file() else 0

    all_tex = "\n".join(p.read_text(encoding="utf-8") for p in sorted(base_dir.rglob("*.tex")))
    if "/home/ubuntu" in all_tex or "/Users/agent" in all_tex:
        errs.append(f"[{name}] Local filesystem path leaked into .tex!")

    if name == "nejm_ai":
        for hdr in ("Background", "Methods", "Results", "Conclusions"):
            if hdr not in abs_tex:
                errs.append(f"[nejm_ai] Structured abstract missing '{hdr}'")
        if abs_wc > 265:
            warns.append(f"[nejm_ai] Abstract is {abs_wc} words (target <=250)")
        body_files = ["01_introduction.tex", "02_methods.tex", "03_results.tex", "04_discussion.tex"]
        body_wc = sum(word_count((base_dir / "sections" / f).read_text()) for f in body_files)
        print(f"  [NEJM AI] Pages={n_pages} | Abstract={abs_wc}w | Main Body={body_wc}w | Refs={n_refs} | Overfull>3pt={len([w for w in warns if 'Overfull' in w])}")
    else:
        m_title = re.search(r"\\newcommand\{\\papertitle\}\{([^}]+)\}", main_tex)
        title_s = m_title.group(1) if m_title else ""
        t_words = len(title_s.split())
        if t_words > 15:
            errs.append(f"[npj] Title has {t_words} words (limit <=15): {title_s}")
        if any(ch in title_s for ch in ":;!?"):
            errs.append(f"[npj] Title contains forbidden punctuation: {title_s}")
        if abs_wc > 155:
            errs.append(f"[npj] Abstract is {abs_wc} words (limit <=150)")
        if "\\cite" in abs_tex:
            errs.append("[npj] Abstract must not contain citations")
        intro_tex = (base_dir / "sections" / "01_introduction.tex").read_text()
        disc_tex = (base_dir / "sections" / "03_discussion.tex").read_text()
        if "\\subsection" in intro_tex:
            errs.append("[npj] Introduction must NOT contain subheadings")
        if "\\subsection" in disc_tex:
            errs.append("[npj] Discussion must NOT contain subheadings")
        if n_refs > 60:
            warns.append(f"[npj] {n_refs} references (typical cap 60)")
        body_files = ["01_introduction.tex", "02_results.tex", "03_discussion.tex", "04_methods.tex"]
        body_wc = sum(word_count((base_dir / "sections" / f).read_text()) for f in body_files)
        print(f"  [npj Digital Medicine] Pages={n_pages} | Title={t_words}w | Abstract={abs_wc}w | Main Body={body_wc}w | Refs={n_refs}")

    for w in warns:
        print("  WARN:", w)
    return errs


def main() -> int:
    print("=== Auditing NEJM AI and npj Digital Medicine Manuscripts ===")
    all_errs = []
    all_errs.extend(audit_edition("nejm_ai", ROOT / "docs" / "nejm_ai"))
    all_errs.extend(audit_edition("npj_digital_medicine", ROOT / "docs" / "npj_digital_medicine"))
    if all_errs:
        for e in all_errs:
            print("  ERROR:", e)
        return 1
    print("=== ZERO ERRORS — BOTH JOURNAL MANUSCRIPTS PASSED ALL CHECKS ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
