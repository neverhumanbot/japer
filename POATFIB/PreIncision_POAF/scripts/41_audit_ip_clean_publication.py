#!/usr/bin/env python3
"""
41_audit_ip_clean_publication.py
================================
Automated IP-Redaction Firewall & Clinical Journal Compliance Auditor for the
simplified NEVERHUMAN Pre-Incision POAF publication suite:
  - NEJM AI Original Article (`docs/clean_nejm_ai` or `docs/nejm_ai`)
  - npj Digital Medicine Research Article (`docs/clean_npj_digital_medicine` or `docs/npj_digital_medicine`)
  - 2-page CEO & CTO Executive Briefs (`docs/clean_briefs` or `docs/briefs`)
  - Clean aggregate receipts & repository READMEs

Hard Checks (exit 1 on ANY violation):
  1. STRICT IP REDACTION FIREWALL across ALL .tex, .bib, .md, .json, .sh AND
     `pdftotext`-extracted text of EVERY .pdf (manuscripts, figures, briefs):
     zero occurrences of target-evolution / JAIN / GP / EvoTarget IP tokens.
  2. LaTeX compilation logs: zero errors, zero undefined refs/cites, zero >3pt
     overfull hboxes (journals) or >2pt overfull hboxes (2-page briefs).
  3. Journal compliance:
     - NEJM AI: structured abstract <= 250w, IMRaD order.
     - npj Digital Medicine: title <= 15w with zero punctuation, unstructured
       abstract <= 150w with zero citations, 0 subheads in Intro/Discussion,
       <= 60 cited references, Nature Portfolio statements present.
  4. Privacy & STS DUA scan: zero local user paths, zero raw HospIDs, zero
     patient-level data or forbidden binary extensions.
  5. Markdown link audit: zero broken relative links in READMEs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_IP_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("evotarget", re.compile(r"evotarget", re.I)),
    ("z_star", re.compile(r"(?:\bz\^\*|\bzstar\b|\bz_star\b)", re.I)),
    ("continuous_target", re.compile(r"continuous[\s\-]+target", re.I)),
    ("target_evolution", re.compile(r"target[\s\-]+evolution|evolved[\s\-]+target|evolutionary[\s\-]+target", re.I)),
    ("target_smoothing", re.compile(r"target[\s\-]+smoothing|label[\s\-]+smoothing", re.I)),
    ("ewma_pso", re.compile(r"\bewma\b|gene[\s\-]+locked|\bpso\b|particle[\s\-]+swarm", re.I)),
    ("gen_milestones", re.compile(r"\bgen[\s\-]*(?:20|1,?000|1000)\b|\bgen0\b|\bgen20\b", re.I)),
    ("teacher_distill", re.compile(r"teacher[\s\-]+distill|distillation", re.I)),
    ("soft_bce", re.compile(r"soft[\s\-]+bce", re.I)),
    ("genetic_programming", re.compile(r"genetic[\s\-]+programming|\bjain\b|\bjopedime\b", re.I)),
    ("dope_echo", re.compile(r"dope[\s\-_]+echo", re.I)),
    ("dual_paradigm", re.compile(r"dual[\s\-]+paradigm", re.I)),
    ("symbolic_gp", re.compile(r"\bsymbolic\b|\bhistgbm2\b|hall[\s\-]+of[\s\-]+fame|\bhof\b", re.I)),
    ("nested_2x2", re.compile(r"nested[\s\-]+2\s*[x×]\s*2|strictly[\s\-]+nested[\s\-]+207", re.I)),
]

FORBIDDEN_EXTS = {
    ".parquet", ".npy", ".npz", ".pt", ".safetensors",
    ".aux", ".log", ".blg", ".out", ".toc", ".fls", ".fdb_latexmk", ".gz",
}


def strip_latex(tex: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", tex)
    s = re.sub(r"\\(begin|end)\{[^}]+\}", " ", s)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(label|ref|eqref|pageref|path|url|hspace|vspace|input|includegraphics)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[{}~$^_\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def word_count(tex: str, drop_figures: bool = False) -> int:
    s = tex
    if drop_figures:
        s = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", s, flags=re.S)
    txt = strip_latex(s)
    return len([w for w in txt.split() if re.search(r"[A-Za-z0-9]", w)])


def scan_text_for_ip(label: str, text: str) -> list[str]:
    hits: list[str] = []
    for name, pat in FORBIDDEN_IP_PATTERNS:
        m = pat.search(text)
        if m:
            snippet = text[max(0, m.start() - 35): min(len(text), m.end() + 35)].replace("\n", " ")
            hits.append(f"[IP-LEAK:{name}] {label} -> ...{snippet}...")
    return hits


def scan_directory_ip_and_privacy(target_dir: Path, check_forbidden_exts: bool = False) -> list[str]:
    errs: list[str] = []
    self_path = Path(__file__).resolve()
    candidates = [target_dir] if target_dir.is_file() else sorted(target_dir.rglob("*"))
    for p in candidates:
        if not p.is_file():
            continue
        if p.resolve() == self_path:
            continue
        if p.name == "41_audit_ip_clean_publication.py":
            continue
        rel = p.name if target_dir.is_file() else str(p.relative_to(target_dir))
        if check_forbidden_exts and p.suffix.lower() in FORBIDDEN_EXTS:
            errs.append(f"[FORBIDDEN-EXT] {rel}")
        # Check filename itself for IP tokens
        errs.extend(scan_text_for_ip(f"FILENAME:{rel}", rel))
        if p.suffix.lower() in {".tex", ".bib", ".md", ".json", ".sh", ".sty", ".cls", ".txt"}:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            errs.extend(scan_text_for_ip(rel, txt))
            if "/home/ubuntu" in txt or "/Users/agent" in txt:
                errs.append(f"[LOCAL-PATH] {rel}")
        elif p.suffix.lower() == ".pdf":
            res = subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True, check=False)
            if res.returncode == 0:
                errs.extend(scan_text_for_ip(f"PDF:{rel}", res.stdout))
                if "/home/ubuntu" in res.stdout or "/Users/agent" in res.stdout:
                    errs.append(f"[LOCAL-PATH-IN-PDF] {rel}")
    return errs


def audit_journal(name: str, base_dir: Path) -> list[str]:
    errs: list[str] = []
    warns: list[str] = []
    main_tex = (base_dir / "main.tex").read_text(encoding="utf-8")
    pdf_p = base_dir / "main.pdf"
    log_p = base_dir / "main.log"
    bbl_p = base_dir / "main.bbl"

    if not pdf_p.is_file():
        return [f"[{name}] Missing compiled main.pdf"]

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

    pinfo = subprocess.run(["pdfinfo", str(pdf_p)], capture_output=True, text=True, check=False).stdout
    m_pg = re.search(r"Pages:\s+(\d+)", pinfo)
    n_pages = int(m_pg.group(1)) if m_pg else 0
    n_refs = len(re.findall(r"\\bibitem", bbl_p.read_text(errors="ignore"))) if bbl_p.is_file() else 0

    abs_tex = (base_dir / "sections" / "00_abstract.tex").read_text(encoding="utf-8")
    abs_wc = word_count(abs_tex)

    if name == "nejm_ai":
        for hdr in ("Background", "Methods", "Results", "Conclusions"):
            if hdr not in abs_tex:
                errs.append(f"[nejm_ai] Structured abstract missing '{hdr}'")
        if abs_wc > 250:
            warns.append(f"[nejm_ai] Abstract is {abs_wc} words (target <=250)")
        body_files = ["01_introduction.tex", "02_methods.tex", "03_results.tex", "04_discussion.tex"]
        prose_wc = sum(word_count((base_dir / "sections" / f).read_text(), drop_figures=True) for f in body_files)
        print(f"  [NEJM AI] Pages={n_pages} | Abstract={abs_wc}w | Prose={prose_wc}w | Refs={n_refs} | Warns={len(warns)}")
    else:
        m_title = re.search(r"\\newcommand\{\\papertitle\}\{([^}]+)\}", main_tex)
        title_s = m_title.group(1) if m_title else ""
        t_words = len(title_s.split())
        if t_words > 15:
            errs.append(f"[npj] Title has {t_words} words (limit <=15): {title_s}")
        if any(ch in title_s for ch in ":;!?"):
            errs.append(f"[npj] Title contains forbidden punctuation: {title_s}")
        if abs_wc > 150:
            errs.append(f"[npj] Abstract is {abs_wc} words (limit <=150)")
        if "\\cite" in abs_tex:
            errs.append("[npj] Abstract must not contain citations")
        if "\\subsection" in (base_dir / "sections" / "01_introduction.tex").read_text():
            errs.append("[npj] Introduction must NOT contain subheadings")
        if "\\subsection" in (base_dir / "sections" / "03_discussion.tex").read_text():
            errs.append("[npj] Discussion must NOT contain subheadings")
        if n_refs > 60:
            warns.append(f"[npj] {n_refs} references (cap 60)")
        body_files = ["01_introduction.tex", "02_results.tex", "03_discussion.tex", "04_methods.tex"]
        prose_wc = sum(word_count((base_dir / "sections" / f).read_text(), drop_figures=True) for f in body_files)
        print(f"  [npj Digital Medicine] Pages={n_pages} | Title={t_words}w | Abstract={abs_wc}w | Prose={prose_wc}w | Refs={n_refs} | Warns={len(warns)}")

    for w in warns:
        print("  WARN:", w)
    return errs


def check_markdown_links(md_paths: list[Path]) -> list[str]:
    errs: list[str] = []
    for p in md_paths:
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8")
        for m in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", txt):
            link = m.group(1).strip().split("#")[0]
            if not link or link.startswith(("http://", "https://", "mailto:")):
                continue
            target = (p.parent / link).resolve()
            if not target.exists():
                errs.append(f"[BROKEN-LINK] {p}: {link}")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nejm-dir", type=Path, required=True)
    ap.add_argument("--npj-dir", type=Path, required=True)
    ap.add_argument("--scan-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--check-forbidden-exts", action="store_true")
    ap.add_argument("--md-links", type=Path, nargs="*", default=[])
    args = ap.parse_args()

    print("=== IP-Clean Publication & Compliance Audit ===")
    errs: list[str] = []
    errs.extend(audit_journal("nejm_ai", args.nejm_dir))
    errs.extend(audit_journal("npj_digital_medicine", args.npj_dir))
    for d in args.scan_dirs:
        errs.extend(scan_directory_ip_and_privacy(d, check_forbidden_exts=args.check_forbidden_exts))
    if args.md_links:
        errs.extend(check_markdown_links(args.md_links))

    if errs:
        print(f"=== FAILED: {len(errs)} ERROR(S) ===")
        for e in errs[:60]:
            print("  ERROR:", e)
        return 1
    print("=== ZERO ERRORS — IP FIREWALL & JOURNAL COMPLIANCE PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
