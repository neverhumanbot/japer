#!/usr/bin/env bash
# Deterministically build (or --check) the 2-page NEVERHUMAN POAF CEO and CTO PDF briefs.
#
# Enforces release gates:
#   - strictly 2 pages per brief
#   - zero undefined references or citations
#   - zero Overfull \hbox warnings > 2.0pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_PUB="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$OUT_PUB/src"
CHECK_MODE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_MODE=1; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1795564800}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/neverhuman-briefs.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

compile_one() {
  local tex_stem="$1"
  local pdf_name="$2"
  local build_dir="$WORK_DIR/$tex_stem"
  mkdir -p "$build_dir"

  if ! (
    cd "$SRC_DIR"
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
      -output-directory="$build_dir" \
      '\pdftrailerid{}\pdfsuppressptexinfo=-1\input{'"$tex_stem"'.tex}' >/dev/null
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
      -output-directory="$build_dir" \
      '\pdftrailerid{}\pdfsuppressptexinfo=-1\input{'"$tex_stem"'.tex}' >/dev/null
  ); then
    echo "error: pdflatex failed for $tex_stem.tex" >&2
    tail -40 "$build_dir/$tex_stem.log" >&2 || true
    exit 1
  fi

  local log_file="$build_dir/$tex_stem.log"
  local pdf_file="$build_dir/$tex_stem.pdf"

  if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" "$log_file" >&2; then
    echo "error: unresolved references or citations in $tex_stem.tex" >&2
    exit 1
  fi

  local overfull
  overfull="$(grep -E 'Overfull \\hbox \([0-9]+\.[0-9]+pt too wide\)' "$log_file" | awk '{
    if (match($0, /\(([0-9]+\.[0-9]+)pt/, m) && (m[1]+0) > 2.0) print $0
  }' || true)"
  if [[ -n "$overfull" ]]; then
    echo "error: overfull hbox > 2.0pt in $tex_stem.tex:" >&2
    printf '%s\n' "$overfull" >&2
    exit 1
  fi

  local pages
  pages="$(pdfinfo "$pdf_file" | awk '/^Pages:/ {print $2}')"
  if [[ "$pages" != "2" ]]; then
    echo "error: expected exactly 2 pages for $pdf_name, got $pages" >&2
    exit 1
  fi

  if [[ "$CHECK_MODE" -eq 1 ]]; then
    if [[ ! -f "$OUT_PUB/$pdf_name" ]]; then
      echo "error: built PDF missing: $OUT_PUB/$pdf_name" >&2
      exit 1
    fi
    local committed_pages
    committed_pages="$(pdfinfo "$OUT_PUB/$pdf_name" | awk '/^Pages:/ {print $2}')"
    if [[ "$committed_pages" != "2" ]]; then
      echo "error: $pdf_name has $committed_pages pages (expected 2)" >&2
      exit 1
    fi
    echo "OK: verified $pdf_name (2 pages, zero undefined refs, zero >2pt overfull hbox)"
  else
    cp "$pdf_file" "$OUT_PUB/$pdf_name"
    echo "Built $pdf_name ($pages pages, $(wc -c < "$OUT_PUB/$pdf_name") bytes)"
  fi
}

compile_one "ceo-brief" "neverhuman-poaf-ceo-brief.pdf"
compile_one "cto-brief" "neverhuman-poaf-cto-brief.pdf"
