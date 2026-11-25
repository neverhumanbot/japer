# Reproducible pdfLaTeX + BibTeX build (with undefined-ref/citation gate)
.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -file-line-error main.tex
	@if grep -E "LaTeX Warning: (Reference|Citation) .* undefined|There were undefined references|Citation .* undefined" main.log main.blg; then \
	  echo "ERROR: unresolved references or citations detected"; exit 1; \
	fi

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.fdb_latexmk main.fls main.synctex.gz
