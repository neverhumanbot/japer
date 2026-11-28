LATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX := bibtex

all: main.pdf

main.pdf: main.tex references.bib sections/*.tex tables/*.tex figures/*.pdf
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc main.lof main.lot

.PHONY: all clean
