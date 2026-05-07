# Local Makefile for Practice 8
LATEXMK = latexmk -cd -pdf -shell-escape -outdir=build -interaction=nonstopmode

practica8:
	@mkdir -p build
	$(LATEXMK) practica8.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: practica8 clean
