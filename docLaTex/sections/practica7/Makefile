# Local Makefile for Practice 7
LATEXMK = latexmk -cd -pdf -shell-escape -outdir=build -interaction=nonstopmode

practica7:
	@mkdir -p build
	$(LATEXMK) practica7.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: practica7 clean
