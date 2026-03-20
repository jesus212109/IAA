# Local Makefile for Practice 1
LATEXMK = latexmk -cd -shell-escape -outdir=build -interaction=nonstopmode

all:
	@mkdir -p build
	$(LATEXMK) practica1.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: all clean
