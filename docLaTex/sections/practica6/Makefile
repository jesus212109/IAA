# Local Makefile for Practice 6
LATEXMK = latexmk -cd -shell-escape -outdir=build -interaction=nonstopmode

all:
	@mkdir -p build
	$(LATEXMK) practica6.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: all clean
