# Local Makefile for Practice 5
LATEXMK = latexmk -cd -shell-escape -outdir=build -interaction=nonstopmode

all:
	@mkdir -p build
	$(LATEXMK) practica5.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: all clean
