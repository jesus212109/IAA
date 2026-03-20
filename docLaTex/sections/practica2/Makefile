# Local Makefile for Practice 2
LATEXMK = latexmk -cd -shell-escape -outdir=build -interaction=nonstopmode

all:
	@mkdir -p build
	$(LATEXMK) practica2.tex

clean:
	rm -rf build/
	rm -rf _minted-*

.PHONY: all clean
