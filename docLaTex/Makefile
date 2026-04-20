# =============================================================================
# Makefile for the IAA Lab Report
# =============================================================================

MAIN = __memoria.tex
BUILD_DIR = build
LATEXMK = latexmk -cd -shell-escape -outdir=$(BUILD_DIR) -interaction=nonstopmode

# Default target: compile the full document
all: practica1 practica2 practica3 practica4 practica5 practica6
	@mkdir -p $(BUILD_DIR)
	$(LATEXMK) $(MAIN)

# Chapter compilation targets
practica1:
	$(MAKE) -C sections/practica1 all

practica2:
	$(MAKE) -C sections/practica2 all

practica3:
	$(MAKE) -C sections/practica3 all

practica4:
	$(MAKE) -C sections/practica4 all

practica5:
	$(MAKE) -C sections/practica5 all

practica6:
	$(MAKE) -C sections/practica6 all

# Full clean: remove all build files and PDF outputs recursively
clean:
	rm -rf $(BUILD_DIR)
	for dir in sections/*/ ; do \
		if [ -f "$$dir/Makefile" ]; then \
			$(MAKE) -C $$dir clean ; \
		fi \
	done
	find . -name "_minted-*" -type d -exec rm -rf {} +
	rm -f __memoria.pdf

help:
	@echo "Available targets:"
	@echo "  make          - Compile full document"
	@echo "  make practica1 - Compile Práctica 1"
	@echo "  make practica2 - Compile Práctica 2"
	@echo "  make practica3 - Compile Práctica 3"
	@echo "  make practica4 - Compile Práctica 4"
	@echo "  make practica5 - Compile Práctica 5"
	@echo "  make practica6 - Compile Práctica 6"
	@echo "  make clean    - Deep clean of all temp files"

.PHONY: all clean help practica1 practica2 practica3 practica4 practica5 practica6
