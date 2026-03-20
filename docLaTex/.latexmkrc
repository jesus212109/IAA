# =============================================================================
# .latexmkrc — Professional build configuration
# =============================================================================

# Redirect all generated files to 'build/' directory
$aux_dir = 'build';
$out_dir = 'build';

# Ensure PDF generation
$pdf_mode = 1;

# Enable shell-escape for minted and force pdflatex explicitly
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode -synctex=1 %O %S';
set_tex_cmds('-shell-escape %O %S');

# Automatically clean up extra extensions
$clean_ext = 'pyg snm nav vrb bbl blg bcf run.xml axp aex';

# Allow bibtex/biber
$bibtex_use = 2; 
