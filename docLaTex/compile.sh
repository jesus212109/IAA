#!/usr/bin/env bash
# =============================================================================
# compile.sh — Compilador unificado de la memoria de prácticas IAA
#
# Uso:
#   ./compile.sh                 Compilar memoria completa
#   ./compile.sh practicaN       Compilar práctica N individual (N=1..8)
#   ./compile.sh --watch         Compilación continua al editar
#   ./compile.sh --check-only    Solo verificar dependencias
#   ./compile.sh --install       Verificar, instalar faltantes y compilar
#   ./compile.sh --clean         Limpiar archivos temporales
#   ./compile.sh --help          Mostrar esta ayuda
# =============================================================================

set -euo pipefail

# ─── Constantes ────────────────────────────────────────────────────────────────
MAIN="__memoria.tex"
BUILD_DIR="build"
LATEXMK_CMD="latexmk -cd -pdf -shell-escape -outdir=%s -interaction=nonstopmode %s"
DOC_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR=""

# ─── Colores ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[*]${RESET} $*"; }
success() { echo -e "${GREEN}[✓]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[!]${RESET} $*"; }
error()   { echo -e "${RED}[✗]${RESET} $*"; }

# ─── Cabecera ──────────────────────────────────────────────────────────────────
print_banner() {
  echo -e "${BOLD}${CYAN}"
  echo "  ╔══════════════════════════════════════╗"
  echo "  ║    IAA — Compilador de la Memoria    ║"
  echo "  ║  Introducción al Aprendizaje Automático ║"
  echo "  ╚══════════════════════════════════════╝"
  echo -e "${RESET}"
}

# ─── Auto-detectar gestor de paquetes ──────────────────────────────────────────
PKG_MANAGER=""
INSTALL_CMD=""
PACKAGE_PREFIX=""
PYTHON_CMD="python3"

if command -v apt &>/dev/null; then
  PKG_MANAGER="apt"
  INSTALL_CMD="sudo apt install -y"
  PACKAGE_PREFIX="deb"
elif command -v dnf &>/dev/null; then
  PKG_MANAGER="dnf"
  INSTALL_CMD="sudo dnf install -y"
  PACKAGE_PREFIX="rpm"
elif command -v pacman &>/dev/null; then
  PKG_MANAGER="pacman"
  INSTALL_CMD="sudo pacman -S --needed --noconfirm"
  PACKAGE_PREFIX="arch"
fi

# ─── Mapa de paquetes por distribución ─────────────────────────────────────────
# Formato: NOMBRE_STY:paquete_deb:paquete_rpm:paquete_arch
STY_MAP=(
  "minted.sty:texlive-latex-extra:texlive-latex-extra:texlive-latexextra"
  "multirow.sty:texlive-latex-extra:texlive-multirow:texlive-latexextra"
  "datetime.sty:texlive-latex-extra:texlive-datetime:texlive-latexextra"
  "eurosym.sty:texlive-latex-extra:texlive-eurosym:texlive-latexextra"
  "fancyhdr.sty:texlive-latex-extra:texlive-fancyhdr:texlive-latexextra"
  "titlesec.sty:texlive-latex-extra:texlive-titlesec:texlive-latexextra"
  "subcaption.sty:texlive-latex-extra:texlive-caption:texlive-latexextra"
  "pdflscape.sty:texlive-latex-extra:texlive-pdflscape:texlive-latexextra"
  "pdfpages.sty:texlive-latex-extra:texlive-pdfpages:texlive-latexextra"
  "appendix.sty:texlive-latex-extra:texlive-appendix:texlive-latexextra"
  "booktabs.sty:texlive-latex-base:texlive-booktabs:texlive-latexextra"
  "natbib.sty:texlive-latex-base:texlive-natbib:texlive-latexextra"
)

BIN_MAP=(
  "pdflatex:texlive-latex-base:texlive-latex:texlive-core"
  "latexmk:texlive-latex-base:texlive-latex:texlive-core"
  "biber:texlive-latex-base:texlive-biber:biber"
)

pkg_for_os() {
  local entry="$1"
  local index
  case "$PACKAGE_PREFIX" in
    deb)  index=1 ;;
    rpm)  index=2 ;;
    arch) index=3 ;;
    *)    echo ""; return ;;
  esac
  echo "$entry" | cut -d':' -f"$((index+1))"
}

# ─── Verificar binarios ────────────────────────────────────────────────────────
check_binaries() {
  local missing=""
  for entry in "${BIN_MAP[@]}"; do
    bin=$(echo "$entry" | cut -d':' -f1)
    if ! command -v "$bin" &>/dev/null; then
      pkg=$(pkg_for_os "$entry")
      missing+=" $bin"
    fi
  done
  echo "$missing"
}

# ─── Verificar paquetes LaTeX ──────────────────────────────────────────────────
check_sty_packages() {
  local missing_sty=""
  local missing_names=""
  for entry in "${STY_MAP[@]}"; do
    sty=$(echo "$entry" | cut -d':' -f1)
    if ! kpsewhich "$sty" &>/dev/null 2>&1; then
      pkg=$(pkg_for_os "$entry")
      missing_sty+=" $sty"
      missing_names+=" $pkg"
    fi
  done
  echo "$missing_names|$missing_sty"
}

# ─── Verificar Pygments (minted) ───────────────────────────────────────────────
check_pygments() {
  if ! "$PYTHON_CMD" -c "import pygments" &>/dev/null 2>&1; then
    return 1
  fi
  return 0
}

PYGMENTS_INSTALL_HINT="  pip3 install Pygments  # o: sudo apt install python3-pygments"

# ─── Ayuda ─────────────────────────────────────────────────────────────────────
show_help() {
  echo ""
  echo "  Uso: ./compile.sh [OPCIÓN | prácticaN]"
  echo ""
  echo "  OPCIONES:"
  echo "    (sin args)     Compilar memoria completa"
  echo "    practicaN      Compilar práctica N individual (N=1..8)"
  echo "    --watch        Compilación continua al editar (latexmk -pvc)"
  echo "    --check-only   Verificar dependencias sin compilar"
  echo "    --install      Verificar, instalar faltantes y compilar"
  echo "    --clean        Limpiar archivos temporales"
  echo "    --help         Mostrar esta ayuda"
  echo ""
  echo "  EJEMPLOS:"
  echo "    ./compile.sh              → build/__memoria.pdf"
  echo "    ./compile.sh practica3    → sections/practica3/build/practica3.pdf"
  echo "    ./compile.sh --watch      → recompila automáticamente al guardar"
  echo ""
}

# ─── Limpieza ──────────────────────────────────────────────────────────────────
do_clean() {
  info "Limpiando archivos temporales..."
  rm -rf "$DOC_DIR/$BUILD_DIR" "$DOC_DIR/__memoria.pdf"
  find "$DOC_DIR" -type d -name '_minted-*' -exec rm -rf {} + 2>/dev/null || true
  for n in 1 2 3 4 5 6 7 8; do
    rm -rf "$DOC_DIR/sections/practica${n}/build" 2>/dev/null || true
  done
  success "Limpieza completada."
}

# ─── Compilar práctica individual ──────────────────────────────────────────────
compile_practice() {
  local num="$1"
  local dir="$DOC_DIR/sections/practica${num}"
  local tex_file="${dir}/practica${num}.tex"

  if [ ! -f "$tex_file" ]; then
    error "No se encuentra la práctica ${num} en:"
    echo "  $tex_file"
    exit 1
  fi

  local outdir="${dir}/${BUILD_DIR}"
  mkdir -p "$outdir"

  info "Compilando práctica ${num}..."

  # shellcheck disable=SC2059
  if printf "$LATEXMK_CMD" "$outdir" "$tex_file" | bash 2>"${outdir}/latexmk_err.log"; then
    local pdf="${outdir}/practica${num}.pdf"
    if [ -f "$pdf" ]; then
      success "Práctica ${num} compilada: ${pdf}"
    else
      warn "Compilación sin errores, pero no se encontró el PDF."
    fi
  else
    error "Error al compilar práctica ${num}."
    extract_errors "${outdir}/practica${num}.log" 5
    exit 1
  fi
}

# ─── Compilar documento completo ──────────────────────────────────────────────
compile_full() {
  local outdir="$DOC_DIR/$BUILD_DIR"
  mkdir -p "$outdir"

  info "Compilando memoria completa..."

  cd "$DOC_DIR"
  # shellcheck disable=SC2059
  if printf "$LATEXMK_CMD" "$outdir" "$MAIN" | bash 2>"${outdir}/latexmk_err.log"; then
    local pdf="${outdir}/${MAIN%.tex}.pdf"
    if [ -f "$pdf" ]; then
      success "Memoria compilada: ${pdf}"
    else
      warn "Compilación sin errores, pero no se encontró el PDF."
    fi
  else
    error "Error al compilar la memoria."
    extract_errors "${outdir}/${MAIN%.tex}.log" 5
    exit 1
  fi
}

# ─── Compilación continua (watch) ──────────────────────────────────────────────
compile_watch() {
  local outdir="$DOC_DIR/$BUILD_DIR"
  mkdir -p "$outdir"

  info "Modo watch activado. Compilando al guardar cambios..."
  info "Pulsa Ctrl+C para detener."
  echo ""

  cd "$DOC_DIR"
  latexmk -cd -pdf -shell-escape -outdir="$outdir" -pvc -interaction=nonstopmode "$MAIN"
}

# ─── Extraer errores del log ──────────────────────────────────────────────────
extract_errors() {
  local log_file="$1"
  local max_lines="${2:-10}"

  if [ -f "$log_file" ]; then
    echo ""
    error "Errores detectados (últimas ${max_lines} líneas relevantes):"
    grep -n "^!" "$log_file" 2>/dev/null | head -"$max_lines" | while IFS=: read -r line msg; do
      echo -e "  ${RED}L${line}:${RESET} ${msg#! }"
    done
    echo ""
    warn "Log completo: ${log_file}"
  fi
}

# ─── Verificar dependencias ────────────────────────────────────────────────────
check_deps() {
  local all_ok=true

  # Binarios
  local missing_bins
  missing_bins=$(check_binaries)
  if [ -n "$missing_bins" ]; then
    warn "Binarios faltantes:$missing_bins"
    all_ok=false
  else
    success "Binarios LaTeX: pdflatex, latexmk, biber"
  fi

  # Paquetes LaTeX
  local sty_result
  sty_result=$(check_sty_packages)
  local missing_names="${sty_result%%|*}"
  local missing_sty="${sty_result##*|}"
  if [ -n "$missing_names" ]; then
    warn "Paquetes LaTeX faltantes:$missing_sty"
    all_ok=false
  else
    success "Paquetes LaTeX: todos encontrados"
  fi

  # Pygments
  if check_pygments; then
    success "Pygments (minted): disponible"
  else
    warn "Pygments (minted) no instalado."
    echo "  $PYGMENTS_INSTALL_HINT"
    all_ok=false
  fi

  # Gestor de paquetes
  if [ -z "$PKG_MANAGER" ]; then
    warn "No se detectó gestor de paquetes (apt/dnf/pacman)."
  else
    success "Gestor de paquetes: $PKG_MANAGER"
  fi

  if $all_ok; then
    success "Entorno completo."
    return 0
  fi
  return 1
}

# ─── Instalar dependencias ─────────────────────────────────────────────────────
install_deps() {
  local needs_install=false
  local install_list=""

  local missing_bins
  missing_bins=$(check_binaries)
  if [ -n "$missing_bins" ]; then
    needs_install=true
    for entry in "${BIN_MAP[@]}"; do
      bin=$(echo "$entry" | cut -d':' -f1)
      if ! command -v "$bin" &>/dev/null; then
        install_list+=" $(pkg_for_os "$entry")"
      fi
    done
  fi

  local sty_result
  sty_result=$(check_sty_packages)
  local missing_names="${sty_result%%|*}"
  if [ -n "$missing_names" ]; then
    needs_install=true
    install_list+=" $missing_names"
  fi

  if ! check_pygments; then
    needs_install=true
    info "Instalando Pygments con pip..."
    "$PYTHON_CMD" -m pip install Pygments 2>&1 | tail -1 || true
  fi

  if $needs_install; then
    if [ -z "$PKG_MANAGER" ]; then
      error "No se detectó gestor de paquetes (apt/dnf/pacman)."
      echo "  Instala manualmente: sudo apt install texlive-latex-extra texlive-publishers texlive-science biber"
      echo "  $PYGMENTS_INSTALL_HINT"
      exit 1
    fi
    install_list=$(echo "$install_list" | tr -s ' ')
    info "Instalando con $PKG_MANAGER:$install_list"
    # shellcheck disable=SC2086
    $INSTALL_CMD $install_list
    success "Dependencias instaladas."
  else
    info "No hay dependencias que instalar."
  fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

cd "$DOC_DIR"
print_banner

case "${1:-}" in
  --help|-h)
    show_help
    ;;

  --check-only)
    check_deps
    ;;

  --install)
    install_deps
    echo ""
    compile_full
    ;;

  --clean)
    do_clean
    ;;

  --watch)
    check_deps || true
    compile_watch
    ;;

  practica[1-8])
    num="${1#practica}"
    # For individual compilation, also check deps first
    check_deps || true
    compile_practice "$num"
    ;;

  ""|--all)
    check_deps || true
    compile_full
    ;;

  *)
    error "Opción desconocida: $1"
    echo ""
    show_help
    exit 1
    ;;
esac
