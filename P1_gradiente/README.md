# Práctica 1 — La Física del Aprendizaje Automático (GD por mini-batches)

Este paquete contiene los archivos de partida para la Práctica 1:
- `optimizacion_energia.jl`: script base con generación de datos + GD por mini-batches + visualización.
- `experimentos_practica1.jl`: runner opcional para generar automáticamente las figuras solicitadas.
- Plantillas de "Cuaderno de Laboratorio" (para que el alumnado entregue **PDF**):
  - `Cuaderno_Laboratorio_P1_template.md` (Markdown)
  - `Cuaderno_Laboratorio_P1_template.tex` (LaTeX)
  - `Cuaderno_Laboratorio_P1_template.ipynb` (Notebook con kernel Julia)
- (Opcional) `optimizacion_energia.py`: versión equivalente en Python.

## Requisitos (Julia)
- Julia 1.9+ (recomendado 1.10+)
- Paquetes: `Plots`, `Statistics`, `Random`

### Instalación rápida
En Julia:
```julia
include("install.jl")
```

## Uso mínimo
Ejecuta el script base:
```bash
julia optimizacion_energia.jl
```

Para generar automáticamente figuras en `outputs/` (valores iniciales que luego el alumnado ajusta):
```bash
julia experimentos_practica1.jl
```

## Exportar a PDF
### Opción A (LaTeX)
```bash
pdflatex Cuaderno_Laboratorio_P1_template.tex
pdflatex Cuaderno_Laboratorio_P1_template.tex
```

### Opción B (Markdown + pandoc)
```bash
pandoc Cuaderno_Laboratorio_P1_template.md -o Cuaderno_Laboratorio_P1.pdf
```

---
Version 1.