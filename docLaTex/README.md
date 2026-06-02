# Memoria de Prácticas — Introducción al Aprendizaje Automático (IAA)

Este repositorio contiene la estructura profesionalizada en LaTeX para la generación de la memoria de prácticas de la asignatura. El proyecto ha sido modularizado para permitir la compilación tanto del documento completo como de capítulos individuales de forma independiente.

---

## Requisitos del sistema

Para compilar necesitas una distribución de **TeX Live** completa o los siguientes paquetes:

- **Binarios:** `pdflatex`, `latexmk`, `biber`
- **LaTeX:** `minted`, `multirow`, `datetime`, `eurosym`, `booktabs`, `subcaption`, `pdfpages`, `pdflscape`, `appendix`, `fancyhdr`, `titlesec`, `natbib`, `tocbibind`
- **Python:** `Pygments` (necesario para `minted`)

## Compilación

Se proporciona un único script `compile.sh` que unifica todo el proceso:

```bash
# Compilar la memoria completa
./compile.sh

# Compilar una práctica individual (con portada, índice y listados propios)
./compile.sh practica3

# Compilación continua (recompila automáticamente al guardar cambios)
./compile.sh --watch

# Verificar dependencias sin compilar
./compile.sh --check-only

# Verificar, instalar dependencias faltantes y compilar
./compile.sh --install

# Limpiar archivos temporales
./compile.sh --clean

# Mostrar ayuda
./compile.sh --help
```

### Salida generada

| Comando | Archivo generado |
|---------|-----------------|
| `./compile.sh` | `build/__memoria.pdf` |
| `./compile.sh practica3` | `sections/practica3/build/practica3.pdf` |

### Compilación individual vs. completa

Al compilar una práctica de forma individual (`./compile.sh practica3`), se genera automáticamente:
- **Portada** propia con título, asignatura y autor
- **Índice** de contenidos de la práctica
- **Listado de figuras** y **listado de tablas**

Al compilar la memoria completa, estos elementos se omiten para cada práctica, apareciendo únicamente en el prefacio del documento.

---

## Estructura del Proyecto

```text
docLaTex/
├── compile.sh                 # Compilador unificado
├── __memoria.tex              # Archivo maestro del proyecto
├── _datos_proyecto.tex        # Metadatos (Título, Autores, Asignatura)
├── preamble.tex               # Configuración central (Paquetes, Estilos)
├── .gitignore                 # Filtro para archivos temporales
├── referencias.bib            # Bibliografía
├── README.md                  # Este archivo
├── Portada/                   # Recursos de la portada principal
├── Imagenes/                  # Recursos gráficos compartidos
├── sections/                  # Capítulos de cada práctica
│   ├── practica1/
│   │   ├── practica1.tex
│   │   ├── img/
│   │   └── build/
│   ├── practica2/
│   └── ...
└── build/                     # Salida del documento completo
```

## Cómo añadir una nueva práctica

1. Crea una carpeta en `sections/practicaN/`.
2. Crea el archivo `.tex` usando la plantilla estándar.
3. Añade la llamada en `__memoria.tex`: `\subfile{sections/practicaN/practicaN}`.
4. Compila con: `./compile.sh practicaN` (no requiere Makefile adicional).

## Convenciones de estilo

Todas las prácticas siguen una estructura homogénea:

- **Título de capítulo:** `\chapter[Short TOC]{Cuaderno de Laboratorio --- Práctica N: ...}`
- **Tareas:** `\section[Tarea N: short]{Tarea N --- descripción completa}`
- **Tablas:** con `\toprule`, `\midrule`, `\bottomrule` (paquete `booktabs`)
- **Figuras:** con `\includegraphics`, `\caption` y `\label`
- **Conclusiones:** `\section*{Conclusiones}` al final de cada práctica
- **Rutas de gráficos:** `\graphicspath` con resolución a 4 niveles
