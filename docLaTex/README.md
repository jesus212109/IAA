# 🎓 Memoria de Prácticas — Introducción al Aprendizaje Automático (IAA)

Este repositorio contiene la estructura profesionalizada en LaTeX para la generación de la memoria de prácticas de la asignatura. El proyecto ha sido modularizado para permitir la compilación tanto del documento completo como de capítulos individuales de forma independiente.

---

## 🚀 Guía de Compilación (Automatizada)

Se han configurado **Makefiles** en múltiples niveles para ofrecer flexibilidad total. **Todos los archivos de salida se generan en carpetas `build/`** para mantener limpio el directorio raíz.

### 1. Compilar todo el proyecto
Desde la raíz del proyecto:
```bash
make
```
Generará `build/__memoria.pdf`.

### 2. Compilar capítulos específicos (Standalone)
Cada práctica puede compilarse de forma independiente (generando su propia portada automática):
```bash
make practica1
make practica2
make practica3
```
O entrando directamente en la carpeta:
```bash
cd sections/practica1
make
```

### 3. Compilar desde la carpeta de salida
Si prefieres trabajar dentro del directorio `build/`:
```bash
cd build
make
```

### 4. Limpieza profunda
Para eliminar todos los archivos temporales, carpetas de caché y PDFs generados recursivamente:
```bash
make clean
```

---

## 📁 Estructura del Proyecto

```text
.
├── __memoria.tex             # Archivo maestro del proyecto
├── _datos_proyecto.tex       # Metadatos (Título, Autores, Asignatura)
├── preamble.tex              # Configuración central (Paquetes, Estilos, Comandos)
├── Makefile                  # Automatización principal
├── .latexmkrc                # Configuración de compilación (build-dir, shell-escape)
├── .gitignore                # Filtro para archivos temporales y PDFs
├── sections/
│   ├── practica1/            # Módulo de la Práctica 1
│   │   ├── practica1.tex     # Código fuente del capítulo
│   │   ├── img/              # Imágenes locales de la P1
│   │   └── build/            # Salida aislada para la P1
│   └── practica2/            # (Misma estructura modular...)
└── Portada/                  # Recursos de la portada principal
```

---

## 🛠️ Herramientas de Automatización

### ⚙️ `.latexmkrc`
Configura `latexmk` para:
- Redirigir **todos** los auxiliares a `build/`.
- Habilitar `-shell-escape` (necesario para el resaltado de código con `minted`).
- Configurar el motor de limpieza recursiva.

### 🧠 `preamble.tex`
Centraliza la lógica del documento. Incluye el comando inteligente `\standalonecover`, que detecta automáticamente si el capítulo se está compilando solo (y necesita portada) o como parte de la memoria completa.

### 🙈 `.gitignore`
Evita que archivos basura (`.log`, `.aux`, `.toc`, etc.) o binarios pesados se suban al control de versiones, manteniendo el repositorio impecable.

---

## 📝 Cómo añadir una nueva práctica

1. Crea una carpeta en `sections/practicaN/`.
2. Crea el archivo `.tex` usando la plantilla:
   ```latex
   \documentclass[../../__memoria.tex]{subfiles}
   \begin{document}
   \standalonecover
   \chapter{Título}
   ... contenido ...
   \end{document}
   ```
3. Añade la llamada en `__memoria.tex`: `\subfile{sections/practicaN/practicaN}`.
4. (Opcional) Copia el `Makefile` de otra sección para habilitar la compilación local.
