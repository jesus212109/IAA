# Práctica 8: Active Learning (Aprendizaje Activo)

El dataset ya está generado y dividido en particiones. La tarea consiste en completar el ciclo de aprendizaje activo y comparar dos estrategias de selección de etiquetas.

## Estructura

```text
practica8_active_learning/
├── data/
│   ├── active_learning_moons.csv   # dataset completo con columna split
│   ├── initial_labeled.csv         # 10 puntos etiquetados iniciales
│   ├── unlabeled_pool.csv          # pool no etiquetado + etiquetas reales como oráculo
│   └── test.csv                    # conjunto de test separado
├── python/
│   ├── practica8_template.py       # plantilla que debe completar el alumnado
│   ├── utils.py                    # funciones auxiliares ya implementadas
│   └── requirements.txt
└── julia/
    ├── practica8_template.jl       # plantilla que debe completar el alumnado
    ├── utils.jl                    # funciones auxiliares ya implementadas
    └── Project.toml
```

## Importante

El fichero `unlabeled_pool.csv` contiene la columna `y`, pero debe interpretarse como el **oráculo**. Es decir, el modelo no puede usar esas etiquetas para entrenar desde el principio. Solo se pueden consultar las etiquetas de los puntos seleccionados en cada iteración.

La práctica parte de:

- 10 puntos etiquetados iniciales;
- un pool de datos no etiquetados;
- consultas de 5 puntos por iteración;
- presupuesto máximo de 50 etiquetas;
- comparación entre selección aleatoria y selección por incertidumbre.

## Versión Python

Desde la carpeta `python/`:

```bash
pip install -r requirements.txt
python practica8_template.py
```

Al principio el script solo calcula el accuracy inicial con 10 etiquetas. Para completar la práctica hay que resolver los `TODO` de `practica8_template.py`.

## Versión Julia

Desde la carpeta `julia/`:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("practica8_template.jl")
```

Al igual que en Python, la plantilla incluye `TODO` y funciones incompletas.

## Qué debe completar el alumnado

1. Entrenar el modelo con los 10 puntos iniciales y medir el accuracy en test.
2. Implementar la selección aleatoria de 5 puntos por iteración.
3. Implementar la selección por incertidumbre usando probabilidad cercana a 0.5.
4. Consultar el oráculo solo para los puntos seleccionados.
5. Actualizar el conjunto de entrenamiento y el pool no etiquetado.
6. Repetir el proceso hasta alcanzar 50 etiquetas.
7. Representar la curva de aprendizaje: accuracy frente a número de etiquetas.
8. Comparar selección aleatoria frente a selección por incertidumbre.
9. Responder razonadamente por qué los puntos cercanos a la frontera de decisión pueden ser más informativos.

## Resultado esperado

El resultado principal debe ser una gráfica comparativa con dos curvas:

- selección aleatoria;
- selección por incertidumbre.

El informe debe interpretar qué estrategia aprende más rápido y por qué.
