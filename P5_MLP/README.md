# Práctica 5 — Arquitectura de Redes Neuronales (MLP)

Material base para el alumnado de **Introducción al Aprendizaje Automático**.

## Qué incluye este paquete

- `src/practica5_mlp_alumnos.py`: script principal con una plantilla guiada.
- `src/experimentos_digitos_guia.py`: apoyo para explorar arquitecturas en dígitos.
- `src/utils_practica5.py`: utilidades para gráficas, tablas y métricas.
- `enunciado_practica5.tex`: borrador del enunciado en LaTeX.
- `informe_plantilla.md`: esquema orientativo para redactar el PDF.
- `requirements.txt`: dependencias mínimas.

## Filosofía del material

Esta carpeta **no pretende dar la práctica resuelta**.

Se proporciona una base para que no perdáis tiempo en tareas mecánicas como:
- cargar datos,
- dividir entrenamiento y prueba,
- entrenar un MLP en `scikit-learn`,
- dibujar fronteras de decisión,
- guardar figuras.

Lo importante de la práctica es que **vosotros interpretéis los resultados** y justifiquéis decisiones de diseño:
- por qué falla un modelo lineal,
- cuándo una capa oculta ayuda,
- qué cambia al variar el número de neuronas,
- qué diferencias aparecen entre Sigmoide y ReLU,
- y qué arquitectura es razonable para alcanzar el 95% en dígitos.

## Cómo empezar

1. Cread un entorno virtual.
2. Instalad dependencias:

```bash
pip install -r requirements.txt
```

3. Ejecutad el script principal:

```bash
python src/practica5_mlp_alumnos.py
```

4. Revisad las figuras generadas en `figures/`.
5. Completad los apartados marcados como `TODO`.

## Qué debéis completar vosotros

En el script principal hay comentarios `TODO` para que:
- añadáis observaciones,
- probéis más configuraciones,
- comparéis resultados,
- y decidáis una arquitectura final para el problema de dígitos.

## Recomendación docente

No entreguéis capturas de código extensas. Priorizad:
- tablas resumidas,
- gráficas claras,
- interpretación de resultados,
- y justificación técnica.
