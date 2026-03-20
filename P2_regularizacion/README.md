# Práctica 2 — Regularización + Validación Cruzada (K-Fold)

## Estructura
- `data/precios_viviendas.csv`: dataset sintético.
- `python/00_generar_dataset.py`: regenera el dataset.
- `python/01_regularizacion_cv.py`: script principal (Ridge/Lasso + K-Fold).
- `python/requirements.txt`: dependencias de Python.
- `julia/regularizacion_cv.jl`: versión Julia.
- `docs/texto_para_moodle.md`: texto breve para publicación.
- `outputs/`: resultados/figuras generadas localmente.

## Ejecución (Python)
Desde `alumnos/`:
```bash
pip install -r python/requirements.txt
python3 python/00_generar_dataset.py
python3 python/01_regularizacion_cv.py
```

## Ejecución (Julia)
Desde `alumnos/`:
```bash
julia julia/regularizacion_cv.jl
```

## Entregable esperado
- Comparativa Ridge vs Lasso para varios valores de `lambda`.
- Comentario sobre el efecto de `K` en RMSE_CV y estabilidad.
- Gráficas/capturas de coeficientes y breve interpretación.

Este paquete de alumnos no incluye scripts de solución del profesorado.
