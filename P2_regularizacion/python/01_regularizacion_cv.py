#!/usr/bin/env python3
"""Práctica 2 — Regularización (Ridge vs Lasso) + Validación Cruzada (K-Fold)

Objetivo
--------
1) Entender cómo cambia el error de validación cruzada al variar K.
2) Ver cómo el parámetro de regularización (alpha/lambda) reduce coeficientes.
3) Comparar Ridge (encoge) vs Lasso (puede anular coeficientes).

Tareas (alumno)
---------------
A) Cambia `metodo` entre "lasso" y "ridge".
B) Prueba varios valores de `valor_lambda` (p.ej., 0.01, 0.1, 1, 10, 100).
C) Prueba varios valores de `k_folds` (p.ej., 2, 5, 10, 20) y comenta estabilidad vs coste.
D) (Opcional) Repite con distintas semillas en KFold para ver variabilidad.

Salida
------
- Por pantalla: RMSE medio de CV
- Figura: barras con coeficientes aprendidos
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "precios_viviendas.csv"

def main() -> None:
    # -----------------------
    # 1) Cargar datos
    # -----------------------
    df = pd.read_csv(DATA_PATH)

    X = df[[f"Var_{i}" for i in range(10)]].to_numpy()
    y = df["Precio"].to_numpy()

    # Normalización (imprescindible para regularización)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # -----------------------
    # 2) Configuración (MODIFICA ESTO)
    # -----------------------
    metodo = "lasso"       # Opciones: "lasso" o "ridge"
    valor_lambda = .5    # Fuerza de la penalización (alpha en sklearn)
    k_folds = 10

    # Consejo: en Lasso a veces necesitas subir max_iter si no converge
    if metodo == "lasso":
        modelo = Lasso(alpha=valor_lambda, max_iter=1000000)
    else:
        modelo = Ridge(alpha=valor_lambda)

    # -----------------------
    # 3) Validación cruzada (K-Fold)
    # -----------------------
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42) #random_state previusly set to 42

    # Start timer
    start_time = time.time()

    # Scoring: sklearn devuelve MSE negativo (porque maximiza scores)
    scores = cross_val_score(modelo, X_std, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_cv = np.sqrt(-np.mean(scores))

    # Stop timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Method={metodo.upper()} | lambda={valor_lambda} | K={k_folds} | RMSE_CV={rmse_cv:.2f} | Time={elapsed_time:.4f}s")
    # -----------------------
    # 4) Entrenar y visualizar coeficientes
    # -----------------------
    modelo.fit(X_std, y)

    plt.figure(figsize=(10, 4))
    plt.bar(range(10), modelo.coef_)
    plt.xticks(range(10), [f"Var_{i}" for i in range(10)])
    plt.title(f"Coeficientes con {metodo.upper()} (λ={valor_lambda}) | RMSE CV: {rmse_cv:.2f}")
    plt.ylabel("Peso del coeficiente")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
