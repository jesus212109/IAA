#!/usr/bin/env python3
"""Genera el dataset sintético para la Práctica 2.

Crea un CSV con 200 muestras y 10 variables (3 relevantes + 7 ruido)
y una variable objetivo (precio).

Salida
------
data/precios_viviendas.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main() -> None:
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    y = 150 + 50*X[:,0] - 30*X[:,1] + 20*X[:,2] + np.random.randn(n_samples)*5

    df = pd.DataFrame(X, columns=[f"Var_{i}" for i in range(10)])
    df["Precio"] = y

    out = Path(__file__).resolve().parent.parent / "data" / "precios_viviendas.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] Dataset guardado en: {out}")

if __name__ == "__main__":
    main()
