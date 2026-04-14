"""Utilidades para la Práctica 5 de MLP.

Este módulo contiene funciones de apoyo para representar fronteras de decisión,
matrices de confusión y tablas simples. No es la parte principal a evaluar en la
práctica; se ofrece para que el alumnado pueda centrarse en el análisis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "resultados"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(filename: str) -> None:
    """Guarda la figura actual en la carpeta `figures`.

    Parameters
    ----------
    filename : str
        Nombre del fichero de salida.
    """
    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"[INFO] Figura guardada en: {output_path}")
    plt.close()



def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str, filename: str | None = None) -> None:
    """Representa la frontera de decisión de un clasificador bidimensional.

    Parameters
    ----------
    model : estimator
        Modelo ya entrenado con método `predict`.
    X : np.ndarray
        Matriz de entrada de forma `(n_samples, 2)`.
    y : np.ndarray
        Vector de etiquetas.
    title : str
        Título de la figura.
    filename : str | None, optional
        Nombre del fichero de salida.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, zz, alpha=0.28)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=28)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    if filename is not None:
        save_figure(filename)



def plot_loss_curve(loss_curve: list[float], title: str, filename: str) -> None:
    """Representa la curva de pérdida de entrenamiento.

    Parameters
    ----------
    loss_curve : list[float]
        Valores de pérdida por iteración.
    title : str
        Título del gráfico.
    filename : str
        Nombre del fichero de salida.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(loss_curve)
    plt.xlabel("Iteración")
    plt.ylabel("Loss")
    plt.title(title)
    save_figure(filename)



def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, filename: str) -> None:
    """Dibuja la matriz de confusión.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.
    title : str
        Título del gráfico.
    filename : str
        Nombre del fichero de salida.
    """
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, colorbar=False)
    plt.title(title)
    save_figure(filename)



def accuracy_summary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Etiquetas reales.
    y_pred : np.ndarray
        Etiquetas predichas.

    Returns
    -------
    float
        Accuracy del clasificador.
    """
    return accuracy_score(y_true, y_pred)



def print_table(rows: Iterable[dict], csv_name: str | None = None) -> pd.DataFrame:
    """Convierte una colección de filas en tabla, la imprime y opcionalmente la guarda.

    Parameters
    ----------
    rows : Iterable[dict]
        Filas con claves homogéneas.
    csv_name : str | None, optional
        Nombre del CSV de salida dentro de `resultados/`.

    Returns
    -------
    pd.DataFrame
        Tabla generada.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        print("[INFO] No hay filas que mostrar.")
        return frame

    print(frame.to_string(index=False))
    if csv_name is not None:
        output_path = RESULTS_DIR / csv_name
        frame.to_csv(output_path, index=False)
        print(f"[INFO] Tabla guardada en: {output_path}")
    return frame
