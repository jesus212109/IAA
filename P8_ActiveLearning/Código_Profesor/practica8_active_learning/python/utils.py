"""
Utilidades para la Práctica 8: Active Learning.

Este fichero contiene funciones auxiliares ya implementadas para que el alumnado
pueda centrarse en el ciclo de consulta y en la comparación de estrategias.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


RANDOM_STATE = 42
N_TREES = 100
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga las particiones predefinidas de la práctica.

    Returns
    -------
    X_initial, y_initial : ndarray
        Los 10 puntos inicialmente etiquetados.
    X_unlabeled, y_unlabeled : ndarray
        Pool no etiquetado y sus etiquetas reales. `y_unlabeled` actúa como
        oráculo: solo debe consultarse para los puntos seleccionados.
    X_test, y_test : ndarray
        Conjunto de test separado.
    """
    initial = pd.read_csv(DATA_DIR / "initial_labeled.csv")
    pool = pd.read_csv(DATA_DIR / "unlabeled_pool.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_initial = initial[["x1", "x2"]].to_numpy()
    y_initial = initial["y"].to_numpy()

    X_unlabeled = pool[["x1", "x2"]].to_numpy()
    y_unlabeled = pool["y"].to_numpy()

    X_test = test[["x1", "x2"]].to_numpy()
    y_test = test["y"].to_numpy()

    return X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Entrena el clasificador base de la práctica."""
    model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def accuracy(model: RandomForestClassifier, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Calcula el accuracy de un modelo sobre el conjunto de test."""
    return accuracy_score(y_test, model.predict(X_test))


def add_queried_points(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_unlabeled: np.ndarray,
    y_unlabeled: np.ndarray,
    query_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mueve los puntos consultados desde el pool al conjunto de entrenamiento."""
    X_query = X_unlabeled[query_idx]
    y_query = y_unlabeled[query_idx]

    X_train = np.vstack([X_train, X_query])
    y_train = np.concatenate([y_train, y_query])

    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0)
    y_unlabeled = np.delete(y_unlabeled, query_idx, axis=0)

    return X_train, y_train, X_unlabeled, y_unlabeled


def plot_learning_curves(
    random_labels: list[int],
    random_acc: list[float],
    uncertainty_labels: list[int],
    uncertainty_acc: list[float],
    output_path: str = "learning_curve_python.png",
) -> None:
    """Representa la curva de aprendizaje de ambas estrategias."""
    plt.figure(figsize=(8, 5))
    plt.plot(random_labels, random_acc, marker="o", label="Selección aleatoria")
    plt.plot(uncertainty_labels, uncertainty_acc, marker="s", label="Selección por incertidumbre")
    plt.xlabel("Número de etiquetas utilizadas")
    plt.ylabel("Accuracy en test")
    plt.title("Curva de aprendizaje: Random vs Active Learning")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()
