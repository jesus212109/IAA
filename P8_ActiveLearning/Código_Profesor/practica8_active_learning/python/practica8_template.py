"""
Práctica 8: Active Learning (Aprendizaje Activo)
Versión recortada para el alumnado.

El dataset ya está preparado en la carpeta ../data.
No tienes que generar los datos: debes completar el entrenamiento inicial,
la selección aleatoria, la selección por incertidumbre y la curva comparativa.
"""

from __future__ import annotations

import numpy as np

from utils import (
    RANDOM_STATE,
    accuracy,
    add_queried_points,
    load_data,
    plot_learning_curves,
    train_model,
)


BATCH_SIZE = 5
MAX_LABELS = 50


def select_random(X_unlabeled: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    n = X_unlabeled.shape[0]
    return rng.choice(n, size=batch_size, replace=False)


def select_by_uncertainty(model, X_unlabeled: np.ndarray, batch_size: int) -> np.ndarray:
    probs = model.predict_proba(X_unlabeled)[:, 1]
    uncertainty = np.abs(probs - 0.5)
    query_idx = np.argsort(uncertainty)[:batch_size]
    return query_idx


def run_query_strategy(strategy: str) -> tuple[list[int], list[float]]:
    """Ejecuta el ciclo de consulta para una estrategia.

    Parameters
    ----------
    strategy : {'random', 'uncertainty'}
        Estrategia de selección de nuevos puntos etiquetados.

    Returns
    -------
    n_labels_history : list of int
        Número de etiquetas usadas después de cada evaluación.
    accuracy_history : list of float
        Accuracy obtenido en test después de cada evaluación.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history: list[int] = []
    accuracy_history: list[float] = []

    while len(y_train) <= MAX_LABELS:
        model = train_model(X_train, y_train)
        acc = accuracy(model, X_test, y_test)
        n_labels_history.append(len(y_train))
        accuracy_history.append(acc)
        if len(y_train) >= MAX_LABELS:
            break
        if strategy == "random":
            query_idx = select_random(X_unlabeled, BATCH_SIZE, rng)
        elif strategy == "uncertainty":
            query_idx = select_by_uncertainty(model, X_unlabeled, BATCH_SIZE)
        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        X_train, y_train, X_unlabeled, y_unlabeled = add_queried_points(
            X_train, y_train, X_unlabeled, y_unlabeled, query_idx
        )

    return n_labels_history, accuracy_history


def main() -> None:
    # Entrenamiento inicial orientativo: puedes usar esta parte para comprobar
    # el rendimiento con solo 10 etiquetas antes de completar los bucles.
    X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    initial_model = train_model(X_initial, y_initial)
    initial_acc = accuracy(initial_model, X_test, y_test)
    print(f"Accuracy inicial con 10 etiquetas: {initial_acc:.4f}")

    random_labels, random_acc = run_query_strategy("random")
    uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

    print("\nResultados finales:")
    print(f"  Random - Accuracy final: {random_acc[-1]:.4f} (con {random_labels[-1]} etiquetas)")
    print(f"  Uncertainty - Accuracy final: {uncertainty_acc[-1]:.4f} (con {uncertainty_labels[-1]} etiquetas)")
    print(f"\nEvolución completa - Random:     {[f'{a:.4f}' for a in random_acc]}")
    print(f"Evolución completa - Uncertainty: {[f'{a:.4f}' for a in uncertainty_acc]}")


if __name__ == "__main__":
    main()
