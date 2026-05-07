"""
Práctica 8: Active Learning — Solución completa para generar resultados reales.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

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

# Output image path for the LaTeX report
IMG_OUTPUT = Path(__file__).resolve().parents[2] / "docLaTex/sections/practica8/img/learning_curve.png"


def select_random(X_unlabeled: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(X_unlabeled), size=batch_size, replace=False)


def select_by_uncertainty(model, X_unlabeled: np.ndarray, batch_size: int) -> np.ndarray:
    proba = model.predict_proba(X_unlabeled)[:, 1]
    uncertainty = np.abs(proba - 0.5)
    return np.argsort(uncertainty)[:batch_size]


def run_query_strategy(strategy: str) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(RANDOM_STATE)
    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history: list[int] = []
    accuracy_history: list[float] = []

    while len(y_train) <= MAX_LABELS:
        model = train_model(X_train, y_train)
        acc = accuracy(model, X_test, y_test)
        accuracy_history.append(acc)
        n_labels_history.append(len(y_train))

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
    # Accuracy inicial
    X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    initial_model = train_model(X_initial, y_initial)
    initial_acc = accuracy(initial_model, X_test, y_test)
    print(f"Accuracy inicial (10 etiquetas): {initial_acc:.4f}")

    # Estrategias
    random_labels, random_acc = run_query_strategy("random")
    uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    print("\n--- Selección aleatoria ---")
    for n, a in zip(random_labels, random_acc):
        print(f"  {n:3d} etiquetas -> accuracy = {a:.4f}")

    print("\n--- Selección por incertidumbre ---")
    for n, a in zip(uncertainty_labels, uncertainty_acc):
        print(f"  {n:3d} etiquetas -> accuracy = {a:.4f}")

    print(f"\nAccuracy final Random:      {random_acc[-1]:.4f}")
    print(f"Accuracy final Uncertainty: {uncertainty_acc[-1]:.4f}")

    # Guardar imagen para LaTeX
    IMG_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(random_labels, random_acc, marker="o", linestyle="--", label="Selección aleatoria", color="steelblue")
    plt.plot(uncertainty_labels, uncertainty_acc, marker="s", linestyle="-", label="Selección por incertidumbre", color="tomato")
    plt.xlabel("Número de etiquetas utilizadas")
    plt.ylabel("Accuracy en test")
    plt.title("Curva de aprendizaje: Aleatorio vs. Aprendizaje Activo")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(IMG_OUTPUT), dpi=200)
    print(f"\nGráfica guardada en: {IMG_OUTPUT}")


if __name__ == "__main__":
    main()
