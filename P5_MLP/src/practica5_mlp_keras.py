"""Ejemplo equivalente usando Keras (TensorFlow) para comparar con la plantilla sklearn.

El script está pensado para mostrar las diferencias de uso respecto a sklearn y
PyTorch: interfaz de alto nivel con `fit`, `evaluate` y `predict`.
"""
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_digits, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils_practica5 import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curve,
    print_table,
    accuracy_summary,
)

RANDOM_STATE = 42
MOONS_NOISE = 0.25
TEST_SIZE_MOONS = 0.30
TEST_SIZE_DIGITS = 0.25
TARGET_ACCURACY = 0.95


@dataclass
class ExperimentResult:
    name: str
    hidden_layers: tuple
    activation: str
    acc_train: float
    acc_test: float
    epochs: int
    final_loss: float


class KerasMLPClassifier:
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], activation: str = 'relu', lr: float = 1e-3, multiclass: bool = False, n_classes: int = 2):
        self.multiclass = multiclass
        self.n_classes = n_classes
        self.model = keras.Sequential()
        for h in hidden_sizes:
            self.model.add(layers.Dense(h, activation=activation, input_shape=(input_dim,)))
            input_dim = h
        if multiclass:
            self.model.add(layers.Dense(n_classes, activation='softmax'))
            self.model.compile(optimizer=keras.optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.add(layers.Dense(1, activation='sigmoid'))
            self.model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
        self.history = None

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, verbose: int = 0):
        self.history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.model.predict(X)
        if self.multiclass:
            return out.argmax(axis=1)
        else:
            return (out.ravel() >= 0.5).astype(int)


def load_moons():
    X, y = make_moons(n_samples=500, noise=MOONS_NOISE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_MOONS, stratify=y, random_state=RANDOM_STATE)
    return X, y, X_train, X_test, y_train, y_test


def task_1_simple_perceptron():
    print('\n=== TAREA 1: Perceptrón simple (Keras) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = KerasMLPClassifier(input_dim=2, hidden_sizes=(), activation='relu', lr=1e-2, multiclass=False)
    model.fit(X_train_s, y_train, epochs=200, batch_size=32)

    y_pred_train = model.predict(scaler.transform(X_train))
    y_pred_test = model.predict(scaler.transform(X_test))
    acc_train = accuracy_summary(y_train, y_pred_train)
    acc_test = accuracy_summary(y_test, y_pred_test)

    print_table([
        {"modelo": "Keras-Perceptron", "arquitectura": (), "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": len(model.history.history['loss'])}
    ])

    plot_decision_boundary(model, X, y, "Keras - Perceptrón simple", "keras_tarea1_perceptron.png")
    plot_confusion_matrix(y_test, y_pred_test, "Keras - Confusión Tarea1", "keras_tarea1_confusion.png")


def task_2_hidden_layer():
    print('\n=== TAREA 2: Capas ocultas (Keras) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    hidden_configs = [(2,), (5,), (20,)]
    rows = []
    for hidden in hidden_configs:
        model = KerasMLPClassifier(input_dim=2, hidden_sizes=hidden, activation='relu', lr=1e-2, multiclass=False)
        model.fit(X_train_s, y_train, epochs=300, batch_size=32)
        y_pred_test = model.predict(scaler.transform(X_test))
        y_pred_train = model.predict(scaler.transform(X_train))
        acc_train = accuracy_summary(y_train, y_pred_train)
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Keras-MLP{hidden}", "arquitectura": hidden, "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": len(model.history.history['loss'])})
        plot_decision_boundary(model, X, y, f"Keras - Una capa oculta {hidden}", f"keras_tarea2_hidden_{'_'.join(map(str,hidden))}.png")

    print_table(rows, csv_name="keras_tarea2_hidden_layer.csv")


def task_3_activation_comparison():
    print('\n=== TAREA 3: Activaciones (Keras) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    arch = (20,)
    activations = ['sigmoid', 'relu']
    rows = []
    for act in activations:
        model = KerasMLPClassifier(input_dim=2, hidden_sizes=arch, activation=act, lr=1e-2, multiclass=False)
        model.fit(scaler.transform(X_train), y_train, epochs=300, batch_size=32)
        y_pred_test = model.predict(scaler.transform(X_test))
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Keras-MLP{arch}-{act}", "arquitectura": arch, "activacion": act, "acc_test": f"{acc_test:.4f}", "epochs": len(model.history.history['loss']), "loss_final": f"{model.history.history['loss'][-1]:.4f}"})
        plot_decision_boundary(model, X, y, f"Keras - Activación: {act}", f"keras_tarea3_activation_{act}.png")
        plot_loss_curve(model.history.history['loss'], f"Keras Loss - {act}", f"keras_tarea3_loss_{act}.png")

    print_table(rows, csv_name="keras_tarea3_activation_comparison.csv")


def task_4_digits():
    print('\n=== TAREA 4: Dígitos (Keras) ===')
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=TEST_SIZE_DIGITS, stratify=digits.target, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    candidate_arch = [(20,), (50,), (100,), (30,15), (50,20), (64,32)]
    rows = []
    for arch in candidate_arch:
        model = KerasMLPClassifier(input_dim=X_train.shape[1], hidden_sizes=arch, activation='relu', lr=1e-3, multiclass=True, n_classes=10)
        model.fit(scaler.transform(X_train), y_train, epochs=200, batch_size=64)
        y_pred_test = model.predict(scaler.transform(X_test))
        y_pred_train = model.predict(scaler.transform(X_train))
        acc_train = accuracy_summary(y_train, y_pred_train)
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Keras-MLP{arch}", "arquitectura": arch, "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": len(model.history.history['loss']), "cumple_95": "sí" if acc_test >= TARGET_ACCURACY else "no"})

    table = print_table(rows, csv_name="keras_tarea4_digits.csv")
    print('\n[INFO] Experimento Keras completado.')


def main():
    task_1_simple_perceptron()
    task_2_hidden_layer()
    task_3_activation_comparison()
    task_4_digits()


if __name__ == "__main__":
    main()
