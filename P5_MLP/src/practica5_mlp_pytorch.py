"""Ejemplo equivalente usando PyTorch para comparar con la plantilla sklearn.

El script repite las tareas principales de la plantilla: problema de lunas,
una capa oculta vs ninguna, comparación de activaciones y un experimento
simple en el dataset de dígitos. Está diseñado para ser didáctico, no para
producción.
"""
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    n_epochs: int
    final_loss: float


class TorchMLPClassifier:
    """Pequeño wrapper para entrenar/predict con PyTorch y exponer `predict`.

    - For binary problems (moons) uses a single logit output and BCEWithLogitsLoss.
    - For multiclass (digits) uses logits with CrossEntropyLoss.
    """

    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...], activation: str = "relu", lr: float = 1e-3):
        sizes = [input_dim] + list(hidden_sizes) + [1]
        self.is_multiclass = False
        # detect multiclass by last layer size decision done later for digits
        layers: List[nn.Module] = []
        act = nn.ReLU if activation == "relu" else nn.Sigmoid if activation == "logistic" else nn.Tanh
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(act())
        # final linear; for binary keep 1 output, for multiclass we will replace
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.lr = lr
        self.loss_curve_ = []

    def fit_binary(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, batch_size: int = 64):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.net.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = self.net(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            self.loss_curve_.append(epoch_loss)
        self.n_epochs_ = epochs
        self.final_loss_ = self.loss_curve_[-1] if self.loss_curve_ else 0.0

    def fit_multiclass(self, X: np.ndarray, y: np.ndarray, epochs: int = 200, batch_size: int = 64, n_classes: int = 10):
        # replace last layer to match n_classes
        in_features = list(self.net.children())[-1].in_features
        # rebuild final layer
        modules = list(self.net.children())[:-1] + [nn.Linear(in_features, n_classes)]
        self.net = nn.Sequential(*modules)
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        opt = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.net.train()
        for ep in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                out = self.net(xb)
                loss = criterion(out, yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            self.loss_curve_.append(epoch_loss)
        self.n_epochs_ = epochs
        self.final_loss_ = self.loss_curve_[-1] if self.loss_curve_ else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        X_t = torch.from_numpy(X.astype(np.float32))
        with torch.no_grad():
            out = self.net(X_t)
        out_np = out.numpy()
        if out_np.ndim == 2 and out_np.shape[1] > 1:
            preds = out_np.argmax(axis=1)
        else:
            preds = (out_np > 0).astype(int).ravel()
        return preds


def load_moons():
    X, y = make_moons(n_samples=500, noise=MOONS_NOISE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_MOONS, stratify=y, random_state=RANDOM_STATE)
    return X, y, X_train, X_test, y_train, y_test


def task_1_simple_perceptron():
    print('\n=== TAREA 1: Perceptrón simple (PyTorch) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = TorchMLPClassifier(input_dim=2, hidden_sizes=(), activation='relu', lr=1e-2)
    model.fit_binary(X_train_s, y_train, epochs=300)

    y_pred_train = model.predict(scaler.transform(X_train))
    y_pred_test = model.predict(scaler.transform(X_test))
    acc_train = accuracy_summary(y_train, y_pred_train)
    acc_test = accuracy_summary(y_test, y_pred_test)

    print_table([
        {"modelo": "Torch-Perceptron", "arquitectura": (), "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": model.n_epochs_}
    ])

    plot_decision_boundary(model, X, y, "PyTorch - Perceptrón simple", "pytorch_tarea1_perceptron.png")
    plot_confusion_matrix(y_test, y_pred_test, "PyTorch - Confusión Tarea1", "pytorch_tarea1_confusion.png")


def task_2_hidden_layer():
    print('\n=== TAREA 2: Capas ocultas (PyTorch) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_s = scaler.transform(X)

    hidden_configs = [(2,), (5,), (20,)]
    rows = []
    for hidden in hidden_configs:
        model = TorchMLPClassifier(input_dim=2, hidden_sizes=hidden, activation='relu', lr=1e-2)
        model.fit_binary(X_train_s, y_train, epochs=400)
        y_pred_test = model.predict(scaler.transform(X_test))
        y_pred_train = model.predict(scaler.transform(X_train))
        acc_train = accuracy_summary(y_train, y_pred_train)
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Torch-MLP{hidden}", "arquitectura": hidden, "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": model.n_epochs_})
        plot_decision_boundary(model, X, y, f"PyTorch - Una capa oculta {hidden}", f"pytorch_tarea2_hidden_{'_'.join(map(str,hidden))}.png")

    print_table(rows, csv_name="pytorch_tarea2_hidden_layer.csv")


def task_3_activation_comparison():
    print('\n=== TAREA 3: Activaciones (PyTorch) ===')
    X, y, X_train, X_test, y_train, y_test = load_moons()
    scaler = StandardScaler().fit(X_train)
    arch = (20,)
    activations = ['logistic', 'relu']
    rows = []
    for act in activations:
        model = TorchMLPClassifier(input_dim=2, hidden_sizes=arch, activation=act, lr=1e-2)
        model.fit_binary(scaler.transform(X_train), y_train, epochs=400)
        y_pred_test = model.predict(scaler.transform(X_test))
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Torch-MLP{arch}-{act}", "arquitectura": arch, "activacion": act, "acc_test": f"{acc_test:.4f}", "epochs": model.n_epochs_, "loss_final": f"{model.final_loss_:.4f}"})
        plot_decision_boundary(model, X, y, f"PyTorch - Activación: {act}", f"pytorch_tarea3_activation_{act}.png")
        plot_loss_curve(model.loss_curve_, f"PyTorch Loss - {act}", f"pytorch_tarea3_loss_{act}.png")

    print_table(rows, csv_name="pytorch_tarea3_activation_comparison.csv")


def task_4_digits():
    print('\n=== TAREA 4: Dígitos (PyTorch) ===')
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=TEST_SIZE_DIGITS, stratify=digits.target, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    candidate_arch = [(20,), (50,), (100,), (30,15), (50,20), (64,32)]
    rows = []
    for arch in candidate_arch:
        model = TorchMLPClassifier(input_dim=X_train.shape[1], hidden_sizes=arch, activation='relu', lr=1e-3)
        model.fit_multiclass(X_train_s, y_train, epochs=300, n_classes=10)
        y_pred_test = model.predict(scaler.transform(X_test))
        y_pred_train = model.predict(scaler.transform(X_train))
        acc_train = accuracy_summary(y_train, y_pred_train)
        acc_test = accuracy_summary(y_test, y_pred_test)
        rows.append({"modelo": f"Torch-MLP{arch}", "arquitectura": arch, "acc_train": f"{acc_train:.4f}", "acc_test": f"{acc_test:.4f}", "epochs": model.n_epochs_, "cumple_95": "sí" if acc_test >= TARGET_ACCURACY else "no"})

    table = print_table(rows, csv_name="pytorch_tarea4_digits.csv")
    print('\n[INFO] Experimento PyTorch completado.')


def main():
    task_1_simple_perceptron()
    task_2_hidden_layer()
    task_3_activation_comparison()
    task_4_digits()


if __name__ == "__main__":
    main()
