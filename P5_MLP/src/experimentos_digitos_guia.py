"""Apoyo opcional para explorar arquitecturas en dígitos.

Este script está pensado como herramienta auxiliar. No sustituye al análisis del
script principal, pero puede ayudar a comparar más combinaciones de forma
ordenada. Conviene usarlo con criterio y no como búsqueda ciega.
"""

from itertools import product

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_practica5 import print_table

RANDOM_STATE = 42
TARGET_ACCURACY = 0.95


def build_model(hidden_layer_sizes: tuple[int, ...], activation: str = "relu") -> Pipeline:
    """Construye un MLP con escalado previo."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver="adam",
                    alpha=1e-4,
                    max_iter=2500,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    """Lanza una rejilla pequeña y razonable de arquitecturas."""
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=0.25,
        stratify=digits.target,
        random_state=RANDOM_STATE,
    )

    shallow = [(n,) for n in [10, 20, 50, 100, 150]]
    deep = [(a, b) for a, b in product([20, 50, 100], [10, 20, 50])]
    candidates = shallow + deep

    rows = []
    for arch in candidates:
        model = build_model(arch)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        mlp = model.named_steps["mlp"]
        rows.append(
            {
                "arquitectura": arch,
                "acc_train": f"{accuracy_score(y_train, y_pred_train):.4f}",
                "acc_test": f"{accuracy_score(y_test, y_pred_test):.4f}",
                "iteraciones": mlp.n_iter_,
                "cumple_95": "sí" if accuracy_score(y_test, y_pred_test) >= TARGET_ACCURACY else "no",
            }
        )

    table = print_table(rows, csv_name="digits_grid_opcional.csv")
    print("\nArquitecturas que alcanzan al menos 95%:")
    print(table[table["cumple_95"] == "sí"].to_string(index=False))

    print("\nRecordatorio: no basta con decir cuál es la mejor. Hay que justificar")
    print("si es una arquitectura mínima razonable, si sobrecomplica el modelo,")
    print("y qué compromiso existe entre anchura, profundidad y generalización.")


if __name__ == "__main__":
    main()
