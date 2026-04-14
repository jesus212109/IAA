"""Plantilla guiada para la Práctica 5: Arquitectura de Redes Neuronales (MLP).

El objetivo de este script es servir como punto de partida. Varias partes ya
están preparadas para ahorrar trabajo mecánico, pero la práctica NO está
cerrada: debéis completar el análisis, interpretar resultados y justificar las
arquitecturas probadas.
"""
from dataclasses import dataclass
from pathlib import Path

from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import utils_practica5 as utils

# ── Redirigir figuras al árbol de LaTeX ───────────────────────────────────────
# Las figuras se guardan en docLaTex/sections/practica5/img/ para que el
# fichero practica5.tex las encuentre sin ningún path adicional.
_REPO_ROOT = Path(__file__).resolve().parents[2]
utils.FIGURES_DIR = _REPO_ROOT / "docLaTex" / "sections" / "practica5" / "img"
utils.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# También actualizamos RESULTS_DIR para los CSV
utils.RESULTS_DIR = _REPO_ROOT / "P5_MLP" / "resultados"
utils.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from utils_practica5 import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curve,
    print_table,
)

RANDOM_STATE = 42
MOONS_NOISE = 0.25
TEST_SIZE_MOONS = 0.30
TEST_SIZE_DIGITS = 0.25
TARGET_ACCURACY = 0.95


@dataclass
class ExperimentResult:
    """Resultado resumido de un experimento con MLP.

    Parameters
    ----------
    name : str
        Nombre corto del experimento.
    hidden_layers : tuple[int, ...]
        Arquitectura de capas ocultas.
    activation : str
        Función de activación.
    acc_train : float
        Accuracy en entrenamiento.
    acc_test : float
        Accuracy en prueba.
    n_iter : int
        Número de iteraciones consumidas.
    final_loss : float
        Última pérdida de entrenamiento.
    """

    name: str
    hidden_layers: tuple
    activation: str
    acc_train: float
    acc_test: float
    n_iter: int
    final_loss: float


def build_mlp(hidden_layer_sizes: tuple, activation: str = "relu", max_iter: int = 2500) -> Pipeline:
    """Construye un pipeline con escalado y clasificador MLP.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Arquitectura del MLP. Si es `()`, no hay capas ocultas.
    activation : str, optional
        Activación de las capas ocultas.
    max_iter : int, optional
        Número máximo de iteraciones.

    Returns
    -------
    Pipeline
        Modelo listo para entrenar.
    """
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
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    activation: str,
    hidden_layers: tuple,
) -> ExperimentResult:
    """Entrena y resume un experimento."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mlp = model.named_steps["mlp"]
    return ExperimentResult(
        name=name,
        hidden_layers=hidden_layers,
        activation=activation,
        acc_train=accuracy_score(y_train, y_pred_train),
        acc_test=accuracy_score(y_test, y_pred_test),
        n_iter=mlp.n_iter_,
        final_loss=mlp.loss_,
    )


def load_moons_dataset():
    """Genera el dataset no lineal de lunas y devuelve su partición."""
    X, y = make_moons(n_samples=500, noise=MOONS_NOISE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE_MOONS,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X, y, X_train, X_test, y_train, y_test


def task_1_simple_perceptron() -> None:
    """Tarea 1: comprobar el límite de un modelo sin capa oculta."""
    print("\n=== TAREA 1: El problema de la no linealidad ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    model = build_mlp(hidden_layer_sizes=(), activation="relu", max_iter=2500)
    result = evaluate_model(
        name="Perceptrón simple",
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        activation="relu",
        hidden_layers=(),
    )

    print_table(
        [
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "activacion": result.activation,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
            }
        ]
    )

    y_pred = model.predict(X_test)
    plot_decision_boundary(model, X, y, "Tarea 1 - Perceptrón simple", "tarea1_perceptron_simple.png")
    plot_confusion_matrix(y_test, y_pred, "Tarea 1 - Matriz de confusión", "tarea1_confusion.png")

    print(f"\n[RESULTADO] acc_train={result.acc_train:.4f}  acc_test={result.acc_test:.4f}")
    print("La frontera aprendida es lineal; no puede separar la geometría de media luna.")


def task_2_hidden_layer() -> None:
    """Tarea 2: comparar el efecto del número de neuronas ocultas."""
    print("\n=== TAREA 2: Diseñando la capa oculta ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    hidden_configs = [(2,), (5,), (20,)]
    rows = []
    for hidden in hidden_configs:
        model = build_mlp(hidden_layer_sizes=hidden, activation="relu", max_iter=3000)
        result = evaluate_model(
            name=f"MLP{hidden}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation="relu",
            hidden_layers=hidden,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
            }
        )
        # Nombre de archivo con el número de neuronas (compatible con LaTeX)
        n = hidden[0]
        filename = f"tarea2_hidden_{n}.png"
        plot_decision_boundary(model, X, y, f"Tarea 2 - Una capa oculta: {n} neuronas", filename)

    print_table(rows, csv_name="tarea2_hidden_layer.csv")
    print("[RESULTADO] Comparación de fronteras con 2, 5 y 20 neuronas completada.")


def task_3_activation_comparison() -> None:
    """Tarea 3: comparar Sigmoide y ReLU con la MISMA arquitectura.

    Nota importante
    ---------------
    Para que la comparación sea justa, aquí NO cambiamos la arquitectura entre
    activaciones. Solo cambia la activación.
    """
    print("\n=== TAREA 3: Funciones de activación ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    fixed_architecture = (20,)
    activations = ["logistic", "relu"]

    rows = []
    for activation in activations:
        model = build_mlp(hidden_layer_sizes=fixed_architecture, activation=activation, max_iter=3000)
        result = evaluate_model(
            name=f"MLP{fixed_architecture}-{activation}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation=activation,
            hidden_layers=fixed_architecture,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "activacion": result.activation,
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
                "loss_final": f"{result.final_loss:.4f}",
            }
        )

        plot_decision_boundary(
            model,
            X,
            y,
            f"Tarea 3 - Activación: {activation}",
            f"tarea3_activation_{activation}.png",
        )
        plot_loss_curve(
            model.named_steps["mlp"].loss_curve_,
            f"Curva de pérdida - {activation}",
            f"tarea3_loss_curve_{activation}.png",
        )

    print_table(rows, csv_name="tarea3_activation_comparison.csv")
    print("[RESULTADO] Comparación Sigmoide vs. ReLU completada.")


def task_4_digits() -> None:
    """Tarea 4: explorar arquitecturas en el dataset de dígitos."""
    print("\n=== TAREA 4: El reto de la caja negra ===")
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=TEST_SIZE_DIGITS,
        stratify=digits.target,
        random_state=RANDOM_STATE,
    )

    candidate_architectures = [
        (20,),
        (50,),
        (100,),
        (30, 15),
        (50, 20),
        (64, 32),
    ]

    rows = []
    best_result = None
    best_model = None

    for arch in candidate_architectures:
        model = build_mlp(hidden_layer_sizes=arch, activation="relu", max_iter=2500)
        result = evaluate_model(
            name=f"MLP{arch}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation="relu",
            hidden_layers=arch,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
                "cumple_95": "sí" if result.acc_test >= TARGET_ACCURACY else "no",
            }
        )
        # Guardar el modelo más simple que supere el 95% (primero en la lista)
        if result.acc_test >= TARGET_ACCURACY and best_result is None:
            best_result = result
            best_model = model

    table = print_table(rows, csv_name="tarea4_digits.csv")
    reached = table[table["cumple_95"] == "sí"]
    if reached.empty:
        print("\nNinguna arquitectura de la lista inicial alcanza el 95%.")
        print("Ampliando búsqueda con arquitecturas adicionales...")
        # Búsqueda adicional si ninguna alcanza el objetivo
        extra_archs = [(128,), (100, 50), (64, 32, 16)]
        for arch in extra_archs:
            model = build_mlp(hidden_layer_sizes=arch, activation="relu", max_iter=3000)
            result = evaluate_model(
                name=f"MLP{arch}",
                model=model,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                activation="relu",
                hidden_layers=arch,
            )
            if result.acc_test >= TARGET_ACCURACY and best_result is None:
                best_result = result
                best_model = model
            rows.append(
                {
                    "modelo": result.name,
                    "arquitectura": result.hidden_layers,
                    "acc_train": f"{result.acc_train:.4f}",
                    "acc_test": f"{result.acc_test:.4f}",
                    "iteraciones": result.n_iter,
                    "cumple_95": "sí" if result.acc_test >= TARGET_ACCURACY else "no",
                }
            )
        print_table(rows, csv_name="tarea4_digits_extended.csv")
    else:
        print("\nArquitecturas que alcanzan el 95%:")
        print(reached.to_string(index=False))

    # Generar matriz de confusión de la arquitectura elegida
    if best_model is not None:
        y_pred_test = best_model.predict(X_test)
        arch_str = str(best_result.hidden_layers)
        plot_confusion_matrix(
            y_test,
            y_pred_test,
            f"Tarea 4 - Arquitectura elegida {arch_str}",
            "tarea4_mejor_arquitectura.png",
        )
        print(f"\n[ARQUITECTURA ELEGIDA] {arch_str}")
        print(f"  acc_train={best_result.acc_train:.4f}  acc_test={best_result.acc_test:.4f}")
        print(f"  Iteraciones={best_result.n_iter}  Loss={best_result.final_loss:.4f}")
        print("  Criterio: mínima complejidad que supera el 95% en test.")
    else:
        print("\n[AVISO] Ninguna arquitectura alcanzó el objetivo del 95%. Revisad la búsqueda.")


def main() -> None:
    """Ejecuta la plantilla completa."""
    print("=" * 60)
    print("PRÁCTICA 5: MLP — Arquitectura de Redes Neuronales")
    print(f"Figuras → {utils.FIGURES_DIR}")
    print("=" * 60)
    task_1_simple_perceptron()
    task_2_hidden_layer()
    task_3_activation_comparison()
    task_4_digits()
    print("\n✓ Todos los experimentos completados. Figuras guardadas.")


if __name__ == "__main__":
    main()
