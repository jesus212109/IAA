"""
PRÁCTICA 7: La Inteligencia Colectiva (Modelos de Ensambles)
Asignatura: Introducción al Aprendizaje Automático (IAA) 26/27

Código base para el alumnado.

Objetivo del script
-------------------
Este archivo proporciona una estructura guiada para que el alumnado implemente
la práctica paso a paso. No es una solución completa: contiene secciones TODO
que deben completarse y analizarse en el informe final.

Dataset utilizado
-----------------
Breast Cancer Wisconsin (incluido en scikit-learn).

Modelos a trabajar
------------------
1. Árbol de decisión simple
2. Random Forest
3. Gradient Boosting (o AdaBoost, si se quiere probar como alternativa)

Recomendación
-------------
Lee cada comentario antes de programar. No se trata solo de obtener métricas,
sino de interpretar los resultados y justificar lo que observas.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
TEST_SIZE = 0.25


# -----------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# -----------------------------------------------------------------------------
def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset Breast Cancer Wisconsin desde scikit-learn.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Etiqueta de clase.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en entrenamiento y prueba.

    TODO para el alumnado:
    - Revisa por qué usamos stratify=y.
    - Comprueba cuántas muestras hay en entrenamiento y en test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# -----------------------------------------------------------------------------
# 2. TAREA 1: ÁRBOL SIMPLE (BASELINE)
# -----------------------------------------------------------------------------
def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """
    Entrena un árbol de decisión sin limitar la profundidad.

    TODO para el alumnado:
    - Comprueba qué ocurre cuando no fijamos max_depth.
    - Relaciona el resultado con el concepto de sobreajuste.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evalúa un modelo en entrenamiento y en prueba.

    Returns
    -------
    results : dict
        Diccionario con accuracy en train y test.

    TODO para el alumnado:
    - Añade otras métricas si lo consideras útil.
    - Interpreta la diferencia entre train_accuracy y test_accuracy.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


# -----------------------------------------------------------------------------
# 3. TAREA 2: RANDOM FOREST
# -----------------------------------------------------------------------------
def random_forest_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trees_list: List[int],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Ejecuta varios experimentos cambiando el número de árboles.

    Parameters
    ----------
    n_trees_list : list of int
        Lista con los valores de n_estimators que se desean probar.

    Returns
    -------
    results_df : pd.DataFrame
        Resultados del experimento.

    TODO para el alumnado:
    - Observa si llega un punto de saturación.
    - Compara estos resultados con el árbol simple.
    """
    rows = []

    for n_trees in n_trees_list:
        start = time.perf_counter()

        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        elapsed = time.perf_counter() - start
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        rows.append(
            {
                "n_estimators": n_trees,
                "test_accuracy": test_accuracy,
                "train_time_seconds": elapsed,
            }
        )

    return pd.DataFrame(rows)


def plot_random_forest_results(results_df: pd.DataFrame) -> None:
    """
    Representa la evolución del accuracy en test frente al número de árboles.

    TODO para el alumnado:
    - Cambia el gráfico para mostrar error en lugar de accuracy si lo prefieres.
    - Interpreta la forma de la curva en el informe.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["n_estimators"],
        results_df["test_accuracy"],
        marker="o",
        linestyle="--",
    )
    plt.title("Random Forest: Accuracy en test vs número de árboles")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy en test")
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to the specified path for the report
    import os
    save_path = "/home/jesus/IAA/docLaTex/sections/practica7/img/rf_accuracy.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Plot saved successfully to {save_path}")


# -----------------------------------------------------------------------------
# 4. TAREA 3: BOOSTING
# -----------------------------------------------------------------------------
def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[GradientBoostingClassifier, float]:
    """
    Entrena un modelo Gradient Boosting y mide el tiempo de entrenamiento.

    TODO para el alumnado:
    - Si quieres, sustituye este modelo por AdaBoost y compara ambos.
    - Interpreta la diferencia entre entrenamiento secuencial (boosting)
      y entrenamiento por ensamble de árboles más independientes (bagging).
    """
    start = time.perf_counter()

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start
    return model, elapsed


# -----------------------------------------------------------------------------
# 5. IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------------------
def show_top_features(model, feature_names: List[str], top_k: int = 3) -> pd.DataFrame:
    """
    Muestra las variables más importantes según el modelo.

    Parameters
    ----------
    model : estimator
        Modelo ya entrenado con atributo feature_importances_.
    feature_names : list of str
        Nombres de las variables.
    top_k : int, default=3
        Número de variables a mostrar.

    Returns
    -------
    top_features : pd.DataFrame
        Tabla con las variables más importantes.

    TODO para el alumnado:
    - Comenta si estas variables tienen sentido en el problema.
    - Compara la importancia obtenida en Random Forest y en Boosting.
    """
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    importances = importances.sort_values("importance", ascending=False)
    return importances.head(top_k)


# -----------------------------------------------------------------------------
# 6. UTILIDADES DE IMPRESIÓN
# -----------------------------------------------------------------------------
def print_confusion_and_report(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Imprime matriz de confusión e informe de clasificación.

    TODO para el alumnado:
    - Explica en el informe qué tipo de error consideras más grave.
    - No copies la salida sin comentarla.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(cm)
    print("\nClassification report:")
    print(report)


# -----------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
def main() -> None:
    # -------------------------------------------------------------------------
    # PASO 1. CARGAR Y DIVIDIR LOS DATOS
    # -------------------------------------------------------------------------
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("=" * 80)
    print("PRÁCTICA 7 - MODELOS DE ENSAMBLES")
    print("=" * 80)
    print(f"Número de muestras totales: {len(X)}")
    print(f"Número de variables: {X.shape[1]}")
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    # -------------------------------------------------------------------------
    # PASO 2. ÁRBOL SIMPLE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 1 - ÁRBOL SIMPLE")
    print("-" * 80)

    tree_model = train_decision_tree(X_train, y_train)
    tree_results = evaluate_model(tree_model, X_train, y_train, X_test, y_test)

    print(f"Accuracy en entrenamiento: {tree_results['train_accuracy']:.4f}")
    print(f"Accuracy en test:          {tree_results['test_accuracy']:.4f}")

    print_confusion_and_report(tree_model, X_test, y_test)

    # TODO para el alumnado:
    # Responde en el informe:
    # 1. ¿Hay señales de sobreajuste?
    # 2. ¿Por qué un árbol muy profundo puede memorizar el entrenamiento?

    # -------------------------------------------------------------------------
    # PASO 3. RANDOM FOREST
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 2 - RANDOM FOREST")
    print("-" * 80)

    n_trees_list = [1, 10, 50, 100]
    rf_results = random_forest_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        n_trees_list=n_trees_list,
    )

    print("\nResultados Random Forest:")
    print(rf_results)

    plot_random_forest_results(rf_results)

    # Entrenamos un modelo final con 100 árboles para analizarlo mejor.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)

    print("\nEvaluación Random Forest (100 árboles):")
    print_confusion_and_report(rf_model, X_test, y_test)

    print("\nTop 3 variables más importantes en Random Forest:")
    print(show_top_features(rf_model, list(X.columns), top_k=3))

    # TODO para el alumnado:
    # Responde en el informe:
    # 1. ¿Mejora Random Forest frente al árbol simple?
    # 2. ¿Se estabiliza el resultado al aumentar el número de árboles?
    # 3. ¿Por qué el bagging reduce la varianza del modelo?

    # -------------------------------------------------------------------------
    # PASO 4. BOOSTING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 3 - BOOSTING")
    print("-" * 80)

    gb_model, gb_train_time = train_gradient_boosting(X_train, y_train)
    gb_results = evaluate_model(gb_model, X_train, y_train, X_test, y_test)

    print(f"Tiempo de entrenamiento:   {gb_train_time:.4f} s")
    print(f"Accuracy en entrenamiento: {gb_results['train_accuracy']:.4f}")
    print(f"Accuracy en test:          {gb_results['test_accuracy']:.4f}")

    print_confusion_and_report(gb_model, X_test, y_test)

    print("\nTop 3 variables más importantes en Gradient Boosting:")
    print(show_top_features(gb_model, list(X.columns), top_k=3))

    # TODO para el alumnado:
    # Responde en el informe:
    # 1. ¿Qué diferencias observas entre Random Forest y Boosting?
    # 2. ¿Cuál parece entrenar más rápido en tu ejecución?
    # 3. ¿Cuál elegirías si priorizas rendimiento? ¿Y si priorizas simplicidad?

    # -------------------------------------------------------------------------
    # PASO 5. TABLA COMPARATIVA FINAL
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TABLA COMPARATIVA FINAL")
    print("-" * 80)

    comparison_df = pd.DataFrame(
        [
            {
                "Modelo": "Árbol simple",
                "Accuracy test": tree_results["test_accuracy"],
                "Ventajas": "TODO",
                "Desventajas": "TODO",
            },
            {
                "Modelo": "Random Forest",
                "Accuracy test": rf_model.score(X_test, y_test),
                "Ventajas": "TODO",
                "Desventajas": "TODO",
            },
            {
                "Modelo": "Gradient Boosting",
                "Accuracy test": gb_results["test_accuracy"],
                "Ventajas": "TODO",
                "Desventajas": "TODO",
            },
        ]
    )

    print(comparison_df)

    # TODO para el alumnado:
    # Completa esta reflexión final en el informe:
    # "Si tuvieras que desplegar un modelo en una aplicación móvil con poca
    # memoria y CPU, ¿cuál elegirías y por qué?"


if __name__ == "__main__":
    main()
