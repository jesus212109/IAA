"""
Práctica 4: Clasificación y el Valor del Error (Matriz de Costes)
Introducción al Aprendizaje Automático

Plantilla para el alumnado.
Objetivo:
- Entrenar un clasificador probabilístico sencillo.
- Analizar cómo cambia el comportamiento del modelo al variar el umbral.
- Minimizar el coste total según una matriz de costes.
- Comparar curva ROC y curva Precision-Recall.

Dataset esperado:
    ../data/fraude_transacciones.csv

Autor: plantilla docente
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# ============================================================
# Configuración general
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Costes del problema
COSTE_FP = 10
COSTE_FN = 200

# Umbrales a estudiar
THRESHOLDS = np.arange(0.0, 1.01, 0.05)

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "fraude_transacciones.csv"
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funciones auxiliares
# ============================================================

def cargar_datos(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset y separa variables de entrada y variable objetivo.

    Parameters
    ----------
    csv_path : Path
        Ruta al fichero CSV.

    Returns
    -------
    X : pd.DataFrame
        Variables predictoras.
    y : pd.Series
        Variable objetivo.
    """
    df = pd.read_csv(csv_path)

    if "fraude" not in df.columns:
        raise ValueError(
            "El dataset debe contener una columna llamada 'fraude'."
        )

    X = df.drop(columns=["fraude"])
    y = df["fraude"]

    return X, y


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """
    Entrena una regresión logística.

    Parameters
    ----------
    X_train : pd.DataFrame
        Datos de entrenamiento.
    y_train : pd.Series
        Etiquetas de entrenamiento.

    Returns
    -------
    LogisticRegression
        Modelo entrenado.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def predecir_con_umbral(
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Convierte probabilidades en predicciones binarias usando un umbral.

    Parameters
    ----------
    y_prob : np.ndarray
        Probabilidades de la clase positiva.
    threshold : float
        Umbral de decisión.

    Returns
    -------
    np.ndarray
        Predicciones binarias.
    """
    return (y_prob >= threshold).astype(int)


def obtener_metricas_confusion(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """
    Calcula TN, FP, FN y TP a partir de las predicciones.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_pred : np.ndarray
        Predicciones binarias.

    Returns
    -------
    dict
        Diccionario con tn, fp, fn, tp.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def calcular_coste(fp: int, fn: int) -> float:
    """
    Calcula el coste total según la matriz de costes del problema.

    Parameters
    ----------
    fp : int
        Número de falsos positivos.
    fn : int
        Número de falsos negativos.

    Returns
    -------
    float
        Coste total.
    """
    return (fp * COSTE_FP) + (fn * COSTE_FN)


def analizar_umbral(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    """
    Evalúa un único umbral de decisión.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_prob : np.ndarray
        Probabilidades de la clase positiva.
    threshold : float
        Umbral a analizar.

    Returns
    -------
    dict
        Resultados del umbral.
    """
    y_pred = predecir_con_umbral(y_prob, threshold)
    metricas = obtener_metricas_confusion(y_true, y_pred)
    coste = calcular_coste(metricas["fp"], metricas["fn"])

    return {
        "threshold": threshold,
        "tn": metricas["tn"],
        "fp": metricas["fp"],
        "fn": metricas["fn"],
        "tp": metricas["tp"],
        "coste_total": coste,
    }


def construir_tabla_umbral(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """
    Construye una tabla con resultados para varios umbrales.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_prob : np.ndarray
        Probabilidades de la clase positiva.
    thresholds : np.ndarray
        Umbrales a evaluar.

    Returns
    -------
    pd.DataFrame
        Tabla resumen por umbral.
    """
    results = []
    for t in thresholds:
        results.append(analizar_umbral(y_true, y_prob, t))

    return pd.DataFrame(results)


def guardar_tabla_resultados(df_resultados: pd.DataFrame, output_path: Path) -> None:
    """
    Guarda la tabla de resultados en CSV.

    Parameters
    ----------
    df_resultados : pd.DataFrame
        Tabla de resultados.
    output_path : Path
        Ruta de salida.
    """
    df_resultados.to_csv(output_path, index=False)


def graficar_coste_vs_umbral(
    df_resultados: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Genera y guarda la gráfica Coste Total vs Umbral.

    Parameters
    ----------
    df_resultados : pd.DataFrame
        Tabla de resultados por umbral.
    output_path : Path
        Ruta del PNG de salida.
    """
    best_idx = df_resultados["coste_total"].idxmin()
    best_row = df_resultados.loc[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(df_resultados["threshold"], df_resultados["coste_total"], marker="o")
    plt.scatter(best_row["threshold"], best_row["coste_total"], s=80)
    plt.xlabel("Umbral")
    plt.ylabel("Coste total (€)")
    plt.title("Coste Total vs Umbral")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def graficar_matriz_confusion(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """
    Genera y guarda la matriz de confusión.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_pred : np.ndarray
        Predicciones binarias.
    output_path : Path
        Ruta del PNG de salida.
    title : str
        Título de la figura.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def graficar_curva_roc(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path,
) -> float:
    """
    Genera y guarda la curva ROC.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_prob : np.ndarray
        Probabilidades de la clase positiva.
    output_path : Path
        Ruta del PNG de salida.

    Returns
    -------
    float
        Área bajo la curva ROC (AUC).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return auc


def graficar_curva_pr(
    y_true: pd.Series,
    y_prob: np.ndarray,
    output_path: Path,
) -> float:
    """
    Genera y guarda la curva Precision-Recall.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas reales.
    y_prob : np.ndarray
        Probabilidades de la clase positiva.
    output_path : Path
        Ruta del PNG de salida.

    Returns
    -------
    float
        Average Precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return ap


def imprimir_resumen_umbral(nombre: str, resultado: dict) -> None:
    """
    Imprime por pantalla un resumen de un umbral concreto.

    Parameters
    ----------
    nombre : str
        Nombre descriptivo.
    resultado : dict
        Resultados del umbral.
    """
    print(f"\n=== {nombre} ===")
    print(f"Umbral: {resultado['threshold']:.2f}")
    print(f"TN: {resultado['tn']}")
    print(f"FP: {resultado['fp']}")
    print(f"FN: {resultado['fn']}")
    print(f"TP: {resultado['tp']}")
    print(f"Coste total: {resultado['coste_total']:.2f} €")


# ============================================================
# Programa principal
# ============================================================

def main() -> None:
    """
    Ejecuta el flujo completo de la práctica.
    """
    print("Cargando datos...")
    X, y = cargar_datos(DATA_PATH)

    print("Separando entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("Entrenando modelo...")
    model = entrenar_modelo(X_train, y_train)

    print("Obteniendo probabilidades...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # ========================================================
    # Tarea 1: Umbral por defecto
    # ========================================================
    resultado_default = analizar_umbral(y_test, y_prob, threshold=0.5)
    imprimir_resumen_umbral("Umbral por defecto (0.5)", resultado_default)

    y_pred_default = predecir_con_umbral(y_prob, 0.5)
    graficar_matriz_confusion(
        y_test,
        y_pred_default,
        OUTPUT_DIR / "matriz_confusion_umbral_0_5.png",
        title="Matriz de confusión (umbral = 0.5)",
    )

    # ========================================================
    # Tarea 2 y 3: Análisis de umbrales y optimización de coste
    # ========================================================
    print("\nAnalizando varios umbrales...")
    df_resultados = construir_tabla_umbral(y_test, y_prob, THRESHOLDS)

    # Orden original por umbral para tablas y gráficas
    df_resultados = df_resultados.sort_values("threshold").reset_index(drop=True)

    guardar_tabla_resultados(
        df_resultados,
        OUTPUT_DIR / "tabla_resultados_umbral.csv",
    )

    best_idx = df_resultados["coste_total"].idxmin()
    best_row = df_resultados.loc[best_idx].to_dict()

    imprimir_resumen_umbral("Mejor umbral según coste", best_row)

    graficar_coste_vs_umbral(
        df_resultados,
        OUTPUT_DIR / "coste_vs_umbral.png",
    )

    # ========================================================
    # Tarea 4: Curvas ROC y PR
    # ========================================================
    print("\nGenerando curvas ROC y Precision-Recall...")
    auc_roc = graficar_curva_roc(
        y_test,
        y_prob,
        OUTPUT_DIR / "curva_roc.png",
    )

    ap_pr = graficar_curva_pr(
        y_test,
        y_prob,
        OUTPUT_DIR / "curva_precision_recall.png",
    )

    print(f"\nAUC ROC: {auc_roc:.4f}")
    print(f"Average Precision: {ap_pr:.4f}")

    # ========================================================
    # Indicaciones para la memoria
    # ========================================================
    print("\n=== INDICACIONES PARA LA MEMORIA ===")
    print("1. Explica si el umbral 0.5 parece razonable en este contexto.")
    print("2. Comenta cómo cambian FP y FN al variar el umbral.")
    print("3. Indica qué umbral minimiza el coste total.")
    print("4. Justifica qué política recomendarías al banco.")
    print("5. Razona si la curva ROC o la curva PR es más informativa.")
    print("6. Usa la tabla y las gráficas generadas en la carpeta outputs/.")


if __name__ == "__main__":
    main()