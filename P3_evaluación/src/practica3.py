"""
=============================================================================
PRACTICE 3: Evaluation Methodology and Scientific Rigor
Subject: Introduction to Machine Learning
3rd Year Computer Engineering - 2025/2026
=============================================================================

This script covers ALL required tasks for Practice 3:

  Task 1 -> The random split lottery
             Shows how many class-1 examples fall in the test set across
             5 runs with different random seeds (no stratification).

  Task 2 -> Stratified split
             Applies StratifiedKFold and verifies that the class-1 proportion
             remains constant across all folds.

  Task 3 -> Variance comparison
             Compares the stability (variance / std) between standard KFold
             and StratifiedKFold using Accuracy, F1-macro, and AUC-ROC.

  Task 4 -> Data Leakage detection
             Introduces an artificial variable (ID_Hospital_Filtro) that leaks
             target information. Trains with and without it and compares results.

  Challenge -> Methodological audit
             Final reflection on the good practices discovered.

Environment requirements:
    pip install scikit-learn pandas numpy matplotlib seaborn scipy
=============================================================================
"""

# ---------------------------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")          # Suppress minor warnings for
                                           # cleaner console output

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # Non-interactive backend (compatible
                                           # with headless environments)
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn models and metrics
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS
# ---------------------------------------------------------------------------
# Path to dataset (relative to script directory)
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "pacientes_riesgo.csv"
)

# Target column (label)
TARGET_COL = "Clase"

# Trap variable that causes data leakage
TRAP_COL = "ID_Hospital_Filtro"

# Number of runs for Task 1
N_RUNS_TASK1 = 5

# Number of folds for StratifiedKFold (Tasks 2 and 3)
N_FOLDS = 10

# Number of cross-validation iterations for Task 3
N_ITER_TASK3 = 10

# Test set fraction for simple split (20%)
TEST_SIZE = 0.20

# Directory where generated figures will be saved
FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs", "figuras"
)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def separador(titulo: str, ancho: int = 70) -> None:
    """
    Print a visual separator with centered title.
    Useful for organizing console output by sections.
    """
    print("\n" + "=" * ancho)
    print(f"  {titulo}")
    print("=" * ancho)


def subseparador(titulo: str, ancho: int = 70) -> None:
    """Second-level separator (dashes)."""
    print("\n" + "-" * ancho)
    print(f"  {titulo}")
    print("-" * ancho)


def cargar_dataset(path: str) -> pd.DataFrame:
    """
    Load the synthetic patient dataset from a CSV file.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame with the loaded data.
    """
    df = pd.read_csv(path)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def crear_modelo() -> Pipeline:
    """
    Create a pipeline with standard scaling and logistic regression.

    Using a Pipeline prevents data leakage between train and test during
    cross-validation: the scaler is fitted ONLY on the train set of each
    fold and then applied to the test set of that fold.

    Returns
    -------
    sklearn.pipeline.Pipeline ready to train.
    """
    modelo = Pipeline([
        ("scaler", StandardScaler()),          # Standardization (mean=0, std=1)
        ("clf", LogisticRegression(
            max_iter=1000,                     # More iterations for convergence
            class_weight="balanced",           # Increases penalty for class-1 errors
            random_state=42,
            solver="lbfgs",
        ))
    ])
    return modelo


# ===========================================================================
# TAREA 1 — LA LOTERÍA DE LA PARTICIÓN ALEATORIA
# ===========================================================================

def tarea1_loteria_particion(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Ejecuta train_test_split 5 veces con semillas distintas y registra
    cuántos ejemplos de clase 1 quedan en cada conjunto de test.

    El objetivo es demostrar que, en datasets muy desbalanceados, una
    partición aleatoria simple puede producir conjuntos de test con muy
    pocos (o ningún) ejemplo positivo, haciendo la evaluación poco fiable.

    Parámetros
    ----------
    X : pd.DataFrame   → Variables predictoras
    y : pd.Series      → Variable objetivo (0/1)

    Retorna
    -------
    pd.DataFrame con el resumen de cada ejecución.
    """
    separador("TAREA 1 — La Lotería de la Partición Aleatoria")

    resultados = []

    for seed in range(N_RUNS_TASK1):
        # Partición aleatoria SIN estratificar (aquí está el problema)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=seed,         # semilla distinta en cada ejecución
            stratify=None              # SIN estratificación → lotería
        )

        # Contamos cuántos positivos (clase 1) hay en el test
        n_positivos_test  = int(y_test.sum())
        n_positivos_train = int(y_train.sum())
        pct_pos_test      = 100.0 * n_positivos_test / len(y_test)

        resultados.append({
            "Ejecución":           seed + 1,
            "Semilla":             seed,
            "Positivos en test":   n_positivos_test,
            "Positivos en train":  n_positivos_train,
            "% clase 1 en test":   round(pct_pos_test, 2),
            "Total en test":       len(y_test),
        })

        print(
            f"  Ejecución {seed+1} (seed={seed}): "
            f"{n_positivos_test} positivos en test "
            f"({pct_pos_test:.1f}%)"
        )

    df_res = pd.DataFrame(resultados)
    print("\n[TABLA RESUMEN - TAREA 1]")
    print(df_res.to_string(index=False))

    # ------------------------------------------------------------------
    # PREGUNTA: ¿Es posible evaluar bien si en test hay 0 o 1 positivo?
    # ------------------------------------------------------------------
    print("\n[ANÁLISIS]")
    print(
        "  Si el test solo contiene 0 o 1 ejemplo positivo, el modelo no puede\n"
        "  ser evaluado con métricas como F1 o Recall para la clase minoritaria.\n"
        "  La Accuracy puede ser 98-100% simplemente prediciendo siempre clase 0,\n"
        "  dando una falsa sensación de buen rendimiento.\n"
        "  → La partición aleatoria es poco fiable con datasets desbalanceados."
    )

    # ------------------------------------------------------------------
    # GRÁFICO: Número de positivos en test por ejecución
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        df_res["Ejecución"], df_res["Positivos en test"],
        color="steelblue", edgecolor="white", linewidth=0.8
    )
    ax.axhline(
        y=y.sum() * TEST_SIZE,            # valor esperado teórico
        color="red", linestyle="--", linewidth=1.5,
        label=f"Esperado (20% de {int(y.sum())} positivos)"
    )
    ax.set_xlabel("Ejecución (semilla)")
    ax.set_ylabel("Nº de ejemplos de clase 1 en test")
    ax.set_title("Tarea 1 — Variabilidad de positivos en test\n(partición aleatoria sin estratificar)")
    ax.legend()
    plt.tight_layout()
    ruta_fig = os.path.join(FIGURES_DIR, "tarea1_loteria.png")
    plt.savefig(ruta_fig, dpi=150)
    plt.close()
    print(f"\n[FIGURA] Guardada en: {ruta_fig}")

    return df_res


# ===========================================================================
# TAREA 2 — PARTICIÓN ESTRATIFICADA
# ===========================================================================

def tarea2_particion_estratificada(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Aplica StratifiedKFold y comprueba que la proporción de clase 1
    se mantiene constante en TODOS los folds.

    La estratificación garantiza que cada fold tenga la misma distribución
    de clases que el dataset original, lo cual es fundamental cuando las
    clases están desbalanceadas.

    Parámetros
    ----------
    X : pd.DataFrame   → Variables predictoras
    y : pd.Series      → Variable objetivo (0/1)

    Retorna
    -------
    pd.DataFrame con la proporción de clase 1 por fold.
    """
    separador("TAREA 2 — Partición Estratificada con StratifiedKFold")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    proporcion_global = y.mean() * 100  # proporción real en el dataset
    print(f"  Proporción global de clase 1 en el dataset: {proporcion_global:.2f}%\n")

    resultados = []

    for i, (idx_train, idx_test) in enumerate(skf.split(X, y), start=1):
        y_fold_test  = y.iloc[idx_test]
        y_fold_train = y.iloc[idx_train]

        pct_test  = 100.0 * y_fold_test.mean()
        pct_train = 100.0 * y_fold_train.mean()

        resultados.append({
            "Fold":                    i,
            "% clase 1 en test":       round(pct_test,  2),
            "% clase 1 en train":      round(pct_train, 2),
            "Ejemplos positivos test": int(y_fold_test.sum()),
            "Tamaño test":             len(y_fold_test),
        })

        print(
            f"  Fold {i:2d}: test={len(y_fold_test):4d} muestras, "
            f"clase1_test={pct_test:.2f}%, "
            f"clase1_train={pct_train:.2f}%"
        )

    df_res = pd.DataFrame(resultados)

    # Comprobación estadística: la desviación de la proporción entre folds
    std_pct = df_res["% clase 1 en test"].std()
    print(f"\n  Desviación típica de la proporción de clase 1 entre folds: {std_pct:.4f}%")
    print("  → Cercana a 0, lo que confirma que la estratificación funciona correctamente.\n")

    print("[TABLA RESUMEN - TAREA 2]")
    print(df_res.to_string(index=False))

    print("\n[ANÁLISIS]")
    print(
        "  Con StratifiedKFold, cada fold mantiene aprox. el mismo porcentaje\n"
        "  de clase 1. Esto garantiza que las métricas de evaluación sean\n"
        "  representativas del rendimiento real del modelo en todas las particiones.\n"
        "  Sin estratificación, la varianza entre folds puede ser enorme,\n"
        "  haciendo la comparación entre modelos poco fiable."
    )

    # ------------------------------------------------------------------
    # GRÁFICO: Proporción de clase 1 por fold
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df_res["Fold"], df_res["% clase 1 en test"],
        marker="o", color="steelblue", linewidth=2, label="% clase 1 en test"
    )
    ax.axhline(
        y=proporcion_global, color="red", linestyle="--",
        linewidth=1.5, label=f"Proporción global ({proporcion_global:.2f}%)"
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("% clase 1 en test")
    ax.set_title(
        "Tarea 2 — Proporción de clase positiva por fold\n"
        "(StratifiedKFold: la proporción se mantiene estable)"
    )
    ax.legend()
    ax.set_xticks(df_res["Fold"])
    plt.tight_layout()
    ruta_fig = os.path.join(FIGURES_DIR, "tarea2_estratificada.png")
    plt.savefig(ruta_fig, dpi=150)
    plt.close()
    print(f"\n[FIGURA] Guardada en: {ruta_fig}")

    return df_res


# ===========================================================================
# TAREA 3 — COMPARATIVA DE LA VARIANZA
# ===========================================================================

def tarea3_comparativa_varianza(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Compare the evaluation stability between standard KFold and StratifiedKFold,
    repeating N_ITER_TASK3 times for each strategy.

    On imbalanced datasets, standard KFold produces higher metric variance
    because each run may have very different class proportions.
    StratifiedKFold produces more stable metrics.

    Metrics evaluated: Accuracy, F1-macro, AUC-ROC.

    Parameters
    ----------
    X : pd.DataFrame   -> Feature matrix
    y : pd.Series      -> Target variable (0/1)

    Returns
    -------
    dict with results for both strategies across all iterations.
    """
    separador("TASK 3 — Variance Comparison (KFold vs StratifiedKFold)")

    modelo = crear_modelo()

    # Store per-iteration results
    acc_kfold,  f1_kfold,  auc_kfold  = [], [], []
    acc_skfold, f1_skfold, auc_skfold = [], [], []

    subseparador("Standard KFold (without stratification)")
    for seed in range(N_ITER_TASK3):
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        # Accuracy using cross_val_score
        acc = cross_val_score(modelo, X, y, cv=kf, scoring="accuracy")
        # F1-macro (averaged across both classes)
        f1  = cross_val_score(modelo, X, y, cv=kf, scoring="f1_macro")
        # AUC-ROC (requires probability estimates)
        auc = cross_val_score(modelo, X, y, cv=kf, scoring="roc_auc")

        acc_kfold.append(acc.mean())
        f1_kfold.append(f1.mean())
        auc_kfold.append(auc.mean())
        print(
            f"  Iter {seed+1:2d} (seed={seed}): "
            f"Acc={acc.mean():.4f}, F1={f1.mean():.4f}, AUC={auc.mean():.4f}"
        )

    subseparador("StratifiedKFold (with stratification)")
    for seed in range(N_ITER_TASK3):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        acc = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")
        f1  = cross_val_score(modelo, X, y, cv=skf, scoring="f1_macro")
        auc = cross_val_score(modelo, X, y, cv=skf, scoring="roc_auc")

        acc_skfold.append(acc.mean())
        f1_skfold.append(f1.mean())
        auc_skfold.append(auc.mean())
        print(
            f"  Iter {seed+1:2d} (seed={seed}): "
            f"Acc={acc.mean():.4f}, F1={f1.mean():.4f}, AUC={auc.mean():.4f}"
        )

    # ------------------------------------------------------------------
    # STATISTICAL SUMMARY
    # ------------------------------------------------------------------
    # Count NaN values: NaN in AUC means a fold had no positive examples
    nan_kfold  = int(np.sum(np.isnan(auc_kfold)))
    nan_skfold = int(np.sum(np.isnan(auc_skfold)))

    print("\n[COMPARATIVE TABLE - TASK 3]")
    print(f"{'Metric':<30} {'KFold':>12} {'StratifiedKFold':>18}")
    print("-" * 62)

    metrics = {
        "Accuracy (mean)":  (np.mean(acc_kfold),       np.mean(acc_skfold)),
        "Accuracy (std)":   (np.std(acc_kfold),        np.std(acc_skfold)),
        "F1-macro (mean)":  (np.mean(f1_kfold),        np.mean(f1_skfold)),
        "F1-macro (std)":   (np.std(f1_kfold),         np.std(f1_skfold)),
        "AUC-ROC (mean)": (np.nanmean(auc_kfold),  np.nanmean(auc_skfold)),
        "AUC-ROC (std)":  (np.nanstd(auc_kfold),   np.nanstd(auc_skfold)),
    }
    for name, (v_kf, v_skf) in metrics.items():
        print(f"  {name:<28} {v_kf:>12.4f} {v_skf:>18.4f}")

    print(f"\n  [WARNING] KFold AUC-ROC was NaN in {nan_kfold}/{N_ITER_TASK3} iterations")
    print(f"            (folds with zero positive examples -> undefined AUC-ROC)")
    print(f"            StratifiedKFold AUC-ROC NaN count: {nan_skfold}/{N_ITER_TASK3}")

    print("\n[CONCLUSION]")
    if np.std(acc_skfold) < np.std(acc_kfold):
        print("  [OK] StratifiedKFold produces Accuracy with LOWER standard deviation.")
    if np.std(f1_skfold) < np.std(f1_kfold):
        print("  [OK] StratifiedKFold produces F1-macro with LOWER standard deviation.")
    if nan_kfold > nan_skfold:
        print(f"  [OK] KFold made AUC-ROC undefined (NaN) in {nan_kfold} iterations vs"
              f" {nan_skfold} in StratifiedKFold.")
    elif np.nanstd(auc_skfold) < np.nanstd(auc_kfold):
        print("  [OK] StratifiedKFold produces AUC-ROC with LOWER standard deviation.")
    print(
        "  -> StratifiedKFold is more appropriate for imbalanced problems because\n"
        "     it guarantees representative partitions, reducing metric variance\n"
        "     and making evaluation more reliable and reproducible.\n"
        "  -> AUC-ROC especially benefits from stratification: it relies on the\n"
        "     ranking of predicted probabilities for the positive class, which\n"
        "     requires at least some positive examples in every test fold."
    )

    # ------------------------------------------------------------------
    # FIGURE: Metric distributions for both strategies (3 subplots)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ---- Accuracy ----
    axes[0].boxplot(
        [acc_kfold, acc_skfold],
        labels=["KFold", "StratifiedKFold"],
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", color="navy"),
        medianprops=dict(color="white", linewidth=2),
    )
    axes[0].set_title("Accuracy", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Mean accuracy per iteration")

    # ---- F1-macro ----
    axes[1].boxplot(
        [f1_kfold, f1_skfold],
        labels=["KFold", "StratifiedKFold"],
        patch_artist=True,
        boxprops=dict(facecolor="darkorange", color="saddlebrown"),
        medianprops=dict(color="white", linewidth=2),
    )
    axes[1].set_title("F1-macro", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Mean F1-macro per iteration")

    # ---- AUC-ROC ----
    axes[2].boxplot(
        [auc_kfold, auc_skfold],
        labels=["KFold", "StratifiedKFold"],
        patch_artist=True,
        boxprops=dict(facecolor="mediumseagreen", color="darkgreen"),
        medianprops=dict(color="white", linewidth=2),
    )
    axes[2].set_title("AUC-ROC", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Mean AUC-ROC per iteration")

    plt.suptitle(
        "Task 3 — Metric Variance: KFold vs StratifiedKFold\n"
        "(10 cross-validation iterations, 10 folds each)",
        y=1.02, fontsize=13
    )
    plt.tight_layout()
    ruta_fig = os.path.join(FIGURES_DIR, "tarea3_varianza.png")
    plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[FIGURE] Saved at: {ruta_fig}")

    return {
        "acc_kfold":   acc_kfold,
        "f1_kfold":    f1_kfold,
        "auc_kfold":   auc_kfold,
        "acc_skfold":  acc_skfold,
        "f1_skfold":   f1_skfold,
        "auc_skfold":  auc_skfold,
    }


# ===========================================================================
# TAREA 4 — DETECCIÓN DE DATA LEAKAGE
# ===========================================================================

def tarea4_data_leakage(
    X_con_trampa: pd.DataFrame,
    X_sin_trampa: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Demuestra el efecto del data leakage introducido por la variable
    'ID_Hospital_Filtro', que filtra información del target al estar
    correlacionada artificialmente con la clase.

    Entrenamos el mismo modelo DOS veces:
      1) Con la variable trampa incluida
      2) Sin la variable trampa

    Y comparamos las métricas obtenidas.

    Parámetros
    ----------
    X_con_trampa : pd.DataFrame  → Todas las variables (incluyendo trampa)
    X_sin_trampa : pd.DataFrame  → Solo variables legítimas
    y            : pd.Series     → Variable objetivo

    Retorna
    -------
    dict con los resultados de ambas configuraciones.
    """
    separador("TAREA 4 — Detección de Data Leakage (La Trampa)")

    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    modelo = crear_modelo()

    # ------------------------------------------------------------------
    # 4.1 — CON la variable trampa
    # ------------------------------------------------------------------
    subseparador("4.1 Entrenando CON la variable trampa (ID_Hospital_Filtro)")
    print(
        "  ADVERTENCIA: Esta variable está correlacionada artificialmente con\n"
        "  la clase objetivo. Su inclusión genera una evaluación engañosa.\n"
    )

    acc_trampa = cross_val_score(modelo, X_con_trampa, y, cv=skf, scoring="accuracy")
    f1_trampa  = cross_val_score(modelo, X_con_trampa, y, cv=skf, scoring="f1_macro")

    print(
        f"  Accuracy con trampa → media={acc_trampa.mean():.4f}, "
        f"std={acc_trampa.std():.4f}"
    )
    print(
        f"  F1-macro con trampa → media={f1_trampa.mean():.4f}, "
        f"std={f1_trampa.std():.4f}"
    )

    if acc_trampa.mean() > 0.90:
        print(
            "\n  ⚠️  RESULTADO SOSPECHOSO: Accuracy cercana o superior al 90%.\n"
            "       En un problema real con estas características, una Accuracy\n"
            "       tan alta casi siempre indica data leakage u otro error."
        )

    # ------------------------------------------------------------------
    # 4.2 — SIN la variable trampa
    # ------------------------------------------------------------------
    subseparador("4.2 Entrenando SIN la variable trampa")
    print(
        "  Ahora se eliminó 'ID_Hospital_Filtro'. Los resultados deben ser\n"
        "  significativamente más bajos y representativos de la dificultad real.\n"
    )

    acc_real = cross_val_score(modelo, X_sin_trampa, y, cv=skf, scoring="accuracy")
    f1_real  = cross_val_score(modelo, X_sin_trampa, y, cv=skf, scoring="f1_macro")

    print(
        f"  Accuracy sin trampa → media={acc_real.mean():.4f}, "
        f"std={acc_real.std():.4f}"
    )
    print(
        f"  F1-macro sin trampa → media={f1_real.mean():.4f}, "
        f"std={f1_real.std():.4f}"
    )

    # ------------------------------------------------------------------
    # 4.3 — REFLEXIÓN comparativa
    # ------------------------------------------------------------------
    subseparador("4.3 Reflexión / Comparativa")
    diff_acc = acc_trampa.mean() - acc_real.mean()
    diff_f1  = f1_trampa.mean()  - f1_real.mean()
    print(f"  Diferencia en Accuracy (trampa - real): {diff_acc:+.4f}")
    print(f"  Diferencia en F1-macro (trampa - real): {diff_f1:+.4f}")
    print(
        "\n  La variable 'ID_Hospital_Filtro' provoca una inflación artificial\n"
        "  del rendimiento. Al eliminarla, el modelo muestra su capacidad real.\n"
        "  Publicar o usar en producción un modelo entrenado con esta variable\n"
        "  llevaría a conclusiones falsas sobre su utilidad clínica."
    )

    # ------------------------------------------------------------------
    # GRÁFICO: Comparativa con y sin trampa
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    categorias  = ["Accuracy\ncon trampa", "Accuracy\nsin trampa",
                   "F1-macro\ncon trampa", "F1-macro\nsin trampa"]
    valores_bar = [
        acc_trampa.mean(), acc_real.mean(),
        f1_trampa.mean(),  f1_real.mean()
    ]
    errores_bar = [
        acc_trampa.std(), acc_real.std(),
        f1_trampa.std(),  f1_real.std()
    ]
    colores = ["#e74c3c", "#2ecc71", "#e74c3c", "#2ecc71"]  # rojo=trampa, verde=real

    bars = ax.bar(categorias, valores_bar, yerr=errores_bar, capsize=5,
                  color=colores, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Puntuación media (±std)")
    ax.set_title(
        "Tarea 4 — Efecto del Data Leakage\n"
        "(rojo=con trampa, verde=sin trampa)"
    )
    # Etiquetas de valor sobre cada barra.
    # Se posicionan por encima del cap de la barra de error (val + std + margen)
    # para evitar solapamiento en barras con desviación grande (p.ej. acc_real).
    for bar, val, err in zip(bars, valores_bar, errores_bar):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + err + 0.04,          # supera el cap de error + margen extra
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    plt.tight_layout()
    ruta_fig = os.path.join(FIGURES_DIR, "tarea4_leakage.png")
    plt.savefig(ruta_fig, dpi=150)
    plt.close()
    print(f"\n[FIGURA] Guardada en: {ruta_fig}")

    return {
        "acc_trampa":  acc_trampa,
        "f1_trampa":   f1_trampa,
        "acc_real":    acc_real,
        "f1_real":     f1_real,
    }


# ===========================================================================
# RETO — AUDITORÍA METODOLÓGICA
# ===========================================================================

def reto_auditoria() -> None:
    """
    Actúa como revisor/auditor de un artículo científico ficticio.
    Identifica las decisiones metodológicas incorrectas descubiertas
    en las tareas anteriores y explica las buenas prácticas a seguir.
    """
    separador("RETO — Auditoría Metodológica")

    print("""
  ══════════════════════════════════════════════════════════════════
  INFORME DE AUDITORÍA METODOLÓGICA
  Experimentos de clasificación binaria desbalanceada (2% clase 1)
  ══════════════════════════════════════════════════════════════════

  ─────────────────────────────────────────────────────────────────
  PROBLEMA 1 — Partición aleatoria simple en datos desbalanceados
  ─────────────────────────────────────────────────────────────────
  ERROR DETECTADO:
    Usar train_test_split sin stratify=y en un dataset con solo un
    2% de la clase positiva puede resultar en conjuntos de test con
    0 o 1 ejemplos positivos, dependiendo de la semilla aleatoria.

  POR QUÉ INVALIDA LA EVALUACIÓN:
    → Con 0 positivos en test, las métricas como F1, Recall o AUC-ROC
      de la clase positiva son matemáticamente indefinidas o triviales.
    → La Accuracy aparente (>97%) no refleja capacidad predictiva real:
      predecir siempre clase 0 ya da ≈98% de Accuracy en este dataset.
    → La reproducibilidad del experimento es nula: cambiando la semilla
      se pueden obtener resultados radicalmente distintos.

  BUENA PRÁCTICA:
    → Usar siempre stratify=y en train_test_split cuando las clases
      están desbalanceadas.

  ─────────────────────────────────────────────────────────────────
  PROBLEMA 2 — KFold estándar vs StratifiedKFold en datasets
               desbalanceados
  ─────────────────────────────────────────────────────────────────
  ERROR DETECTADO:
    Aplicar KFold sin estratificación en datos desbalanceados produce
    folds con proporciones de clase muy variables.

  POR QUÉ INVALIDA LA EVALUACIÓN:
    → Una alta varianza entre folds hace que elegir el "mejor modelo"
      sea dependiente del azar, no del mérito real del clasificador.
    → Las comparaciones entre modelos pueden llevar a elegir el modelo
      incorrecto simplemente porque le tocaron folds "más fáciles".

  BUENA PRÁCTICA:
    → Usar StratifiedKFold para garantizar que cada fold tenga la
      misma distribución de clases que el dataset completo.
    → Reportar media ± desviación típica de las métricas para
      cuantificar la incertidumbre de la evaluación.

  ─────────────────────────────────────────────────────────────────
  PROBLEMA 3 — Data Leakage por variable correlacionada con el target
  ─────────────────────────────────────────────────────────────────
  ERROR DETECTADO:
    La variable 'ID_Hospital_Filtro' está artificialmente correlacionada
    con la variable objetivo. Al incluirla durante el entrenamiento y la
    evaluación, el modelo "aprende" implícitamente el target a través de
    esta variable en lugar de aprender el problema real.

  EFECTO OBSERVADO:
    → Las métricas de evaluación son anormalmente altas (>90% Accuracy,
      F1-macro >>0.8), muy por encima de lo esperado para este problema.
    → Al eliminar la variable trampa, las métricas caen a niveles más
      realistas y acordes a la dificultad del dataset.

  POR QUÉ CONDUCE A CONCLUSIONES FALSAS:
    → Un modelo con data leakage tiene un rendimiento excelente en el
      sistema de evaluación pero fallará estrepitosamente en producción,
      donde la variable trampa no estaría disponible (o no debería estarlo).
    → En un contexto clínico, desplegar ese modelo podría causar errores
      médicos graves al no detectar casos de riesgo real.

  BUENA PRÁCTICA:
    → Auditar todas las variables del dataset antes de entrenar.
    → Verificar que ninguna feature es una transformación o derivada
      del target (ni temporal ni artificialmente).
    → Aplicar una matriz de correlación y análisis de importancias de
      características para detectar variables sospechosamente fuertes.

  ─────────────────────────────────────────────────────────────────
  RESUMEN DE BUENAS PRÁCTICAS PARA FUTUROS EXPERIMENTOS
  ─────────────────────────────────────────────────────────────────
    1. Siempre estratificar las particiones en problemas desbalanceados.
    2. Usar StratifiedKFold en lugar de KFold estándar.
    3. Reportar métricas orientadas a clases minoritarias: F1, Recall,
       Precision y AUC-ROC además de la Accuracy global.
    4. Auditar el dataset en busca de variables con data leakage.
    5. Usar Pipelines para asegurar que los preprocesados no filtran
       información del test al train durante la validación cruzada.
    6. Repetir experimentos con distintas semillas y reportar la
       desviación típica para cuantificar la incertidumbre.
    7. Documentar y compartir el código de evaluación para asegurar
       la reproducibilidad del experimento.
  ══════════════════════════════════════════════════════════════════
    """)


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================

def main() -> None:
    """
    Punto de entrada del programa. Ejecuta todas las tareas en orden y
    presenta un resumen final de los hallazgos.
    """
    print("=" * 70)
    print("  PRÁCTICA 3 — Metodología de Evaluación y Rigor Científico")
    print("  Introducción al Aprendizaje Automático — Curso 2025/2026")
    print("=" * 70)

    # ------------------------------------------------------------------
    # CARGA DEL DATASET
    # ------------------------------------------------------------------
    df = cargar_dataset(DATASET_PATH)

    # Información general del dataset
    print(f"\n  Columnas: {list(df.columns)}")
    print(f"  Distribución de clases:\n{df[TARGET_COL].value_counts().to_string()}")
    print(f"  Proporción clase 1: {df[TARGET_COL].mean()*100:.2f}%")

    # ------------------------------------------------------------------
    # SEPARAR FEATURES Y TARGET
    # ------------------------------------------------------------------
    # Variables predictoras legítimas (sin incluir la trampa ni el target)
    features_legit = [c for c in df.columns if c not in [TARGET_COL, TRAP_COL]]

    # Todas las variables (incluyendo la trampa → para Tarea 4)
    features_todos = [c for c in df.columns if c != TARGET_COL]

    X_sin_trampa  = df[features_legit]          # Solo variables reales
    X_con_trampa  = df[features_todos]          # Variables reales + ID_Hospital_Filtro
    y             = df[TARGET_COL]

    print(f"\n  Variables legítimas ({len(features_legit)}): {features_legit}")
    print(f"  Variable trampa: {TRAP_COL}")

    # ------------------------------------------------------------------
    # EJECUCIÓN DE LAS TAREAS
    # ------------------------------------------------------------------

    # ── Tarea 1 ──────────────────────────────────────────────────────
    df_t1 = tarea1_loteria_particion(X_sin_trampa, y)

    # ── Tarea 2 ──────────────────────────────────────────────────────
    df_t2 = tarea2_particion_estratificada(X_sin_trampa, y)

    # ── Tarea 3 ──────────────────────────────────────────────────────
    resultados_t3 = tarea3_comparativa_varianza(X_sin_trampa, y)

    # ── Tarea 4 ──────────────────────────────────────────────────────
    resultados_t4 = tarea4_data_leakage(X_con_trampa, X_sin_trampa, y)

    # ── Reto: Auditoría ──────────────────────────────────────────────
    reto_auditoria()

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    separador("FINAL EXECUTIVE SUMMARY")
    print(f"""
  Dataset: {df.shape[0]} patients, {df[TARGET_COL].sum()} positives ({df[TARGET_COL].mean()*100:.1f}%)

  TASK 1 — Random split lottery:
    · Range of positives in test:       {df_t1["Positivos en test"].min()} – {df_t1["Positivos en test"].max()}
    · Standard deviation:               {df_t1["Positivos en test"].std():.2f}
    -> High variability -> random split is NOT reliable here.

  TASK 2 — Stratified split:
    · Std of class-1 proportion:        {df_t2["% clase 1 en test"].std():.4f}%
    -> Near zero -> StratifiedKFold guarantees representativeness.

  TASK 3 — Variance comparison (KFold vs StratifiedKFold):
    · Accuracy std  KFold:              {np.std(resultados_t3["acc_kfold"]):.4f}
    · Accuracy std  StratifiedKFold:    {np.std(resultados_t3["acc_skfold"]):.4f}
    · F1-macro std  KFold:              {np.std(resultados_t3["f1_kfold"]):.4f}
    · F1-macro std  StratifiedKFold:    {np.std(resultados_t3["f1_skfold"]):.4f}
    · AUC-ROC NaN iters KFold:          {int(np.sum(np.isnan(resultados_t3["auc_kfold"])))}/{N_ITER_TASK3} (folds with zero positives)
    · AUC-ROC mean  KFold (valid only): {np.nanmean(resultados_t3["auc_kfold"]):.4f}
    · AUC-ROC std   KFold (valid only): {np.nanstd(resultados_t3["auc_kfold"]):.4f}
    · AUC-ROC mean  StratifiedKFold:    {np.nanmean(resultados_t3["auc_skfold"]):.4f}
    · AUC-ROC std   StratifiedKFold:    {np.nanstd(resultados_t3["auc_skfold"]):.4f}
    -> KFold AUC-ROC is undefined in most iterations: empirical proof of the problem.
    -> StratifiedKFold produces computable, stable metrics across all three indicators.

  TASK 4 — Data Leakage:
    · Accuracy WITH trap:  {resultados_t4["acc_trampa"].mean():.4f} +/- {resultados_t4["acc_trampa"].std():.4f}
    · Accuracy WITHOUT trap: {resultados_t4["acc_real"].mean():.4f} +/- {resultados_t4["acc_real"].std():.4f}
    · F1-macro WITH trap:  {resultados_t4["f1_trampa"].mean():.4f} +/- {resultados_t4["f1_trampa"].std():.4f}
    · F1-macro WITHOUT trap: {resultados_t4["f1_real"].mean():.4f} +/- {resultados_t4["f1_real"].std():.4f}
    -> Variable 'ID_Hospital_Filtro' artificially inflates all metrics.

  Figures saved in: {FIGURES_DIR}
    """)


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
