"""
Práctica 6: Clasificación de Texto y Filtros Anti-Spam (versión base para alumnos)

Objetivo:
    Completar un clasificador de texto basado en Naïve Bayes para distinguir
    entre mensajes legítimos (ham) y spam.

Instrucciones:
    1. Lee el código completo antes de empezar.
    2. Completa las zonas marcadas con TODO.
    3. Ejecuta el script y revisa los resultados.
    4. Responde en tu informe a las preguntas planteadas en los comentarios.

Dependencias:
    pip install pandas scikit-learn matplotlib

Ejecución:
    python practica_6_naive_bayes_spam_base_alumno.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================================
# PARTE 1. CARGA Y LIMPIEZA DEL DATASET
# ============================================================================

def clean_text(text: str) -> str:
    """
    Limpia un texto de entrada.

    TODO:
        Completa esta función para que haga, al menos, lo siguiente:
        - pasar el texto a minúsculas;
        - eliminar signos de puntuación y símbolos extraños;
        - eliminar espacios múltiples.

    Pista:
        Puedes usar expresiones regulares con re.sub(...).
    """
    text = str(text)

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset y prepara una etiqueta numérica.

    Se espera un CSV con las columnas:
        - Category
        - Message

    ham  -> 0
    spam -> 1
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero: {csv_path.resolve()}\n"
            "Asegúrate de que 'spam.csv' está en la misma carpeta que este script."
        )

    df = pd.read_csv(csv_path)

    print("\nColumnas detectadas en el CSV:")
    print(list(df.columns))

    # Nos quedamos solo con las columnas necesarias.
    df = df[["Category", "Message"]].copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Message"] = df["Message"].astype(str)

    df["label_num"] = df["Category"].map({"ham": 0, "spam": 1})

    return df


# ============================================================================
# PARTE 2. EXPLORACIÓN INICIAL
# ============================================================================

def show_basic_info(df: pd.DataFrame) -> None:
    """Muestra información básica del dataset."""
    print("\n" + "=" * 80)
    print("INFORMACIÓN BÁSICA DEL DATASET")
    print("=" * 80)
    print(f"Número total de mensajes: {len(df)}")
    print("\nDistribución de clases:")
    print(df["Category"].value_counts())
    print("\nPrimeros 5 mensajes:")
    print(df.head())

    # Pregunta para el informe:
    # ¿El dataset está balanceado o hay bastantes más mensajes ham que spam?


# ============================================================================
# PARTE 3. PARTICIÓN TRAIN / TEST Y BOLSA DE PALABRAS
# ============================================================================

def prepare_data(df: pd.DataFrame):
    """
    Prepara los datos para el entrenamiento.

    TODO:
        - Separar variables de entrada X y etiquetas y.
        - Dividir en entrenamiento y test.
        - Crear un CountVectorizer usando clean_text como preprocessor.
        - Ajustar el vectorizador con train y transformar train/test.
    """
    X = df["Message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = CountVectorizer(preprocessor=clean_text)

    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer, X_train_dtm, X_test_dtm


# ============================================================================
# PARTE 4. ENTRENAMIENTO DEL MODELO
# ============================================================================

def train_model(X_train_dtm, y_train):
    """
    Entrena un modelo Multinomial Naïve Bayes.

    TODO:
        Crear el modelo con alpha=1.0 y ajustarlo con fit(...).

    Pregunta para el informe:
        ¿Por qué es importante usar alpha=1.0 en vez de dejar que una palabra
        no observada provoque probabilidad cero?
    """
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_dtm, y_train)

    return model


# ============================================================================
# PARTE 5. EVALUACIÓN
# ============================================================================

def evaluate_model(model, X_test_dtm, y_test, output_dir: str | Path | None = None):
    """
    Evalúa el modelo sobre el conjunto de prueba.

    TODO:
        - Obtener predicciones.
        - Mostrar classification_report.
        - Calcular la matriz de confusión.
        - Guardar una imagen con la matriz de confusión.
    """
    y_pred = model.predict(X_test_dtm)

    print("\n" + "=" * 80)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 80)

    print("Métricas de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    cm = confusion_matrix(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("Matriz de Confusión - Naïve Bayes")
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / "matriz_confusion_spam_base.png"

        plt.savefig(fig_path, dpi=300)

        print(f"\nFigura guardada en: {fig_path}")

    # Cerrar la figura si no hay entorno gráfico.
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close()

    return y_pred


# ============================================================================
# PARTE 6. PALABRAS MÁS ASOCIADAS AL SPAM
# ============================================================================

def get_top_spam_words(model, vectorizer, top_n: int = 5) -> pd.DataFrame:
    """
    Devuelve las palabras más asociadas a la clase spam.

    Idea:
        Puedes comparar las probabilidades logarítmicas aprendidas para spam y ham.
        Cuanto mayor sea la diferencia logP(word|spam) - logP(word|ham),
        más indicativa será esa palabra del spam.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())

    log_prob_ham = model.feature_log_prob_[0]
    log_prob_spam = model.feature_log_prob_[1]

    score = log_prob_spam - log_prob_ham
    top_idx = np.argsort(score)[::-1][:top_n]

    result = pd.DataFrame(
        {
            "word": feature_names[top_idx],
            "logP(word|spam)": log_prob_spam[top_idx],
            "logP(word|ham)": log_prob_ham[top_idx],
            "spam_minus_ham": score[top_idx],
        }
    )
    return result


# ============================================================================
# PARTE 7. MENSAJES DE PRUEBA INVENTADOS POR EL ALUMNO
# ============================================================================

def classify_custom_messages(messages: list[str], model, vectorizer) -> pd.DataFrame:
    """
    Clasifica mensajes inventados manualmente.

    TODO:
        - Transformar los mensajes con el vectorizador.
        - Predecir la clase.
        - Obtener predict_proba.
        - Devolver una tabla clara con resultados.
    """
    X_new = vectorizer.transform(messages)
    predicted_class = model.predict(X_new)
    predicted_proba = model.predict_proba(X_new)

    results = pd.DataFrame(
        {
            "message": messages,
            "predicted_label": ["spam" if label == 1 else "ham" for label in predicted_class],
            "P(ham)": predicted_proba[:, 0],
            "P(spam)": predicted_proba[:, 1],
        }
    )
    return results


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main() -> None:
    """Ejecuta toda la práctica."""
    csv_path = Path(__file__).with_name("spam.csv")

    # 1. Cargar dataset
    df = load_dataset(csv_path)
    show_basic_info(df)

    # 2. Preparar datos
    (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
        X_train_dtm,
        X_test_dtm,
    ) = prepare_data(df)

    # 3. Tamaño del vocabulario
    print("\n" + "=" * 80)
    print("INFORMACIÓN DEL VOCABULARIO")
    print("=" * 80)

    print(f"Tamaño del vocabulario: {len(vectorizer.vocabulary_)} palabras únicas.")

    # 4. Entrenar modelo
    model = train_model(X_train_dtm, y_train)

    # 5. Mostrar probabilidades a priori
    print("\n" + "=" * 80)
    print("PROBABILIDADES A PRIORI")
    print("=" * 80)

    prior_probs = np.exp(model.class_log_prior_)
    print(f"P(ham)  = {prior_probs[0]:.4f}")
    print(f"P(spam) = {prior_probs[1]:.4f}")

    # 6. Evaluación
    evaluate_model(model, X_test_dtm, y_test, output_dir=Path(__file__).parent)

    # 7. Palabras más asociadas al spam
    print("\n" + "=" * 80)
    print("PALABRAS MÁS CARACTERÍSTICAS DEL SPAM")
    print("=" * 80)
    top_words = get_top_spam_words(model, vectorizer, top_n=5)
    print(top_words)

    # 8. Prueba con mensajes inventados
    custom_messages = [
        "Hi, are we still meeting tomorrow at the library?",
        "Congratulations! You have won a free vacation. Claim your prize now!",
        "Hello, we have a special offer for you if you reply today.",
    ]

    print("\n" + "=" * 80)
    print("CLASIFICACIÓN DE MENSAJES DE PRUEBA")
    print("=" * 80)
    custom_results = classify_custom_messages(custom_messages, model, vectorizer)
    print(custom_results)

    # 9. Pregunta final para el informe
    print("\n" + "=" * 80)
    print("PREGUNTA FINAL")
    print("=" * 80)
    print(
        "Explica qué ocurre si intentas clasificar una palabra como 'oferta' "
        "cuando no ha aparecido en el entrenamiento y cómo ayuda el "
        "suavizado de Laplace en ese caso."
    )


if __name__ == "__main__":
    main()
