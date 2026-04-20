"""
Práctica 6: Clasificación de Texto y Filtros Anti-Spam con Naïve Bayes en Julia.
Versión base para alumnos.

Objetivo:
    Completar un clasificador de texto basado en Naïve Bayes para distinguir
    entre mensajes legítimos (ham) y spam.

Instrucciones:
    1. Lee el código completo antes de empezar.
    2. Completa las zonas marcadas con TODO.
    3. Ejecuta el script y revisa los resultados.
    4. Responde en tu informe a las preguntas planteadas en los comentarios.

Dependencias de Julia:
    using Pkg
    Pkg.add(["CSV", "DataFrames"])

Ejecución:
    julia practica_6_naive_bayes_spam_base_alumno.jl
"""

using CSV
using DataFrames
using Printf
using Random
using SparseArrays
using Statistics
using LinearAlgebra

struct MultinomialNBModel
    alpha::Float64
    class_log_prior::Vector{Float64}
    feature_log_prob::Matrix{Float64}
    vocabulary::Dict{String, Int}
    index_to_word::Vector{String}
end

# ============================================================================
# PARTE 1. CARGA Y LIMPIEZA DEL DATASET
# ============================================================================

"""
    clean_text(text)

Limpia un texto de entrada.

TODO:
- pasar el texto a minúsculas;
- eliminar signos de puntuación y símbolos extraños;
- eliminar espacios múltiples.
"""
function clean_text(text)::String
    text = String(text)

    # TODO: pasar a minúsculas
    # text = ...

    # TODO: reemplazar caracteres no alfanuméricos por espacios
    # text = ...

    # TODO: colapsar espacios múltiples y hacer strip
    # text = ...

    return text
end

"""
    tokenize(text)

Divide un texto en palabras.
"""
function tokenize(text)::Vector{String}
    cleaned = clean_text(text)

    # TODO:
    # Si cleaned está vacío, devolver String[]
    # En caso contrario, dividir con split(cleaned)
    return String[]
end

"""
    load_dataset(csv_path)

Carga el CSV y crea la columna label_num:
- ham  -> 0
- spam -> 1
"""
function load_dataset(csv_path::AbstractString)::DataFrame
    if !isfile(csv_path)
        error("No se encontró el fichero: $(abspath(csv_path))")
    end

    df = CSV.read(csv_path, DataFrame)

    # TODO:
    # Comprobar que existen las columnas "Category" y "Message"

    df = select(df, ["Category", "Message"])
    df.Category = lowercase.(strip.(string.(df.Category)))
    df.Message = string.(df.Message)

    # TODO:
    # Crear la columna label_num con 0 para ham y 1 para spam
    # Ejemplo orientativo:
    # df.label_num = [...]

    return df
end

# ============================================================================
# PARTE 2. EXPLORACIÓN INICIAL
# ============================================================================

function show_basic_dataset_info(df::DataFrame)
    println(repeat("=", 80))
    println("INFORMACIÓN BÁSICA DEL DATASET")
    println(repeat("=", 80))
    println("Número total de mensajes: ", nrow(df))
    println("\nDistribución de clases:")
    println(combine(groupby(df, :Category), nrow => :count))
    println("\nPrimeros 5 ejemplos:")
    show(first(df, 5), allrows=true, allcols=true)
    println("\n")

    # Pregunta para el informe:
    # ¿El dataset está balanceado o hay muchos más ham que spam?
end

# ============================================================================
# PARTE 3. TRAIN / TEST
# ============================================================================

"""
    stratified_train_test_split(messages, labels; test_size=0.25, seed=42)

TODO:
- separar índices de ham y spam;
- barajarlos con Random.seed! y shuffle!;
- reservar una parte para test;
- devolver X_train, X_test, y_train, y_test.
"""
function stratified_train_test_split(
    messages::Vector{String},
    labels::Vector{Int};
    test_size::Float64=0.25,
    seed::Int=42
)
    Random.seed!(seed)

    # TODO:
    ham_idx = Int[]
    spam_idx = Int[]

    # TODO:
    # shuffle!(...)

    # TODO:
    n_test_ham = 0
    n_test_spam = 0

    # TODO:
    test_idx = Int[]
    train_idx = Int[]

    # TODO:
    X_train = String[]
    X_test = String[]
    y_train = Int[]
    y_test = Int[]

    return X_train, X_test, y_train, y_test
end

# ============================================================================
# PARTE 4. VOCABULARIO Y BAG OF WORDS
# ============================================================================

"""
    build_vocabulary(messages)

Construye el vocabulario a partir del conjunto de entrenamiento.
"""
function build_vocabulary(messages::Vector{String})
    counts = Dict{String, Int}()

    # TODO:
    # Recorrer los mensajes, tokenizar y contar palabras

    # TODO:
    words = String[]
    vocabulary = Dict{String, Int}()

    return vocabulary, words
end

"""
    vectorize_messages(messages, vocabulary)

Convierte mensajes en una matriz dispersa Bag of Words.
"""
function vectorize_messages(messages::Vector{String}, vocabulary::Dict{String, Int})
    row_idx = Int[]
    col_idx = Int[]
    values = Int[]

    # TODO:
    # Para cada mensaje:
    #   - contar las palabras que estén en el vocabulario;
    #   - guardar fila, columna y frecuencia;
    #   - crear al final la sparse matrix.

    return sparse(row_idx, col_idx, values, length(messages), length(vocabulary))
end

# ============================================================================
# PARTE 5. MODELO NAÏVE BAYES
# ============================================================================

"""
    train_multinomial_nb(X_train, y_train, vocabulary, index_to_word; alpha=1.0)

Entrena un Multinomial Naïve Bayes con suavizado de Laplace.

Pregunta para el informe:
¿Por qué es importante sumar alpha = 1.0 a los recuentos?
"""
function train_multinomial_nb(
    X_train::SparseMatrixCSC{Int, Int},
    y_train::Vector{Int},
    vocabulary::Dict{String, Int},
    index_to_word::Vector{String};
    alpha::Float64=1.0
)
    n_docs = size(X_train, 1)
    vocab_size = size(X_train, 2)

    # TODO:
    ham_rows = Int[]
    spam_rows = Int[]

    # TODO:
    ham_counts = zeros(Int, vocab_size)
    spam_counts = zeros(Int, vocab_size)

    # TODO:
    total_ham_tokens = 0
    total_spam_tokens = 0

    # TODO:
    p_ham = 0.0
    p_spam = 0.0
    class_log_prior = zeros(2)

    # TODO:
    prob_word_given_ham = zeros(vocab_size)
    prob_word_given_spam = zeros(vocab_size)

    # TODO:
    feature_log_prob = zeros(2, vocab_size)

    return MultinomialNBModel(
        alpha,
        class_log_prior,
        feature_log_prob,
        vocabulary,
        index_to_word
    )
end

# ============================================================================
# PARTE 6. PREDICCIÓN
# ============================================================================

function predict_log_proba(model::MultinomialNBModel, X::SparseMatrixCSC{Int, Int})
    scores = Matrix{Float64}(undef, size(X, 1), 2)

    # TODO:
    # Para cada documento y para cada clase:
    # score = log prior + suma(frecuencia * log probabilidad)
    return scores
end

function softmax_row(log_scores::AbstractVector{<:Real})
    # TODO:
    # Implementar softmax estable numéricamente
    return zeros(Float64, length(log_scores))
end

function predict_proba(model::MultinomialNBModel, X::SparseMatrixCSC{Int, Int})
    log_scores = predict_log_proba(model, X)
    probs = Matrix{Float64}(undef, size(log_scores, 1), 2)

    # TODO:
    # Aplicar softmax_row a cada fila
    return probs
end

function predict(model::MultinomialNBModel, X::SparseMatrixCSC{Int, Int})
    probs = predict_proba(model, X)

    # TODO:
    # Devolver 0 para ham y 1 para spam según la probabilidad mayor
    return Int[]
end

# ============================================================================
# PARTE 7. EVALUACIÓN
# ============================================================================

function confusion_matrix_binary(y_true::Vector{Int}, y_pred::Vector{Int})
    # TODO:
    # Calcular TN, FP, FN y TP
    return [0 0; 0 0]
end

safe_div(num, den) = den == 0 ? 0.0 : num / den

function print_classification_report(y_true::Vector{Int}, y_pred::Vector{Int})
    cm = confusion_matrix_binary(y_true, y_pred)

    # TODO:
    # Extraer tn, fp, fn, tp
    # Calcular precision, recall y F1 para ham y spam
    # Calcular accuracy
    # Imprimirlo con formato legible

    return cm
end

function save_confusion_matrix_csv(cm::Matrix{Int}, path::AbstractString)
    df_cm = DataFrame(
        real_ham = [cm[1, 1], cm[2, 1]],
        real_spam = [cm[1, 2], cm[2, 2]]
    )
    df_cm.predicted = ["ham", "spam"]
    select!(df_cm, [:predicted, :real_ham, :real_spam])
    CSV.write(path, df_cm)
end

# ============================================================================
# PARTE 8. INTERPRETACIÓN DEL MODELO
# ============================================================================

function get_top_spam_words(model::MultinomialNBModel; top_n::Int=5)
    # TODO:
    # Comparar logP(word|spam) y logP(word|ham)
    # Ordenar por la diferencia
    return DataFrame(
        word = String[],
        logP_word_spam = Float64[],
        logP_word_ham = Float64[],
        spam_minus_ham = Float64[]
    )
end

function classify_custom_messages(messages::Vector{String}, model::MultinomialNBModel)
    # TODO:
    # Vectorizar mensajes, obtener probabilidades y etiquetas
    return DataFrame(
        message = messages,
        predicted_label = String[],
        P_ham = Float64[],
        P_spam = Float64[]
    )
end

# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

function main()
    csv_path = joinpath(@__DIR__, "spam.csv")

    # 1. Carga de datos
    df = load_dataset(csv_path)
    show_basic_dataset_info(df)

    messages = Vector{String}(df.Message)
    labels = Vector{Int}(df.label_num)

    # 2. Train / test
    X_train, X_test, y_train, y_test = stratified_train_test_split(messages, labels)

    # 3. Vocabulario y vectorización
    vocabulary, index_to_word = build_vocabulary(X_train)
    X_train_bow = vectorize_messages(X_train, vocabulary)
    X_test_bow = vectorize_messages(X_test, vocabulary)

    println(repeat("=", 80))
    println("INFORMACIÓN DEL VOCABULARIO")
    println(repeat("=", 80))

    # TODO:
    # Mostrar tamaño del vocabulario
    println()

    # 4. Entrenamiento
    model = train_multinomial_nb(
        X_train_bow,
        y_train,
        vocabulary,
        index_to_word;
        alpha=1.0
    )

    # 5. Probabilidades a priori
    println(repeat("=", 80))
    println("PROBABILIDADES A PRIORI")
    println(repeat("=", 80))

    # TODO:
    # Convertir model.class_log_prior a probabilidades con exp.(...)
    # e imprimir P(ham) y P(spam)
    println()

    # 6. Evaluación
    y_pred = predict(model, X_test_bow)
    cm = print_classification_report(y_test, y_pred)

    cm_path = joinpath(@__DIR__, "matriz_confusion_spam_julia_base.csv")
    save_confusion_matrix_csv(cm, cm_path)
    println("Matriz de confusión guardada en: ", cm_path)
    println()

    # 7. Palabras más asociadas a spam
    println(repeat("=", 80))
    println("PALABRAS MÁS CARACTERÍSTICAS DEL SPAM")
    println(repeat("=", 80))
    top_spam_words = get_top_spam_words(model, top_n=5)
    show(top_spam_words, allrows=true, allcols=true)
    println("\n")

    # 8. Mensajes de prueba
    custom_messages = [
        "Hi, are we still meeting tomorrow at the library?",
        "Congratulations! You have won a free vacation. Claim your prize now!",
        "Hello, we have a special offer for you if you reply today."
    ]

    println(repeat("=", 80))
    println("CLASIFICACIÓN DE MENSAJES DE PRUEBA")
    println(repeat("=", 80))
    custom_results = classify_custom_messages(custom_messages, model)
    show(custom_results, allrows=true, allcols=true)
    println()

    println(repeat("=", 80))
    println("PREGUNTA FINAL")
    println(repeat("=", 80))
    println(
        "Explica qué ocurre si intentas clasificar una palabra como 'oferta' " *
        "cuando no ha aparecido en el entrenamiento y cómo ayuda el " *
        "suavizado de Laplace en ese caso."
    )
end

main()
