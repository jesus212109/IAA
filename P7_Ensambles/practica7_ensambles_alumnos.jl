"""
PRÁCTICA 7: La Inteligencia Colectiva (Modelos de Ensambles)
Asignatura: Introducción al Aprendizaje Automático (IAA) 26/27

Versión en Julia del código base.

Notas importantes
-----------------
- Esta versión utiliza un CSV local con el dataset Breast Cancer Wisconsin.
- El fichero `breast_cancer_wisconsin.csv` debe estar en la misma carpeta que este script.
- El modelo de boosting usado aquí es AdaBoost, que encaja con la práctica
  porque en el guión se permite usar Gradient Boosting o AdaBoost.

Paquetes necesarios
-------------------
using Pkg
Pkg.add(["CSV", "DataFrames", "Plots", "DecisionTree"])

Objetivo
--------
El script ofrece una estructura guiada para que el alumnado implemente,
complete y analice la práctica. No es una solución cerrada: mantiene
comentarios y preguntas orientadas al informe final.
"""

using CSV
using DataFrames
using DecisionTree
using Plots
using Printf
using Random
using Statistics

const RANDOM_STATE = 42
const TEST_SIZE = 0.25
const DATA_PATH = joinpath(@__DIR__, "breast_cancer_wisconsin.csv")
const RF_PLOT_PATH = joinpath(@__DIR__, "random_forest_accuracy_julia.png")


# -----------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# -----------------------------------------------------------------------------
function load_dataset(path::String=DATA_PATH)
    """
    Carga el dataset desde un CSV local.

    Returns
    -------
    X : Matrix{Float64}
        Variables predictoras.
    y : Vector{String}
        Etiquetas de clase.
    feature_names : Vector{String}
        Nombres de las variables.
    """
    df = CSV.read(path, DataFrame)
    feature_names = String.(names(df)[1:end-1])
    X = Matrix{Float64}(df[:, feature_names])
    y = String.(df[:, :target])
    return X, y, feature_names
end


function stratified_split(
    X::Matrix{Float64},
    y::Vector{String};
    test_size::Float64=TEST_SIZE,
    random_state::Int=RANDOM_STATE,
)
    """
    Divide el dataset en entrenamiento y prueba preservando la proporción de clases.

    TODO para el alumnado:
    - Revisa por qué conviene mantener la proporción de clases.
    - Comprueba cuántas muestras de cada clase quedan en train y en test.
    """
    rng = MersenneTwister(random_state)
    train_idx = Int[]
    test_idx = Int[]

    for cls in unique(y)
        idx = findall(==(cls), y)
        shuffle!(rng, idx)

        n_test = round(Int, length(idx) * test_size)
        n_test = clamp(n_test, 1, length(idx) - 1)

        append!(test_idx, idx[1:n_test])
        append!(train_idx, idx[n_test+1:end])
    end

    shuffle!(rng, train_idx)
    shuffle!(rng, test_idx)

    return X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]
end


# -----------------------------------------------------------------------------
# 2. MÉTRICAS Y UTILIDADES DE EVALUACIÓN
# -----------------------------------------------------------------------------
function accuracy_score(y_true::Vector{String}, y_pred::Vector{String})
    return mean(y_true .== y_pred)
end


function confusion_matrix_manual(y_true::Vector{String}, y_pred::Vector{String})
    labels = sort(unique(vcat(y_true, y_pred)))
    label_to_idx = Dict(label => i for (i, label) in enumerate(labels))
    cm = zeros(Int, length(labels), length(labels))

    for (yt, yp) in zip(y_true, y_pred)
        cm[label_to_idx[yt], label_to_idx[yp]] += 1
    end

    return labels, cm
end


function classification_report_df(y_true::Vector{String}, y_pred::Vector{String})
    labels = sort(unique(vcat(y_true, y_pred)))
    rows = NamedTuple[]

    for label in labels
        tp = sum((y_true .== label) .& (y_pred .== label))
        fp = sum((y_true .!= label) .& (y_pred .== label))
        fn = sum((y_true .== label) .& (y_pred .!= label))
        support = sum(y_true .== label)

        precision = (tp + fp) == 0 ? 0.0 : tp / (tp + fp)
        recall = (tp + fn) == 0 ? 0.0 : tp / (tp + fn)
        f1 = (precision + recall) == 0 ? 0.0 : 2 * precision * recall / (precision + recall)

        push!(rows, (
            class=label,
            precision=precision,
            recall=recall,
            f1_score=f1,
            support=support,
        ))
    end

    return DataFrame(rows)
end


function evaluate_model(model, X_train, y_train, X_test, y_test; predictor=predict)
    """
    Evalúa un modelo en entrenamiento y en prueba.

    Returns
    -------
    results : NamedTuple
        Accuracy en entrenamiento y prueba.

    TODO para el alumnado:
    - Añade otras métricas si lo consideras útil.
    - Interpreta la diferencia entre train_accuracy y test_accuracy.
    """
    y_train_pred = String.(predictor(model, X_train))
    y_test_pred = String.(predictor(model, X_test))

    return (
        train_accuracy = accuracy_score(y_train, y_train_pred),
        test_accuracy = accuracy_score(y_test, y_test_pred),
        y_train_pred = y_train_pred,
        y_test_pred = y_test_pred,
    )
end


function print_confusion_and_report(model, X_test, y_test; predictor=predict)
    """
    Imprime matriz de confusión e informe de clasificación simplificado.

    TODO para el alumnado:
    - Explica en el informe qué tipo de error consideras más grave.
    - No copies la salida sin comentarla.
    """
    y_pred = String.(predictor(model, X_test))
    labels, cm = confusion_matrix_manual(y_test, y_pred)
    report = classification_report_df(y_test, y_pred)

    println("\nEtiquetas (filas = real, columnas = predicción): ", labels)
    println("Matriz de confusión:")
    println(cm)

    println("\nClassification report:")
    show(report, allrows=true, allcols=true)
    println()
end


# -----------------------------------------------------------------------------
# 3. TAREA 1: ÁRBOL SIMPLE (BASELINE)
# -----------------------------------------------------------------------------
function train_decision_tree(X_train, y_train)
    """
    Entrena un árbol de decisión sin limitar la profundidad.

    TODO para el alumnado:
    - Comprueba qué ocurre cuando no fijamos max_depth.
    - Relaciona el resultado con el concepto de sobreajuste.
    """
    model = DecisionTreeClassifier()
    DecisionTree.fit!(model, X_train, y_train)
    return model
end


# -----------------------------------------------------------------------------
# 4. TAREA 2: RANDOM FOREST
# -----------------------------------------------------------------------------
function train_random_forest(X_train, y_train; n_trees::Int=100)
    model = RandomForestClassifier(n_trees=n_trees)
    DecisionTree.fit!(model, X_train, y_train)
    return model
end


function random_forest_experiment(X_train, y_train, X_test, y_test, n_trees_list)
    """
    Ejecuta varios experimentos variando el número de árboles.

    TODO para el alumnado:
    - Observa si llega un punto de saturación.
    - Compara estos resultados con el árbol simple.
    """
    rows = NamedTuple[]

    for n_trees in n_trees_list
        start_ns = time_ns()
        model = train_random_forest(X_train, y_train; n_trees=n_trees)
        elapsed = (time_ns() - start_ns) / 1e9

        y_test_pred = String.(DecisionTree.predict(model, X_test))
        test_accuracy = accuracy_score(y_test, y_test_pred)

        push!(rows, (
            n_estimators=n_trees,
            test_accuracy=test_accuracy,
            train_time_seconds=elapsed,
        ))
    end

    return DataFrame(rows)
end


function plot_random_forest_results(results_df::DataFrame; save_path::String=RF_PLOT_PATH)
    """
    Representa la evolución del accuracy en test frente al número de árboles.

    TODO para el alumnado:
    - Cambia el gráfico para mostrar error en lugar de accuracy si lo prefieres.
    - Interpreta la forma de la curva en el informe.
    """
    plt = plot(
        results_df.n_estimators,
        results_df.test_accuracy,
        marker=:circle,
        linestyle=:dash,
        xlabel="n_estimators",
        ylabel="Accuracy en test",
        title="Random Forest: Accuracy en test vs número de árboles",
        legend=false,
        grid=true,
        size=(800, 500),
    )

    savefig(plt, save_path)
    display(plt)
    println("\nGráfica guardada en: $(save_path)")
end


# -----------------------------------------------------------------------------
# 5. TAREA 3: BOOSTING (ADABOOST)
# -----------------------------------------------------------------------------
function train_adaboost(X_train, y_train; n_iterations::Int=100)
    """
    Entrena un modelo AdaBoost de stumps y mide el tiempo de entrenamiento.

    Nota didáctica
    --------------
    En la versión Python del guión se proponía GradientBoostingClassifier o,
    alternativamente, AdaBoostClassifier. Aquí usamos AdaBoostStumpClassifier,
    que es una opción natural dentro de DecisionTree.jl.

    TODO para el alumnado:
    - Interpreta la diferencia entre entrenamiento secuencial (boosting)
      y entrenamiento más independiente de bagging.
    - Compara el comportamiento frente a Random Forest.
    """
    start_ns = time_ns()
    model = AdaBoostStumpClassifier(n_iterations=n_iterations)
    DecisionTree.fit!(model, X_train, y_train)
    elapsed = (time_ns() - start_ns) / 1e9
    return model, elapsed
end


# -----------------------------------------------------------------------------
# 6. IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------------------
function _importance_vector(model, n_features::Int)
    raw_importance = try
        DecisionTree.impurity_importance(model)
    catch
        DecisionTree.split_importance(model)
    end

    if raw_importance isa AbstractVector
        return Float64.(raw_importance)
    elseif raw_importance isa AbstractDict
        values = zeros(Float64, n_features)
        for (k, v) in raw_importance
            values[Int(k)] = Float64(v)
        end
        return values
    else
        error("No se pudo interpretar el formato de importancia de variables.")
    end
end


function show_top_features(model, feature_names::Vector{String}; top_k::Int=3)
    """
    Devuelve las variables más importantes según el modelo.

    TODO para el alumnado:
    - Comenta si estas variables tienen sentido en el problema.
    - Compara la importancia obtenida en Random Forest y en Boosting.
    """
    importances = _importance_vector(model, length(feature_names))
    df = DataFrame(feature=feature_names, importance=importances)
    sort!(df, :importance, rev=true)
    return first(df, min(top_k, nrow(df)))
end


# -----------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
function main()
    # -------------------------------------------------------------------------
    # PASO 1. CARGAR Y DIVIDIR LOS DATOS
    # -------------------------------------------------------------------------
    X, y, feature_names = load_dataset()
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    println(repeat("=", 80))
    println("PRÁCTICA 7 - MODELOS DE ENSAMBLES (JULIA)")
    println(repeat("=", 80))
    println("Número de muestras totales: $(size(X, 1))")
    println("Número de variables: $(size(X, 2))")
    println("Muestras de entrenamiento: $(size(X_train, 1))")
    println("Muestras de prueba: $(size(X_test, 1))")

    # -------------------------------------------------------------------------
    # PASO 2. ÁRBOL SIMPLE
    # -------------------------------------------------------------------------
    println("\n" * repeat("-", 80))
    println("TAREA 1 - ÁRBOL SIMPLE")
    println(repeat("-", 80))

    tree_start_ns = time_ns()
    tree_model = train_decision_tree(X_train, y_train)
    tree_train_time = (time_ns() - tree_start_ns) / 1e9
    tree_results = evaluate_model(tree_model, X_train, y_train, X_test, y_test; predictor=DecisionTree.predict)

    @printf("Tiempo de entrenamiento:   %.4f s\n", tree_train_time)
    @printf("Accuracy en entrenamiento: %.4f\n", tree_results.train_accuracy)
    @printf("Accuracy en test:          %.4f\n", tree_results.test_accuracy)

    print_confusion_and_report(tree_model, X_test, y_test; predictor=DecisionTree.predict)

    # TODO para el alumnado:
    # 1. ¿Hay señales de sobreajuste?
    # 2. ¿Por qué un árbol muy profundo puede memorizar el entrenamiento?

    # -------------------------------------------------------------------------
    # PASO 3. RANDOM FOREST
    # -------------------------------------------------------------------------
    println("\n" * repeat("-", 80))
    println("TAREA 2 - RANDOM FOREST")
    println(repeat("-", 80))

    n_trees_list = [1, 10, 50, 100]
    rf_curve_results = random_forest_experiment(X_train, y_train, X_test, y_test, n_trees_list)

    println("\nResultados Random Forest:")
    show(rf_curve_results, allrows=true, allcols=true)
    println()

    plot_random_forest_results(rf_curve_results)

    rf_start_ns = time_ns()
    rf_model = train_random_forest(X_train, y_train; n_trees=100)
    rf_train_time = (time_ns() - rf_start_ns) / 1e9
    rf_results = evaluate_model(rf_model, X_train, y_train, X_test, y_test; predictor=DecisionTree.predict)

    println("\nEvaluación Random Forest (100 árboles):")
    @printf("Tiempo de entrenamiento:   %.4f s\n", rf_train_time)
    @printf("Accuracy en entrenamiento: %.4f\n", rf_results.train_accuracy)
    @printf("Accuracy en test:          %.4f\n", rf_results.test_accuracy)

    print_confusion_and_report(rf_model, X_test, y_test; predictor=DecisionTree.predict)

    println("\nTop 3 variables más importantes en Random Forest:")
    show(show_top_features(rf_model, feature_names; top_k=3), allrows=true, allcols=true)
    println()

    # TODO para el alumnado:
    # 1. ¿Mejora Random Forest frente al árbol simple?
    # 2. ¿Se estabiliza el resultado al aumentar el número de árboles?
    # 3. ¿Por qué el bagging reduce la varianza del modelo?

    # -------------------------------------------------------------------------
    # PASO 4. BOOSTING
    # -------------------------------------------------------------------------
    println("\n" * repeat("-", 80))
    println("TAREA 3 - BOOSTING (ADABOOST)")
    println(repeat("-", 80))

    boost_model, boost_train_time = train_adaboost(X_train, y_train; n_iterations=100)
    boost_results = evaluate_model(boost_model, X_train, y_train, X_test, y_test; predictor=DecisionTree.predict)

    @printf("Tiempo de entrenamiento:   %.4f s\n", boost_train_time)
    @printf("Accuracy en entrenamiento: %.4f\n", boost_results.train_accuracy)
    @printf("Accuracy en test:          %.4f\n", boost_results.test_accuracy)

    print_confusion_and_report(boost_model, X_test, y_test; predictor=DecisionTree.predict)

    println("\nTop 3 variables más importantes en AdaBoost:")
    show(show_top_features(boost_model, feature_names; top_k=3), allrows=true, allcols=true)
    println()

    # TODO para el alumnado:
    # 1. ¿Qué diferencias observas entre Random Forest y Boosting?
    # 2. ¿Cuál parece entrenar más rápido en tu ejecución?
    # 3. ¿Cuál elegirías si priorizas rendimiento? ¿Y si priorizas simplicidad?

    # -------------------------------------------------------------------------
    # PASO 5. TABLA COMPARATIVA FINAL
    # -------------------------------------------------------------------------
    println("\n" * repeat("-", 80))
    println("TABLA COMPARATIVA FINAL")
    println(repeat("-", 80))

    comparison_df = DataFrame(
        Modelo=["Árbol simple", "Random Forest", "AdaBoost"],
        Accuracy_train=[tree_results.train_accuracy, rf_results.train_accuracy, boost_results.train_accuracy],
        Accuracy_test=[tree_results.test_accuracy, rf_results.test_accuracy, boost_results.test_accuracy],
        Tiempo_entrenamiento_s=[tree_train_time, rf_train_time, boost_train_time],
        Ventajas=["TODO", "TODO", "TODO"],
        Desventajas=["TODO", "TODO", "TODO"],
    )

    show(comparison_df, allrows=true, allcols=true)
    println()

    # TODO para el alumnado:
    # Completa esta reflexión final en el informe:
    # "Si tuvieras que desplegar un modelo en una aplicación móvil con poca
    # memoria y CPU, ¿cuál elegirías y por qué?"
end


main()
