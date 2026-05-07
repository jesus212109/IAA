# Utilidades para la Práctica 8: Active Learning.
# Este fichero contiene funciones auxiliares ya implementadas para que el alumnado
# pueda centrarse en el ciclo de consulta y la comparación de estrategias.

using CSV
using DataFrames
using DecisionTree
using Plots
using Random
using Statistics

const RANDOM_STATE = 42
const N_TREES = 100
const DATA_DIR = joinpath(@__DIR__, "..", "data")

"""
    load_data()

Carga las particiones predefinidas de la práctica.

Devuelve:
- `X_initial`, `y_initial`: los 10 puntos inicialmente etiquetados.
- `X_unlabeled`, `y_unlabeled`: pool no etiquetado y etiquetas reales.
  `y_unlabeled` actúa como oráculo: solo debe consultarse para los puntos seleccionados.
- `X_test`, `y_test`: conjunto de test separado.
"""
function load_data()
    initial = CSV.read(joinpath(DATA_DIR, "initial_labeled.csv"), DataFrame)
    pool = CSV.read(joinpath(DATA_DIR, "unlabeled_pool.csv"), DataFrame)
    test = CSV.read(joinpath(DATA_DIR, "test.csv"), DataFrame)

    X_initial = Matrix(initial[:, [:x1, :x2]])
    y_initial = Vector{Int}(initial.y)

    X_unlabeled = Matrix(pool[:, [:x1, :x2]])
    y_unlabeled = Vector{Int}(pool.y)

    X_test = Matrix(test[:, [:x1, :x2]])
    y_test = Vector{Int}(test.y)

    return X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test
end

"""
    train_model(X_train, y_train)

Entrena el clasificador base de la práctica.
"""
function train_model(X_train, y_train)
    n_subfeatures = 2
    partial_sampling = 0.7
    max_depth = -1
    min_samples_leaf = 1
    min_samples_split = 2
    min_purity_increase = 0.0

    model = build_forest(
        y_train,
        X_train,
        n_subfeatures,
        N_TREES,
        partial_sampling,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        min_purity_increase;
        rng=Random.MersenneTwister(RANDOM_STATE),
    )
    return model
end

"""
    accuracy(model, X_test, y_test)

Calcula el accuracy de un modelo sobre el conjunto de test.
"""
function accuracy(model, X_test, y_test)
    y_pred = apply_forest(model, X_test)
    return mean(y_pred .== y_test)
end

"""
    add_queried_points(X_train, y_train, X_unlabeled, y_unlabeled, query_idx)

Mueve los puntos consultados desde el pool al conjunto de entrenamiento.
"""
function add_queried_points(X_train, y_train, X_unlabeled, y_unlabeled, query_idx)
    X_query = X_unlabeled[query_idx, :]
    y_query = y_unlabeled[query_idx]

    X_train_new = vcat(X_train, X_query)
    y_train_new = vcat(y_train, y_query)

    mask = trues(size(X_unlabeled, 1))
    mask[query_idx] .= false

    X_unlabeled_new = X_unlabeled[mask, :]
    y_unlabeled_new = y_unlabeled[mask]

    return X_train_new, y_train_new, X_unlabeled_new, y_unlabeled_new
end

"""
    plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

Representa la curva de aprendizaje de ambas estrategias.
"""
function plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc; output_path="learning_curve_julia.png")
    plt = plot(
        random_labels,
        random_acc,
        marker=:circle,
        label="Selección aleatoria",
        xlabel="Número de etiquetas utilizadas",
        ylabel="Accuracy en test",
        title="Curva de aprendizaje: Random vs Active Learning",
        legend=:bottomright,
    )
    plot!(plt, uncertainty_labels, uncertainty_acc, marker=:square, label="Selección por incertidumbre")
    savefig(plt, output_path)
    display(plt)
end
