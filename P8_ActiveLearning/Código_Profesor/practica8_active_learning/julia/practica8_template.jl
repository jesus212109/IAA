#=
Práctica 8: Active Learning (Aprendizaje Activo)
Versión recortada para el alumnado.

El dataset ya está preparado en la carpeta ../data.
No tienes que generar los datos: debes completar el entrenamiento inicial,
la selección aleatoria, la selección por incertidumbre y la curva comparativa.
=#

include("utils.jl")

const BATCH_SIZE = 5
const MAX_LABELS = 50

"""
    select_random(X_unlabeled, batch_size, rng)

Selecciona aleatoriamente `batch_size` índices del pool no etiquetado.

Debes devolver posiciones dentro de `X_unlabeled`, no los puntos completos.

Pistas:
- Puedes usar `randperm(rng, n)`.
- No selecciones el mismo índice dos veces en la misma iteración.
"""
function select_random(X_unlabeled, batch_size, rng)
    # TODO: completa esta función.
    error("Completa select_random")
end

"""
    select_by_uncertainty(model, X_unlabeled, batch_size)

Selecciona los puntos donde el modelo tiene más incertidumbre.

En clasificación binaria, la incertidumbre es mayor cuando la probabilidad
predicha para la clase 1 está cerca de 0.5.

Pistas:
- Usa `apply_forest_proba(model, X_unlabeled, [0, 1])`.
- Extrae la probabilidad de la clase 1.
- Calcula `abs.(probs .- 0.5)`.
- Ordena de menor a mayor: los valores más pequeños son los más inciertos.
- Devuelve los `batch_size` índices más inciertos.
"""
function select_by_uncertainty(model, X_unlabeled, batch_size)
    # TODO: completa esta función.
    error("Completa select_by_uncertainty")
end

"""
    run_query_strategy(strategy)

Ejecuta el ciclo de consulta para una estrategia.

`strategy` debe ser `"random"` o `"uncertainty"`.
"""
function run_query_strategy(strategy::String)
    rng = Random.MersenneTwister(RANDOM_STATE)

    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history = Int[]
    accuracy_history = Float64[]

    while length(y_train) <= MAX_LABELS
        # TODO 1: entrena el modelo con X_train, y_train.

        # TODO 2: evalúa el modelo en X_test, y_test y guarda el accuracy.

        # TODO 3: guarda también el número actual de etiquetas.

        # TODO 4: si ya has llegado a MAX_LABELS, termina el bucle.

        # TODO 5: selecciona los puntos a consultar según la estrategia.
        # - Si strategy == "random", usa select_random.
        # - Si strategy == "uncertainty", usa select_by_uncertainty.
        # - Si strategy tiene otro valor, lanza un error.

        # TODO 6: consulta el oráculo y actualiza train/pool usando add_queried_points.

        error("Completa el bucle de aprendizaje activo")
    end

    return n_labels_history, accuracy_history
end

function main()
    # Entrenamiento inicial orientativo: puedes usar esta parte para comprobar
    # el rendimiento con solo 10 etiquetas antes de completar los bucles.
    X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    initial_model = train_model(X_initial, y_initial)
    initial_acc = accuracy(initial_model, X_test, y_test)
    println("Accuracy inicial con 10 etiquetas: $(round(initial_acc, digits=4))")

    # TODO 7: ejecuta la estrategia aleatoria.
    # random_labels, random_acc = run_query_strategy("random")

    # TODO 8: ejecuta la estrategia por incertidumbre.
    # uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    # TODO 9: representa ambas curvas de aprendizaje.
    # plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

    # TODO 10: imprime o comenta los resultados finales.
end

main()
