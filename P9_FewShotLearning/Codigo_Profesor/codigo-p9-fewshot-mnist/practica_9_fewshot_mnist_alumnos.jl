# Práctica 9 - Few-shot Learning con MNIST
# Código base para alumnos (Julia)
#
# Objetivo:
# 1. Entrenar una CNN sin ver el dígito 7.
# 2. Reutilizar la CNN como extractor de características.
# 3. Crear prototipos a partir de pocos ejemplos.
# 4. Clasificar imágenes por distancia al prototipo más cercano.
# 5. Comparar 1-shot frente a 5-shot.
#
# Paquetes necesarios:
# ] add Flux MLDatasets Statistics Random LinearAlgebra StatsBase OneHotArrays MultivariateStats Plots

using Flux
using MLDatasets
using Statistics
using Random
using LinearAlgebra
using StatsBase
using OneHotArrays
using MultivariateStats
using Plots

const SEED = 42
Random.seed!(SEED)

const NOVEL_CLASS = 7
const KNOWN_CLASSES = [0, 1, 2, 3, 4, 5, 6, 8, 9]
const COMPACT_CLASSES = collect(1:length(KNOWN_CLASSES))

"""
Carga MNIST y separa el mundo conocido del dígito nuevo.

El modelo se entrenará sin ver ningún 7. Los 7 del conjunto de entrenamiento
se reservan para simular la aparición de una clase nueva con pocos ejemplos.
"""
function load_mnist_world_without_sevens()
    train_x, train_y = MNIST(:train)[:]
    test_x, test_y = MNIST(:test)[:]

    # MLDatasets devuelve imágenes como 28 x 28 x N.
    # Flux espera tensores para convolución como W x H x C x N.
    X_train = Float32.(reshape(train_x, 28, 28, 1, :)) ./ 255f0
    X_test = Float32.(reshape(test_x, 28, 28, 1, :)) ./ 255f0
    y_train = Int.(train_y)
    y_test = Int.(test_y)

    known_mask = y_train .!= NOVEL_CLASS
    seven_mask = y_train .== NOVEL_CLASS

    X_train_known = X_train[:, :, :, known_mask]
    y_train_known = y_train[known_mask]
    X_train_seven = X_train[:, :, :, seven_mask]
    y_train_seven = y_train[seven_mask]

    return X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test
end

"""
Remapea las etiquetas conocidas de MNIST a 1..9 para entrenar la CNN.

Ejemplo: las clases originales son [0,1,2,3,4,5,6,8,9], pero Flux trabaja
más cómodamente con índices compactos [1,2,3,4,5,6,7,8,9].
"""
function remap_known_labels(y)
    mapping = Dict(label => i for (i, label) in enumerate(KNOWN_CLASSES))
    return [mapping[label] for label in y]
end

"""
Construye una CNN sencilla con una capa de embedding explícita.

La penúltima capa Dense(64) se usará después como extractor de características.
"""
function build_classifier(num_classes::Int)
    model = Chain(
        Conv((3, 3), 1 => 32, relu),
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu),
        MaxPool((2, 2)),
        Flux.flatten,
        Dense(1600 => 64, relu),      # embedding
        Dense(64 => num_classes)      # logits
    )
    return model
end

"""
Entrena la CNN solo con las clases conocidas.
"""
function train_classifier!(model, X_train_known, y_train_known; epochs=3, batchsize=128, lr=1f-3)
    y_compact = remap_known_labels(y_train_known)
    y_onehot = onehotbatch(y_compact, COMPACT_CLASSES)

    data = Flux.DataLoader((X_train_known, y_onehot), batchsize=batchsize, shuffle=true)
    opt_state = Flux.setup(Adam(lr), model)

    for epoch in 1:epochs
        total_loss = 0f0
        nbatches = 0

        for (xb, yb) in data
            loss, grads = Flux.withgradient(model) do m
                logits = m(xb)
                Flux.logitcrossentropy(logits, yb)
            end
            Flux.update!(opt_state, model, grads[1])
            total_loss += Float32(loss)
            nbatches += 1
        end

        println("Epoch $(epoch) - loss = ", round(total_loss / nbatches, digits=4))
    end
end

"""
Evalúa el clasificador inicial solo sobre las clases conocidas del test.
"""
function evaluate_known_classifier(model, X_test, y_test)
    known_mask = y_test .!= NOVEL_CLASS
    X = X_test[:, :, :, known_mask]
    y_original = y_test[known_mask]
    y_compact = remap_known_labels(y_original)

    logits = model(X)
    pred_compact = onecold(logits, COMPACT_CLASSES)
    return mean(pred_compact .== y_compact)
end

"""
Devuelve un extractor de características eliminando la última capa del clasificador.
"""
function create_feature_extractor(model)
    # TODO: devuelve una Chain sin la última capa Dense de clasificación.
    # Pista: Chain(model.layers[1:end-1]...)
    return nothing
end

"""
Selecciona n ejemplos de una clase concreta.
"""
function sample_class_examples(X, y, class_label::Int, n::Int)
    idx = findall(==(class_label), y)
    selected = sample(idx, n; replace=false)
    return X[:, :, :, selected], y[selected]
end

"""
Construye un episodio N-way K-shot usando los dígitos 0..9.

Support set: pocos ejemplos por clase.
Query set: ejemplos de test para evaluar.
"""
function build_fewshot_episode(X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test; n_shots=5, n_query_per_class=100)
    support_x_parts = Vector{Array{Float32, 4}}()
    support_y_parts = Vector{Vector{Int}}()
    query_x_parts = Vector{Array{Float32, 4}}()
    query_y_parts = Vector{Vector{Int}}()

    for class_label in 0:9
        if class_label == NOVEL_CLASS
            Xs, ys = sample_class_examples(X_train_seven, y_train_seven, class_label, n_shots)
        else
            Xs, ys = sample_class_examples(X_train_known, y_train_known, class_label, n_shots)
        end

        Xq, yq = sample_class_examples(X_test, y_test, class_label, n_query_per_class)

        push!(support_x_parts, Xs)
        push!(support_y_parts, ys)
        push!(query_x_parts, Xq)
        push!(query_y_parts, yq)
    end

    X_support = cat(support_x_parts...; dims=4)
    y_support = vcat(support_y_parts...)
    X_query = cat(query_x_parts...; dims=4)
    y_query = vcat(query_y_parts...)

    return X_support, y_support, X_query, y_query
end

"""
Calcula los prototipos de cada clase como la media de sus embeddings.
"""
function compute_prototypes(feature_extractor, X_support, y_support)
    # TODO:
    # 1. Obtén los embeddings del support set con feature_extractor(X_support).
    # 2. Para cada clase, calcula la media de sus embeddings.
    # 3. Devuelve una matriz de prototipos y las etiquetas asociadas.
    return nothing, nothing
end

"""
Clasifica las imágenes de consulta por distancia euclídea al prototipo más cercano.
"""
function classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
    # TODO:
    # 1. Obtén los embeddings de X_query.
    # 2. Calcula la distancia euclídea de cada embedding a cada prototipo.
    # 3. Asigna a cada imagen la etiqueta del prototipo más cercano.
    return nothing
end

"""
Ejecuta un episodio few-shot completo.
"""
function run_episode(feature_extractor, X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test; n_shots=5)
    X_support, y_support, X_query, y_query = build_fewshot_episode(
        X_train_known, y_train_known,
        X_train_seven, y_train_seven,
        X_test, y_test;
        n_shots=n_shots,
        n_query_per_class=100,
    )

    prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
    y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
    acc = mean(y_pred .== y_query)

    return acc, y_pred, y_query, X_query, prototypes, prototype_labels
end

function main()
    X_train_known, y_train_known, X_train_seven, y_train_seven, X_test, y_test = load_mnist_world_without_sevens()

    println("Clases conocidas: ", KNOWN_CLASSES)
    println("Clase nueva: ", NOVEL_CLASS)
    println("Ejemplos de entrenamiento sin sietes: ", size(X_train_known, 4))
    println("Sietes reservados para few-shot: ", size(X_train_seven, 4))

    model = build_classifier(length(KNOWN_CLASSES))
    train_classifier!(model, X_train_known, y_train_known; epochs=3)

    known_acc = evaluate_known_classifier(model, X_test, y_test)
    println("Accuracy inicial en clases conocidas: ", round(known_acc, digits=4))

    feature_extractor = create_feature_extractor(model)

    # TODO: Ejecuta y compara los episodios 1-shot y 5-shot.
    # Pista:
    # acc_1shot, _, _, _, _, _ = run_episode(...; n_shots=1)
    # acc_5shot, _, _, _, _, _ = run_episode(...; n_shots=5)

    println("Completa los TODO para obtener los resultados finales.")
end

main()
