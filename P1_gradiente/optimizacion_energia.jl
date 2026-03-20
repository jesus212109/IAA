
using Plots, Statistics, Random

# =============================
# Práctica 1 — Script de partida
# =============================
# Objetivo: estudiar la dinámica de convergencia del gradiente descendente (mini-batches)
# y cómo influyen los hiperparámetros (α, batch_size) y un criterio de parada.

# -----------------------------
# 1) Generación de datos (consumo energético vs carga de CPU)
# -----------------------------
function make_dataset(; n::Int=500, seed::Int=42)
    Random.seed!(seed)
    # Carga CPU en [0, 100] (%)
    X = 100 .* rand(n, 1)

    # Modelo lineal "real" + ruido (simula consumo energético en W)
    y = 50 .+ 1.2 .* X .+ randn(n, 1) .* 5
    return X, y
end

# Normalización z-score (solo para X; el término de sesgo va aparte)
function normalize_zscore(X)
    return (X .- mean(X)) ./ std(X)
end

# Añade columna de 1s para el sesgo (bias)
function add_bias(X_norm)
    n = size(X_norm, 1)
    return hcat(ones(n, 1), X_norm)
end

# -----------------------------
# 2) Gradiente descendente por mini-batches
# -----------------------------
# Ejecuta GD por mini-batches para regresión lineal (MSE).
#
# Devuelve:
#   θ            : pesos aprendidos (incluye bias)
#   history_loss : coste por época
#   epochs_run   : épocas realmente ejecutadas (puede ser < n_epochs por early stop)
function minibatch_gd(X_b, y; α=0.5, n_epochs=100, batch_size=32, seed=42, tol=nothing)
    m, n = size(X_b)
    Random.seed!(seed)

    θ = randn(n, 1)
    history_loss = Float64[]
    prev_loss = Inf

    for epoch in 1:n_epochs
        indices = shuffle(1:m)
        X_shuffled = X_b[indices, :]
        y_shuffled = y[indices, :]

        for i in 1:batch_size:m
            idx_end = min(i + batch_size - 1, m)
            xi = X_shuffled[i:idx_end, :]
            yi = y_shuffled[i:idx_end, :]

            # ∇J(θ) para MSE: (2/b) Xᵀ(Xθ - y)
            ∇ = (2 / size(xi, 1)) .* xi' * (xi * θ .- yi)
            θ .-= α .* ∇
        end

        loss = mean((X_b * θ .- y).^2)
        push!(history_loss, loss)

        # Criterio de parada (Reto)
        if tol !== nothing
            if abs(loss - prev_loss) < tol
                return θ, history_loss, epoch
            end
            prev_loss = loss
        end
    end

    return θ, history_loss, n_epochs
end

# -----------------------------
# 3) Ejecución por defecto (puedes cambiar α y batch_size aquí)
# -----------------------------
X, y = make_dataset()
X_norm = normalize_zscore(X)
X_b = add_bias(X_norm)

α = 0.5
n_epochs = 100
batch_size = 32
tol = nothing  # pon, por ejemplo, 1e-5 para activar el criterio de parada

θ, history_loss, epochs_run = minibatch_gd(X_b, y; α=α, n_epochs=n_epochs, batch_size=batch_size, tol=tol)

# -----------------------------
# 4) Visualización
# -----------------------------
p = plot(history_loss,
    title="Convergencia en Julia (α=$α, batch=$batch_size)",
    xlabel="Épocas", ylabel="Coste J(θ)", lw=2, legend=false)

display(p)

# Guarda la figura (útil para el Cuaderno de Laboratorio)
mkpath("outputs")
savefig(p, joinpath("outputs", "loss_alpha_$(α)_batch_$(batch_size).png"))
println("Figura guardada en outputs/")
println("Épocas ejecutadas: ", epochs_run)
