
using Plots
include("optimizacion_energia.jl")  # reutiliza funciones y dataset

# ======================================================
# Runner opcional para generar las figuras de la Práctica 1
# ======================================================
# Edita los valores de α para provocar los cuatro escenarios solicitados:
# - αslow: curva casi plana descendente
# - αopt : "curva de codo" (baja rápido y se estabiliza)
# - αosc : oscila antes de estabilizar
# - αfail: diverge (crece el error)

mkpath("outputs")

# 1) Dataset base
X, y = make_dataset()
X_norm = normalize_zscore(X)
X_b = add_bias(X_norm)

# 2) Experimentos α (batch fijo 32)
batch_fixed = 32
n_epochs = 120

alphas = Dict(
    "A_alpha_slow" => 0.01,
    "B_alpha_opt"  => 0.2,
    "C_alpha_osc"  => 0.8,
    "D_alpha_fail" => 2.0
)

for (tag, α) in alphas
    θ, history_loss, epochs_run = minibatch_gd(X_b, y; α=α, n_epochs=n_epochs, batch_size=batch_fixed, tol=nothing)
    p = plot(history_loss,
        title="Convergencia ($tag, α=$α, batch=$batch_fixed)",
        xlabel="Épocas", ylabel="Coste J(θ)", lw=2, legend=false)
    savefig(p, joinpath("outputs", "alpha_" * tag * "_a$(α).png"))
end

# 3) Experimentos batch (usa αopt)
αopt = alphas["B_alpha_opt"]
batches = Dict(
    "Batch_completo" => size(X_b, 1),
    "Mini_batch_32"  => 32,
    "Estocastico_1"  => 1
)

for (tag, bsz) in batches
    θ, history_loss, epochs_run = minibatch_gd(X_b, y; α=αopt, n_epochs=n_epochs, batch_size=bsz, tol=nothing)
    p = plot(history_loss,
        title="Convergencia ($tag, α=$αopt, batch=$bsz)",
        xlabel="Épocas", ylabel="Coste J(θ)", lw=2, legend=false)
    savefig(p, joinpath("outputs", "batch_" * tag * "_b$(bsz).png"))
end

# 4) Reto: criterio de parada (ejemplo)
tol = 1e-5
θ, history_loss, epochs_run = minibatch_gd(X_b, y; α=αopt, n_epochs=10_000, batch_size=32, tol=tol)
p = plot(history_loss,
    title="Early stop (α=$αopt, batch=32, tol=$tol)",
    xlabel="Épocas", ylabel="Coste J(θ)", lw=2, legend=false)
savefig(p, joinpath("outputs", "reto_early_stop.png"))

println("Listo. Figuras generadas en outputs/")
