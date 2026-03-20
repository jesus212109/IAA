# Práctica 2 — Regularización + Validación Cruzada (Julia / MLJ)
#
# Objetivos:
# 1) Ver el efecto de λ en Lasso (coeficientes -> 0)
# 2) Cambiar K (nfolds) y observar estabilidad vs coste
#
# Requisitos:
# ] add MLJ DataFrames Random Plots MLJLinearModels
#
using MLJ
using DataFrames
using Random
using Plots
using CSV

# -----------------------
# 1) Cargar datos (CSV generado)
# -----------------------
df = CSV.read(joinpath(@__DIR__, "..", "data", "precios_viviendas.csv"), DataFrame)

X = select(df, Not(:Precio))
y = df.Precio

# -----------------------
# 2) Configuración (MODIFICA ESTO)
# -----------------------
metodo = "lasso"  # "lasso" o "ridge"
λ = 1.0           # parámetro de regularización
k = 10            # folds

# -----------------------
# 3) Modelo + Validación
# -----------------------
if metodo == "lasso"
    LassoRegressor = @load LassoRegressor pkg=MLJLinearModels
    model = LassoRegressor(lambda=λ)
else
    RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
    model = RidgeRegressor(lambda=λ)
end

mach = machine(model, X, y)
res = evaluate!(mach, resampling=CV(nfolds=k, shuffle=true, rng=42), measure=rmse)

println("Metodo=$(uppercase(metodo)) | λ=$λ | K=$k | RMSE_CV=$(res.measurement[1])")

# -----------------------
# 4) Ajuste y visualización de coeficientes
# -----------------------
fit!(mach)

fp = fitted_params(mach)

# Extraemos los valores numéricos de los coeficientes (ya que vienen en formato Par(Nombre=>Valor))
if hasproperty(fp, :coefs)
    coef_values = last.(fp.coefs)
    
    bar(1:length(coef_values), coef_values, title="Coeficientes $(uppercase(metodo)) (λ=$λ)",
        xticks=(1:length(coef_values), names(X)), legend=false)
else
    @warn "No se pudieron extraer coeficientes automáticamente. Inspecciona `fitted_params(mach)`."
end
