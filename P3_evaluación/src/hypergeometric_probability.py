"""
Comprobación estadística — Tarea 1, Práctica 3
Introducción al Aprendizaje Automático, 2025/2026

Calcula la probabilidad exacta de obtener 0 (o pocos) positivos
en el conjunto de test cuando se hace una partición aleatoria
sin estratificar, usando la distribución hipergeométrica.

  N = 1000  (tamaño del dataset)
  K = 20    (positivos, 2% del total)
  n = 200   (tamaño del conjunto de test, 20%)
"""

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Parámetros del problema
# ---------------------------------------------------------------------------
N = 1000   # pacientes totales
K = 20     # positivos en el dataset
n = 200    # ejemplos en el conjunto de test

# scipy: hypergeom(M, n, N) → M=población, n=positivos, N=extracciones
rv = stats.hypergeom(M=N, n=K, N=n)

# ---------------------------------------------------------------------------
# Probabilidades puntuales
# ---------------------------------------------------------------------------
p_zero        = rv.pmf(0)    # P(X = 0)
p_one         = rv.pmf(1)    # P(X = 1)
p_two         = rv.pmf(2)    # P(X = 2)
p_at_most_one = rv.cdf(1)    # P(X <= 1)

media = rv.mean()   # = n*K/N = 4.0
sigma = rv.std()

# ---------------------------------------------------------------------------
# Salida
# ---------------------------------------------------------------------------
print("=" * 65)
print("  Distribución hipergeométrica — partición aleatoria")
print("  N={}, K={} positivos, test_size={}".format(N, K, n))
print("=" * 65)

print("\n[Parámetros]")
print(f"  Media esperada de positivos en test  : {media:.4f}")
print(f"  Desv. típica                          : {sigma:.4f}")

print("\n[Probabilidades]")
print(f"  P(X = 0) — ningún positivo en test   : {p_zero:.6f}  ({100*p_zero:.4f}%)")
print(f"  P(X = 1) — exactamente uno            : {p_one:.6f}  ({100*p_one:.4f}%)")
print(f"  P(X = 2) — exactamente dos            : {p_two:.6f}  ({100*p_two:.4f}%)")
print(f"  P(X ≤ 1) — a lo sumo uno              : {p_at_most_one:.6f}  ({100*p_at_most_one:.4f}%)")

print("\n[Consecuencia]")
print(
    f"  Hay un {100*p_zero:.4f}% de probabilidad de que el test quede sin\n"
    f"  ningún ejemplo positivo. En ese caso, Recall, F1-score\n"
    f"  y AUC-ROC son indefinidos o carecen de sentido estadístico.\n"
    f"\n"
    f"  P(X ≤ 1) = {100*p_at_most_one:.4f}% → en casi 1 de cada 15 ejecuciones\n"
    f"  se obtiene 0 o 1 positivo en test con este dataset.\n"
    f"\n"
    f"  Con estratificación: siempre round({K} * {n}/{N}) = {round(K*n/N)} positivos."
)

print("\n[Fórmula LaTeX para la memoria]")
print(f"  P(X=0) = binom({K},0)*binom({N-K},{n}) / binom({N},{n}) ≈ {p_zero:.4f}")
print("=" * 65)
