### Análisis del Coste Computacional ($K=10$ vs $K=100$)

* **Incremento de Tiempo:** El paso de $K=10$ a $K=100$ supone un aumento drástico en el tiempo de ejecución. Con $K=10$ la media ronda los **0.016s**, mientras que con $K=100$ asciende a unos **0.131s**, multiplicando el coste computacional por 8.
* **Estabilidad del RMSE:** Con $K=10$ los valores ya son muy estables (rango 5.16 - 5.23) frente a la alta varianza de $K=2$. Con $K=100$, la fluctuación se reduce (5.17 - 5.19), pero la ganancia en precisión es casi imperceptible.

### ¿Merece la pena el tiempo extra?

**No merece la pena.** La mejora en la precisión del RMSE entre $K=10$ y $K=100$ es marginal (apenas 0.01 unidades), lo cual no justifica un coste temporal 8 veces mayor. El valor $K=10$ se confirma como el estándar óptimo, ofreciendo el mejor equilibrio entre robustez estadística y eficiencia computacional para este modelo.