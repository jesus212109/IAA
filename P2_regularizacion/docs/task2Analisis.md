## Tarea 2: Regularización Ridge vs. Lasso

Para esta tarea se ha fijado el número de particiones a $K=10$ y se han evaluado distintos niveles de penalización ($\lambda$) para ambos métodos.

### 1. Ridge Regression (Penalización L2)
Se han probado los valores $\lambda = 0.1, 10, 100$, obteniendo los siguientes resultados de RMSE:
* $\lambda = 0.1 \rightarrow RMSE \approx 5.01$
* $\lambda = 10 \rightarrow RMSE \approx 5.96$
* $\lambda = 100 \rightarrow RMSE \approx 22.52$

**Análisis de coeficientes:** Al observar las gráficas generadas, visualmente parece que para $\lambda = 0.1$ la variable `Var_8` vale cero, y para $\lambda = 10$ tanto `Var_4` como `Var_8` parecen desaparecer. Sin embargo, a nivel técnico, Ridge rara vez hace los coeficientes exactamente cero; simplemente reduce drásticamente su magnitud, pero no elimina por completo las variables del modelo.

### 2. Lasso Regression (Penalización L1)
Se han realizado pruebas con los mismos valores de $\lambda$ para observar la limpieza del modelo:
* **Identificación del valor óptimo:** El valor $\lambda = 1.0$ (RMSE $\approx 5.27$) es uno de los muchos que consiguen eliminar (poner a cero) las 7 variables ruidosas, conservando únicamente las 3 importantes. En la práctica, con cualquier valor de $\lambda$ en el rango de 0.5 a 2.0 también consiguí esta selección de atributos con facilidad.
* **Comportamiento del error:** Con una penalización baja ($\lambda = 0.1$), el RMSE es menor (5.00), pero el modelo aún no anula todas las variables de ruido.

### 3. El dilema de la complejidad
Al aplicar un $\lambda$ demasiado alto (por ejemplo, $\lambda = 100$), el RMSE se dispara hasta 62.98. El modelo se vuelve demasiado rígido, los coeficientes se reducen casi a cero y la predicción pierde la señal real. Esto provoca que entremos en una zona de **sesgo alto (underfitting)**.

### Reto: El "Cazador de Variables"

* [cite_start]**Valor de λ elegido:** 0.5 [cite: 304]
* **Variables seleccionadas:** `Var_0`, `Var_1` y `Var_2`. El modelo ha conseguido anular por completo los coeficientes de las variables de ruido (`Var_3` a `Var_9`)[cite: 304].
* **RMSE obtenido:** 5.03

**Justificación del compromiso óptimo:**
Tras el ajuste manual del parámetro en la consola (probando valores como 1.5, 0.6, 0.55, 0.45 y 0.5), el valor λ = 0.5 demuestra ser el punto de equilibrio perfecto[cite: 304]. 

Logra la máxima simplicidad al descartar 7 atributos, reduciendo el coste de mantenimiento del modelo[cite: 302, 303]. Al mismo tiempo, conserva un rendimiento excelente (RMSE de 5.03), una precisión casi idéntica a la de modelos menos restrictivos (como λ = 0.1 con RMSE 5.00), pero sin arrastrar el ruido.

---