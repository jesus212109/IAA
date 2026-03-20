# Cuaderno de Laboratorio — Práctica 1: La Física del Aprendizaje Automático

**Asignatura:** Introducción al Aprendizaje Automático (3º Ing. Informática)  
**Curso:** 2025/2026  
**Alumno/a:** ___________________________  
**Fecha:** ______________________________

## Objetivo
Comprender la dinámica de la optimización mediante gradiente descendente y el impacto de los hiperparámetros en la convergencia.

---

## 1. Tarea 1 — La importancia de la tasa de aprendizaje (α)

### 1.1 Escenario A (αslow): convergencia lenta
- **Valor elegido:** α = _______
- **Figura:** `outputs/alpha_A_alpha_slow_...png`

![αslow](outputs/alpha_A_alpha_slow_a0.01.png)

**Análisis (obligatorio):**
- ¿Por qué la curva baja tan lentamente?
- ¿Cuántas iteraciones estimas que harían falta para llegar cerca del mínimo? Justifica.

### 1.2 Escenario B (αopt): “curva de codo”
- **Valor elegido:** α = _______
- **Figura:** `outputs/alpha_B_alpha_opt_...png`

![αopt](outputs/alpha_B_alpha_opt_a0.2.png)

**Análisis (obligatorio):**
- Describe la “curva de codo” y qué te dice sobre estabilidad y rapidez.

### 1.3 Escenario C (αosc): oscilación amortiguada
- **Valor elegido:** α = _______
- **Figura:** `outputs/alpha_C_alpha_osc_...png`

![αosc](outputs/alpha_C_alpha_osc_a0.8.png)

**Análisis (obligatorio):**
- ¿Qué está ocurriendo “físicamente” con los pesos θ en el espacio de búsqueda?
- Explica por qué puede oscilar y aun así estabilizarse.

### 1.4 Escenario D (αfail): divergencia
- **Valor elegido:** α = _______
- **Figura:** `outputs/alpha_D_alpha_fail_...png`

![αfail](outputs/alpha_D_alpha_fail_a2.0.png)

**Análisis (obligatorio):**
- ¿Por qué el error crece? Relaciónalo con “pasarse” del mínimo.

---

## 2. Tarea 2 — El compromiso del Mini-batch

Usa tu **αopt** (del escenario B) y compara:

### 2.1 Batch completo
![Batch completo](outputs/batch_Batch_completo_b500.png)

### 2.2 Mini-batch (32 o 16)
![Mini-batch](outputs/batch_Mini_batch_32_b32.png)

### 2.3 Estocástico puro (batch=1)
![Estocástico](outputs/batch_Estocastico_1_b1.png)

**Preguntas de reflexión (obligatorio):**
1. ¿Cuál de las tres curvas es más “ruidosa” y por qué?
2. A nivel de tiempo de ejecución (CPU), ¿cuál ha sido más eficiente?  
   Justifica basándote en computación vectorial y número de actualizaciones.

---

## 3. El reto — “Ajuste de Precisión” (criterio de parada)

Modifica el script para que el entrenamiento se detenga automáticamente cuando:

\[ |J_t - J_{t-1}| < 10^{-5} \]

- **Mejor combinación encontrada:** α = _______ ; batch = _______
- **Épocas hasta parar:** _______
- **Figura (opcional):**
![Early stop](outputs/reto_early_stop.png)

**Conclusión:**
- ¿Qué combinación para antes con error aceptable?
- ¿Qué sacrificas (si algo) para conseguirlo?

---

## 4. Conclusiones finales (obligatorio)
Resume en 8–12 líneas lo que has aprendido sobre:
- relación entre α y estabilidad,
- efecto del batch en ruido/velocidad,
- utilidad de un criterio de parada.
