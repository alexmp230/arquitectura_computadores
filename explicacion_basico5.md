# Guía Definitiva y Explicación Línea por Línea: `basico5.cu`
### *Sincronización de Hilos y Reducción Paralela: Aproximación del Número $\pi$*

---

## 1. El Salto Conceptual: De Hilos Independientes a Hilos Cooperativos

En los ejercicios anteriores (**BÁSICO 1, 2, 3 y 4**), cada hilo de ejecución trabajaba de forma **totalmente aislada**:
* El hilo $i$ leía exclusivamente la posición $i$ del vector o matriz.
* Escribía su resultado exclusivamente en la posición $i$.
* No le importaba en absoluto qué hacían los demás hilos ni en qué orden terminaban.

Sin embargo, muchos problemas fundamentales de la computación requieren que los hilos **colaboren entre sí**, compartiendo datos intermedios para condensar un conjunto grande de datos en un único resultado (sumatorios, máximos, mínimos, productos escalares).

A esta familia de algoritmos se la conoce como **Reducción Paralela**.

---

## 2. El Fundamento Matemático: El Problema de Basilea

Para poner en práctica la cooperación entre hilos, calculamos el valor aproximado del número $\pi$ mediante la famosa serie de Basilea demostrada por Leonhard Euler en 1734:

$$\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{1}{1^2} + \frac{1}{2^2} + \frac{1}{3^2} + \frac{1}{4^2} + \dots = \frac{\pi^2}{6}$$

Despejando $\pi$ para un número finito de $N$ términos:

$$\pi \approx \sqrt{6 \times \sum_{n=1}^{N} \frac{1}{n^2}}$$

### División del trabajo en la solución propuesta:
1. **Fase 1 (Generación paralela de términos)**:
   Cada hilo $myID$ (desde $0$ hasta $N-1$) calcula su término individual de la sucesión dentro del kernel:
   $$n = myID + 1 \implies \text{vector}[myID] = \frac{1.0f}{n^2}$$
2. **Fase 2 (Suma cooperativa por reducción paralela)**:
   Todos los hilos del bloque colaboran sumando los $N$ términos mediante saltos sucesivos a la mitad (`salto = blockDim.x / 2`) hasta que el total acumulado queda en `vector[0]`.
3. **Fase 3 (Cálculo del resultado final en el Device)**:
   El hilo 0 (`myID == 0`) calcula directamente en el kernel la raíz cuadrada del resultado escalado por 6:
   $$\text{*pi} = \sqrt{6 \times \text{vector}[0]}$$
4. **Fase 4 (Evaluación en el Host)**:
   Se copia el valor de $\pi$ al Host y se compara con la constante `#define PI 3.141593f` para obtener el error relativo porcentual.

---

## 3. ¿Por qué es Necesaria la Función `__syncthreads()`?

Los hilos dentro de una GPU se ejecutan en pequeños grupos de 32 llamados **warps**. El hardware no garantiza que todos los warps avancen a la misma velocidad: unos pueden ir por la instrucción 10 mientras otros aún están en la 3.

Si un hilo necesita leer un dato que otro hilo está calculando, surge un **riesgo de datos RAW (*Read After Write*)**:
* Si el hilo lector lee **antes** de que el hilo escritor termine de escribir, leerá basura o datos obsoletos.
* El resultado final sería aleatorio y dependería de la suerte de la planificación del silicio (*condición de carrera / Race Condition*).

### La Barrera de Sincronización: `__syncthreads()`
* Es una instrucción hardware que actúa como un **muro o semáforo**: ningún hilo de ese bloque puede cruzar la línea hasta que **todos y cada uno de los hilos del bloque hayan llegado a ese punto**.
* Garantiza que todas las escrituras en memoria realizadas antes del `__syncthreads()` son visibles y seguras para todos los hilos después de superarlo.

---

## 4. El Algoritmo de Reducción Paralela en Árbol ($O(\log_2 N)$)

En una CPU secuencial, sumar $N$ números requiere un bucle de $N$ pasos ($O(N)$).  
En una GPU paralela con $N$ hilos, organizamos la suma en un **árbol binario** que solo necesita $\mathbf{\log_2(N)}$ **pasos**.

### Ejemplo con $N = 8$ elementos (requiere $\log_2(8) = 3$ pasos):

```
Posición inicial:   [0]      [1]      [2]      [3]   |   [4]      [5]      [6]      [7]
Valores:             t1       t2       t3       t4   |    t5       t6       t7       t8
-----------------------------------------------------------------------------------------
PASO 1 (salto = 4): Hilos 0, 1, 2, 3 trabajan:
  Hilo 0: [0] = [0] + [4]   --> (t1 + t5)
  Hilo 1: [1] = [1] + [5]   --> (t2 + t6)
  Hilo 2: [2] = [2] + [6]   --> (t3 + t7)
  Hilo 3: [3] = [3] + [7]   --> (t4 + t8)
  --> BARRERA: __syncthreads();
-----------------------------------------------------------------------------------------
PASO 2 (salto = 2): Hilos 0, 1 trabajan:
  Hilo 0: [0] = [0] + [2]   --> (t1+t5) + (t3+t7)
  Hilo 1: [1] = [1] + [3]   --> (t2+t6) + (t4+t8)
  --> BARRERA: __syncthreads();
-----------------------------------------------------------------------------------------
PASO 3 (salto = 1): Solo el Hilo 0 trabaja:
  Hilo 0: [0] = [0] + [1]   --> SUMA TOTAL DE LOS 8 TÉRMINOS
  --> BARRERA: __syncthreads();
-----------------------------------------------------------------------------------------
Resultado final: En vector[0].
```

Por este motivo se exige que $N$ sea **potencia de 2** ($32, 64, 128, 256, 512, 1024$), para que en cada iteración la división por 2 sea exacta.

---

## 5. Análisis Detallado del Código

### 5.1. El Kernel CUDA: `calculoPi`

```cuda
__global__ void calculoPi(float *vector, float *pi)
{
    // KERNEL CON 1 SOLO BLOQUE
    // identificacion de cada hilo
    int myID = threadIdx.x;

    // cada hilo genera su denominador y escribe su termino
    float n = (float)(myID + 1);
    vector[myID] = 1.0f / (n * n);

    // sincronizamos todos los hilos para evitar riesgos RAW
    __syncthreads();

    // ******************
    // REDUCCION PARALELA
    // ******************
    int salto = blockDim.x / 2;
    while (salto > 0)
    {
        // en cada paso solo trabajan la mitad de los hilos
        if (myID < salto)
        {
            vector[myID] = vector[myID] + vector[myID + salto];
        }
        // sincronizamos los hilos para evitar riesgos de tipo RAW
        __syncthreads();
        salto = salto / 2;
    }

    // ******************
    // Solo el hilo no.'0' escribe el resultado final:
    // evitamos los riesgos estructurales por el acceso a la memoria
    if (myID == 0)
    {
        *pi = sqrtf(6.0f * vector[0]);
    }
}
```

* **`float n = (float)(myID + 1);`**: Como `threadIdx.x` va de $0$ a $\text{blockDim.x}-1$, se suma $1$ para generar el denominador $n \in [1, N]$.
* **Primer `__syncthreads()`**: Imprescindible. Si no estuviera, un hilo podría intentar sumar `vector[myID + salto]` antes de que el hilo `myID + salto` haya escrito su término inicial.
* **Segundo `__syncthreads()` (en el bucle `while`)**: Imprescindible. Asegura que ningún hilo divida `salto` y comience la siguiente ronda de sumas antes de que todos los hilos activos hayan completado la ronda actual.
* **`if (myID == 0)`**: Evita riesgos estructurales y colisiones en memoria: únicamente el hilo 0 realiza la raíz cuadrada y guarda el escalar en `*pi`.

---

### 5.2. Función `propiedades_Device()` y Compatibilidad de GPU

La función cumple dos misiones clave:
1. Imprime las características físicas del dispositivo (nombre, capacidad de cómputo, arquitectura, SMs, CUDA cores totales y máximo de hilos por bloque).
2. **Retorna `deviceProp.maxThreadsPerBlock`** (`int`), permitiendo validar dinámicamente si el número de términos solicitado cabe en un único bloque.

#### Soporte ampliado para GPUs modernas:
En la plantilla del profesor, el `switch(major)` solo contemplaba hasta la arquitectura Ampere (con 64 cores asignados de forma genérica). Se ha mantenido la compatibilidad retroactiva y añadido:
* **Ada Lovelace (Compute Capability 8.9)**: Detecta tarjetas de la serie RTX 40 asignando correctamente 128 CUDA Cores por SM.
* **Hopper (CC 9.0)** y **Blackwell (CC 10.x / 12.x)**: Previene que aparezca como "DESCONOCIDA" y calcula los cores adecuadamente.

---

### 5.3. El Programa Principal (`main`)

1. **Detección y Obtención del Límite de Hilos**:
   ```cuda
   maxThreads = propiedades_Device();
   ```
2. **Validación de la Entrada**:
   ```cuda
   while (terminos > maxThreads)
   {
       printf("> Se ha superado el maximo numero de hilos por bloque (%d hilos)\n", maxThreads);
       printf(" Introduce de nuevo el numero de terminos (potencia de 2): ");
       scanf("%d", &terminos);
       getchar();
   }
   ```
   Evita errores de ejecución en CUDA al impedir lanzamientos con más hilos de los admitidos físicamente por el hardware (típicamente 1024 hilos por bloque).
3. **Reserva y Lanzamiento (1 Bloque de $N$ Hilos)**:
   ```cuda
   calculoPi <<< 1, terminos >>> (dev_vector, dev_pi);
   ```
   Se lanza en **1 solo bloque** porque la primitiva `__syncthreads()` únicamente sincroniza hilos pertenecientes al mismo bloque.
4. **Cálculo del Error Relativo**:
   ```cuda
   float error = 100.0f * (*hst_pi - PI) / PI;
   ```
5. **Liberación de Recursos y Marca Temporal**:
   - Se libera la memoria tanto en el host (`free(hst_pi)`) como en el device (`cudaFree(...)`).
   - Se imprime la fecha y hora de ejecución mediante `time_t` y `ctime(&fecha)`.

---

## 6. Traza Numérica con $N = 512$ Hilos

Al ejecutar el programa con $N = 512$:

* Suma parcial de los 512 términos: $\sum_{n=1}^{512} \frac{1}{n^2} \approx 1.642981$.
* Multiplicación por 6: $6 \times 1.642981 = 9.857885$.
* Raíz cuadrada calculada en GPU: $\sqrt{9.857885} = \mathbf{3.139729}$.
* Error relativo respecto a $3.141593$:
  $$\frac{3.139729 - 3.141593}{3.141593} \times 100 = \mathbf{-0.059332\%}$$

El programa coincide exactamente con la solución oficial de la asignatura y respeta al 100% los nombres, flujo y estándares exigidos.
