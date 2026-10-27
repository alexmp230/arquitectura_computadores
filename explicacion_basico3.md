# Guía Definitiva y Explicación Línea por Línea: `basico3.cu`
### *De Bloque Único a la Jerarquía de Hilos y Bloques: Procesando Datos de Cualquier Tamaño*

---

## 1. El gran salto conceptual: Superando la barrera física de los 1024 hilos

En el ejercicio anterior (**BÁSICO 2**) logramos que la tarjeta gráfica ejecutara cálculo real en paralelo. Sin embargo, teníamos una limitación física severa:
> **Solo podíamos lanzar 1 único bloque.** Y por diseño del hardware de NVIDIA, ningún bloque puede tener más de **1024 hilos** (`maxThreadsPerBlock = 1024`).

Si queríamos procesar un vector de 2.000 elementos, 100.000 o 10 millones, el programa de Básico 2 fallaba o daba error.

En este ejercicio (**BÁSICO 3**) alcanzamos la verdadera arquitectura escalable de CUDA: **agrupar hilos en múltiples bloques formando una malla (*Grid*)**.

```
                +-------------------------------------------------------+
                |                    GRID (Malla CUDA)                  |
                |                 Total: gridDim.x bloques              |
                +-------------------------------------------------------+
                     |                     |                        |
        +------------v-----------+  +------v-----------------+  +---v--------------------+
        |   BLOQUE 0             |  |   BLOQUE 1             |  |   BLOQUE 2             |
        |   (blockIdx.x = 0)     |  |   (blockIdx.x = 1)     |  |   (blockIdx.x = 2)     |
        +------------------------+  +------------------------+  +------------------------+
        | Hilos [0, 1, ..., 9]   |  | Hilos [0, 1, ..., 9]   |  | Hilos [0, 1, ..., 9]   |
        +------------------------+  +------------------------+  +------------------------+
              |                                   |                           |
         idGlobal: 0 a 9                    idGlobal: 10 a 19           idGlobal: 20 a 29
```

### ¿Por qué la GPU se organiza en Bloques y no en un millón de hilos planos?

La tarjeta gráfica no es un procesador gigante homogéneo; está dividida internamente en varias unidades de cómputo independientes llamadas **Streaming Multiprocessors (SM)**:

- Tu **GeForce RTX 5060 Ti** tiene **36 SMs** independientes.
- La GPU no puede repartir hilos sueltos de uno en uno a cada SM. En su lugar, el planificador hardware (*GigaThread Engine*) **asigna bloques completos a cada SM**.
- Cuando un SM termina de procesar un bloque, inmediatamente toma el siguiente bloque en cola.
- **Ventaja de diseño (*Escalabilidad Transparente*)**: El mismo código compilado puede ejecutarse en una modesta GPU con 2 SMs (procesando 2 bloques a la vez) o en una GPU de centro de datos con 144 SMs (procesando 144 bloques a la vez) **sin cambiar ni una sola línea de código C++**.

---

## 2. El Sistema de Coordenadas de CUDA: Las 4 Variables Mágicas

Dentro de un kernel, NVIDIA proporciona 4 variables predefinidas integradas en el hardware que definen la geometría del lanzamiento:

| Variable | Tipo | ¿Qué representa? | Valor en nuestro ejemplo ($N=33$, bloque de 10) |
| :--- | :--- | :--- | :--- |
| `gridDim.x` | `int` | **Tamaño de la malla**: Cuántos bloques se han lanzado en total en el eje $x$. | `4` bloques |
| `blockIdx.x` | `int` | **Índice del bloque actual**: Identificador del bloque que ejecuta este hilo. | Varía de `0` a `3` |
| `blockDim.x` | `int` | **Tamaño del bloque**: Cuántos hilos componen cada bloque. | `10` hilos |
| `threadIdx.x` | `int` | **Índice local del hilo**: Posición del hilo dentro de su propio bloque. | Varía de `0` a `9` |

---

### La Fórmula de Oro: El Cálculo del `idGlobal`

Dado que `threadIdx.x` se reinicia a `0` al comienzo de cada bloque, **no podemos usar `threadIdx.x` directamente para indexar vectores largos**. Si lo hiciéramos, el hilo 0 del bloque 0 y el hilo 0 del bloque 1 escribirían ambos en la posición `[0]` del array, destruyendo los datos (*Data Race*).

Para obtener un índice unívoco global (del `0` al `N-1`), usamos:

$$\mathbf{idGlobal} = \mathbf{threadIdx.x} + (\mathbf{blockDim.x} \times \mathbf{blockIdx.x})$$

#### Ejemplo numérico con $N = 33$ y bloques de 10 hilos:

- **Bloque 0** (`blockIdx.x = 0`):
  - Hilo 0: $0 + (10 \times 0) = \mathbf{0}$
  - Hilo 1: $1 + (10 \times 0) = \mathbf{1}$
  - ...
  - Hilo 9: $9 + (10 \times 0) = \mathbf{9}$

- **Bloque 1** (`blockIdx.x = 1`):
  - Hilo 0: $0 + (10 \times 1) = \mathbf{10}$
  - Hilo 1: $1 + (10 \times 1) = \mathbf{11}$
  - ...
  - Hilo 9: $9 + (10 \times 1) = \mathbf{19}$

- **Bloque 2** (`blockIdx.x = 2`):
  - Hilo 0: $0 + (10 \times 2) = \mathbf{20}$
  - ...
  - Hilo 9: $9 + (10 \times 2) = \mathbf{29}$

- **Bloque 3** (`blockIdx.x = 3`):
  - Hilo 0: $0 + (10 \times 3) = \mathbf{30}$
  - Hilo 1: $1 + (10 \times 3) = \mathbf{31}$
  - Hilo 2: $2 + (10 \times 3) = \mathbf{32}$
  - Hilo 3: $3 + (10 \times 3) = \mathbf{33} \longrightarrow$ **¡CUIDADO! Fuera de rango.**

---

### La Cláusula de Salvaguarda: `if (id < n)`

Como los bloques tienen tamaño fijo (10 hilos), para procesar 33 elementos necesitamos lanzar **4 bloques** ($4 \times 10 = 40$ hilos).
Esto significa que los hilos con índices globales **33, 34, 35, 36, 37, 38 y 39 no tienen datos válidos que procesar**.

```c
if (id < n)
{
    // Solo los hilos del 0 al 32 ejecutan estas lineas
    vector_2[id] = vector_1[n - 1 - id];
    vector_suma[id] = vector_1[id] + vector_2[id];
}
// Los hilos del 33 al 39 no entran al if y terminan pacificamente
```

> [!WARNING]
> Si olvidas el `if (id < n)`, los hilos 33 a 39 escribirán en zonas de memoria no reservadas de la VRAM, provocando corrupción de memoria (*Memory Corruption*), resultados erróneos o un fallo de ejecución en CUDA (`cudaErrorIllegalAddress`).

---

## 3. Desglose Exhaustivo Línea por Línea de `basico3.cu`

Analizamos cada sección del archivo [basico3.cu](file:///c:/Users/Alex/source/repos/Plantilla%20Cuda%2012.8/basico3.cu):

```c
1: /**
2:  * ARQUITECTURA DE COMPUTADORES
3:  * 2º Grado en Ingenieria Informatica
4:  *
5:  * EJERCICIO: "BÁSICO 3: Hilos y Bloques"
6:  * Lanzar un kernel con múltiples hilos o threads agrupados en bloques.
7:  *
8:  * Alumno: Alejandro Muñoz Pérez
9:  * Fecha: 28/09/2026
10:  */
```

---

### Inclusión de cabeceras y constantes

```c
12: #include <stdio.h>
13: #include <stdlib.h>
14: #include <time.h>
15: #include <cuda_runtime.h>
16: #include <device_launch_parameters.h>
17: #include "propiedades.h"
18: 
19: // Tamaño fijo de bloque (10 hilos por bloque según especificación)
20: #define HILOS_POR_BLOQUE 10
```

- **Línea 16 (`<device_launch_parameters.h>`)**: Declara las variables de hardware `threadIdx`, `blockIdx`, `blockDim` y `gridDim`.
- **Línea 20 (`#define HILOS_POR_BLOQUE 10`)**: Establece el tamaño de bloque exigido por la práctica (*"bloques de tamaño fijo, por ejemplo 10 hilos"*).

---

### El Kernel CUDA con Indexación Global

```c
22: ///////////////////////////////////////////////////////////////////////////
23: // KERNEL: Funcion ejecutada en la GPU (Device)
24: __global__ void suma_GPU(int* vector_1, int* vector_2, int* vector_suma, int n)
25: {
26: 	// Identificador global unívoco del hilo a lo largo de toda la malla (grid)
27: 	int id = threadIdx.x + blockDim.x * blockIdx.x;
28: 
29: 	// Cláusula de seguridad: comprueba que el índice no supere el tamaño del vector
30: 	if (id < n)
31: 	{
32: 		// El segundo vector se genera en el device invirtiendo el primer vector
33: 		vector_2[id] = vector_1[n - 1 - id];
34: 
35: 		// Suma de ambos vectores elemento a elemento
36: 		vector_suma[id] = vector_1[id] + vector_2[id];
37: 	}
38: }
```

- **Línea 27**: La traslación de coordenadas de dos niveles (Bloque + Hilo Local $\to$ Índice Global).
- **Línea 30**: Cláusula de guardia fundamental para no desbordar el vector de tamaño `n`.
- **Línea 33**: Inversión en paralelo del vector: el hilo en `id` lee la posición complementaria `n - 1 - id`.
- **Línea 36**: Suma directa sin bucles.

---

### Consulta y Muestra de Propiedades de Hardware (Figura 3.1)

El enunciado exige mostrar:
> *"Las principales propiedades del dispositivo seleccionado, el máximo número de hilos permitidos en cada eje y el máximo número de bloques permitidos en cada eje."*

```c
40: void mostrar_Propiedades(int deviceID)
41: {
42: 	cudaDeviceProp deviceProp;
43: 	cudaGetDeviceProperties(&deviceProp, deviceID);
...
68: 	printf("***************************************************\n");
69: 	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
70: 	printf("***************************************************\n");
71: 	printf("> Capacidad de Computo            : %d.%d\n", major, minor);
72: 	printf("> No. de MultiProcesadores        : %d\n", SM);
73: 	printf("> No. de CUDA Cores (%dx%d)       : %d\n", cudaCores, SM, cudaCores * SM);
74: 	printf("> No. maximo de Hilos (por bloque): %d\n", deviceProp.maxThreadsPerBlock);
75: 	printf("  [eje x -> %d]\n", deviceProp.maxThreadsDim[0]);
76: 	printf("  [eje y -> %d]\n", deviceProp.maxThreadsDim[1]);
77: 	printf("  [eje z -> %d]\n", deviceProp.maxThreadsDim[2]);
78: 	printf("> No. maximo de Bloques (por eje):\n");
79: 	printf("  [eje x -> %d]\n", deviceProp.maxGridSize[0]);
80: 	printf("  [eje y -> %d]\n", deviceProp.maxGridSize[1]);
81: 	printf("  [eje z -> %d]\n", deviceProp.maxGridSize[2]);
82: 	printf("***************************************************\n");
83: }
```

#### Análisis de los Límites Físicos Reportados:
1. **`deviceProp.maxThreadsPerBlock` (1024)**:
   - Número máximo total de hilos que caben en 1 bloque (el producto de las 3 dimensiones $X \times Y \times Z$ nunca puede superar este valor).
2. **`deviceProp.maxThreadsDim[3]`**:
   - `eje x -> 1024`: Un bloque puede ser unidimensional de hasta 1024 hilos.
   - `eje y -> 1024`: Un bloque puede tener hasta 1024 en $Y$ (si $X=1, Z=1$).
   - `eje z -> 64`: En el eje $Z$, el silicio limita a un máximo de 64 hilos.
3. **`deviceProp.maxGridSize[3]`**:
   - `eje x -> 2147483647` ($2^{31}-1$): ¡Podemos lanzar más de **dos mil millones de bloques** a lo largo del eje $X$!
   - `eje y -> 65535` ($2^{16}-1$).
   - `eje z -> 65535` ($2^{16}-1$).

---

### Cálculo Dinámico del Número de Bloques

En la función `main`:

```c
113: 	// 3. Calculo del numero de bloques necesarios con bloques de tamaño fijo
114: 	int hilosB = HILOS_POR_BLOQUE;
115: 	int Nbloques = (N + hilosB - 1) / hilosB;
116: 
117: 	printf("> Vector de %d elementos\n", N);
118: 	printf("> Lanzamiento con %d bloques de %d hilos (%d hilos)\n", Nbloques, hilosB, Nbloques * hilosB);
```

#### ¿Por qué la fórmula `(N + hilosB - 1) / hilosB`?
Es el patrón clásico de **redondeo hacia arriba (*Ceiling Division*) usando aritmética entera pura**:

| $N$ | Operación entera `(N + 9) / 10` | Resultado `Nbloques` | Hilos totales lanzados | ¿Cubre todos los datos? |
| :---: | :---: | :---: | :---: | :---: |
| **10** | `(10 + 9) / 10 = 19 / 10` | **1** | 10 | Exacto |
| **11** | `(11 + 9) / 10 = 20 / 10` | **2** | 20 | Sí (sobran 9 hilos protegidos) |
| **33** | `(33 + 9) / 10 = 42 / 10` | **4** | 40 | Sí (sobran 7 hilos protegidos) |

> [!TIP]
> **¿Por qué no usar `ceil((float)N / hilosB)`?**
> Porque la conversión a número flotante (`float`) y la llamada a la función matemática `ceil()` son computacionalmente más lentas y pueden sufrir imprecisiones de coma flotante con enteros grandes. La fórmula entera `(N + B - 1) / B` es instantánea (1 ciclo de CPU).

---

### Lanzamiento del Kernel Multi-Bloque

```c
141: 	// 8. Lanzamiento del Kernel con Nbloques y hilosB
142: 	suma_GPU<<<Nbloques, hilosB>>>(dev_vector1, dev_vector2, dev_resultado, N);
143: 
144: 	// Sincronizacion para asegurar la finalizacion de todos los bloques
145: 	cudaDeviceSynchronize();
```

- En **Básico 2** lanzábamos: `suma_GPU<<<1, N>>>`.
- En **Básico 3** lanzamos: `suma_GPU<<<Nbloques, hilosB>>>` (ej. `<<<4, 10>>>`).
- La GPU crea la malla bidimensional abstracta y reparte los 4 bloques entre sus SMs disponibles.

---

## 4. Tabla Comparativa: Básico 2 vs Básico 3

| Característica | BÁSICO 2 (`basico2.cu`) | BÁSICO 3 (`basico3.cu`) |
| :--- | :--- | :--- |
| **Organización de Hilos** | 1 solo bloque con $N$ hilos | Múltiples bloques con tamaño fijo (10 hilos/bloque) |
| **Configuración de Kernel** | `<<<1, N>>>` | `<<<Nbloques, hilosB>>>` |
| **Límite de Elementos ($N$)** | Máximo **1024 elementos** (tope por bloque) | **Sin límite práctico** (hasta $2 \times 10^9 \times 10$ hilos) |
| **Cálculo del Índice** | `id = threadIdx.x;` | `id = threadIdx.x + blockDim.x * blockIdx.x;` |
| **Hilos Inactivos / Sobrantes** | Ninguno (hilos lanzados = $N$) | Sí en el último bloque (controlados con `if (id < n)`) |
| **Distribución Hardware** | Corre en un **único SM** | Se reparte en paralelo entre **todos los SMs** de la GPU |

---

## 5. Preguntas Clave para el Examen y la Memoria Académica

### 1. Si tu GPU tiene 36 SMs y lanzas un kernel con `<<<1, 1024>>>` (Básico 2), ¿qué porcentaje de la GPU estás aprovechando?
Estás utilizando **solo 1 de los 36 SMs**. El 97.2% de los núcleos de tu GPU se quedan completamente ociosos sin hacer nada.
Por el contrario, con `<<<M, N>>>` (Básico 3), los bloques se reparten entre los 36 SMs, permitiendo el aprovechamiento del 100% de la capacidad de cómputo del chip.

### 2. ¿En qué orden se ejecutan los bloques de una malla?
**En un orden totalmente arbitrario e impredecible**. Los bloques son completamente independientes entre sí. El hardware puede ejecutar el Bloque 3 antes que el Bloque 0 según la disponibilidad de recursos en los SMs. Por tanto, el código de un bloque **nunca debe depender** de los resultados producidos por otro bloque en el mismo kernel.

### 3. ¿Por qué en la industria real se usan bloques de 128, 256 o 512 hilos en lugar de 10?
En la arquitectura de hardware de NVIDIA, la unidad física mínima de ejecución es el **Warp** (compuesto siempre por **32 hilos**).
Si configuras un bloque con 10 hilos:
- El planificador de hardware debe asignar 1 Warp de 32 hilos de todos modos.
- 10 hilos trabajarán y los otros **22 hilos del Warp estarán desactivados (*Warp Underutilization*)**.
- Para este ejercicio académico se pide un bloque de 10 hilos para poder visualizar y verificar visualmente en la consola los índices en vectores pequeños (como 33 elementos). En aplicaciones de producción, siempre se eligen múltiplos de 32 (típicamente 128 o 256).

### 4. ¿Qué diferencia conceptual existe entre `maxThreadsDim[0]` y `maxGridSize[0]`?
- `maxThreadsDim[0]` (1024): Es el límite de hilos que pueden cooperar y comunicarse a través de memoria compartida dentro de un mismo bloque local en el eje $X$.
- `maxGridSize[0]` (2.147.483.647): Es el número máximo de bloques independientes que la GPU puede admitir en la cola global de trabajo en el eje $X$.
