# Guía Definitiva y Explicación Línea por Línea: `basico2.cu`
### *De Python a CUDA C++: Tu primer Kernel masivamente paralelo*

---

## 1. El gran cambio mental: De bucles secuenciales al paralelismo masivo

En el ejercicio anterior (**BÁSICO 1**) aprendiste a reservar memoria en la tarjeta gráfica (`cudaMalloc`) y a mover datos entre la CPU y la GPU (`cudaMemcpy`). Sin embargo, ¡la GPU todavía no había calculado absolutamente nada! Solo la usamos como un "disco duro" ultrarrápido donde guardar números.

En este ejercicio (**BÁSICO 2**) damos el verdadero salto revolucionario: **ejecutar código de cálculo dentro de la GPU**.

### ¿Cómo piensa un programador de Python vs un programador de CUDA?

Imagina que tienes una lista de $N$ números y quieres sumar cada posición o invertir el orden:

| Enfoque | En Python 🐍 (Secuencial) | En CUDA C++ ⚙️ (Paralelismo Masivo) |
| :--- | :--- | :--- |
| **Mecanismo** | Un bucle `for i in range(N):`. | **Sin bucle**. Se lanzan $N$ hilos (*threads*) a la vez. |
| **Tiempo** | Si hay 1.000 elementos, la CPU da **1.000 pasos seguidos**, uno tras otro. | Los 1.000 elementos se procesan **en un solo instante**, cada uno en su propio micro-núcleo de la GPU. |
| **Filosofía** | *"Un solo trabajador muy rápido haciendo 1.000 tareas por orden"*. | *"Un ejército de 1.000 trabajadores haciendo una micro-tarea cada uno a la vez"*. |
| **El concepto clave** | Iterador (`i`). | **Bucle desenrollado (*Loop Unrolling*)** gracias a `threadIdx.x`. |

```
En Python / CPU Tradicional:
  Paso 0: vector_suma[0] = vector_1[0] + vector_2[0]
  Paso 1: vector_suma[1] = vector_1[1] + vector_2[1]
  ...
  Paso 23: vector_suma[23] = vector_1[23] + vector_2[23]
  (Tarda 24 unidades de tiempo)

En CUDA / GPU:
  Hilo 0  ---> Calcula posicion [0]  ---|
  Hilo 1  ---> Calcula posicion [1]  ---|  ¡Todos a la vez al mismo ciclo de reloj!
  ...                                ---|  (Tarda 1 unidad de tiempo)
  Hilo 23 ---> Calcula posicion [23] ---|
```

---

## 2. Diccionario de supervivencia: Nuevos símbolos y conceptos de CUDA

Antes de destripar el código, comprende estos 4 elementos fundamentales:

1. **`__global__ void miFuncion(...)`**:
   - Es un **especificador de función**. Le grita al compilador `nvcc`: *"Esta función se llama desde la CPU (Host), pero se compila para ejecutarse dentro de la GPU (Device)"*.
   - A estas funciones se las llama **Kernels**.
   - **Regla de oro**: Un kernel **SIEMPRE devuelve `void`** (nada). No puede hacer `return resultado;` porque se ejecutan cientos o miles de copias a la vez. Los resultados se devuelven modificando directamente la memoria apuntada por los punteros que le pasas.

2. **La sintaxis mágica `<<<num_bloques, num_hilos>>>`**:
   - Es la llamada de configuración de ejecución (*Kernel Launch Execution Configuration*).
   - En este ejercicio usamos `<<<1, N>>>`: significa *"Lanza **1 solo bloque** que contenga **N hilos** en paralelo"*.

3. **`threadIdx.x` (El DNI del hilo)**:
   - Variable interna de CUDA que existe únicamente dentro de un kernel.
   - Cada uno de los hilos que se están ejecutando tiene su propio valor:
     - El hilo 0 ve `threadIdx.x == 0`.
     - El hilo 1 ve `threadIdx.x == 1`.
     - ...
     - El hilo $N-1$ ve `threadIdx.x == N - 1`.
   - Con este índice, cada hilo sabe exactamente qué casilla de los arrays le toca procesar sin pisar a los demás.

4. **`cudaDeviceSynchronize()`**:
   - Como la CPU lanza el kernel a la GPU y sigue su camino sin esperar (ejecución asíncrona), necesitamos un "semáforo en rojo".
   - `cudaDeviceSynchronize()` le dice a la CPU: *"Quédate congelada aquí hasta que todos los hilos de la GPU hayan terminado su trabajo por completo"*.

---

## 3. Desglose Exhaustivo Línea por Línea de `basico2.cu`

Vamos a analizar el código fuente paso a paso.

```c
1: /**
2:  * ARQUITECTURA DE COMPUTADORES
3:  * 2º Grado en Ingenieria Informatica
4:  *
5:  * EJERCICIO: "BÁSICO 2: Función kernel"
6:  * Objetivo: Lanzar un kernel de un solo bloque con múltiples hilos o threads
7:  * para calcular la suma de dos vectores.
8:  *
9:  * Alumno: Alejandro Muñoz Pérez
10:  * Fecha: 25/09/2026
11:  */
```
- **Líneas 1 a 11**: Encabezado formal de la práctica con tus datos, la fecha y el objetivo.

---

### Inclusión de cabeceras

```c
13: #include <stdio.h>
14: #include <stdlib.h>
15: #include <time.h>
16: #include <cuda_runtime.h>
17: #include <device_launch_parameters.h>
18: #include "propiedades.h"
```

- **Línea 13 (`<stdio.h>`)**: Para `printf()` y `scanf()` (entrada y salida por consola).
- **Línea 14 (`<stdlib.h>`)**: Para `malloc()`, `free()`, `rand()` y `srand()`.
- **Línea 15 (`<time.h>`)**: Para capturar el reloj del sistema (`time(NULL)`).
- **Línea 16 (`<cuda_runtime.h>`)**: La API runtime de CUDA (`cudaMalloc`, `cudaMemcpy`, etc.).
- **Línea 17 (`<device_launch_parameters.h>`)**: **¡Novedad de este ejercicio!** Contiene las definiciones de palabras clave de hardware como `threadIdx`, `blockIdx` y `blockDim`. Si no se incluye, Visual Studio a menudo subraya `threadIdx.x` en rojo con error falso de IntelliSense.
- **Línea 18 (`"propiedades.h"`)**: Nuestra biblioteca propia donde implementamos la función `propiedades_Device(0)` que detecta tu gráfica y sus multiprocesadores.

---

### El Kernel CUDA: `suma_GPU`

```c
20: ///////////////////////////////////////////////////////////////////////////
21: // KERNEL: Funcion ejecutada en la GPU (Device)
22: __global__ void suma_GPU(int* vector_1, int* vector_2, int* vector_suma, int n)
23: {
24: 	// Identificador del hilo dentro del bloque
25: 	int id = threadIdx.x;
26: 
27: 	if (id < n)
28: 	{
29: 		// El segundo vector se genera en el device invirtiendo el primer vector
30: 		vector_2[id] = vector_1[n - 1 - id];
31: 
32: 		// Suma de ambos vectores
33: 		vector_suma[id] = vector_1[id] + vector_2[id];
34: 	}
35: }
```

Este pequeño fragmento de 15 líneas es el núcleo de toda la computación paralela. Analicemos cada detalle:

- **Línea 22 (`__global__ void suma_GPU(...)`)**:
  - `__global__`: Marca la función para ejecutarse en los multiprocesadores de la RTX 5060 Ti.
  - Recibe 4 parámetros:
    1. `int* vector_1`: Puntero a la memoria de la GPU con el vector original.
    2. `int* vector_2`: Puntero donde este kernel guardará el vector invertido.
    3. `int* vector_suma`: Puntero donde se guardará el resultado de la suma.
    4. `int n`: Tamaño de los vectores (pasado por valor).
- **Línea 25 (`int id = threadIdx.x;`)**:
  - Cada hilo consulta su propio número. Si lanzamos 24 hilos, habrá 24 instancias de esta función corriendo al unísono, y cada una tendrá su propio valor de `id` (del 0 al 23).
- **Línea 27 (`if (id < n)`)**:
  - **Cláusula de guardia imprescindible**: Evita que un hilo intente acceder a posiciones de memoria fuera del array (*Buffer Overflow* o fallo de segmentación en VRAM).
- **Línea 30 (`vector_2[id] = vector_1[n - 1 - id];`)**:
  - Cumple el requisito: *"El segundo vector se debe generar en el device y será el inverso del primer vector"*.
  - **Ejemplo con $N = 24$**:
    - El hilo `id = 0` leerá la casilla `24 - 1 - 0 = 23` de `vector_1` y la escribirá en la casilla `0` de `vector_2`.
    - El hilo `id = 1` leerá la casilla `24 - 1 - 1 = 22` y la escribirá en la casilla `1`.
    - ...
    - El hilo `id = 23` leerá la casilla `0` y la escribirá en la casilla `23`.
    - ¡Todo el vector queda invertido en 1 solo ciclo de cómputo!
- **Línea 33 (`vector_suma[id] = vector_1[id] + vector_2[id];`)**:
  - El hilo toma el elemento de la posición `id` del primer vector, le suma el que acaba de calcular en el segundo vector y guarda el resultado en `vector_suma[id]`.
  - **No hay bucle `for`**: La repetición la sustituye la multitud de hilos.

---

### La función `main`: Host

```c
39: int main(int argc, char** argv)
40: {
41: 	int deviceID = 0;
42: 
43: 	// 1. Mostrar propiedades del dispositivo seleccionado
44: 	propiedades_Device(deviceID);
```
- **Líneas 41 a 44**: Elegimos la GPU 0 y llamamos a `propiedades_Device(0)` para imprimir el bloque de especificaciones técnicas: Arquitectura, Compute Capability, Multiprocesadores (SM), CUDA Cores y Memoria Global.

---

### Consulta del límite de hilos por bloque de la GPU

```c
46: 	// Obtener propiedades para conocer el limite maximo de hilos por bloque
47: 	cudaDeviceProp deviceProp;
48: 	cudaGetDeviceProperties(&deviceProp, deviceID);
49: 	int maxThreads = deviceProp.maxThreadsPerBlock;
```
- **Línea 47**: `cudaDeviceProp deviceProp;` es una estructura oficial de NVIDIA que contiene más de 50 campos sobre la arquitectura de la tarjeta gráfica.
- **Línea 48**: Rellena esa estructura con los datos reales de tu GPU.
- **Línea 49**: Extraemos `deviceProp.maxThreadsPerBlock`. En tu RTX 5060 Ti (y en todas las GPUs modernas con Compute Capability $\ge 2.0$), este valor es **1024**.
  - Este dato es el límite físico del silicio: un bloque de CUDA nunca puede alojar más de 1024 hilos.

---

### Entrada interactiva con control de errores

```c
51: 	// 2. Lectura interactiva de N con validacion de hilos maximos
52: 	int N = 0;
53: 	while (1)
54: 	{
55: 		printf("  Introduce el numero de elementos: ");
56: 		if (scanf("%d", &N) != 1)
57: 		{
58: 			while (getchar() != '\n');
59: 			continue;
60: 		}
61: 
62: 		if (N <= 0)
63: 		{
64: 			printf("> ERROR: el numero de elementos debe ser mayor que 0!\n");
65: 		}
66: 		else if (N > maxThreads)
67: 		{
68: 			printf("> ERROR: numero maximo de hilos superado! [%d hilos]\n", maxThreads);
69: 		}
70: 		else
71: 		{
72: 			break;
73: 		}
74: 	}
```
- **Línea 53 (`while (1)`)**: Bucle infinito. No saldremos hasta que el usuario introduzca un número válido (usando `break`).
- **Línea 56 (`scanf("%d", &N) != 1`)**:
  - `scanf` intenta leer un número entero desde el teclado.
  - Si el usuario teclea letras (ej. `"hola"`), `scanf` falla y devuelve 0. En las líneas 58-59 vaciamos el buffer para evitar que el programa se quede en un bucle loco.
- **Líneas 62 a 65**: Comprobación de que $N$ sea positivo.
- **Líneas 66 a 69**: **El requisito del enunciado**. Si el usuario introduce `2000`, como $2000 > 1024$, imprimirá:
  ```text
  > ERROR: numero maximo de hilos superado! [1024 hilos]
  ```
  y volverá a solicitar `Introduce el numero de elementos:`.
- **Líneas 70 a 73**: Si el número está entre 1 y 1024 (por ejemplo `24`), se ejecuta el `break` y continúa el programa.

```c
75: 	// Limpiar el salto de linea residual en el buffer de entrada
76: 	while (getchar() != '\n');
77: 
78: 	printf("> Vector de %d elementos\n", N);
79: 	printf("> Lanzamiento con 1 bloque de %d hilos\n", N);
```
- Limpia el carácter `\n` que queda en el teclado para que el `getchar()` del final del programa no se cierre solo.
- Muestra el encabezado del lanzamiento idéntico a la Figura 2.1 del enunciado.

---

### Reserva de memoria en Host y Device

```c
80: 	size_t size = N * sizeof(int);
81: 
82: 	// 3. Reserva de memoria en el Host
83: 	int* hst_vector1 = (int*)malloc(size);
84: 	int* hst_vector2 = (int*)malloc(size);
85: 	int* hst_resultado = (int*)malloc(size);
86: 
87: 	// 4. Reserva de memoria en el Device
88: 	int* dev_vector1 = NULL;
89: 	int* dev_vector2 = NULL;
90: 	int* dev_resultado = NULL;
91: 	cudaMalloc((void**)&dev_vector1, size);
92: 	cudaMalloc((void**)&dev_vector2, size);
93: 	cudaMalloc((void**)&dev_resultado, size);
```
- **Línea 80**: Calculamos los bytes necesarios. Si $N = 24$ y cada `int` ocupa 4 bytes, `size = 24 * 4 = 96 bytes`.
- **Líneas 83 a 85**: Reservamos en la memoria RAM del ordenador (`malloc`):
  - `hst_vector1`: Para generar los aleatorios.
  - `hst_vector2`: Para recibir el vector invertido de la GPU.
  - `hst_resultado`: Para recibir la suma de la GPU.
- **Líneas 91 a 93**: Reservamos en los chips VRAM de la RTX 5060 Ti (`cudaMalloc`):
  - `dev_vector1`: Copia del vector 1 en la gráfica.
  - `dev_vector2`: El vector 2 que construirá el kernel en la gráfica.
  - `dev_resultado`: El vector donde el kernel sumará ambos.

---

### Generación de valores aleatorios y copia a la GPU

```c
95: 	// 5. Inicializacion del primer vector en el Host con enteros aleatorios [0, 9]
96: 	srand((unsigned int)time(NULL));
97: 	for (int i = 0; i < N; i++)
98: 	{
99: 		hst_vector1[i] = rand() % 10;
100: 	}
101: 
102: 	// 6. Copia del vector 1 desde el Host hacia el Device
103: 	cudaMemcpy(dev_vector1, hst_vector1, size, cudaMemcpyHostToDevice);
```
- **Línea 96**: Semilla aleatoria basada en la hora actual.
- **Línea 99 (`rand() % 10`)**: 
  - El operador `%` (módulo o resto de la división entera) fuerza que el resultado esté siempre en el rango **[0, 9]** (el resto de dividir cualquier entero entre 10 solo puede ser 0, 1, 2, ..., 9).
  - Cumple estrictamente el requisito: *"enteros de tipo int comprendidos entre 0 y 9"*.
- **Línea 103**: Copiamos los $N$ enteros desde la CPU a la memoria de la GPU con `cudaMemcpyHostToDevice`.

---

### Lanzamiento del Kernel y Sincronización

```c
105: 	// 7. Lanzamiento del Kernel (1 bloque con N hilos)
106: 	suma_GPU<<<1, N>>>(dev_vector1, dev_vector2, dev_resultado, N);
107: 
108: 	// Sincronizacion para asegurar la finalizacion del kernel
109: 	cudaDeviceSynchronize();
```
- **Línea 106 (`suma_GPU<<<1, N>>>(...)`)**:
  - **`1`**: Queremos un único bloque de hilos.
  - **`N`**: Queremos exactamente $N$ hilos dentro de ese bloque (ej. 24 hilos).
  - La tarjeta gráfica despierta 24 hilos en paralelo. Cada uno ejecutará la función `suma_GPU` simultáneamente.
- **Línea 109 (`cudaDeviceSynchronize()`)**:
  - Hace una pausa en el procesador principal (CPU) hasta que los 24 hilos de la GPU hayan terminado de escribir en `dev_vector2` y `dev_resultado`.

---

### Recogida de datos y presentación en pantalla

```c
111: 	// 8. Copia de resultados desde el Device hacia el Host
112: 	cudaMemcpy(hst_vector2, dev_vector2, size, cudaMemcpyDeviceToHost);
113: 	cudaMemcpy(hst_resultado, dev_resultado, size, cudaMemcpyDeviceToHost);
```
- Traemos de vuelta desde la GPU hacia la memoria RAM de la CPU los dos vectores calculados con `cudaMemcpyDeviceToHost`.

```c
115: 	// 9. Mostrar vectores iniciales y resultado final
116: 	printf("VECTOR 1:\n");
117: 	for (int i = 0; i < N; i++)
118: 	{
119: 		printf("%2d ", hst_vector1[i]);
120: 	}
121: 	printf("\n");
122: 
123: 	printf("VECTOR 2:\n");
124: 	for (int i = 0; i < N; i++)
125: 	{
126: 		printf("%2d ", hst_vector2[i]);
127: 	}
128: 	printf("\n");
129: 
130: 	printf("SUMA:\n");
131: 	for (int i = 0; i < N; i++)
132: 	{
133: 		printf("%2d ", hst_resultado[i]);
134: 	}
135: 	printf("\n");
```
- **Líneas 119, 126, 133 (`printf("%2d ", ...)`)**:
  - `%2d` imprime el número entero reservando un ancho fijo de 2 espacios.
  - Esto garantiza que los números de una cifra (`2`) y los de dos cifras (`13`) queden perfectamente alineados verticalmente en columnas, exactamente como en la captura del profesor.

---

### Limpieza de memoria y salida

```c
137: 	// 10. Liberacion de memoria
138: 	free(hst_vector1);
139: 	free(hst_vector2);
140: 	free(hst_resultado);
141: 	cudaFree(dev_vector1);
142: 	cudaFree(dev_vector2);
143: 	cudaFree(dev_resultado);
144: 
145: 	// 11. Finalizacion
146: 	printf("***************************************************\n");
147: 	printf("<pulsa [INTRO] para finalizar>");
148: 	getchar();
149: 
150: 	return 0;
151: }
```
- **Líneas 138 a 143**: Devolvemos religiosamente la memoria RAM al sistema operativo (`free`) y la memoria VRAM a la tarjeta gráfica (`cudaFree`).
- **Líneas 148 a 150**: Espera a que pulses `Enter` antes de cerrar la consola y devuelve `0` indicando éxito total.

---

## 4. Resumen: ¿Cómo se traduciría este ejercicio a Python conceptual?

Para un programador que viene de Python o librerías científicas como **NumPy** o **PyTorch**, este ejercicio equivale conceptualmente a lo siguiente:

```python
import random
import torch  # La librería de Deep Learning que usa CUDA por debajo

# 1. Entrada de datos y validación de hilos
MAX_THREADS = 1024

while True:
    try:
        N = int(input("Introduce el numero de elementos: "))
        if N <= 0:
            print("> ERROR: el numero de elementos debe ser mayor que 0!")
        elif N > MAX_THREADS:
            print(f"> ERROR: numero maximo de hilos superado! [{MAX_THREADS} hilos]")
        else:
            break
    except ValueError:
        pass

# 2. Generar Vector 1 en la CPU
vector_1_cpu = [random.randint(0, 9) for _ in range(N)]

# 3. Enviar a la GPU (VRAM de la RTX 5060 Ti)
dev_v1 = torch.tensor(vector_1_cpu, device="cuda")

# 4. Kernel en la GPU: Invertir y sumar en paralelo
dev_v2 = torch.flip(dev_v1, dims=[0])   # Inversión instantánea en GPU
dev_suma = dev_v1 + dev_v2              # Suma paralela elemento a elemento

# 5. Descargar resultados de vuelta a la CPU (Host)
vector_2_cpu = dev_v2.cpu().tolist()
vector_suma_cpu = dev_suma.cpu().tolist()

# 6. Mostrar resultados
print("VECTOR 1:\n", " ".join(f"{x:2d}" for x in vector_1_cpu))
print("VECTOR 2:\n", " ".join(f"{x:2d}" for x in vector_2_cpu))
print("SUMA:\n",     " ".join(f"{x:2d}" for x in vector_suma_cpu))
```

---

## 5. Preguntas Clave para el Examen o la Memoria de Prácticas

### 1. ¿Por qué usamos `suma_GPU<<<1, N>>>` y no lanzamos más de un bloque?
En este ejercicio básico el objetivo específico era aprender a usar **un único bloque** identificando hilos mediante `threadIdx.x`. Si quisiéramos procesar vectores de un millón de elementos ($N = 1.000.000$), un solo bloque no bastaría (el tope es 1024), y tendríamos que usar una cuadrícula (*grid*) con múltiples bloques, introduciendo variables como `blockIdx.x` y `blockDim.x`.

### 2. ¿Qué ventaja tiene generar el segundo vector en el Device en vez de en el Host?
Si hubiésemos invertido el vector en la CPU, habríamos tenido que hacer una segunda transferencia `cudaMemcpy` a través del bus PCI-Express para enviarlo a la GPU. Al generarlo directamente en la GPU con el kernel (`vector_2[id] = vector_1[n - 1 - id]`), nos ahorramos tiempo de tráfico en el bus PCIe, que es el principal cuello de botella de las aplicaciones CUDA.

### 3. ¿Por qué la variable `id` no necesita un bucle `for`?
Porque cada hilo hardware de la GPU ejecuta de forma independiente la misma instrucción del kernel sobre un dato diferente. Es el paradigma **SIMT (*Single Instruction, Multiple Threads*)** característico de la arquitectura NVIDIA CUDA.
