# Guía Definitiva y Explicación Línea por Línea: `basico1.cu`
### *De Python a CUDA C++: Entendiendo la memoria de la GPU desde cero*

---

## 1. El gran cambio mental: De Python a C / CUDA

Si tu único lenguaje de programación previo es **Python**, entrar en C y CUDA puede parecer un choque cultural tremendo. En Python, el intérprete hace casi todo el trabajo sucio por ti en segundo plano:

| Concepto | En Python 🐍 | En C / CUDA ⚙️ |
| :--- | :--- | :--- |
| **Tipado** | Dinámico (`x = 5.0` y luego `x = "hola"`). | Estricto (`float x = 5.0f;`). No puedes cambiar el tipo nunca. |
| **Memoria** | Automática. Un *Garbage Collector* borra variables que no se usan. | **Manual**. Tú pides los bytes exactos y tú tienes que liberarlos a mano. |
| **Listas / Arrays** | `lista = [0.1, 0.2]`. Crece sola y almacena cualquier cosa. | Bloque continuo de memoria contigua de tamaño fijo. |
| **Dónde vive el dato** | Todo en la memoria RAM principal de la CPU. | Hay **dos mundos separados**: la RAM de la CPU (Host) y la VRAM de la GPU (Device). |
| **Punteros** | Ocultos (todas las variables en Python son referencias por debajo). | **Explícitos**. Manejas directamente direcciones de memoria en hexadecimal. |

### La regla de oro de CUDA: Dos mundos desconectados
Tu ordenador tiene dos cerebros con sus propias memorias independientes comunicadas por un puente (el bus PCI-Express):
1. **HOST (El Anfitrión)**: Es la CPU (Intel/AMD) y la memoria RAM principal de tu ordenador.
2. **DEVICE (El Dispositivo)**: Es tu tarjeta gráfica (NVIDIA GeForce RTX 5060 Ti) y su memoria propia de vídeo ultra-rápida (VRAM de 8 GB).

> ⚠️ **Importante**: La GPU **NO** puede leer directamente una variable que esté en la RAM de la CPU, ni la CPU puede leer directamente una variable que esté en la VRAM de la GPU. Para que la GPU trabaje con datos, **tienes que copiarlos expresamente por el bus PCIe**.

---

## 2. Diccionario de supervivencia: Símbolos raros de C/C++

Antes de leer el código, grábate el significado de estos 3 símbolos:

1. **El asterisco `*` en una declaración (Puntero)**:
   - `float *p;` significa: *"La variable `p` no guarda un número float; guarda la dirección de memoria (como el número de portal de una casa) donde empieza un float"*.
2. **El ampersand `&` (Operador "Dirección de")**:
   - Si `x` es una variable normal, `&x` significa: *"Dime en qué posición exacta de la memoria está guardada `x`"*.
3. **El asterisco `*` al usar una variable (Desreferenciación)**:
   - `*p` significa: *"Ve a la dirección de memoria que guarda `p` y dime qué valor hay guardado dentro de esa casa"*.

---

## 3. Desglose Exhaustivo Línea por Línea de `basico1.cu`

Vamos a analizar el código paso a paso, línea a línea, explicando qué hace y por qué existe.

```c
1: /*
2: * ARQUITECTURA DE COMPUTADORES
3: * 2º Grado en Ingenieria Informatica
4: *
5: * EJERCICIO: "BÁSICO 1: Memoria Global"
6: * >> Transferencia de datos entre Host y Device (Global Memory)
7: *
8: */
```
- **Líneas 1 a 8**: Comentario multilínea. En C/C++ todo lo que esté entre `/*` y `*/` es ignorado por el compilador. Es el equivalente a las triples comillas en Python (`""" comentario """`).

---

### Inclusión de librerías (Los `import` de C)

```c
10: // includes
11: #include <stdio.h>
12: #include <stdlib.h>
13: #include <time.h>
14: #include <cuda_runtime.h>
15: #include "propiedades.h"
```

En Python usas `import modulo`. En C usas `#include <archivo.h>`, que literalmente copia y pega el contenido de ese archivo de cabecera antes de compilar:
- **Línea 11 (`#include <stdio.h>`)**: *Standard Input/Output*. Contiene funciones básicas de consola como `printf()` (el `print()` de C) y `getchar()` (el `input()` de C).
- **Línea 12 (`#include <stdlib.h>`)**: *Standard Library*. Contiene la función `malloc()` para pedir memoria RAM, `free()` para devolverla, y `rand()` para generar números aleatorios.
- **Línea 13 (`#include <time.h>`)**: Contiene funciones para medir el tiempo y leer el reloj del sistema (`time()`), que usaremos para que los números aleatorios cambien en cada ejecución.
- **Línea 14 (`#include <cuda_runtime.h>`)**: La librería fundamental de NVIDIA CUDA. Nos da acceso a todas las funciones que empiezan por `cuda...` (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetDeviceCount`, etc.).
- **Línea 15 (`#include "propiedades.h"`)**: Comillas dobles `""` en vez de `<>` indican que es un archivo local creado por nosotros (en la misma carpeta del proyecto). Aquí guardamos la función que imprime los datos de tu RTX 5060 Ti.

---

### Constantes Globales

```c
17: #define N 8
```
- **Línea 17**: `#define` es una directiva del preprocesador. Le dice al compilador: *"Cada vez que veas la letra `N` en el código, sustitúyela por el número `8` antes de compilar"*.
- **En Python**: Sería el equivalente a escribir una constante global en mayúsculas: `N = 8`.

---

### La función `main`: El punto de entrada

```c
20: // MAIN: rutina principal ejecutada en el host
21: int main(int argc, char** argv)
22: {
```
- **Línea 21**: En C, el programa **siempre** empieza a ejecutarse en la función llamada `main`. 
  - En Python haces:
    ```python
    if __name__ == "__main__":
        # tu código aquí
    ```
  - En C es obligatorio definir `int main(...)`.
  - `int`: Significa que la función `main` devolverá un número entero al sistema operativo cuando termine (típicamente `0` significa *"todo terminó con éxito"* y cualquier otro número como `1` significa *"hubo un error"*).
  - `int argc, char** argv`: Son los argumentos que se le pueden pasar al programa si lo ejecutas desde la consola (ej. `programa.exe argumento1`). No los usamos en este ejercicio, pero se ponen por convenio estándar.
- **Línea 22 (`{`)**: En Python usas sangrías (indentación) con tabuladores para abrir bloques. En C/C++ los bloques de código se abren con una llave `{` y se cierran con `}`.

---

### Paso 1: Comprobación del Hardware CUDA

```c
23: 	// 1. Busqueda y presentacion de propiedades del dispositivo
24: 	int deviceCount = 0;
25: 	cudaError_t err = cudaGetDeviceCount(&deviceCount);
```
- **Línea 24**: Creamos una variable entera llamada `deviceCount` e inicializada a 0. Aquí guardaremos cuántas tarjetas gráficas compatibles con CUDA tiene tu ordenador.
- **Línea 25**: 
  - Llamamos a `cudaGetDeviceCount(&deviceCount)`.
  - **¿Por qué lleva `&`?**: Porque en C las funciones no pueden devolver múltiples valores fácilmente como en Python (`count, err = get_devices()`). En su lugar, le pasamos a la función la **dirección de memoria** de nuestra variable (`&deviceCount`) para que la función de NVIDIA vaya a esa dirección y escriba directamente el resultado ahí dentro (en tu caso escribirá un `1`).
  - La función devuelve un código de estado de tipo `cudaError_t`. Si todo fue bien, devolverá `cudaSuccess` (un 0).

```c
26: 	if (err != cudaSuccess || deviceCount == 0)
27: 	{
28: 		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
29: 		printf("<pulsa [INTRO] para finalizar>");
30: 		getchar();
31: 		return 1;
32: 	}
```
- **Línea 26**: `!=` significa "distinto de" y `||` significa "O lógico" (el `or` de Python). Dice: *"Si hubo un error al hablar con el driver o el número de dispositivos es 0..."*.
- **Línea 28 y 29**: `printf` muestra texto por consola. El caracter especial `\n` representa un salto de línea (como pulsar Enter).
- **Línea 30 (`getchar()`)**: Espera a que el usuario pulse una tecla en el teclado. Esto evita que la ventana negra de la consola de Windows se cierre en un milisegundo antes de que te dé tiempo a leer el mensaje.
- **Línea 31 (`return 1`)**: Termina el programa inmediatamente avisando al sistema operativo de que ha habido un error (código 1).

```c
34: 	// Mostramos propiedades del dispositivo seleccionado (device 0)
35: 	propiedades_Device(0);
```
- **Línea 35**: Llama a la función que tenemos en [propiedades.h](file:///c:/Users/Alex/source/repos/Plantilla%20Cuda%2012.8/propiedades.h), pasándole el identificador `0` (la primera y única GPU de tu PC). Esto imprime por pantalla el bloque informativo con los 4.608 núcleos CUDA, la memoria VRAM y la arquitectura Blackwell.

---

### Paso 2: Declaración de Punteros

```c
37: 	// 2. Declaracion de punteros
38: 	float *hst_A = NULL;
39: 	float *hst_B = NULL;
40: 	float *dev_A = NULL;
41: 	float *dev_B = NULL;
42: 
43: 	size_t size = N * sizeof(float);
```
- **Líneas 38 a 41**:
  - Declaramos cuatro punteros a números en coma flotante (`float *`).
  - **Convención de nombres**:
    - `hst_` significa **Host** (vivirá en la memoria RAM de tu CPU).
    - `dev_` significa **Device** (vivirá en la memoria VRAM de tu GPU).
  - `NULL`: Es como el `None` de Python. Inicializamos los punteros a `NULL` (dirección 0) para que no apunten a basura de la memoria antes de utilizarlos.
- **Línea 43**:
  - `sizeof(float)`: En C, un número decimal `float` ocupa exactamente **4 bytes** en la memoria.
  - `N * sizeof(float)`: Si queremos guardar 8 floats, necesitamos `8 * 4 = 32 bytes` de memoria contigua.
  - `size_t`: Es simplemente un tipo de número entero positivo grande que se usa en C para representar tamaños en bytes.

---

### Paso 3: Reserva de Memoria en la CPU (Host)

```c
45: 	// 3. Reserva de memoria en el Host
46: 	hst_A = (float*)malloc(size);
47: 	hst_B = (float*)malloc(size);
```
- **Líneas 46 y 47**:
  - `malloc(size)`: Viene de *Memory Allocate*. Le pide al sistema operativo: *"Oye Windows, búscame un hueco libre de 32 bytes en la memoria RAM y devuélveme la dirección de memoria donde empieza"*.
  - `(float*)`: Se llama un *casting* (conversión de tipo). `malloc` devuelve memoria genérica (`void*`), y con `(float*)` le decimos al compilador que trate esa dirección como el inicio de un array de números `float`.
  - **En Python**: Sería el equivalente a hacer:
    ```python
    hst_A = [0.0] * N
    hst_B = [0.0] * N
    ```

```c
49: 	if (hst_A == NULL || hst_B == NULL)
50: 	{
51: 		printf("Error: Fallo al reservar memoria en el Host\n");
52: 		return 1;
53: 	}
```
- **Líneas 49 a 53**: Buenas prácticas en C. Si tu ordenador estuviera saturado y no le quedara ni 1 byte de RAM libre, `malloc` devolvería `NULL`. Aquí comprobamos que la reserva tuvo éxito antes de continuar.

---

### Paso 4: Reserva de Memoria en la GPU (Device)

```c
55: 	// 4. Reserva de memoria en la memoria global del Device
56: 	cudaMalloc((void**)&dev_A, size);
57: 	cudaMalloc((void**)&dev_B, size);
```
- **Líneas 56 y 57**: Aquí ocurre la magia de CUDA.
  - `cudaMalloc`: Es el hermano gemelo de `malloc`, pero en vez de pedirle memoria RAM a Windows, **le pide memoria VRAM al procesador gráfico de tu RTX 5060 Ti**.
  - **¿Por qué la sintaxis tan rara `(void**)&dev_A`?**:
    - `dev_A` ya es un puntero (`float*`), es decir, una variable pensada para guardar una dirección.
    - Pero para que la función `cudaMalloc` pueda modificar `dev_A` y escribir en él la dirección que ha reservado en la GPU, necesita saber la dirección de esa variable: `&dev_A` (un puntero a un puntero).
    - `(void**)` convierte ese doble puntero a genérico para que la API de CUDA lo acepte.
    - Le pasamos `size` (32 bytes). Al terminar esta línea, `dev_A` apunta a un bloque de 32 bytes dentro de los 8 GB de tu gráfica.

---

### Paso 5: Inicializar los datos en el Host

```c
59: 	// 5. Inicializacion del array hst_A en el host con numeros aleatorios en [0, 1]
60: 	srand((unsigned int)time(NULL));
61: 	for (int i = 0; i < N; i++)
62: 	{
63: 		hst_A[i] = (float)rand() / (float)RAND_MAX;
64: 		hst_B[i] = 0.0f; // Inicializamos hst_B a cero para comprobar la copia
65: 	}
```
- **Línea 60**: `time(NULL)` obtiene la hora actual en segundos. `srand(...)` usa esa hora como "semilla" para el generador de números aleatorios. 
  - **En Python**: Equivale a `random.seed()`. Si no pusieras esta línea, cada vez que abras el programa saldrían exactamente los mismos números.
- **Línea 61**: El bucle `for` clásico de C:
  - `int i = 0`: Empieza con la variable `i` valiendo 0.
  - `i < N`: Sigue repitiendo mientras `i` sea menor que 8 (es decir, de 0 a 7).
  - `i++`: En cada vuelta, suma 1 a `i` (equivalente a `i += 1` en Python).
  - **En Python**: Equivale a `for i in range(N):`.
- **Línea 63**:
  - `rand()` genera un número entero aleatorio gigantesco entre 0 y `RAND_MAX` (que suele ser 32767).
  - Al dividir `(float)rand() / (float)RAND_MAX`, obtenemos un número decimal comprendido entre `0.0` y `1.0` (por ejemplo `0.43`).
  - **En Python**: Equivale a `hst_A[i] = random.random()`.
- **Línea 64**: Ponemos todos los elementos de `hst_B` a cero absoluto (`0.0f`). Así, cuando más adelante veamos que `hst_B` tiene los mismos números que `hst_A`, estaremos 100% seguros de que la GPU los ha copiado de verdad y no estaban ahí por casualidad.

---

### Paso 6: El viaje de los datos (Las transferencias `cudaMemcpy`)

Este es el corazón del ejercicio: transferir datos según el esquema de la práctica:

```
[RAM del PC]               [VRAM de la RTX 5060 Ti]
   hst_A   ---(Paso 1)--->   dev_A
                               |
                           (Paso 2)
                               v
   hst_B   <--(Paso 3)---    dev_B
```

```c
67: 	// 6. Transferencias de memoria de acuerdo al esquema:
68: 	//    hst_A (Host) -> dev_A (Device)
69: 	//    dev_A (Device) -> dev_B (Device)
70: 	//    dev_B (Device) -> hst_B (Host)
71: 	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
72: 	cudaMemcpy(dev_B, dev_A, size, cudaMemcpyDeviceToDevice);
73: 	cudaMemcpy(hst_B, dev_B, size, cudaMemcpyDeviceToHost);
```

La función `cudaMemcpy` siempre recibe 4 argumentos:
`cudaMemcpy(DESTINO, ORIGEN, CUÁNTOS_BYTES, DIRECCIÓN_DEL_TRÁFICO);`

- **Línea 71 (Paso 1: CPU $\rightarrow$ GPU)**:
  - Destino: `dev_A` (GPU).
  - Origen: `hst_A` (CPU).
  - Sentido: `cudaMemcpyHostToDevice` (Del Host al Device).
  - *Los 8 números aleatorios viajan por el cable/bus PCIe hacia la tarjeta gráfica*.
- **Línea 72 (Paso 2: GPU $\rightarrow$ GPU)**:
  - Destino: `dev_B` (GPU).
  - Origen: `dev_A` (GPU).
  - Sentido: `cudaMemcpyDeviceToDevice` (De una zona de la GPU a otra zona de la GPU).
  - *La copia ocurre a velocidades astronómicas internamente dentro de los chips de memoria GDDR7 de tu gráfica*.
- **Línea 73 (Paso 3: GPU $\rightarrow$ CPU)**:
  - Destino: `hst_B` (CPU).
  - Origen: `dev_B` (GPU).
  - Sentido: `cudaMemcpyDeviceToHost` (Del Device de vuelta al Host).
  - *Los datos vuelven por el bus PCIe a la memoria RAM de tu ordenador y se guardan en `hst_B`*.

---

### Paso 7: Mostrar los resultados en pantalla

```c
75: 	// 7. Visualizacion de resultados
76: 	printf("\n===================================================\n");
77: 	printf("RESULTADOS DE LA TRANSFERENCIA (N = %d elementos):\n", N);
78: 	printf("===================================================\n");
79: 	printf("%-10s %-15s %-15s\n", "Indice", "hst_A (Origen)", "hst_B (Destino)");
80: 	printf("---------------------------------------------------\n");
81: 	for (int i = 0; i < N; i++)
82: 	{
83: 		printf("[%2d]       %-15.2f %-15.2f\n", i, hst_A[i], hst_B[i]);
84: 	}
85: 	printf("---------------------------------------------------\n");
```
- **Línea 77**: `%d` es un marcador de posición para un número entero (`int`). Le pasamos `N`, así que imprimirá: `RESULTADOS DE LA TRANSFERENCIA (N = 8 elementos):`.
- **Línea 79 y 83**: Secretos del formateo con `printf`:
  - `%-10s`: Imprime un texto (`s` = string) reservando 10 caracteres de ancho y alineándolo a la izquierda (el signo `-`).
  - `[%2d]`: Imprime el índice `i` ocupando al menos 2 huecos (para que el número `7` y el número `10` queden perfectamente alineados en columnas verticales).
  - `%-15.2f`: Imprime un número decimal (`f` = float) con **exactamente 2 cifras decimales** (`.2`), ocupando 15 espacios de ancho.
  - **En Python**: Equivale a usar f-strings formateadas:
    ```python
    print(f"[{i:2d}]       {hst_A[i]:-15.2f} {hst_B[i]:-15.2f}")
    ```

---

### Paso 8: Liberación de recursos (Limpieza obligatoria)

```c
87: 	// 8. Liberacion de memoria
88: 	free(hst_A);
89: 	free(hst_B);
90: 	cudaFree(dev_A);
91: 	cudaFree(dev_B);
```
- **Líneas 88 y 89 (`free`)**:
  - En C **no hay recolector de basura automático**. Cada vez que llamas a `malloc()`, estás "secuestrando" un trozo de RAM del sistema.
  - Con `free(hst_A)` y `free(hst_B)` le dices al sistema operativo: *"Ya he terminado con esta memoria, márcala como disponible para otros programas"*.
- **Líneas 90 y 91 (`cudaFree`)**:
  - Hace exactamente lo mismo, pero con la memoria VRAM de la GPU.
  - Si un programa CUDA hace muchos `cudaMalloc` dentro de un bucle y nunca hace `cudaFree`, la tarjeta gráfica se quedará sin memoria (error `Out of Memory`).

---

### Paso 9: Finalización elegante

```c
93: 	// 9. Salida del programa con fecha
94: 	time_t fecha;
95: 	time(&fecha);
96: 	printf("\n***************************************************\n");
97: 	printf("Programa ejecutado el: %s", ctime(&fecha));
98: 	printf("***************************************************\n");
99: 	printf("<pulsa [INTRO] para finalizar>");
100: 	getchar();
101: 
102: 	return 0;
103: }
```
- **Líneas 94 y 95**: `time_t fecha; time(&fecha);` lee la hora y fecha del sistema operativo.
- **Línea 97**: `ctime(&fecha)` convierte esa fecha numérica en una frase legible para humanos en inglés (ej: `Thu Sep 17 18:30:49 2026\n`).
- **Línea 100 (`getchar()`)**: Pausa el programa hasta que el usuario presione la tecla Enter.
- **Línea 102 (`return 0`)**: Informa a Windows de que la función `main` ha concluido con éxito (código de salida 0).

---

## 4. Resumen: ¿Cómo se traduciría todo este programa a Python conceptual?

Para que termines de conectar todos los puntos en tu cabeza, este programa de C/CUDA hace exactamente lo mismo que este hipotético código en Python:

```python
import random
import time

N = 8

# 1. Comprobar que hay GPU y mostrar sus datos
print("DEVICE 0: NVIDIA GeForce RTX 5060 Ti (4608 núcleos CUDA)")

# 2. Reservar e inicializar en CPU (Host)
hst_A = [random.random() for _ in range(N)]
hst_B = [0.0] * N

# 3. Enviar a GPU, moverlo internamente y traerlo de vuelta
# (En librerías de Python como PyTorch se haría con .to('cuda') y .to('cpu'))
dev_A = copiar_a_gpu(hst_A)         # cudaMemcpyHostToDevice
dev_B = copiar_en_gpu(dev_A)         # cudaMemcpyDeviceToDevice
hst_B = descargar_de_gpu(dev_B)      # cudaMemcpyDeviceToHost

# 4. Imprimir para verificar que hst_A == hst_B
print("Indice     hst_A (Origen)  hst_B (Destino)")
for i in range(N):
    print(f"[{i:2d}]       {hst_A[i]:-15.2f} {hst_B[i]:-15.2f}")

# 5. En Python la memoria se limpia sola al terminar
print("Programa finalizado con éxito.")
```

Con esta base, ¡ya tienes los conceptos clave para afrontar los siguientes ejercicios de CUDA (donde empezaremos a ejecutar código en paralelo dentro de la propia GPU usando *Kernels*)!
