# Guía Definitiva y Explicación Línea por Línea: `basico4.cu`
### *Arrays Multidimensionales: Organización Espacial de Hilos (2D) y Aplanamiento de Matrices en Memoria*

---

## 1. El Salto Conceptual: Del Vector Unidimensional (1D) a la Matriz Bidimensional (2D)

En las prácticas anteriores (**BÁSICO 1, 2 y 3**) organizamos los hilos a lo largo de un único eje: el **eje $x$**. Esto resultaba natural porque los datos a procesar eran vectores lineales (arrays 1D).

Sin embargo, en computación científica, procesamiento de imágenes, redes neuronales y gráficos por ordenador, los problemas suelen estructurarse de manera natural en **dos o tres dimensiones** (matrices, imágenes píxel a píxel, mallas volumétricas).

```
          MATRIZ 2D (Concepto Lógico)                  DISTRIBUCIÓN DE HILOS 2D
       Columna 0   Columna 1   Columna 2            (threadIdx.x = Columna, threadIdx.y = Fila)
      +-----------+-----------+-----------+              +------------+------------+------------+
Fila 0|   (0,0)   |   (0,1)   |   (0,2)   |        Fila 0| Hilo (0,0) | Hilo (1,0) | Hilo (2,0) |
      +-----------+-----------+-----------+              +------------+------------+------------+
Fila 1|   (1,0)   |   (1,1)   |   (1,2)   |        Fila 1| Hilo (0,1) | Hilo (1,1) | Hilo (2,1) |
      +-----------+-----------+-----------+              +------------+------------+------------+
Fila 2|   (2,0)   |   (2,1)   |   (2,2)   |        Fila 2| Hilo (0,2) | Hilo (1,2) | Hilo (2,2) |
      +-----------+-----------+-----------+              +------------+------------+------------+
```

### ¿Por qué CUDA permite lanzar hilos en 2D o 3D si la memoria física sigue siendo lineal?
1. **Claridad algorítmica**: Permite que el programador piense directamente en coordenadas de fila y columna $(y, x)$ sin tener que calcular manualmente cómo repartir un número total de hilos.
2. **Coherencia con el hardware**: Permite optimizar el acceso a memoria y la correspondencia con la topología de los datos.

> [!NOTE]
> **Limitación de este ejercicio (BÁSICO 4)**:
> El enunciado especifica lanzar **un solo bloque bidimensional**. Al ser un único bloque, estamos sujetos a la restricción física del hardware: **el producto $F \times C$ no puede superar los 1024 hilos** (`maxThreadsPerBlock = 1024`). En ejercicios posteriores se combinarán múltiples bloques con hilos bidimensionales para procesar matrices de cualquier tamaño.

---

## 2. El Tipo de Dato `dim3` y el Sistema de Coordenadas de CUDA

CUDA introduce el tipo estructurado `dim3` para definir dimensiones tridimensionales enteras:
```c
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};
```
Cualquier componente no especificada toma automáticamente el valor por defecto de `1`.

### La Convención Fundamental: ¿Por qué el eje $x$ son las Columnas y el eje $y$ son las Filas?

En matemáticas y programación solemos nombrar las matrices como $M[\text{fila}][\text{columna}]$. Sin embargo, en CUDA asociamos:
* **Eje $x$ $\rightarrow$ Columnas** (ancho / coordenada horizontal)
* **Eje $y$ $\rightarrow$ Filas** (alto / coordenada vertical)

```c
dim3 hilosB(COLUMNAS, FILAS); // x = COLUMNAS, y = FILAS, z = 1
```

#### ¿Por qué esta asignación y no al revés?
Hay dos razones críticas de arquitectura:
1. **Orden en memoria de C/C++ (*Row-Major Order*)**: Los elementos contiguos de una misma fila están juntos en memoria. Avanzar en una columna contigua significa avanzar exactamente 1 posición de memoria (`sizeof(int)` bytes).
2. **Warps de la GPU y Coalescencia de Memoria (*Memory Coalescing*)**:
   - La GPU agrupa los hilos de un bloque en paquetes de 32 llamados **warps**.
   - Los hilos de un warp se forman con hilos contiguos en el **eje $x$** (`threadIdx.x` de 0 a 31).
   - Si el eje $x$ es la columna, los 32 hilos del warp leerán 32 enteros consecutivos en memoria física, permitiendo al controlador de memoria de la GPU realizar una **única transacción combinada ultrarrápida (acceso coalescente)**. Si invirtiéramos los ejes, cada hilo leería una fila distinta, saltando en memoria y destruyendo el rendimiento.

---

## 3. Mapeo de Memoria: De la Matriz 2D al Vector Lineal 1D

Tanto `malloc()` en la CPU como `cudaMalloc()` en la GPU reservan un **bloque lineal y continuo de bytes**. No existe en la memoria física una cuadrícula 2D con filas separadas.

Para una matriz de $F = 3$ filas y $C = 4$ columnas (12 elementos en total):

```
VISTA MATRICIAL 2D:
            Col 0   Col 1   Col 2   Col 3
  Fila 0: [  (0,0),  (0,1),  (0,2),  (0,3) ]
  Fila 1: [  (1,0),  (1,1),  (1,2),  (1,3) ]
  Fila 2: [  (2,0),  (2,1),  (2,2),  (2,3) ]

VISTA EN MEMORIA FÍSICA LINEAL:
 Índice:   0      1      2      3   |   4      5      6      7   |   8      9     10     11
 Elemento:(0,0)  (0,1)  (0,2)  (0,3)| (1,0)  (1,1)  (1,2)  (1,3)| (2,0)  (2,1)  (2,2)  (2,3)
           <----- Fila 0 ----->       <----- Fila 1 ----->       <----- Fila 2 ----->
```

### La Fórmula de Indexación Lineal

Para encontrar la posición en el array plano correspondiente a la celda $(\text{fila}, \text{columna})$, aplicamos:

$$\mathbf{globalID} = \mathbf{columna} + (\mathbf{fila} \times \mathbf{COLUMNAS})$$

* Cada fila completa que saltamos ocupa exactamente $\mathbf{COLUMNAS}$ elementos en memoria.
* Dentro de esa fila, nos desplazamos $\mathbf{columna}$ posiciones.

---

## 4. Análisis Detallado del Código Línea por Línea

### 4.1. El Kernel CUDA: `MatFinal`

```cuda
__global__ void MatFinal(int* entrada, int* salida, int columnas)
{
    // Indice de columna: EJE x
    int columna = threadIdx.x;
    // Indice de fila: EJE y
    int fila = threadIdx.y;

    // Indice lineal en memoria para acceder a la matriz plana
    int globalID = columna + fila * columnas;

    // Mantenemos las columnas pares y asignamos 0 en el resto
    if (columna % 2 == 0)
    {
        salida[globalID] = entrada[globalID];
    }
    else
    {
        salida[globalID] = 0;
    }
}
```

* `__global__`: Especificador que indica que la función se compila para la GPU (*Device*) y se invoca desde la CPU (*Host*).
* `threadIdx.x`: Identificador del hilo a lo largo del ancho del bloque. Corresponde al número de columna (de `0` a `COLUMNAS - 1`).
* `threadIdx.y`: Identificador del hilo a lo largo del alto del bloque. Corresponde al número de fila (de `0` a `FILAS - 1`).
* `columna % 2 == 0`: Operación módulo.
  - Columnas `0, 2, 4, 6...`: Condición verdadera $\rightarrow$ copia el valor original: `salida[globalID] = entrada[globalID]`.
  - Columnas `1, 3, 5, 7...`: Condición falsa $\rightarrow$ se asigna cero: `salida[globalID] = 0`.

---

### 4.2. Función `main`: Preparación del Host y Lanzamiento

#### Comprobación de dispositivos y propiedades
```cuda
int deviceCount = 0;
cudaGetDeviceCount(&deviceCount);
if (deviceCount == 0) { ... return 1; }
else { propiedades_Device(0); }
```
Verifica que el equipo dispone de hardware CUDA compatible antes de intentar reservar memoria o lanzar kernels, mostrando los límites físicos del dispositivo.

#### Lectura de Dimensiones
```cuda
int FILAS, COLUMNAS;
printf("Introduce el numero de filas: ");
scanf("%d", &FILAS);
printf("Introduce el numero de columnas: ");
scanf("%d", &COLUMNAS);
getchar();
```
Lectura directa por teclado de las dimensiones de la matriz requeridas por el enunciado.

#### Reserva de Memoria
```cuda
size_t size = totalElementos * sizeof(int);

int* hst_Entrada = (int*)malloc(size);
int* hst_Salida  = (int*)malloc(size);

int* dev_Entrada = NULL;
int* dev_Salida  = NULL;
cudaMalloc((void**)&dev_Entrada, size);
cudaMalloc((void**)&dev_Salida, size);
```
Se reservan arrays unidimensionales contiguos tanto en la RAM del Host como en la VRAM de la GPU.

#### Inicialización Aleatoria [1..9]
```cuda
srand((unsigned int)time(NULL));
for (int i = 0; i < totalElementos; i++)
{
    hst_Entrada[i] = 1 + rand() % 9;
    hst_Salida[i]  = 0;
}
```
Genera números enteros pseudoaleatorios en el rango cerrado $[1, 9]$ cumpliendo estrictamente con el enunciado.

#### Configuración de la Geometría y Lanzamiento del Kernel
```cuda
dim3 Nbloques(1);
dim3 hilosB(COLUMNAS, FILAS);

printf("> KERNEL de 1 BLOQUE con %d HILOS:\n", totalElementos);
printf("  eje x -> %2d hilos\n  eje y -> %2d hilos\n", COLUMNAS, FILAS);

MatFinal<<<Nbloques, hilosB>>>(dev_Entrada, dev_Salida, COLUMNAS);
cudaDeviceSynchronize();
```
* `Nbloques(1)`: Se lanza una malla de 1 único bloque ($1 \times 1 \times 1$).
* `hilosB(COLUMNAS, FILAS)`: El bloque tiene dimensiones $x = \text{COLUMNAS}$, $y = \text{FILAS}$, $z = 1$.
* Cada hilo procesa exactamente una posición de la matriz sin necesidad de bucles dentro del kernel.

#### Impresión de Resultados en Formato Matricial
```cuda
for (int i = 0; i < FILAS; i++)
{
    for (int j = 0; j < COLUMNAS; j++)
    {
        printf("%2d ", hst_Entrada[j + i * COLUMNAS]);
    }
    printf("\n");
}
```
Recorre la memoria plana reconstruyendo la visualización bidimensional mediante saltos de línea al terminar cada fila.

#### Finalización del Programa
```cuda
time_t fecha;
time(&fecha);
printf("***************************************************\n");
printf("Programa ejecutado el: %s", ctime(&fecha));
printf("<pulsa [INTRO] para finalizar>");
getchar();
return 0;
```
Muestra la marca de tiempo de la ejecución y espera una pulsación de tecla antes de terminar. Al finalizar el proceso, el sistema operativo y el driver de NVIDIA reclaman automáticamente toda la memoria reservada.

---

## 5. Traza Numérica de Ejemplo

Imaginemos una ejecución con $F = 3$ filas y $C = 5$ columnas ($15$ hilos en total):

### Matriz Original Inicializada en el Host:
```text
           Col 0   Col 1   Col 2   Col 3   Col 4
 Fila 0:     7       3       8       1       9
 Fila 1:     4       6       2       5       7
 Fila 2:     9       8       3       4       2
```

### Ejecución de los Hilos en la GPU:
* Los hilos con `threadIdx.x == 0, 2, 4` (columnas pares) copian su dato correspondiente.
* Los hilos con `threadIdx.x == 1, 3` (columnas impares) escriben `0`.

### Matriz Final Obtenida:
```text
           Col 0   Col 1   Col 2   Col 3   Col 4
 Fila 0:     7       0       8       0       9
 Fila 1:     4       0       2       0       7
 Fila 2:     9       0       3       0       2
```

---

## 6. Resumen de Puntos Clave para Exámenes y Prácticas

1. **`dim3`**: Tipo de tres enteros `(x, y, z)`. Si no se especifican $y$ o $z$, valen $1$.
2. **Eje $x$ = Columnas, Eje $y$ = Filas**: Fundamental para preservar el orden contiguo en memoria y garantizar accesos coalescentes en los warps.
3. **Indexación plana**: $\text{índice} = x + (y \times \text{ancho})$.
4. **Límite de un único bloque**: $F \times C \le 1024$.
