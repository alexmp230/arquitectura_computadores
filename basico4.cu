/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
* 
* Basico 4: "Arrays multidimensionales" 
* Utilizar los hilos de ejecución de un kernel lanzado con dos dimensiones.
*
* Alumno: Alejandro Muñoz Pérez
* Fecha: 03/10/2026
*/

///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/////////////////////////////////////////////////////////////////////////// 
// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host
__host__ void propiedades_Device(); 

// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void MatFinal(int *entrada, int *salida, int COLUMNAS) 
{ 
	// KERNEL BIDIMENSIONAL DE UN SOLO BLOQUE: (X,Y)
	int columna = threadIdx.x; 
	int fila = threadIdx.y; 
	int globalID = columna + fila * COLUMNAS; 

	if (columna % 2 == 0) 
	{ 
		salida[globalID] = entrada[globalID]; 
	} 
	else
	{ 
		salida[globalID] = 0; 
	} 
} 

///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv) 
{ 
	// buscando dispositivos
	int deviceCount; 
	cudaGetDeviceCount(&deviceCount); 
	if (deviceCount == 0) 
	{ 
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n"); 
		printf("<pulsa [INTRO] para finalizar>"); 
		getchar(); 
		return 1; 
	} 
	else
	{ 
		propiedades_Device(); 
	} 

	// dimensiones de la matriz
	int FILAS, COLUMNAS;
	printf("> Introduce el numero de filas: ");
	scanf("%d", &FILAS);
	printf("> Introduce el numero de columnas: ");
	scanf("%d", &COLUMNAS);
	getchar();

	// declaraciones
	int *hst_Entrada, *hst_Salida; 
	int *dev_Entrada, *dev_Salida; 

	// reserva en el host
	hst_Entrada = (int*)malloc(FILAS * COLUMNAS * sizeof(int)); 
	hst_Salida = (int*)malloc(FILAS * COLUMNAS * sizeof(int)); 

	// reserva en el device
	cudaMalloc((void**)&dev_Entrada, FILAS * COLUMNAS * sizeof(int)); 
	cudaMalloc((void**)&dev_Salida, FILAS * COLUMNAS * sizeof(int)); 

	// inicializacion
	for (int i = 0; i < FILAS * COLUMNAS; i++) 
	{ 
		hst_Entrada[i] = 1 + rand() % 9; 
		hst_Salida[i] = 0; 
	} 

	// copia de datos hacia el device
	cudaMemcpy(dev_Entrada, hst_Entrada, FILAS * COLUMNAS * sizeof(int), cudaMemcpyHostToDevice); 

	// dimensiones del kernel
	dim3 Nbloques(1); 
	dim3 hilosB(COLUMNAS, FILAS); 

	// numero de hilos
	printf("> KERNEL de 1 BLOQUE con %d HILOS:\n", COLUMNAS * FILAS); 
	printf("  eje x -> %2d hilos\n  eje y -> %2d hilos\n", COLUMNAS, FILAS); 

	// llamada al kernel
	MatFinal<<<Nbloques, hilosB>>>(dev_Entrada, dev_Salida, COLUMNAS); 

	// recogida de datos desde el device
	cudaMemcpy(hst_Salida, dev_Salida, FILAS * COLUMNAS * sizeof(int), cudaMemcpyDeviceToHost); 

	// impresion de resultados
	printf("> MATRIZ ORIGINAL:\n"); 
	for (int i = 0; i < FILAS; i++) 
	{ 
		for (int j = 0; j < COLUMNAS; j++) 
		{ 
			printf("%2d ", hst_Entrada[j + i * COLUMNAS]); 
		} 
		printf("\n"); 
	} 
	printf("\n"); 

	printf("> MATRIZ FINAL:\n"); 
	for (int i = 0; i < FILAS; i++) 
	{ 
		for (int j = 0; j < COLUMNAS; j++) 
		{ 
			printf("%2d ", hst_Salida[j + i * COLUMNAS]); 
		} 
		printf("\n"); 
	} 

	// salida del programa
	time_t fecha; 
	time(&fecha); 
	printf("***************************************************\n"); 
	printf("Programa ejecutado el: %s", ctime(&fecha)); 
	printf("<pulsa [INTRO] para finalizar>"); 
	getchar(); 

	return 0; 
} 

///////////////////////////////////////////////////////////////////////////
//    Función que imprime las propiedades de mi gráfica
__host__ void propiedades_Device() 
{ 
	int deviceID; 
	cudaGetDevice(&deviceID); 

	int runtimeVersion; 
	cudaRuntimeGetVersion(&runtimeVersion); 

	cudaDeviceProp deviceProp; 
	cudaGetDeviceProperties(&deviceProp, deviceID); 

	int cudaCores = 0; 
	int SM = deviceProp.multiProcessorCount; 
	int major = deviceProp.major; 
	int minor = deviceProp.minor; 
	const char *archName; 

	switch (major) 
	{ 
	case 1: 
		archName = "TESLA"; 
		cudaCores = 8; 
		break; 
	case 2: 
		archName = "FERMI"; 
		if (minor == 0) 
			cudaCores = 32; 
		else
			cudaCores = 48; 
		break; 
	case 3: 
		archName = "KEPLER"; 
		cudaCores = 192; 
		break; 
	case 5: 
		archName = "MAXWELL"; 
		cudaCores = 128; 
		break; 
	case 6: 
		archName = "PASCAL"; 
		cudaCores = 64; 
		break; 
	case 7: 
		cudaCores = 64; 
		if (minor == 0) 
			archName = "VOLTA"; 
		else
			archName = "TURING"; 
		break; 
	case 8: 
		archName = "AMPERE"; 
		cudaCores = 64; 
		break; 
	case 10:
	case 12:
		archName = "BLACKWELL";
		cudaCores = 128;
		break;
	default: 
		archName = "DESCONOCIDA"; 
		cudaCores = 0; 
	} 

	// presentacion de propiedades
	printf("***************************************************\n"); 
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name); 
	printf("***************************************************\n"); 
	printf("> CUDA Toolkit                         : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10); 
	printf("> Capacidad de Computo                 : %d.%d\n", major, minor); 
	printf("> Arquitectura CUDA                    : %s\n", archName); 
	printf("> No. de MultiProcesadores             : %d\n", SM); 
	printf("> No. de CUDA Cores (%dx%d)            : %d\n", cudaCores, SM, cudaCores * SM); 
	printf("> No. maximo de Hilos (por bloque)     : %d\n", deviceProp.maxThreadsPerBlock); 
	printf("> Memoria Global (total)               : %zu MiB\n", (size_t)(deviceProp.totalGlobalMem / (1024 * 1024))); 
	printf("***************************************************\n"); 
}
