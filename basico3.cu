/**
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 *
 * EJERCICIO: "BÁSICO 3: Hilos y Bloques"
 * Lanzar un kernel con múltiples hilos o threads agrupados en bloques.
 *
 * Alumno: Alejandro Muñoz Pérez
 * Fecha: 28/09/2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>


// Tamaño fijo de bloque (10 hilos por bloque según especificación)
#define HILOS_POR_BLOQUE 10

///////////////////////////////////////////////////////////////////////////
// KERNEL: Funcion ejecutada en la GPU (Device)
__global__ void suma_GPU(int* vector_1, int* vector_2, int* vector_suma, int n)
{
	// Identificador global unívoco del hilo a lo largo de toda la malla (grid)
	int id = threadIdx.x + blockDim.x * blockIdx.x;

	// Cláusula de seguridad: comprueba que el índice no supere el tamaño del vector
	if (id < n)
	{
		// El segundo vector se genera en el device invirtiendo el primer vector
		vector_2[id] = vector_1[n - 1 - id];

		// Suma de ambos vectores elemento a elemento
		vector_suma[id] = vector_1[id] + vector_2[id];
	}
}

///////////////////////////////////////////////////////////////////////////
// HOST: Muestra las propiedades del dispositivo con limites por eje (Figura 3.1)
void mostrar_Propiedades(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;

	switch (major)
	{
	case 1:
		cudaCores = 8;
		break;
	case 2:
		cudaCores = (minor == 0) ? 32 : 48;
		break;
	case 3:
		cudaCores = 192;
		break;
	case 5:
		cudaCores = 128;
		break;
	case 6:
		cudaCores = 64;
		break;
	case 7:
		cudaCores = 64;
		break;
	case 8:
		cudaCores = (minor != 9) ? 64 : 128;
		break;
	case 9:
		cudaCores = 128;
		break;
	case 10:
	case 12:
		cudaCores = 128;
		break;
	default:
		cudaCores = 128;
		break;
	}

	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> Capacidad de Computo            : %d.%d\n", major, minor);
	printf("> No. de MultiProcesadores        : %d\n", SM);
	printf("> No. de CUDA Cores (%dx%d)       : %d\n", cudaCores, SM, cudaCores * SM);
	printf("> No. maximo de Hilos (por bloque): %d\n", deviceProp.maxThreadsPerBlock);
	printf("  [eje x -> %d]\n", deviceProp.maxThreadsDim[0]);
	printf("  [eje y -> %d]\n", deviceProp.maxThreadsDim[1]);
	printf("  [eje z -> %d]\n", deviceProp.maxThreadsDim[2]);
	printf("> No. maximo de Bloques (por eje):\n");
	printf("  [eje x -> %d]\n", deviceProp.maxGridSize[0]);
	printf("  [eje y -> %d]\n", deviceProp.maxGridSize[1]);
	printf("  [eje z -> %d]\n", deviceProp.maxGridSize[2]);
	printf("***************************************************\n");
}

///////////////////////////////////////////////////////////////////////////
// MAIN: Funcion principal ejecutada en el Host
int main(int argc, char** argv)
{
	int deviceID = 0;

	// 1. Mostrar propiedades del dispositivo seleccionado y limites por eje
	mostrar_Propiedades(deviceID);

	// 2. Lectura interactiva de N con validacion
	int N = 0;
	while (1)
	{
		printf("  Introduce el numero de elementos: ");
		if (scanf("%d", &N) != 1)
		{
			while (getchar() != '\n');
			continue;
		}

		if (N <= 0)
		{
			printf("> ERROR: el numero de elementos debe ser mayor que 0!\n");
		}
		else
		{
			break;
		}
	}

	// Limpiar el salto de linea residual en el buffer de entrada
	while (getchar() != '\n');

	int hilosB = HILOS_POR_BLOQUE;
	int Nbloques = N / hilosB;
	if (N % hilosB != 0)
	{
		Nbloques = Nbloques + 1;
	}

	printf("> Vector de %d elementos\n", N);
	printf("> Lanzamiento con %d bloques de %d hilos (%d hilos)\n", Nbloques, hilosB, Nbloques * hilosB);

	size_t size = N * sizeof(int);

	// 4. Reserva de memoria en el Host
	int* hst_vector1 = (int*)malloc(size);
	int* hst_vector2 = (int*)malloc(size);
	int* hst_resultado = (int*)malloc(size);

	// 5. Reserva de memoria en el Device
	int* dev_vector1 = NULL;
	int* dev_vector2 = NULL;
	int* dev_resultado = NULL;
	cudaMalloc((void**)&dev_vector1, size);
	cudaMalloc((void**)&dev_vector2, size);
	cudaMalloc((void**)&dev_resultado, size);

	// 6. Inicializacion del primer vector en el Host con enteros aleatorios [0, 9]
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++)
	{
		hst_vector1[i] = rand() % 10;
	}

	// 7. Copia del vector 1 desde el Host hacia el Device
	cudaMemcpy(dev_vector1, hst_vector1, size, cudaMemcpyHostToDevice);

	// 8. Lanzamiento del Kernel con Nbloques y hilosB
	suma_GPU<<<Nbloques, hilosB>>>(dev_vector1, dev_vector2, dev_resultado, N);



	// 9. Copia de resultados desde el Device hacia el Host
	cudaMemcpy(hst_vector2, dev_vector2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, size, cudaMemcpyDeviceToHost);

	// 10. Mostrar vectores iniciales y resultado final
	printf("VECTOR 1:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_vector1[i]);
	}
	printf("\n");

	printf("VECTOR 2:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_vector2[i]);
	}
	printf("\n");

	printf("SUMA:\n");
	for (int i = 0; i < N; i++)
	{
		printf("%2d ", hst_resultado[i]);
	}
	printf("\n");

	// 11. Liberacion de memoria
	free(hst_vector1);
	free(hst_vector2);
	free(hst_resultado);
	cudaFree(dev_vector1);
	cudaFree(dev_vector2);
	cudaFree(dev_resultado);

	// 12. Finalizacion
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();

	return 0;
}