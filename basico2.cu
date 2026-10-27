/**
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 *
 * EJERCICIO: "BÁSICO 2: Función kernel"
 * Objetivo: Lanzar un kernel (función ejecutada en la tarjeta gráfica) de un solo bloque con
 * múltiples hilos o threads.
 *
 * Alumno: Alejandro Muñoz Pérez
 * Fecha: 25/09/2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "propiedades.h"

///////////////////////////////////////////////////////////////////////////
// KERNEL: Funcion ejecutada en la GPU (Device)
__global__ void suma_GPU(int* vector_1, int* vector_2, int* vector_suma, int n)
{
	// Identificador del hilo dentro del bloque
	int id = threadIdx.x;

	if (id < n)
	{
		// El segundo vector se genera en el device invirtiendo el primer vector
		vector_2[id] = vector_1[n - 1 - id];

		// Suma de ambos vectores
		vector_suma[id] = vector_1[id] + vector_2[id];
	}
}

///////////////////////////////////////////////////////////////////////////
// MAIN: Funcion principal ejecutada en el Host
int main(int argc, char** argv)
{
	int deviceID = 0;

	// 1. Mostrar propiedades del dispositivo seleccionado
	propiedades_Device(deviceID);

	// Obtener propiedades para conocer el limite maximo de hilos por bloque
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	int maxThreads = deviceProp.maxThreadsPerBlock;

	// 2. Lectura interactiva de N con validacion de hilos maximos
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
		else if (N > maxThreads)
		{
			printf("> ERROR: numero maximo de hilos superado! [%d hilos]\n", maxThreads);
		}
		else
		{
			break;
		}
	}

	// Limpiar el salto de linea residual en el buffer de entrada
	while (getchar() != '\n');

	printf("> Vector de %d elementos\n", N);
	printf("> Lanzamiento con 1 bloque de %d hilos\n", N);

	size_t size = N * sizeof(int);

	// 3. Reserva de memoria en el Host
	int* hst_vector1 = (int*)malloc(size);
	int* hst_vector2 = (int*)malloc(size);
	int* hst_resultado = (int*)malloc(size);

	// 4. Reserva de memoria en el Device
	int* dev_vector1 = NULL;
	int* dev_vector2 = NULL;
	int* dev_resultado = NULL;
	cudaMalloc((void**)&dev_vector1, size);
	cudaMalloc((void**)&dev_vector2, size);
	cudaMalloc((void**)&dev_resultado, size);

	// 5. Inicializacion del primer vector en el Host con enteros aleatorios [0, 9]
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++)
	{
		hst_vector1[i] = rand() % 10;
	}

	// 6. Copia del vector 1 desde el Host hacia el Device
	cudaMemcpy(dev_vector1, hst_vector1, size, cudaMemcpyHostToDevice);

	// 7. Lanzamiento del Kernel (1 bloque con N hilos)
	suma_GPU<<<1, N>>>(dev_vector1, dev_vector2, dev_resultado, N);

	// Sincronizacion para asegurar la finalizacion del kernel
	cudaDeviceSynchronize();

	// 8. Copia de resultados desde el Device hacia el Host
	cudaMemcpy(hst_vector2, dev_vector2, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, size, cudaMemcpyDeviceToHost);

	// 9. Mostrar vectores iniciales y resultado final
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

	// 10. Liberacion de memoria
	free(hst_vector1);
	free(hst_vector2);
	free(hst_resultado);
	cudaFree(dev_vector1);
	cudaFree(dev_vector2);
	cudaFree(dev_resultado);

	// 11. Finalizacion
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();

	return 0;
}