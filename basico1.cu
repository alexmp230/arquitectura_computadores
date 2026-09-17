/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* EJERCICIO: "BÁSICO 1: Memoria Global"
* >> Transferencia de datos entre Host y Device (Global Memory)
*
*/
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "propiedades.h"

#define N 8

///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// 1. Busqueda y presentacion de propiedades del dispositivo
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}

	// Mostramos propiedades del dispositivo seleccionado (device 0)
	propiedades_Device(0);

	// 2. Declaracion de punteros
	float *hst_A = NULL;
	float *hst_B = NULL;
	float *dev_A = NULL;
	float *dev_B = NULL;

	size_t size = N * sizeof(float);

	// 3. Reserva de memoria en el Host
	hst_A = (float*)malloc(size);
	hst_B = (float*)malloc(size);

	if (hst_A == NULL || hst_B == NULL)
	{
		printf("Error: Fallo al reservar memoria en el Host\n");
		return 1;
	}

	// 4. Reserva de memoria en la memoria global del Device
	cudaMalloc((void**)&dev_A, size);
	cudaMalloc((void**)&dev_B, size);

	// 5. Inicializacion del array hst_A en el host con numeros aleatorios en [0, 1]
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++)
	{
		hst_A[i] = (float)rand() / (float)RAND_MAX;
		hst_B[i] = 0.0f; // Inicializamos hst_B a cero para comprobar la copia
	}

	// 6. Transferencias de memoria de acuerdo al esquema:
	//    hst_A (Host) -> dev_A (Device)
	//    dev_A (Device) -> dev_B (Device)
	//    dev_B (Device) -> hst_B (Host)
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, dev_A, size, cudaMemcpyDeviceToDevice);
	cudaMemcpy(hst_B, dev_B, size, cudaMemcpyDeviceToHost);

	// 7. Visualizacion de resultados
	printf("\n===================================================\n");
	printf("RESULTADOS DE LA TRANSFERENCIA (N = %d elementos):\n", N);
	printf("===================================================\n");
	printf("%-10s %-15s %-15s\n", "Indice", "hst_A (Origen)", "hst_B (Destino)");
	printf("---------------------------------------------------\n");
	for (int i = 0; i < N; i++)
	{
		printf("[%2d]       %-15.2f %-15.2f\n", i, hst_A[i], hst_B[i]);
	}
	printf("---------------------------------------------------\n");

	// 8. Liberacion de memoria
	free(hst_A);
	free(hst_B);
	cudaFree(dev_A);
	cudaFree(dev_B);

	// 9. Salida del programa con fecha
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el: %s", ctime(&fecha));
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();

	return 0;
}
