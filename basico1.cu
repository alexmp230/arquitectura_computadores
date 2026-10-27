/**
 * ARQUITECTURA DE COMPUTADORES
 * 2º Grado en Ingenieria Informatica
 *
 * EJERCICIO: "BÁSICO 1: Memoria Global"
 * Utilizar la memoria global del device para leer y escribir datos.
 *
 * Alumno: Alejandro Muñoz Pérez
 * Fecha: 18/09/2026
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "propiedades.h"

 ///////////////////////////////////////////////////////////////////////////
 // defines

#define N 8

///////////////////////////////////////////////////////////////////////////
// declaracion de funciones

int main(int argc, char** argv)
{
	// 1. Propiedades del dispositivo
	propiedades_Device(0);

	// 2. Reserva de memoria en Host
	float* hst_A = (float*)malloc(N * sizeof(float));
	float* hst_B = (float*)malloc(N * sizeof(float));

	// 3. Reserva de memoria en Device
	float* dev_A, * dev_B;
	cudaMalloc((void**)&dev_A, N * sizeof(float));
	cudaMalloc((void**)&dev_B, N * sizeof(float));

	// 4. Inicializacion de datos en el Host
	srand((unsigned int)time(NULL));
	for (int i = 0; i < N; i++)
	{
		hst_A[i] = (float)rand() / (float)RAND_MAX;
		hst_B[i] = 0.0f;
	}

	// 5. Transferencias: Host -> Device -> Device -> Host
	cudaMemcpy(dev_A, hst_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, dev_A, N * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(hst_B, dev_B, N * sizeof(float), cudaMemcpyDeviceToHost);

	// 6. Mostrar resultados
	printf("\nVector original (hst_A):\n");
	for (int i = 0; i < N; i++)
	{
		printf("%.2f ", hst_A[i]);
	}

	printf("\n\nVector copiado (hst_B):\n");
	for (int i = 0; i < N; i++)
	{
		printf("%.2f ", hst_B[i]);
	}
	printf("\n");

	// 7. Liberar memoria
	free(hst_A);
	free(hst_B);
	cudaFree(dev_A);
	cudaFree(dev_B);

	// 8. Salida del programa con fecha
	time_t fecha;
	time(&fecha);
	printf("\n***************************************************\n");
	printf("Programa ejecutado el: %s", ctime(&fecha));
	printf("***************************************************\n");
	printf("<pulsa [INTRO] para finalizar>");
	getchar();

	return 0;
}