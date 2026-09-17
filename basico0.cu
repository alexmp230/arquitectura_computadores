/*
* ARQUITECTURA DE COMPUTADORES
* 2º Grado en Ingenieria Informatica
*
* EJEMPLO: "Dispositivos CUDA"
* >> Propiedades de un dispositivo CUDA
*
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
__host__ void propiedades_Device(int deviceID);
/* El significado de la etiqueta __host__ se explica en el ejercicio "Básico 2" */
///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
/* Se puede prescindir de los argumentos "int argc" y "char** argv".
Solo se utilizan cuando se tiene previsto llamar al ejecutable .exe desde la consola de
comandos con parámetros adicionales (por ejemplo find –name archivo.txt en linux).
*/
{
	// Buscando dispositivos
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount);
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}

	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	// En algunos casos puede ser necesario usar la función getchar() para que el IDE de
	// VisualStudio no cierre la consola de salida hasta que hayamos pulsado una tecla:
	getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////
__host__ void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	// Cálculo del numero total de cores (SP)
	// dependiendo de la capacidad de computo de la GPU
	int cudaCores;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char *archName;
	switch (major)
	{
	case 1:
		//TESLA
		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		archName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA(7.0) //TURING(7.5)
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 8:
		//AMPERE(8.0) //ADA LOVELACE(8.9)
		if (minor != 9){
			archName = "AMPERE";
			cudaCores = 64;}
		else{
			archName = "ADA LOVELACE";
			cudaCores = 128;}
		break;
	case 9:
		//HOPPER
		archName = "HOPPER";
		cudaCores = 128;
		break;
	case 10:
	case 12:
		//BLACKWELL (10.x Datacenter / 12.x GeForce RTX serie 50)
		archName = "BLACKWELL";
		cudaCores = 128;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		archName = "DESCONOCIDA";
		cudaCores = 0;
	}
	int rtV;
	cudaRuntimeGetVersion(&rtV);
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t\t: %d.%d\n", rtV/1000,(rtV%1000)/10);
	printf("> Arquitectura CUDA \t: %s\n", archName);
	printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t: %d\n", cudaCores, SM, cudaCores*SM);
	printf("> Memoria Global (total) \t: %llu MiB\n",
		(unsigned long long)(deviceProp.totalGlobalMem / (1024 * 1024)));
	printf("***************************************************\n");
}
///////////////////////////////////////////////////////////////////////////