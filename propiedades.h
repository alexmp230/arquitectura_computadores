#ifndef PROPIEDADES_H
#define PROPIEDADES_H

#include <stdio.h>
#include <cuda_runtime.h>

///////////////////////////////////////////////////////////////////////////
// HOST: Muestra las caracteristicas principales del dispositivo CUDA
__host__ inline void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);

	int cudaCores;
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
		if (minor != 9) {
			archName = "AMPERE";
			cudaCores = 64;
		} else {
			archName = "ADA LOVELACE";
			cudaCores = 128;
		}
		break;
	case 9:
		archName = "HOPPER";
		cudaCores = 128;
		break;
	case 10:
	case 12:
		// BLACKWELL (10.x Datacenter / 12.x GeForce RTX serie 50)
		archName = "BLACKWELL";
		cudaCores = 128;
		break;
	default:
		archName = "DESCONOCIDA";
		cudaCores = 0;
	}

	int rtV;
	cudaRuntimeGetVersion(&rtV);

	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA \t: %s\n", archName);
	printf("> Capacidad de Computo \t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> Memoria Global (total) \t: %llu MiB\n",
		(unsigned long long)(deviceProp.totalGlobalMem / (1024 * 1024)));
	printf("***************************************************\n");
}

#endif // PROPIEDADES_H
