# Entorno de Prácticas CUDA - Estructura de Computadores

Este repositorio contiene las prácticas y ejercicios de CUDA para la asignatura de **Estructura de Computadores** (2º Grado en Ingeniería Informática).

---

## 1. Especificaciones del Hardware y Entorno

En esta máquina se dispone de hardware moderno de última generación:

- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **Arquitectura CUDA**: Blackwell (GeForce RTX Serie 50)
- **Capacidad de Cómputo (Compute Capability)**: `12.0` (`sm_120`)
- **Multiprocesadores (SM)**: `36`
- **Núcleos CUDA por SM**: `128` (FP32)
- **Número Total de Núcleos CUDA**: `4.608` (36 SM × 128 cores/SM)
- **Memoria Global VRAM**: ~8 GB GDDR7 (`8.123 MiB` reportados por driver)
- **CUDA Toolkit**: `12.8`
- **Entorno de Compilación**: Visual Studio con Platform Toolset `v143` (C++20, x64)

---

## 2. Pautas para Adaptar Ejercicios Antiguos

Muchos ejercicios académicos fueron diseñados para arquitecturas anteriores (Tesla, Fermi, Kepler, Maxwell, Pascal, Volta, Turing o Ampere). Al trabajar con la **RTX 5060 Ti (Blackwell / CC 12.0)** se deben tener en cuenta las siguientes consideraciones:

### 2.1 Identificación de Arquitectura en Código
En programas que lean propiedades con `cudaGetDeviceProperties(&prop, id)`:
- `prop.major = 12` corresponde a la serie de consumo **Blackwell (GeForce RTX 50)**.
- `prop.major = 10` corresponde a chips datacenter Blackwell (B100/B200).
- Cada SM dispone de **128 núcleos FP32**.

Fragmento de referencia para switches:
```cpp
case 10:
case 12:
    // BLACKWELL (10.x Datacenter / 12.x GeForce RTX serie 50)
    archName = "BLACKWELL";
    cudaCores = 128;
    break;
```

### 2.2 Tamaños de Bloque y Warps
- **Warp size**: Sigue siendo **32 hilos**.
- **Hilos máximos por bloque**: 1024.
- **Configuración recomendada de bloques**: Múltiplos de 32 (típicamente 128, 256 o 512 hilos por bloque).

---

## 3. Métodos de Compilación y Ejecución

### Opción A: Desde Visual Studio (Recomendada)
1. Abrir la solución `Plantilla Cuda 12.8.slnx` o el proyecto `Plantilla Cuda 12.8.vcxproj`.
2. Para cambiar el archivo `.cu` que se compila:
   - En el Explorador de Soluciones, hacer clic derecho sobre el archivo actual (`basicotest.cu`) -> *Propiedades* -> cambiar *Tipo de elemento* a `Ninguno` o excluirlo del proyecto.
   - Añadir o incluir el nuevo archivo (ej. `ejercicio2.cu`) y asegurarse de que en sus propiedades el *Tipo de elemento* sea **CUDA C/C++**.
3. Compilar y ejecutar pulsando **F5** o **Ctrl + F5**.

---

### Opción B: Comprobación Manual desde Consola (PowerShell)

En Windows, invocar `nvcc` directamente en un PowerShell estándar no detecta el compilador de host `cl.exe` (MSVC). Para compilar y ejecutar de forma idéntica a Visual Studio, se utiliza `MSBuild` cargando el entorno de desarrollo:

#### Script de compilación y ejecución rápida:
```powershell
# 1. Cargar el entorno de desarrollo de Visual Studio en PowerShell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
$devShell = Join-Path $vs "Common7\Tools\Launch-VsDevShell.ps1"
& $devShell -Arch amd64

# 2. Compilar el proyecto en configuración Debug x64
msbuild "Plantilla Cuda 12.8.vcxproj" /p:Configuration=Debug /p:Platform=x64

# 3. Ejecutar el binario generado
& ".\x64\Debug\Plantilla Cuda 12.8.exe"
```

#### Cómo cambiar el archivo activo en el `.vcxproj`:
En el archivo `Plantilla Cuda 12.8.vcxproj`, la etiqueta `<ItemGroup>` define el archivo CUDA a compilar:
```xml
<ItemGroup>
  <CudaCompile Include="basico1.cu" />
</ItemGroup>
```
Basta con reemplazar el nombre del archivo por el del nuevo ejercicio.

---

### Opción C: Compilación Directa por Consola de un Archivo `.cu` Suelto (nvcc)

Si se desea compilar un archivo individual (como `basico1.cu`) directamente con `nvcc` sin tocar el `.vcxproj`:

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" -vcvars_ver=14.44.35207 && nvcc -o salida.exe basico1.cu && salida.exe"
```
> **Nota técnica**: Se utiliza `-vcvars_ver=14.44.35207` (MSVC toolset `v143`) porque CUDA 12.8 requiere compatibilidad con MSVC v143 en lugar de la versión de previsualización v145.

---

## 4. Modularización de Código Común (`propiedades.h`)

Para evitar duplicar código repetitivo en cada ejercicio y poder centrarse únicamente en lo que pide cada enunciado, la función de consulta de propiedades se ha extraído a una cabecera:

- **Archivo**: `propiedades.h`
- **Función**: `propiedades_Device(int deviceID)`

### Cómo usarla en ejercicios futuros (`basico2.cu`, `basico3.cu`, etc.):
1. Añadir el include en la cabecera:
   ```cpp
   #include "propiedades.h"
   ```
2. Invocarla directamente en `main`:
   ```cpp
   propiedades_Device(0);
   ```

### Consideración para entregas académicas:
Si para una entrega en la universidad el profesor solicita subir **únicamente el archivo `.cu` individual**, basta con copiar el cuerpo de `propiedades_Device` al final del `.cu` antes de subirlo. Mientras se trabaje en este entorno local, se mantendrá separada para mayor legibilidad y modularidad.

