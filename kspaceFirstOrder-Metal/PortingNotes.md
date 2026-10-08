# Metal port of kspaceFirstOrder-CUDA: starting notes

Notes from an initial look at porting the k-Wave CUDA code to Apple Metal so that it runs on the GPU of Apple silicon Macs. This folder holds the untouched CUDA sources (commit `22376b1`), which are the starting point for the port. Everything below was checked against the code in this folder unless marked as *to verify*.

## Goal

Build a Metal version of `kspaceFirstOrder-CUDA` (k-Wave 1.3, CUDA code version 3.6) that reads and writes the same HDF5 files and accepts the same command line, so that it can be called from MATLAB through `kspaceFirstOrder2DG` / `kspaceFirstOrder3DG` (or `*DC` with `'BinaryName'`), exactly like the CUDA binary.

Suggested first step is a 1 to 2 day spike to check the FFT library and measure the speed-up before committing to the full port (see the plan below).

## Context

- This repo (private GitHub repo `NeuroHarmonics/k-wave-cpp-mac`, local folder `~/Documents/Local-Repos/k-wave-cpp`) holds both codes. `../kspaceFirstOrder-OMP` is the finished macOS port of the OpenMP (CPU) code 1.3, released as `v1.3-macos` with an arm64 binary. Commit `b86d796` shows every change that was needed for macOS (`git show b86d796`, the files were at the repo root then) and is a useful reference, since this CUDA code shares most of its host side (Hdf5, Logger, Parameters, main.cpp) with the OpenMP code.
- Release tags are per code. The existing `v1.3-macos` is the OpenMP build. Use prefixed tags from now on, for example `omp-v1.3-macos-2` and `metal-v0.1`.
- The k-Wave MATLAB toolbox is at `~/Documents/Local-Repos/k-wave/k-Wave`. MATLAB R2025b is at `/Applications/MATLAB_R2025b.app` and runs headless with `matlab -batch <script>`.
- Machine is an M3 Pro (6 performance + 6 efficiency CPU cores, 18 core GPU, Metal 3), 36 GB RAM, macOS 15.7, Xcode 26.3 with Apple Clang 17.
- Homebrew has `hdf5` (2.2.0, works with this code), `libomp`, `fftw` and `libaec` installed.
- The **Metal shader compiler is not installed**. `xcrun metal` fails with "missing Metal Toolchain". Either run `xcodebuild -downloadComponent MetalToolchain` (needs the user's OK, it is a large download) or compile the shaders at runtime with `MTLDevice::newLibrary(source, ...)`, which needs no offline compiler and is probably simpler for a start.
- The user wants the diff to the original sources to stay easy to read. The untouched CUDA sources are committed in `22376b1`, so `git diff 22376b1 -- kspaceFirstOrder-Metal` shows everything changed by the port. This notes file is the only addition so far.
- Commit messages for this user are plain and short, with no colons, no co-author line and no mention of Claude.

## How the CUDA code is structured

About 24k lines in total, but the GPU specific part is small and well separated.

- Kernels are launched (`<<<...>>>`) only in three files
  - `KSpaceSolver/SolverCudaKernels.cu`, 30 kernels, about 2,150 lines
  - `MatrixClasses/TransposeCudaKernels.cu`, 2 kernels (XY and XZ transposes of 3D real matrices)
  - `OutputStreams/OutputStreamsCudaKernels.cu`, 4 kernels (sensor sampling for index, cuboid and whole domain outputs)
- `KSpaceSolver/KSpaceFirstOrderSolver.cpp` (about 2,300 lines) never launches kernels. It calls about 29 wrapper functions in the `SolverKernels` namespace, and otherwise uses the CUDA API only for version queries, `cudaMemGetInfo` and `cudaDeviceReset`.
- Other host files that touch the CUDA API and need edits
  - `MatrixClasses/BaseFloatMatrix.cpp`, `BaseIndexMatrix.cpp` (`cudaMalloc`, `cudaHostRegister`, `cudaMemcpy` host to device and back)
  - `MatrixClasses/CufftComplexMatrix.cpp/.h` (all cuFFT use, about 700 lines)
  - `Containers/MatrixContainer.cpp/.h`, `Containers/CudaMatrixContainer.cu/.cuh`
  - `OutputStreams/BaseOutputStream.cpp`, `IndexOutputStream.cpp/.h`, `CuboidOutputStream.cpp/.h` (mapped host buffers)
  - `Parameters/CudaParameters.cpp/.h` (device selection, block and grid sizes), `Parameters/CudaDeviceConstants.cu/.cuh`
  - `Utils/CudaUtils.cuh`, `Logger/Logger.h`, `main.cpp`

### Kernel style

- Kernels are grid stride loops over the whole grid (`for (i = getIndex(); i < nElements; i += getStride())`), mostly element wise. The 3D coordinates are recomputed from the linear index where PML or shift vectors are needed.
- Kernels are templated on simulation dimension (2D or 3D) and on flags such as scalar or heterogeneous medium properties.
- Each kernel gets its matrix pointers from a `__constant__` table of device pointers (`cudaMatrixContainer`, accessed via `getRealData(MI::kRhoX)` and so on) and its scalar parameters from a `__constant__` struct (`cudaDeviceConstants`, 33 fields). Kernels mostly take no arguments.
- Device features used. No `__syncthreads`, no atomics, no warp shuffles or votes, no textures, no streams (everything is on the default stream). Shared memory and `__syncwarp` only in the two transpose kernels, which use 32x33 tiles with the warp size as a template parameter.
- The only maths function in the kernels is `sqrt`. Absorption operators (power law terms) are precomputed on the host, so GPU fast math should not cause the infinity problem described below, but keep it in mind.
- All data is `float`, complex data is `float2`. Apple GPUs have no FP64, which is fine.
- Sensor aggregates (`p_max`, `p_min`, `p_rms`, and the velocity versions) are accumulated per sensor point over time, so no cross thread reductions are needed.
- The CUDA code has no axisymmetric mode (the OpenMP code has one). A Metal port would inherit this gap.

### FFTs

`CufftComplexMatrix` uses
- `cufftPlan3d` / `cufftPlan2d` real to complex and complex to real for the main N-D transforms
- `cufftPlanMany` batched 1D real to complex and complex to real transforms along X, Y and Z, used for the non staggered (shifted) velocity outputs. The Y and Z versions transpose the data first with the transpose kernels and then do 1D transforms along the fastest dimension.

## Proposed mapping to Metal

| CUDA | Metal | Notes |
|---|---|---|
| `__global__` kernels with templates | Metal Shading Language kernels with templates, one explicit instantiation per variant via `[[host_name("...")]]`, or function constants | Mostly mechanical translation |
| `__constant__` table of device pointers | Argument buffer holding `MTLBuffer::gpuAddress()` values, read in the shader as `device float*` | Needs Metal 3 on Apple silicon (macOS 13+). All buffers referenced this way must be made resident with `useResource`/`useResources` on the encoder, or with a residency set (macOS 15+) |
| `__constant__ cudaDeviceConstants` | A `constant` buffer bound to every kernel | Easy |
| `cudaMalloc` + `cudaHostRegister` + `cudaMemcpy` | `MTLBuffer` with shared storage. Host and GPU can use the same memory, so copies become no-ops. `newBuffer(bytesNoCopy...)` can wrap existing page aligned host allocations | Simplifies things, but check where the code relies on separate host and device copies (for example checkpointing and output flushing) |
| Mapped host output buffers (`cudaHostRegisterMapped`) | Shared storage buffers | Easy |
| Default stream ordering | One command buffer per time step (or per chunk of steps), one serial compute encoder so each dispatch sees the previous one's writes | Synchronise before the CPU reads sensor data or writes output |
| Shared memory tiles and `__syncwarp` in transposes | `threadgroup` memory with `threadgroup_barrier(mem_flags::mem_threadgroup)` | Apple SIMD group width is also 32, but use proper barriers rather than relying on it |
| cuFFT | VkFFT (Metal backend, header only) is the most likely candidate. Alternatives are MPSGraph FFT ops or writing transforms by hand | Main risk, see below |
| nvcc Makefile | clang++ with `metal-cpp` headers, link `-framework Metal -framework Foundation` (plus `QuartzCore` if metal-cpp needs it) | Shader source either compiled at runtime or with the Metal toolchain once installed |

### Risks and things to verify

- *To verify* VkFFT Metal backend maturity for single precision 3D and 2D real to complex and complex to real, for batched strided 1D real transforms, and at the sizes k-Wave uses (often not powers of two, chosen to have small prime factors). Check how it records into an existing command buffer or encoder, so FFTs and kernels can share one command buffer.
- *To verify* performance of passing pointers through argument buffers versus binding buffers directly. Direct binding means changing every kernel signature, so the argument buffer route keeps the port closest to the original.
- Metal shader compilation uses fast math by default. Consider turning it off (`MTLCompileOptions` math mode or `fastMathEnabled = false`) at least while validating, then measure whether it matters.
- Large grids. A single `MTLBuffer` has a maximum length (`MTLDevice::maxBufferLength`), and there is a GPU working set limit (`recommendedMaxWorkingSetSize`). Check both against the biggest simulations that should be supported.
- 32 bit indices. The CUDA kernels index with `unsigned int`, which caps the grid at 2^32 elements as on CUDA. Keep the same for now.

## Lessons from the macOS OpenMP port (`../kspaceFirstOrder-OMP`)

These will probably come up again for the host side of this code.

- Many host files are guarded with `#ifdef __linux__` (getopt, unistd, getrusage, and the message headers). They need `|| defined(__APPLE__)`. On macOS `ru_maxrss` is in bytes, not kilobytes.
- The processor name is read with inline `cpuid` assembly, which does not exist on ARM. The OpenMP port uses `sysctlbyname("machdep.cpu.brand_string")` on macOS.
- `<immintrin.h>` was included only for `_mm_malloc`/`_mm_free`. The OpenMP port added `Utils/AlignedMemory.h`, which falls back to `posix_memalign` on non x86.
- Checks like `(defined(__GNUC__) ...) && !(defined(__clang__) || ...)` were used to mean "FFTW build" and silently disabled code under Clang. Check what similar guards mean here before trusting them.
- With `-ffast-math` Clang deletes comparisons against infinity. The OpenMP code relies on `x == infinity()` checks when building the absorption operators, so the port adds `-fno-finite-math-only`. The CUDA host code has the same checks (`KSpaceSolver/KSpaceFirstOrderSolver.cpp` lines 2001 and 2002), so the same flag is needed when compiling the host side with Clang.
- macOS ships GNU Make 3.81, which does not support `.RECIPEPREFIX`. Recipes must start with a tab.
- Homebrew FFTW is built without NEON on Apple silicon. Not relevant to the GPU path, but if the host side ever uses FFTW, use `../kspaceFirstOrder-OMP/build-fftw-macos.sh`.
- Files under `~/Downloads` cannot be read from this environment (macOS privacy). Ask the user to copy files elsewhere.

## Validation and benchmarking

Scripts in `../tests`
- `compare_with_matlab.m` runs a 2D and a 3D heterogeneous, absorbing, nonlinear simulation through the MATLAB solver and through a binary, and prints the maximum relative error. Choose the binary with `bin_name` at the top (the OpenMP build by default, it expects the binary inside its own folder in this repo). The macOS OpenMP build gives errors of about 1e-6 to 3e-6, and a correct Metal build should be similar.
- `make_bench_inputs.m` writes `bench_128.h5` and `bench_256.h5` (3D, heterogeneous, absorbing, nonlinear, 200 time steps) into the current folder. The 256 file is about 400 MB, and `*.h5` files are ignored by git.
- `bench.sh <binary> <N> [threads]` runs one benchmark and prints total and time stepping times.

Baseline with the OpenMP binary on this M3 Pro (`../kspaceFirstOrder-OMP/kspaceFirstOrder-OMP`, NEON FFTW, 6 to 12 threads, numbers vary by about 10 percent between runs)

| Case | Time stepping | Total |
|---|---|---|
| 128^3, 200 steps | 2.5 to 3.7 s | 3.3 to 4.9 s |
| 256^3, 200 steps | 27 to 28 s | 31 s |
| MATLAB `kspaceFirstOrder3D`, 128^3, single | | 19 s |

The code is memory bandwidth bound. The M3 Pro has about 150 GB/s, and the CPU cores already use a good share of it, so the expected GPU gain on this machine is modest (a guess of 1.5 to 3x). The gain should be larger on Max and Ultra chips (about 400 to 800 GB/s).

## Suggested plan

1. **Import into git.** Done. The untouched CUDA sources are in commit `22376b1`.
2. **Spike (1 to 2 days).** Write a small standalone program, not yet inside k-Wave, that allocates 256^3 buffers, runs VkFFT forward and inverse 3D real transforms plus three or four representative kernels (for example the velocity, density and pressure updates of a 3D linear lossless step), and times 200 steps. Compare against the OpenMP baseline above. Decide whether to continue.
3. **Host side macOS fixes.** Apply the same kind of fixes as in the OpenMP port so the non GPU code builds with Clang on macOS.
4. **GPU layer.** Replace the CUDA runtime use with a small Metal context (device, queue, pipeline cache, argument buffer for matrix pointers, constants buffer). Port `CudaMatrixContainer`, `CudaDeviceConstants`, the base matrix classes and the output stream buffers.
5. **Kernels.** Port the 36 kernels, keeping the same wrapper function names in the `SolverKernels` namespace so `KSpaceFirstOrderSolver.cpp` barely changes.
6. **FFT.** Replace `CufftComplexMatrix` with a VkFFT based class, including the batched 1D transforms for shifted velocity.
7. **Validate** with `compare_with_matlab.m`, covering the source and sensor types (`p`, `p_max`, `p_min`, `p_rms`, `u`, `u_non_staggered_raw`, which MATLAB also uses to compute intensity, cuboid sensors and checkpointing), then benchmark.
