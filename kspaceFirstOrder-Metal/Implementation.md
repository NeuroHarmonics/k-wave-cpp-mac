# Metal port, notes for developers

This folder started as the unchanged k-Wave 1.3 CUDA code (commit `22376b1`). Running `git diff 22376b1 -- kspaceFirstOrder-Metal` shows every change. Class, file and function names were kept from the CUDA code (`SolverCudaKernels`, `CufftComplexMatrix`, `CudaParameters`) to keep the diff readable.

## How it works

- **Metal layer.** `Utils/MetalContext` replaces the CUDA runtime: device selection, buffer allocation, kernel launches, events and synchronisation.
- **Memory.** Each matrix is one shared storage `MTLBuffer`. The host pointer is `contents()`, the device pointer is the buffer's `gpuAddress()`, which behaves like a CUDA device pointer (stored in tables, offset, passed to kernels, never dereferenced on the CPU). `copyToDevice` is a no-op and `copyFromDevice` waits for the GPU. This relies on the host writing matrices only before the time loop or after a synchronisation.
- **Kernels.** Each CUDA `.cu` file was split into a `.metal` file (kernels) and a `.cpp` file (launch wrappers). `Utils/CudaUtils.metal` is the shared header. The Makefile concatenates the `.metal` files into `Utils/MetalKernelsSource.cpp`, and Metal compiles them when the program starts.
- **Kernel arguments.** Buffer 0 holds the device constants and buffer 1 the table of matrix pointers (the CUDA `__constant__` variables). Other arguments follow from index 2: pointers are bound as buffers, values with `setBytes`. All buffers are made resident with `useResources`, since kernels reach matrices through the table.
- **Templates.** The solver kernels' template parameters (`simulationDimension`, `rho0ScalarFlag`, `bOnAScalarFlag`, `c0ScalarFlag`, `alphaCoefScalarFlag`) are function constants with the same names, so the kernel bodies did not change. Pipelines are specialised on first use. The other templates are explicit instances named like `cudaSampleIndex_kRms`.
- **Ordering.** All kernels and FFTs go into one serial compute encoder, which keeps the order of the CUDA default stream. The command buffer is committed once per time step. Events let the CPU write the sensor data of step t to HDF5 while the GPU computes step t+1.
- **FFTs.** `CufftComplexMatrix` uses VkFFT (Metal backend, fetched by `fetch-deps-macos.sh`), one plan per cuFFT plan, recorded into the same encoder. ND transforms are out of place in the cuFFT layout. Batched 1D transforms use the full grid with the other axes switched off (`omitDimension`), in place on transposed data for y and z, as in CUDA.

## Things that are easy to break

- **Nyquist frequency in the shifted velocity.** The half cell shift makes the Nyquist frequency of even sized dimensions imaginary. cuFFT ignores that imaginary part in complex to real transforms, VkFFT does not, so the shift kernels zero it. Without this, `u_non_staggered` and intensity are wrong by up to 10 percent on even grids.
- **VkFFT coalescing.** `createPlan` tries the default `coalescedMemory`, then 16 and 8 bytes, and keeps the first that does every axis in one pass. The default splits the y axis of 2D grids of 1536^2 and more into two passes, which is 25 to 35 percent slower.
- **VkFFT launch offsets.** `specifyOffsetsAtLaunch` generates invalid Metal code in VkFFT 1.3.4, so FFT buffers must start at offset 0 (checked in `executePlan`).
- **`vkFFT.h`** defines the metal-cpp implementation, so it is included in `CufftComplexMatrix.cpp` only.
- **Device constants and matrix indices** exist twice: `Parameters/CudaDeviceConstants.cuh` and `Utils/CudaUtils.metal`, and `MatrixContainer::MatrixIdx` and `MI`. A `static_assert` checks the number of matrices, the rest must be kept in sync by hand.
- **Explicit template instantiations** must come after the member definitions (Clang does not instantiate members defined later, nvcc did).
- **Function constants.** Every kernel in the library needs the function constant values, even kernels that do not use them, so `getPipeline` always supplies them.
- **Metal Shading Language** has no lambdas, and `host_name` must be a plain identifier.
- **Transposes** use `simdgroup_barrier` in place of `__syncwarp`, which relies on 32 thread SIMD groups (true on Apple GPUs).

## Validation

Run from `../tests` with `matlab -batch <script>`.

- `compare_with_matlab.m` (set `KWAVE_BINARY=kspaceFirstOrder-Metal`) gives 1e-6 to 3e-6.
- `compare_features.m` runs every sensor output, source type and absorption model through MATLAB, the OpenMP binary and the Metal binary. Metal should be within a factor of about 2 of the OpenMP error (1e-7 to 4e-6). Errors are relative to the maximum of each output, so outputs that are small compared with the field (`p_final`, sensors the wave barely reaches) show larger errors from the same round off.
- Checkpoint and restart gives output identical to an uninterrupted run. The Metal binary is deterministic, the OpenMP binary is not (FFTW chooses plans by timing).

## Profiling

- `KWAVE_METAL_PROFILE=1` prints the GPU time per kernel and FFT (each runs in its own command buffer, so only the split is meaningful). `KWAVE_METAL_PROFILE=gpu` prints how long the GPU was busy.
- `../tools/fftbench nx ny [nz]` (build with `make` in `../tools`) times VkFFT transforms with tuning options from environment variables.
- Everything runs at about 130 GB/s on an M3 Pro, so further gains need fewer bytes moved. Options not done, a few percent each: fuse `computeDensity*` with `computePressureTerms*` (about 5 percent of a 3D absorbing step), one inverse FFT instead of two for absorption in homogeneous media (about 5 percent), shifted velocity in y and z from the 3D spectrum instead of transposes and 1D FFTs (about 5 percent when recording `u_non_staggered` or intensity).

## Limitations

- No axisymmetric mode (as the CUDA code).
- Transducer transmit apodization is ignored, as in all k-Wave C++ codes (the input file has no dataset for it).
- Apple silicon only, macOS 15 or newer for the release build (the minimum of the Homebrew libraries).
