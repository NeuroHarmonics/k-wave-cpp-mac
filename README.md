# k-Wave C++ codes for macOS

macOS port and builds of the C++ simulation codes from the [k-Wave toolbox](http://www.k-wave.org) (release 1.3).

## Repository contents

| Folder | Description |
|---|---|
| `kspaceFirstOrder-OMP` | OpenMP (CPU) code, ported from the `kspaceFirstOrder-OMP` Linux source |
| `kspaceFirstOrder-Metal` | GPU code for Apple silicon, ported from the `kspaceFirstOrder-CUDA` Linux source |
| `tools` | VkFFT timing tool used to tune the FFTs of the Metal code |
| `tests` | MATLAB comparison against the k-Wave MATLAB solvers, and benchmark scripts |

## Using the binaries

The binaries run on Apple silicon Macs with macOS 15 or newer. Prebuilt binaries are attached to [releases](https://github.com/NeuroHarmonics/k-wave-cpp-mac/releases). All libraries are linked statically, so nothing else needs to be installed.

- **OpenMP.** Copy `kspaceFirstOrder-OMP` into the k-Wave `binaries` folder, or pass its location to `kspaceFirstOrder2DC`, `kspaceFirstOrder3DC` or `kspaceFirstOrderASC` with the `BinaryPath` option.
- **Metal.** The binary takes the same inputs and command line as the CUDA code. Copy it into the k-Wave `binaries` folder renamed to `kspaceFirstOrder-CUDA`, and `kspaceFirstOrder2DG` and `kspaceFirstOrder3DG` use it without any changes. Alternatively keep the name and pass `'BinaryName', 'kspaceFirstOrder-Metal'` (and `BinaryPath` if needed).

:warning: macOS may block a downloaded binary, in which case remove the quarantine flag with `xattr -d com.apple.quarantine <binary>`.

## How the ports work

- **OpenMP.** The original OpenMP code with platform changes only. Linux-specific code paths are extended to macOS, x86-specific calls have Arm equivalents, and FFTW is built with NEON. The numerical code is unchanged.

- **Metal.** The CUDA code with its GPU layer replaced. The CUDA kernels are translated to the Metal Shading Language, cuFFT is replaced by [VkFFT](https://github.com/DTolm/VkFFT), and the CUDA runtime by a small Metal layer. The solver, file handling and command line are unchanged, so the binary reads and writes the same HDF5 files as the CUDA code. CPU and GPU share memory on Apple silicon, so each matrix is held once rather than as a host and a device copy. See [kspaceFirstOrder-Metal/Implementation.md](kspaceFirstOrder-Metal/Implementation.md) for details.

## Performance

M3 Pro (18 core GPU, 150 GB/s), time stepping, heterogeneous absorbing nonlinear medium:

| Grid | OpenMP (best of 6 and 12 threads) | Metal | Speed-up |
|---|---|---|---|
| 2D 256^2, 1000 steps | 0.84 s | 0.20 s | 4x |
| 2D 2048^2, 1000 steps | 27 s | 12 s | 2.2x |
| 3D 128^3, 200 steps | 2.5 s | 1.5 s | 1.7x |
| 3D 256^3, 200 steps | 28 s | 15 s | 1.9x |

The code is memory bandwidth bound on both CPU and GPU. Profiling the Metal binary (`KWAVE_METAL_PROFILE=1`) at 256^3 shows about 60 percent of the GPU time in FFTs (3.2 ms per 3D transform) and 40 percent in the element-wise kernels. Both run at about 130 GB/s, and the GPU is busy 100 percent of the time, so the speed-up on this machine is close to the bandwidth limit. Chips with more bandwidth (Max, Ultra) should gain more.

## Building

OpenMP code, see `kspaceFirstOrder-OMP/Readme.md`, section "Compiling the C++ code on macOS". In short

```bash
brew install fftw hdf5 libomp
cd kspaceFirstOrder-OMP
./build-fftw-macos.sh
make -j CPU_ARCH=ARM64
```

Metal code, see `kspaceFirstOrder-Metal/Readme.md`, section "Compilation". In short

```bash
brew install hdf5 libomp
cd kspaceFirstOrder-Metal
./fetch-deps-macos.sh
make -j CPU_ARCH=ARM64
```

Use `CPU_ARCH=ARM64` for binaries that run on all Apple silicon Macs (the default optimises for the build machine).

## Testing

`tests/compare_with_matlab.m` runs 2D and 3D simulations through the MATLAB solver and through a binary (select it with the environment variable `KWAVE_BINARY`) and prints the maximum relative difference, which should be around 1e-6 to 3e-6. `tests/compare_features.m` covers all sensor outputs, source types and absorption models and runs both binaries side by side. `tests/make_bench_inputs.m` and `tests/bench.sh` create benchmark inputs and time a binary.

## License

The k-Wave codes are distributed under the GNU Lesser General Public License, see `License.md` in each folder.
