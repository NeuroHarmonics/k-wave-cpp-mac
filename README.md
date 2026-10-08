# k-Wave C++ codes for macOS

macOS builds of the C++ simulation codes from the [k-Wave toolbox](http://www.k-wave.org) (release 1.3). Each code lives in its own folder, imported unchanged from the k-Wave release first and then modified, so the git history shows exactly what was changed from the original.

| Folder | Description | Status |
|---|---|---|
| `kspaceFirstOrder-OMP` | OpenMP (CPU) code built with Apple Clang | Working, released as `v1.3-macos` (Apple silicon) |
| `kspaceFirstOrder-Metal` | GPU code for Apple silicon, ported from the k-Wave CUDA code | Not started, see `kspaceFirstOrder-Metal/PortingNotes.md` |
| `tests` | MATLAB comparison against the k-Wave MATLAB solvers, and benchmark scripts | |

## Using the binaries

Download the binary from the [releases](https://github.com/NeuroHarmonics/k-wave-cpp-mac/releases), and either copy it into the k-Wave `binaries` folder or pass its location to `kspaceFirstOrder2DC`, `kspaceFirstOrder3DC` or `kspaceFirstOrderASC` with the `BinaryPath` option.

## Building

See `kspaceFirstOrder-OMP/Readme.md`, section "Compiling the C++ code on macOS". In short

```bash
brew install fftw hdf5 libomp
cd kspaceFirstOrder-OMP
./build-fftw-macos.sh
make -j CPU_ARCH=ARM64
```

## Testing

`tests/compare_with_matlab.m` runs 2D and 3D simulations through the MATLAB solver and through a binary and prints the maximum relative difference, which should be around 1e-6 to 3e-6. `tests/make_bench_inputs.m` and `tests/bench.sh` create benchmark inputs and time a binary.

## License

The k-Wave codes are distributed under the GNU Lesser General Public License, see `License.md` in each folder.
