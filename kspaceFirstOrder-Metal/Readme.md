## Overview

k-Wave is an open source MATLAB toolbox designed for the time-domain simulation of propagating acoustic waves in 1D, 2D, or 3D. The toolbox has a wide range of functionality, but at its heart is an advanced numerical model that can account for both linear or nonlinear wave propagation, an arbitrary distribution of heterogeneous material parameters, and power law acoustic absorption. See k-Wave website(http://www.k-wave.org).

This project is a part of the k-Wave toolbox accelerating 2D/3D simulations using an optimized CUDA/C++ implementation to run small to moderate grid sizes (e.g., 128x128 to 10,000x10,000 in 2D or 64x64x64 to 512x512x512 in 3D) on systems with a single NVidia GPU. Axisymmetric coordinate systems are not supported.

This folder holds a port of the CUDA code to Apple Metal, so that it runs on the GPU of Apple silicon Macs. It reads and writes the same HDF5 files and accepts the same command line as `kspaceFirstOrder-CUDA`. The CUDA kernels were translated to the Metal Shading Language (`*.metal`), cuFFT was replaced by VkFFT, and the CUDA runtime by a small Metal layer (`Utils/MetalContext`). Class and file names were kept from the CUDA code so the changes are easy to follow. See `Implementation.md` for the details.


## Repository structure

    .
    +--Containers         - Matrix and output stream containers
    +--GetoptWin64        - Windows version of the getopt routine
    +--Hdf5               - HDF5 classes (file access)
    +--KSpaceSolver       - Solver classes with all the kernels
    +--Logger             - Logger class for reporting progress and errors
    +--MatrixClasses      - Matrix classes holding simulation data
    +--OutputStreams      - Output streams for sampling data
    +--Parameters         - Parameters of the simulation
    +--Utils              - Utility routines, Metal layer (MetalContext)
    +--ThirdParty         - VkFFT and metal-cpp, downloaded by fetch-deps-macos.sh
    Changelog.md          - Change log
    License.md            - License file
    Makefile              - GNU Makefile
    Readme.md             - Read me
    Doxyfile              - Doxygen documentation file
    fetch-deps-macos.sh   - Downloads the third party libraries
    Implementation.md     - Notes for developers
    header_bg.png         - Doxygen logo
    main.cpp              - Main file of the project


## Compilation

The code is compiled on macOS 13 or newer with Apple Clang and libraries from [Homebrew](https://brew.sh). It runs on Macs with a Metal 3 GPU, which includes all Apple silicon Macs. The GPU kernels are embedded in the binary as source code and compiled by Metal when the program starts (in about a second the first time, Metal caches the result), so the Metal toolchain is not needed.

 1. Install the Xcode command line tools and the libraries:
    ```bash
    xcode-select --install
    brew install hdf5 libomp
    ```

 2. Download the header only libraries VkFFT (FFTs on the GPU) and metal-cpp (C++ interface to Metal) into `ThirdParty`:
    ```bash
    ./fetch-deps-macos.sh
    ```

 3. Compile the source code by typing:
    ```bash
    make -j
    ```
    By default, the host code is optimized for the CPU it is compiled on. To create a binary that runs on all Apple silicon Macs (M1 and newer), type:
    ```bash
    make -j CPU_ARCH=ARM64
    ```
    Static linking (default) links HDF5, libaec and libomp into the binary, so it only depends on system libraries and frameworks and can be copied to other Macs without Homebrew. Note that the minimum macOS version is the one the Homebrew libraries were built for (usually the macOS version of the build machine).

To use the binary from MATLAB, copy it into the k-Wave `binaries` folder renamed to `kspaceFirstOrder-CUDA`, and `kspaceFirstOrder2DG` and `kspaceFirstOrder3DG` use it without any changes. Alternatively keep the name and pass `'BinaryName', 'kspaceFirstOrder-Metal'` (and `BinaryPath` if needed).


## Usage

The CUDA codes offers a lot of parameters and output flags to be used. For more information, please type:

```bash
./kspaceFirstOrder-Metal --help
```
