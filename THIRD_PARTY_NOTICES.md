# Third party notices

The k-Wave C++ codes are distributed under the GNU Lesser General Public License version 3 (`LICENSE`, which builds on the GNU General Public License version 3 in `licenses/GPL-3.0.txt`).

The release binaries link the following libraries statically. Their license texts are in the `licenses` folder.

| Library | Version | License | Used in | License text | Source |
|---|---|---|---|---|---|
| FFTW | 3.3.10 | GPL-2.0-or-later | OpenMP | `licenses/FFTW.txt` | https://www.fftw.org/fftw-3.3.10.tar.gz |
| HDF5 | 2.2.0 | BSD-3-Clause | OpenMP, Metal | `licenses/HDF5.txt` | https://github.com/HDFGroup/hdf5 |
| libaec | 1.1.7 | BSD-2-Clause | OpenMP, Metal | `licenses/libaec.txt` | https://gitlab.dkrz.de/k202009/libaec |
| LLVM OpenMP runtime (libomp) | 23.1.3 | Apache-2.0 WITH LLVM-exception | OpenMP, Metal | `licenses/LLVM-OpenMP.txt` | https://github.com/llvm/llvm-project |
| VkFFT | 1.3.4 | MIT | Metal | `licenses/VkFFT.txt` | https://github.com/DTolm/VkFFT |
| metal-cpp | macOS 15 / iOS 18 | Apache-2.0 | Metal | `licenses/metal-cpp.txt` | https://developer.apple.com/metal/cpp/ |

FFTW is Copyright (c) 2003, 2007-14 Matteo Frigo and Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology. It is used unmodified, built by `kspaceFirstOrder-OMP/build-fftw-macos.sh` from the source above. Because FFTW is licensed under the GPL, the OpenMP binary as a whole is distributed under the terms of the GNU General Public License version 3. The complete source code for the binaries is in this repository.

zlib, the C and C++ runtime libraries and the Metal framework are part of macOS and are not included in the binaries.
