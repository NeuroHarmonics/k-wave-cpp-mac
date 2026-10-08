#!/bin/bash
#
# Build a single precision FFTW library with NEON (Apple silicon) or AVX/AVX2 (Intel) SIMD and OpenMP support for
# the macOS build of kspaceFirstOrder-OMP. The Homebrew FFTW bottle is built without NEON on Apple silicon, which
# makes the FFTs (and hence the whole simulation) considerably slower.
#
# The library is installed into ThirdParty/fftw, which is picked up automatically by the Makefile.
#
# Requirements: Xcode command line tools, brew install libomp
#
# Usage: ./build-fftw-macos.sh

set -euo pipefail

FFTW_VERSION=3.3.10
FFTW_MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${ROOT_DIR}/ThirdParty/fftw"
BUILD_DIR="${ROOT_DIR}/ThirdParty/build"
OMP_DIR="$(brew --prefix)/opt/libomp"

if [ ! -f "${OMP_DIR}/include/omp.h" ]; then
  echo "libomp not found, please run: brew install libomp" >&2
  exit 1
fi

# Select SIMD instruction set
if [ "$(uname -m)" = "arm64" ]; then
  ARCH_FLAGS="-mcpu=apple-m1"
  SIMD_ARGS="--enable-neon --enable-armv8-cntvct-el0"
else
  ARCH_FLAGS=""
  SIMD_ARGS="--enable-sse2 --enable-avx --enable-avx2"
fi

# Download and verify the sources
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
curl -sSLO "https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
if [ "$(md5 -q fftw-${FFTW_VERSION}.tar.gz)" != "${FFTW_MD5}" ]; then
  echo "Checksum of fftw-${FFTW_VERSION}.tar.gz does not match" >&2
  exit 1
fi
tar xzf "fftw-${FFTW_VERSION}.tar.gz"
cd "fftw-${FFTW_VERSION}"

# Apple Clang needs the OpenMP flags passed to make, configure does not detect them correctly
OPENMP_FLAGS="-Xclang -fopenmp -I${OMP_DIR}/include"

./configure --prefix="${PREFIX}"                                         \
            --enable-single --enable-openmp --enable-threads ${SIMD_ARGS} \
            --enable-static --disable-shared --disable-fortran --disable-doc \
            CC=clang CFLAGS="-O3 ${ARCH_FLAGS}"                          \
            LDFLAGS="-L${OMP_DIR}/lib -lomp"

make -j "$(sysctl -n hw.ncpu)" OPENMP_CFLAGS="${OPENMP_FLAGS}"
make install OPENMP_CFLAGS="${OPENMP_FLAGS}"

cd "${ROOT_DIR}"
rm -rf "${BUILD_DIR}"

echo "FFTW ${FFTW_VERSION} installed into ${PREFIX}"
