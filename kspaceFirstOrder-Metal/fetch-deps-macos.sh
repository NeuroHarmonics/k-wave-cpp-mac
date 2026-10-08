#!/bin/zsh
# Download the header only libraries used by the Metal build into ThirdParty/
#   VkFFT      FFT library with a Metal backend (MIT license)
#   metal-cpp  Apple's C++ interface to Metal (Apache 2.0 license)
#
# Usage: ./fetch-deps-macos.sh   (run from this folder)

set -e

VKFFT_VERSION=v1.3.4
METALCPP_ZIP=metal-cpp_macOS15_iOS18.zip

mkdir -p ThirdParty
cd ThirdParty

if [ ! -d VkFFT ]; then
  curl -sSL https://github.com/DTolm/VkFFT/archive/refs/tags/$VKFFT_VERSION.tar.gz | tar xz
  mv VkFFT-${VKFFT_VERSION#v} VkFFT
fi

if [ ! -d metal-cpp ]; then
  curl -sSLO https://developer.apple.com/metal/cpp/files/$METALCPP_ZIP
  unzip -q $METALCPP_ZIP
  rm $METALCPP_ZIP
fi

echo "VkFFT $VKFFT_VERSION and metal-cpp are in $(pwd)"
