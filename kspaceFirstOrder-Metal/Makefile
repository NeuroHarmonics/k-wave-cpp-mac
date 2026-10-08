# /**
# * @file      Makefile
# * @author    Jiri Jaros
# *            Faculty of Information Technology
# *            Brno University of Technology
# * @email     jarosjir@fit.vutbr.cz
# * @comments  macOS makefile for the Metal port of the CUDA code (Apple silicon)
# *
# * @tool      kspaceFirstOrder 3.6
# * @created   02 December  2014, 12:32
# * @lastModif 11 February  2020, 16:34
# *
# * @copyright Copyright (C) 2014 - 2020 SC\@FIT Research Group,
# *            Brno University of Technology, Brno, CZ.
# *
# * This file is part of the C++ extension of the k-Wave Toolbox
# * (http://www.k-wave.org).
# *
# * k-Wave is free software: you can redistribute it and/or modify it under the
# * terms of the GNU Lesser General Public License as published by the Free
# * Software Foundation, either version 3 of the License, or (at your option)
# * any later version.
# *
# * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY
# * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# * FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# * more details.
# *
# * You should have received a copy of the GNU Lesser General Public License
# * along with k-Wave.
# * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
# */

################################################################################
#                                                                              #
# The source codes can be compiled under macOS 13 or newer with Apple Clang    #
# (Xcode or the Command Line Tools). The binary runs on Macs with a Metal 3    #
# GPU, i.e., all Apple silicon Macs.                                           #
#                                                                              #
# The GPU kernels (*.metal) are embedded in the binary as source code and      #
# compiled when the program starts, so the Metal toolchain is not needed.      #
#                                                                              #
# Necessary libraries:                                                         #
#  - HDF5, libaec, libomp from Homebrew: brew install hdf5 libomp              #
#  - VkFFT and metal-cpp headers: ./fetch-deps-macos.sh                        #
#                                                                              #
# This makefile uses static linking by default. All libraries but the system   #
# ones (libSystem, libc++, libz and the Metal frameworks) are linked into the  #
# binary.                                                                      #
#                                                                              #
# How to compile the code                                                      #
#  1. ./fetch-deps-macos.sh                                                    #
#  2. Select one of the linking possibilities.                                 #
#  3. Select CPU architecture (native is default).                             #
#  4. make -j                                                                  #
#                                                                              #
################################################################################


################################################################################
#         Set following flags based on your compiler and library paths         #
################################################################################

# Static linking is default, so the binary can be copied to other Macs
LINKING = STATIC
#LINKING = DYNAMIC

# Library paths (Homebrew)
BREW_DIR    = $(shell brew --prefix)
HDF5_DIR    = $(BREW_DIR)/opt/hdf5
SZIP_DIR    = $(BREW_DIR)/opt/libaec
OMP_DIR     = $(BREW_DIR)/opt/libomp
THIRD_PARTY = ThirdParty

# Select CPU architecture of the host code.
# The native architecture will compile and optimize the code for the underlying
# processor. ARM64 runs on all Apple silicon (M1 and newer).
CPU_ARCH = native
#CPU_ARCH = ARM64

############################### Common flags ###################################
# Git hash of release 1.3
GIT_HASH       = -D__KWAVE_GIT_HASH__=\"468dc31c2842a7df5f2a07c3a13c16c9b0b2b770\"

########################### Apple Clang + Metal ################################
# C++ compiler
CXX       = clang++

# C++ standard
CPP_STD   = -std=c++17

# Enable OpenMP for the host code (Apple Clang needs the external libomp runtime)
OPENMP    = -Xpreprocessor -fopenmp

# Set CPU architecture
ifeq ($(CPU_ARCH), ARM64)
  CPU_FLAGS = -arch arm64 -mcpu=apple-m1
else
  CPU_FLAGS = -mcpu=native
endif

# Use maximum optimization. Finite math must stay off since Clang would otherwise remove the infinity checks in the
# absorption operators (KSpaceFirstOrderSolver::generateKappaAndNablas).
CPU_OPT   = -O3 -ffast-math -fno-finite-math-only

# Debug flags
CPU_DEBUG =
# Profile flags
PROFILE   =
# C++ warning flags
WARNING   = -Wall -Wno-pass-failed

# Add include directories, the third party headers are system headers to hide their warnings
INCLUDES  = -I. -I$(HDF5_DIR)/include -I$(OMP_DIR)/include \
            -isystem $(THIRD_PARTY)/metal-cpp -isystem $(THIRD_PARTY)/VkFFT/vkFFT
# Add library directories
LIB_PATHS = -L$(HDF5_DIR)/lib -L$(SZIP_DIR)/lib -L$(OMP_DIR)/lib

# Frameworks
FRAMEWORKS = -framework Metal -framework Foundation -framework QuartzCore

# Set compiler flags and header files directories
CXXFLAGS  = $(CPU_FLAGS) $(CPU_OPT) $(OPENMP) $(CPU_DEBUG) $(PROFILE) $(WARNING) \
            $(CPP_STD) $(GIT_HASH) $(INCLUDES) -MMD -MP

# Set linker flags and library files directories
ifeq ($(LINKING), STATIC)
  # Static link of all non-system libraries (macOS does not support fully static binaries).
  LDFLAGS = $(CPU_FLAGS) $(CPU_DEBUG) $(PROFILE) $(CPP_STD) $(LIB_PATHS)
  LDLIBS  = $(HDF5_DIR)/lib/libhdf5_hl.a \
            $(HDF5_DIR)/lib/libhdf5.a    \
            $(SZIP_DIR)/lib/libsz.a      \
            $(SZIP_DIR)/lib/libaec.a     \
            $(OMP_DIR)/lib/libomp.a      \
            -lz -lm $(FRAMEWORKS)
else
  # Dynamic link with runtime paths
  LDFLAGS = $(CPU_FLAGS) $(CPU_DEBUG) $(PROFILE) $(CPP_STD) $(LIB_PATHS) \
            -Wl,-rpath,$(HDF5_DIR)/lib -Wl,-rpath,$(OMP_DIR)/lib
  LDLIBS  = -lhdf5_hl -lhdf5 -lomp -lm -lz $(FRAMEWORKS)
endif

################################### Build ######################################
# Target binary name
TARGET       = kspaceFirstOrder-Metal

# Units to be compiled
DEPENDENCIES = main.o                                   \
               Containers/MatrixContainer.o             \
               Containers/CudaMatrixContainer.o         \
               Containers/OutputStreamContainer.o       \
               Hdf5/Hdf5File.o                          \
               Hdf5/Hdf5FileHeader.o                    \
               KSpaceSolver/KSpaceFirstOrderSolver.o    \
               KSpaceSolver/SolverCudaKernels.o         \
               Logger/Logger.o                          \
               MatrixClasses/BaseFloatMatrix.o          \
               MatrixClasses/BaseIndexMatrix.o          \
               MatrixClasses/CufftComplexMatrix.o       \
               MatrixClasses/ComplexMatrix.o            \
               MatrixClasses/IndexMatrix.o              \
               MatrixClasses/RealMatrix.o               \
               MatrixClasses/TransposeCudaKernels.o     \
               OutputStreams/BaseOutputStream.o         \
               OutputStreams/IndexOutputStream.o        \
               OutputStreams/CuboidOutputStream.o       \
               OutputStreams/WholeDomainOutputStream.o  \
               OutputStreams/OutputStreamsCudaKernels.o \
               Parameters/CommandLineParameters.o       \
               Parameters/Parameters.o                  \
               Parameters/CudaParameters.o              \
               Parameters/CudaDeviceConstants.o         \
               Utils/MetalContext.o                     \
               Utils/MetalKernelsSource.o

# Metal kernels, CudaUtils.metal must be the first one
METAL_SOURCES = Utils/CudaUtils.metal                     \
                KSpaceSolver/SolverCudaKernels.metal      \
                MatrixClasses/TransposeCudaKernels.metal  \
                OutputStreams/OutputStreamsCudaKernels.metal

# Build target
all: $(TARGET)

# Link target
$(TARGET): $(DEPENDENCIES)
	$(CXX) $(LDFLAGS) $(DEPENDENCIES) $(LDLIBS) -o $@

# Compile units
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

# Embed the source of the Metal kernels into the binary
Utils/MetalKernelsSource.cpp: $(METAL_SOURCES)
	( printf '// Generated by the Makefile from the .metal files, do not edit.\n'; \
	  printf 'extern const char kMetalKernelsSource[];\n'; \
	  printf 'const char kMetalKernelsSource[] = R"__METAL__(\n'; \
	  cat $(METAL_SOURCES); \
	  printf ')__METAL__";\n' ) > $@

# Header dependencies
-include $(DEPENDENCIES:.o=.d)

# Clean repository
.PHONY: clean
clean:
	rm -f $(DEPENDENCIES) $(DEPENDENCIES:.o=.d) $(TARGET) Utils/MetalKernelsSource.cpp
