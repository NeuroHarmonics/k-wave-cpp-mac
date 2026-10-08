/**
 * @file      CudaUtils.metal
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The Metal header with utility functions and the definitions shared by all kernels. The Makefile puts
 *            this file in front of the other .metal files, the result is compiled at run time.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      22 March     2016, 15:25 (created) \n
 *            11 February  2020, 16:24 (revised)
 *
 * @copyright Copyright (C) 2016 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
 *
 * This file is part of the C++ extension of the [k-Wave Toolbox](http://www.k-wave.org).
 *
 * k-Wave is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * k-Wave is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
 * more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along with k-Wave.
 * If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
 */

#include <metal_stdlib>

using namespace metal;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Types shared with the host ---------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @struct dim3
 * @brief  Three unsigned integers, same layout as dim3 in Utils/CudaTypes.h.
 */
struct dim3
{
  /// Constructor, same defaults as CUDA.
  dim3(const uint x = 1, const uint y = 1, const uint z = 1) : x(x), y(y), z(z) {}

  /// x component.
  uint x;
  /// y component.
  uint y;
  /// z component.
  uint z;
};// end of dim3

/// Single precision complex number.
typedef float2 cuFloatComplex;

/**
 * @struct Parameters
 * @brief  Enumerations of Parameters (Parameters/Parameters.h) used by the kernels.
 */
struct Parameters
{
  /// Simulation dimension.
  enum class SimulationDimension
  {
    /// 2D simulation.
    k2D,
    /// 3D simulation.
    k3D
  };

  /// Source mode.
  enum class SourceMode
  {
    /// Dirichlet source condition.
    kDirichlet            = 0,
    /// Additive-no-correction source condition.
    kAdditiveNoCorrection = 1,
    /// Additive source condition.
    kAdditive             = 2
  };
};// end of Parameters

/// Shortcut for Simulation dimensions.
using SD = Parameters::SimulationDimension;

/**
 * @struct CudaDeviceConstants
 * @brief  Simulation constants, same layout as CudaDeviceConstants in Parameters/CudaDeviceConstants.cuh.
 */
struct CudaDeviceConstants
{
  /// Is the simulation 2D or 3D.
  Parameters::SimulationDimension simulationDimension;
  /// Size of x dimension.
  uint nx;
  /// Size of y dimension.
  uint ny;
  /// Size of z dimension.
  uint nz;
  /// Total number of elements.
  uint nElements;
  /// Size of complex x dimension.
  uint nxComplex;
  /// Size of complex y dimension.
  uint nyComplex;
  /// Size of complex z dimension.
  uint nzComplex;
  /// Complex number of elements.
  uint nElementsComplex;
  /// Normalization constant for 3D FFT.
  float fftDivider;
  /// Normalization constant for 1D FFT over x.
  float fftDividerX;
  /// Normalization constant for 1D FFT over y.
  float fftDividerY;
  /// Normalization constant for 1D FFT over z.
  float fftDividerZ;
  /// dt
  float dt;
  /// 2.0 * dt
  float dtBy2;
  /// c^2
  float c2;
  /// rho0 in homogeneous case.
  float rho0;
  /// dt * rho0 in homogeneous case.
  float dtRho0;
  /// dt / rho0Sgx in homogeneous case.
  float dtRho0Sgx;
  /// dt / rho0Sgy in homogeneous case.
  float dtRho0Sgy;
  /// dt / rho0Sgz in homogeneous case.
  float dtRho0Sgz;
  /// B/A value for homogeneous case.
  float bOnA;
  /// AbsorbTau value for homogeneous case.
  float absorbTau;
  /// AbsorbEta value for homogeneous case.
  float absorbEta;
  /// Size of the velocity source.
  uint velocitySourceSize;
  /// Velocity source mode.
  Parameters::SourceMode velocitySourceMode;
  /// Velocity source many.
  uint velocitySourceMany;
  /// Size of the pressure source mask.
  uint presureSourceSize;
  /// Pressure source mode.
  Parameters::SourceMode presureSourceMode;
  /// Pressure source many.
  uint presureSourceMany;
};// end of CudaDeviceConstants

/**
 * @enum  MI
 * @brief Matrix identifiers, same order as MatrixContainer::MatrixIdx in Containers/MatrixContainer.h.
 */
enum class MI : uint
{
  kKappa = 0,
  kSourceKappa,
  kC2,
  kP,

  kRhoX,
  kRhoY,
  kRhoZ,

  kUxSgx,
  kUySgy,
  kUzSgz,

  kDuxdx,
  kDuydy,
  kDuzdz,

  kRho0,
  kDtRho0Sgx,
  kDtRho0Sgy,
  kDtRho0Sgz,

  kDdxKShiftPosR,
  kDdyKShiftPos,
  kDdzKShiftPos,

  kDdxKShiftNegR,
  kDdyKShiftNeg,
  kDdzKShiftNeg,

  kPmlXSgx,
  kPmlYSgy,
  kPmlZSgz,
  kPmlX,
  kPmlY,
  kPmlZ,

  kBOnA,
  kAbsorbTau,
  kAbsorbEta,
  kAbsorbNabla1,
  kAbsorbNabla2,

  kSensorMaskIndex,
  kSensorMaskCorners,

  kInitialPressureSourceInput,
  kPressureSourceInput,
  kTransducerSourceInput,
  kVelocityXSourceInput,
  kVelocityYSourceInput,
  kVelocityZSourceInput,
  kPressureSourceIndex,
  kVelocitySourceIndex,
  kDelayMask,

  kDxudxn,
  kDyudyn,
  kDzudzn,
  kDxudxnSgx,
  kDyudynSgy,
  kDzudznSgz,

  kUxShifted,
  kUyShifted,
  kUzShifted,

  kXShiftNegR,
  kYShiftNegR,
  kZShiftNegR,

  kTemp1RealND,
  kTemp2RealND,
  kTemp3RealND,
  kTempCufftX,
  kTempCufftY,
  kTempCufftZ,
  kTempCufftShift
};// end of MI

// The host passes the number of matrices in MatrixContainer::MatrixIdx
static_assert(uint(MI::kTempCufftShift) + 1 == MATRIX_IDX_COUNT, "MI does not match MatrixContainer::MatrixIdx");

/**
 * @struct CudaMatrixContainer
 * @brief  Device pointers to all matrices, same layout as CudaMatrixContainer in Containers/CudaMatrixContainer.cuh.
 *         Missing matrices have a null pointer.
 */
struct CudaMatrixContainer
{
  /// Device pointers indexed by MI.
  device float* matrices[MATRIX_IDX_COUNT];
};// end of CudaMatrixContainer

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Function constants ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// The template parameters of the CUDA solver kernels are function constants in Metal. The pipelines are specialized
// with them at run time (MetalContext::getPipeline), so only the variants used by the simulation are built.

/// Dimensionality of the simulation as an integer.
constant int  simulationDimensionIdx [[function_constant(0)]];
/// Is density homogeneous?
constant bool rho0ScalarFlag         [[function_constant(1)]];
/// Is nonlinearity homogeneous?
constant bool bOnAScalarFlag         [[function_constant(2)]];
/// Is sound speed homogeneous?
constant bool c0ScalarFlag           [[function_constant(3)]];
/// Is absorption homogeneous?
constant bool alphaCoefScalarFlag    [[function_constant(4)]];

/// Dimensionality of the simulation.
constant SD   simulationDimension = SD(simulationDimensionIdx);

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------------- Variables ------------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Arguments every kernel gets. The simulation constants (cudaDeviceConstants) and the device pointers to all matrices
 * (cudaMatrixContainer) are bound by MetalContext at buffer indices 0 and 1, other kernel arguments start at 2.
 */
#define KERNEL_ARGS constant CudaDeviceConstants& cudaDeviceConstants [[buffer(0)]],    \
                    constant CudaMatrixContainer& cudaMatrixContainer [[buffer(1)]],    \
                    uint                          kernelIndex         [[thread_position_in_grid]], \
                    uint                          kernelStride        [[threads_per_grid]]

/// Get pointer to real matrix data from the cudaMatrixContainer.
#define getRealData(matrixIdx)    (cudaMatrixContainer.matrices[uint(matrixIdx)])
/// Get pointer to complex matrix data from the cudaMatrixContainer.
#define getComplexData(matrixIdx) (reinterpret_cast<device cuFloatComplex*>(cudaMatrixContainer.matrices[uint(matrixIdx)]))
/// Get pointer to index matrix data from the cudaMatrixContainer.
#define getIndexData(matrixIdx)   (reinterpret_cast<device size_t*>(cudaMatrixContainer.matrices[uint(matrixIdx)]))

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Index routines ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get global 1D coordinate for 1D CUDA block.
 * @return X coordinate for 1D CUDA block.
 */
#define getIndex() (kernelIndex)

/**
 * @brief  Get x-stride for 3D CUDA block (for processing multiple grid points by a single thread).
 * @return X stride for 3D CUDA block.
 */
#define getStride() (kernelStride)

/**
 * @brief  Get 3D coordinates for a real matrix form a 1D index.
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - 1D index.
 * @return 3D coordinates.
 */
inline dim3 getReal3DCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return dim3( i % cudaDeviceConstants.nx,
              (i / cudaDeviceConstants.nx) % cudaDeviceConstants.ny,
               i / (cudaDeviceConstants.nx * cudaDeviceConstants.ny));
}// end of getReal3DCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get 3D coordinates for a complex matrix form a 1D index.
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - 1D index.
 * @return 3D coordinates.
 */
inline dim3 getComplex3DCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return dim3( i % cudaDeviceConstants.nxComplex,
              (i / cudaDeviceConstants.nxComplex) % cudaDeviceConstants.nyComplex,
               i / (cudaDeviceConstants.nxComplex * cudaDeviceConstants.nyComplex));
}// end of getComplex3DCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get 2D coordinates for a real matrix form a 1D index.
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - 1D index.
 * @return 2D coordinates, z == 1.
 */
inline dim3 getReal2DCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return dim3(i % cudaDeviceConstants.nx, i / cudaDeviceConstants.nx, 1);
}// end of getReal2DCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get 2D coordinates for a complex matrix form a 1D index.
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - 1D index.
 * @return 2D coordinates, z == 1.
 */
inline dim3 getComplex2DCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return dim3(i % cudaDeviceConstants.nxComplex, i / cudaDeviceConstants.nxComplex, 1);
}// end of getComplex2DCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting 2D or 3D coordinates. The dimension is the function constant simulationDimension, the
 *         counterpart of the template parameter in CUDA.
 *
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - index.
 *
 * @return 2D/3D coordinates.
 */
inline dim3 getRealCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return (simulationDimension == SD::k3D) ? getReal3DCoords(cudaDeviceConstants, i)
                                          : getReal2DCoords(cudaDeviceConstants, i);
}// end of getRealCoords
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Shortcut for getting 2D or 3D coordinates. The dimension is the function constant simulationDimension, the
 *         counterpart of the template parameter in CUDA.
 *
 * @param  [in] cudaDeviceConstants - Simulation constants.
 * @param  [in] i                   - index.
 *
 * @return 2D/3D coordinates.
 */
inline dim3 getComplexCoords(constant CudaDeviceConstants& cudaDeviceConstants, const uint i)
{
  return (simulationDimension == SD::k3D) ? getComplex3DCoords(cudaDeviceConstants, i)
                                          : getComplex2DCoords(cudaDeviceConstants, i);
}// end of getComplexCoords
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Complex arithmetic -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// The per element operators *, *=, + and += for float2 that CUDA lacks are built into Metal.

/**
 * @brief  Complex multiplication, as cuCmulf from cuComplex.h.
 * @param  [in] a - First operand.
 * @param  [in] b - Second operand.
 * @return a * b
 */
inline cuFloatComplex cuCmulf(const cuFloatComplex a, const cuFloatComplex b)
{
  return cuFloatComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}// end of cuCmulf
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Utility kernels --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Fill memory with a 32 bit value, used by MetalContext::memset (the counterpart of cudaMemset).
 *
 * @param [out] data   - Memory to fill.
 * @param [in]  value  - Value of each 32 bit word.
 * @param [in]  nWords - Number of words.
 */
kernel void cudaMemset(device uint*    data   [[buffer(2)]],
                       constant uint&  value  [[buffer(3)]],
                       constant size_t& nWords [[buffer(4)]],
                       KERNEL_ARGS)
{
  for (size_t i = getIndex(); i < nWords; i += getStride())
  {
    data[i] = value;
  }
}// end of cudaMemset
//----------------------------------------------------------------------------------------------------------------------
