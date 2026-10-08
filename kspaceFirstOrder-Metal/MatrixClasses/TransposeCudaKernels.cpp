/**
 * @file      TransposeCudaKernels.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file for the interface to CUDA transpose kernels for 3D FFTs. The kernels are in
 *            TransposeCudaKernels.metal and launched through MetalContext.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      02 August    2019, 14:49 (created) \n
 *            11 February  2020, 16:17 (revised)
 *
 * @copyright Copyright (C) 2019 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <string>

#include <MatrixClasses/TransposeCudaKernels.cuh>
#include <Parameters/Parameters.h>

#include <Logger/Logger.h>
#include <Utils/MetalContext.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get block size for the transposition kernels.
 * @return 3D grid size.
 */
inline dim3 getSolverTransposeBlockSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeBlockSize();
};// end of getSolverTransposeBlockSize()
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get grid size for complex 3D kernels
 * @return 3D grid size
 */
inline dim3 GetSolverTransposeGirdSize()
{
  return Parameters::getInstance().getCudaParameters().getSolverTransposeGirdSize();
};// end of getSolverTransposeGirdSize()
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get the name of a transposition kernel instance (the instances are listed in TransposeCudaKernels.metal).
 * @param  [in] kernelName - Name of the kernel template.
 * @param  [in] padding    - Which matrices are padded.
 * @return Kernel name.
 */
inline std::string getKernelName(const std::string& kernelName, const TransposeCudaKernels::TransposePadding padding)
{
  using TP = TransposeCudaKernels::TransposePadding;
  switch (padding)
  {
    case TP::kInput:       return kernelName + "_kInput";
    case TP::kOutput:      return kernelName + "_kOutput";
    case TP::kInputOutput: return kernelName + "_kInputOutput";
    default:               return kernelName + "_kNone";
  }
}// end of getKernelName
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Transpose a real 3D matrix in the X-Y direction. It is done out-of-place.
 * As long as the blockSize.z == 1, the transposition works also for 2D case.
 */
template<TransposeCudaKernels::TransposePadding padding>
void TransposeCudaKernels::trasposeReal3DMatrixXY(float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes)
{
  // Fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  MetalContext::getInstance().launchKernel(getKernelName("cudaTrasnposeReal3DMatrixXY", padding),
                                           GetSolverTransposeGirdSize(),
                                           getSolverTransposeBlockSize(),
                                           outputMatrix,
                                           inputMatrix,
                                           dimSizes);
}// end of trasposeReal3DMatrixXY
//----------------------------------------------------------------------------------------------------------------------

//---------------------------------- Explicit instances of TrasposeReal3DMatrixXY ------------------------------------//
/// Transpose a real 3D matrix in the X-Y direction, input matrix padded, output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input matrix compact, output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Y direction, input and output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kNone>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInputOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------


/**
 * Transpose a real 3D matrix in the X-Z direction. It is done out-of-place.
 */
template<TransposeCudaKernels::TransposePadding padding>
void TransposeCudaKernels::trasposeReal3DMatrixXZ(float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes)
{
  // Fixed size at the moment, may be tuned based on the domain shape in the future
  // warpSize set to 32, and 4 tiles processed at once
  MetalContext::getInstance().launchKernel(getKernelName("cudaTrasnposeReal3DMatrixXZ", padding),
                                           GetSolverTransposeGirdSize(),
                                           getSolverTransposeBlockSize(),
                                           outputMatrix,
                                           inputMatrix,
                                           dimSizes);
}// end of trasposeReal3DMatrixXZ
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------- Explicit instances of TrasposeReal3DMatrixXZ -------------------------------------//
/// Transpose a real 3D matrix in the X-Z direction, input matrix padded, output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input matrix compact, output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix compact.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kNone>
                                                (float*       outputMatrix,
                                                 const float* inputMatrix,
                                                 const dim3&  dimSizes);

/// Transpose a real 3D matrix in the X-Z direction, input and output matrix padded.
template
void TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInputOutput>
                                                 (float*       outputMatrix,
                                                  const float* inputMatrix,
                                                  const dim3&  dimSizes);
//----------------------------------------------------------------------------------------------------------------------
