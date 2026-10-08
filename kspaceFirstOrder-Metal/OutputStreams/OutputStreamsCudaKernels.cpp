/**
 * @file      OutputStreamsCudaKernels.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of the interface to cuda kernels used for data sampling (output streams).
 *            The kernels are in OutputStreamsCudaKernels.metal and launched through MetalContext.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      27 January   2015, 17:21 (created) \n
 *            11 February  2020, 16:21 (revised)
 *
 * @copyright Copyright (C) 2015 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <OutputStreams/BaseOutputStream.h>
#include <OutputStreams/OutputStreamsCudaKernels.cuh>

#include <Parameters/Parameters.h>
#include <Logger/Logger.h>
#include <Utils/MetalContext.h>

//--------------------------------------------------------------------------------------------------------------------//
//----------------------------------------------- Global routines ----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get the name of a sampling kernel instance (the instances are listed in OutputStreamsCudaKernels.metal).
 * @param  [in] kernelName - Name of the kernel template.
 * @param  [in] reduceOp   - Reduction operator.
 * @return Kernel name.
 */
inline std::string getKernelName(const std::string& kernelName, const BaseOutputStream::ReduceOperator reduceOp)
{
  using RO = BaseOutputStream::ReduceOperator;
  switch (reduceOp)
  {
    case RO::kRms: return kernelName + "_kRms";
    case RO::kMax: return kernelName + "_kMax";
    case RO::kMin: return kernelName + "_kMin";
    default:       return kernelName + "_kNone";
  }
}// end of getKernelName
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Index mask sampling --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Sample the source matrix using the index sensor mask and store data in buffer.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleIndex(float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples)
{
  MetalContext::getInstance().launchKernel(getKernelName("cudaSampleIndex", reduceOp),
                                           nullptr,
                                           nSamples,
                                           samplingBuffer,
                                           sourceData,
                                           sensorData,
                                           nSamples);
}// end of sampleIndex
//----------------------------------------------------------------------------------------------------------------------

/// Sample the source matrix using the index sensor mask, no post-processing.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kNone>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the root mean square.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kRms>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the maximum.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kMax>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);
/// Sample the source matrix using the index sensor mask, take the minimum.
template
void OutputStreamsCudaKernels::sampleIndex<BaseOutputStream::ReduceOperator::kMin>
                                          (float*        samplingBuffer,
                                           const float*  sourceData,
                                           const size_t* sensorData,
                                           const size_t  nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Cuboid mask sampling -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Sample data inside one cuboid and store it to buffer. The operation is given in the template parameter.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleCuboid(float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples)
{
  MetalContext::getInstance().launchKernel(getKernelName("cudaSampleCuboid", reduceOp),
                                           nullptr,
                                           nSamples,
                                           samplingBuffer,
                                           sourceData,
                                           topLeftCorner,
                                           bottomRightCorner,
                                           matrixSize,
                                           nSamples);
}// end of sampleCuboid
//----------------------------------------------------------------------------------------------------------------------

///Sample data inside one cuboid and store it to buffer, no post-processing.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kNone>
                                           (float*       samplingBuffer,
                                            const float*  sourceData,
                                            const dim3    topLeftCorner,
                                            const dim3    bottomRightCorner,
                                            const dim3    matrixSize,
                                            const size_t  nSamples);
/// Sample data inside one cuboid and store it to buffer, take the root mean square.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kRms>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
/// Sample data inside one cuboid and store it to buffer, take the maximum.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kMax>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);
/// Sample data inside one cuboid and store it to buffer, take the minimum.
template
void OutputStreamsCudaKernels::sampleCuboid<BaseOutputStream::ReduceOperator::kMin>
                                           (float*       samplingBuffer,
                                            const float* sourceData,
                                            const dim3   topLeftCorner,
                                            const dim3   bottomRightCorner,
                                            const dim3   matrixSize,
                                            const size_t nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Whole domain based sampling --------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Sample and the whole domain and apply a defined operator.
 */
template<BaseOutputStream::ReduceOperator reduceOp>
void OutputStreamsCudaKernels::sampleAll(float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples)
{
  MetalContext::getInstance().launchKernel(getKernelName("cudaSampleAll", reduceOp),
                                           nullptr,
                                           nSamples,
                                           samplingBuffer,
                                           sourceData,
                                           nSamples);
}// end of sampleMaxAll
//----------------------------------------------------------------------------------------------------------------------

/// Sample and the whole domain and apply a defined operator, take the root mean square.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kRms>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
/// Sample and the whole domain and apply a defined operator, take the maximum.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kMax>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);
/// Sample and the whole domain and apply a defined operator, take the minimum.
template
void OutputStreamsCudaKernels::sampleAll<BaseOutputStream::ReduceOperator::kMin>
                                        (float*       samplingBuffer,
                                         const float* sourceData,
                                         const size_t nSamples);

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Post-processing -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Calculate post-processing for RMS.
 */
void OutputStreamsCudaKernels::postProcessingRms(float*       samplingBuffer,
                                                 const float  scalingCoeff,
                                                 const size_t nSamples)
{
  MetalContext::getInstance().launchKernel("cudaPostProcessingRms",
                                           nullptr,
                                           nSamples,
                                           samplingBuffer,
                                           scalingCoeff,
                                           nSamples);
}// end of postProcessingRms
//----------------------------------------------------------------------------------------------------------------------
