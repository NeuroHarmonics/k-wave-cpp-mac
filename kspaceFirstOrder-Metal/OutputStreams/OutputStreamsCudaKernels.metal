/**
 * @file      OutputStreamsCudaKernels.metal
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file of cuda kernels used for data sampling (output streams), ported to Metal. The
 *            host interface is in OutputStreamsCudaKernels.cpp.
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

/**
 * @struct BaseOutputStream
 * @brief  Reduction operators of BaseOutputStream (OutputStreams/BaseOutputStream.h) used by the kernels.
 */
struct BaseOutputStream
{
  /// How to aggregate data.
  enum class ReduceOperator
  {
    /// Store actual data (time series).
    kNone,
    /// Calculate root mean square.
    kRms,
    /// Store maximum.
    kMax,
    /// Store minimum.
    kMin
  };
};// end of BaseOutputStream

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Index mask sampling --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to sample data based on index sensor mask. The operator is given by the template parameter.
 *
 * @param [out] samplingBuffer - Buffer to sample data in.
 * @param [in]  sourceData     - Source matrix.
 * @param [in]  sensorData     - Sensor mask.
 * @param [in]  nSamples       - Number of sampled points.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
kernel void cudaSampleIndex(device float*        samplingBuffer [[buffer(2)]],
                            const device float*  sourceData     [[buffer(3)]],
                            const device size_t* sensorData     [[buffer(4)]],
                            constant size_t&     nSamples       [[buffer(5)]],
                            KERNEL_ARGS)
{
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kNone:
      {
        samplingBuffer[i] = sourceData[sensorData[i]];
        break;
      }

      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[sensorData[i]] * sourceData[sensorData[i]]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[sensorData[i]]);
        break;
      }
    }// switch
  }// for
}// end of cudaSampleIndex
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//--------------------------------------------- Cuboid mask sampling -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Transform 3D coordinates within the cuboid into 1D coordinates within the matrix being sampled.
 *
 * @param [in] cuboidIdx         - Cuboid index.
 * @param [in] topLeftCorner     - Top left corner.
 * @param [in] bottomRightCorner - Bottom right corner.
 * @param [in] matrixSize        - Size of the matrix being sampled.
 * @return 1D index into the matrix being sampled.
 */
inline size_t transformCoordinates(const size_t cuboidIdx,
                                   const dim3   topLeftCorner,
                                   const dim3   bottomRightCorner,
                                   const dim3   matrixSize)
{
  dim3 localPosition;
  // Calculate the cuboid size
  dim3 cuboidSize(bottomRightCorner.x - topLeftCorner.x + 1,
                  bottomRightCorner.y - topLeftCorner.y + 1,
                  bottomRightCorner.z - topLeftCorner.z + 1);

  // Find coordinates within the cuboid
  size_t slabSize = cuboidSize.x * cuboidSize.y;
  localPosition.z =  cuboidIdx / slabSize;
  localPosition.y = (cuboidIdx % slabSize) / cuboidSize.x;
  localPosition.x = (cuboidIdx % slabSize) % cuboidSize.x;

  // Transform the coordinates to the global dimensions
  dim3 globalPosition(localPosition);
  globalPosition.z += topLeftCorner.z;
  globalPosition.y += topLeftCorner.y;
  globalPosition.x += topLeftCorner.x;

  // Calculate 1D index
  return (globalPosition.z * matrixSize.x * matrixSize.y +
          globalPosition.y * matrixSize.x +
          globalPosition.x);
}// end of transformCoordinates
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to sample data inside one cuboid, operation is selected by a template parameter.

 * @param [out] samplingBuffer    - Buffer to sample data in.
 * @param [in]  sourceData        - Source matrix.
 * @param [in]  topLeftCorner     - Top left corner of the cuboid.
 * @param [in]  bottomRightCorner - Bottom right corner of the cuboid.
 * @param [in]  matrixSize        - Dimension sizes of the matrix being sampled.
 * @param [in]  nSamples          - Number of grid points inside the cuboid.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
kernel void cudaSampleCuboid(device float*       samplingBuffer    [[buffer(2)]],
                             const device float* sourceData        [[buffer(3)]],
                             constant dim3&      topLeftCorner     [[buffer(4)]],
                             constant dim3&      bottomRightCorner [[buffer(5)]],
                             constant dim3&      matrixSize        [[buffer(6)]],
                             constant size_t&    nSamples          [[buffer(7)]],
                             KERNEL_ARGS)
{
  for (auto i = getIndex(); i < nSamples; i += getStride())
  {
    auto Position = transformCoordinates(i, topLeftCorner, bottomRightCorner, matrixSize);
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kNone:
      {
        samplingBuffer[i] = sourceData[Position];
        break;
      }

      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[Position] * sourceData[Position]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[Position]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[Position]);
        break;
      }
    }// switch
  }// for
}// end of cudaSampleCuboid
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Whole domain based sampling --------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to sample and aggregate the source matrix on the whole domain and apply a reduce operator.
 *
 * @param [in,out] samplingBuffer - Buffer to sample data in.
 * @param [in]     sourceData     - Source matrix.
 * @param [in]     nSamples       - Number of sampled points.
 */
template <BaseOutputStream::ReduceOperator reduceOp>
kernel void cudaSampleAll(device float*       samplingBuffer [[buffer(2)]],
                          const device float* sourceData     [[buffer(3)]],
                          constant size_t&    nSamples       [[buffer(4)]],
                          KERNEL_ARGS)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    switch (reduceOp)
    {
      case BaseOutputStream::ReduceOperator::kRms:
      {
        samplingBuffer[i] += (sourceData[i] * sourceData[i]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMax:
      {
        samplingBuffer[i] = max(samplingBuffer[i], sourceData[i]);
        break;
      }

      case BaseOutputStream::ReduceOperator::kMin:
      {
        samplingBuffer[i] = min(samplingBuffer[i], sourceData[i]);
        break;
      }
    }
  }
}// end of cudaSampleAll
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//-------------------------------------------------- Post-processing -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * CUDA kernel to apply post-processing for RMS.
 *
 * @param [in, out] samplingBuffer - Buffer to apply post-processing on.
 * @param [in]      scalingCoeff   - Scaling coeficinet for RMS.
 * @param [in]      nSamples       - Number of elements.
 */
kernel void cudaPostProcessingRms(device float*    samplingBuffer [[buffer(2)]],
                                  constant float&  scalingCoeff   [[buffer(3)]],
                                  constant size_t& nSamples       [[buffer(4)]],
                                  KERNEL_ARGS)
{
  for (size_t i = getIndex(); i < nSamples; i += getStride())
  {
    samplingBuffer[i] = sqrt(samplingBuffer[i] * scalingCoeff);
  }
}// end of cudaPostProcessingRMS
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Explicit kernel instances ----------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Kernel arguments of the index sampling.
#define SAMPLE_INDEX_ARGS device float*        samplingBuffer [[buffer(2)]], \
                          const device float*  sourceData     [[buffer(3)]], \
                          const device size_t* sensorData     [[buffer(4)]], \
                          constant size_t&     nSamples       [[buffer(5)]], \
                          KERNEL_ARGS
/// Kernel arguments of the cuboid sampling.
#define SAMPLE_CUBOID_ARGS device float*       samplingBuffer    [[buffer(2)]], \
                           const device float* sourceData        [[buffer(3)]], \
                           constant dim3&      topLeftCorner     [[buffer(4)]], \
                           constant dim3&      bottomRightCorner [[buffer(5)]], \
                           constant dim3&      matrixSize        [[buffer(6)]], \
                           constant size_t&    nSamples          [[buffer(7)]], \
                           KERNEL_ARGS
/// Kernel arguments of the whole domain sampling.
#define SAMPLE_ALL_ARGS device float*       samplingBuffer [[buffer(2)]], \
                        const device float* sourceData     [[buffer(3)]], \
                        constant size_t&    nSamples       [[buffer(4)]], \
                        KERNEL_ARGS

/// Shortcut for the reduction operators.
using ReduceOp = BaseOutputStream::ReduceOperator;

/// Sample the source matrix using the index sensor mask, no post-processing.
template [[host_name("cudaSampleIndex_kNone")]] kernel void cudaSampleIndex<ReduceOp::kNone>(SAMPLE_INDEX_ARGS);
/// Sample the source matrix using the index sensor mask, take the root mean square.
template [[host_name("cudaSampleIndex_kRms")]]  kernel void cudaSampleIndex<ReduceOp::kRms>(SAMPLE_INDEX_ARGS);
/// Sample the source matrix using the index sensor mask, take the maximum.
template [[host_name("cudaSampleIndex_kMax")]]  kernel void cudaSampleIndex<ReduceOp::kMax>(SAMPLE_INDEX_ARGS);
/// Sample the source matrix using the index sensor mask, take the minimum.
template [[host_name("cudaSampleIndex_kMin")]]  kernel void cudaSampleIndex<ReduceOp::kMin>(SAMPLE_INDEX_ARGS);

/// Sample data inside one cuboid and store it to buffer, no post-processing.
template [[host_name("cudaSampleCuboid_kNone")]] kernel void cudaSampleCuboid<ReduceOp::kNone>(SAMPLE_CUBOID_ARGS);
/// Sample data inside one cuboid and store it to buffer, take the root mean square.
template [[host_name("cudaSampleCuboid_kRms")]]  kernel void cudaSampleCuboid<ReduceOp::kRms>(SAMPLE_CUBOID_ARGS);
/// Sample data inside one cuboid and store it to buffer, take the maximum.
template [[host_name("cudaSampleCuboid_kMax")]]  kernel void cudaSampleCuboid<ReduceOp::kMax>(SAMPLE_CUBOID_ARGS);
/// Sample data inside one cuboid and store it to buffer, take the minimum.
template [[host_name("cudaSampleCuboid_kMin")]]  kernel void cudaSampleCuboid<ReduceOp::kMin>(SAMPLE_CUBOID_ARGS);

/// Sample and the whole domain and apply a defined operator, take the root mean square.
template [[host_name("cudaSampleAll_kRms")]] kernel void cudaSampleAll<ReduceOp::kRms>(SAMPLE_ALL_ARGS);
/// Sample and the whole domain and apply a defined operator, take the maximum.
template [[host_name("cudaSampleAll_kMax")]] kernel void cudaSampleAll<ReduceOp::kMax>(SAMPLE_ALL_ARGS);
/// Sample and the whole domain and apply a defined operator, take the minimum.
template [[host_name("cudaSampleAll_kMin")]] kernel void cudaSampleAll<ReduceOp::kMin>(SAMPLE_ALL_ARGS);
//----------------------------------------------------------------------------------------------------------------------
