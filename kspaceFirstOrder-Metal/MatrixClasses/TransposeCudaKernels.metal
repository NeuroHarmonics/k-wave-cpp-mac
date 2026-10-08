/**
 * @file      TransposeCudaKernels.metal
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file for CUDA transpose kernels for 3D FFTs, ported to Metal. The host interface is in
 *            TransposeCudaKernels.cpp.
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

/**
 * @namespace TransposeCudaKernels
 * @brief     Types used by the transposition kernels, same as in TransposeCudaKernels.cuh.
 */
namespace TransposeCudaKernels
{
  /**
   * @enum TransposePadding
   * @brief How the data is padded during matrix transposition.
   */
  enum class TransposePadding
  {
    /// No padding.
    kNone,
    /// Input matrix is padded.
    kInput,
    /// Output matrix is padded.
    kOutput,
    /// Both matrices are padded.
    kInputOutput
  };
}// end of TransposeCudaKernels

/**
 * Thread and thread group indices of the transposition kernels, named as in CUDA.
 */
#define TRANSPOSE_KERNEL_ARGS uint3 threadIdx [[thread_position_in_threadgroup]], \
                              uint3 blockIdx  [[threadgroup_position_in_grid]],   \
                              uint3 blockDim  [[threads_per_threadgroup]],        \
                              uint3 gridDim   [[threadgroups_per_grid]]

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Cuda kernel to transpose a 3D matrix of any dimension sizes in XY planes.\n
 * Every CUDA block in a 1D grid transposes a few slabs.
 * Every CUDA block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the shared memory padding is to eliminate bank
 * conflicts.
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 * In Metal, a warp is a SIMD group. Apple GPUs execute 32 threads per SIMD group, and the threads of a thread group
 * with the same threadIdx.y form one SIMD group, since blockDim.x is 32. The lambda functions of the CUDA code are
 * written out, as Metal does not support lambda expressions.
 *
 * @tparam      padding      - Which matrices are padded.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 * @tparam      tilesAtOnce  - How many tiles to transpose at once (4 is default).
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to be of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1 \n
 *          Grid has to be organized (N, 1 ,1)
 *
 */
template<TransposeCudaKernels::TransposePadding padding,
         int                                    warpSize,
         int                                    tilesAtOnce>
kernel void cudaTrasnposeReal3DMatrixXY(device float*       outputMatrix [[buffer(2)]],
                                        const device float* inputMatrix  [[buffer(3)]],
                                        constant dim3&      dimSizes     [[buffer(4)]],
                                        TRANSPOSE_KERNEL_ARGS)
{
  // We transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts.
  threadgroup float sharedTile[tilesAtOnce][warpSize][warpSize + 1];

  using TP = TransposeCudaKernels::TransposePadding;
  // Pad input and output dimension by 1 complex element if necessary
  const uint nxPadded = ((padding == TP::kInput)  || (padding == TP::kInputOutput))
                          ? 2 * (dimSizes.x / 2 + 1) : dimSizes.x;
  const uint nyPadded = ((padding == TP::kOutput) || (padding == TP::kInputOutput))
                          ? 2 * (dimSizes.y / 2 + 1) : dimSizes.y;

  const dim3 tileCount = {dimSizes.x / warpSize + 1, dimSizes.y / warpSize + 1, 1};

  // Run over all slabs. Each CUDA block processes a single slab.
  for (auto slabIdx = blockIdx.x; slabIdx < dimSizes.z; slabIdx += gridDim.x)
  {
    // Calculate offset of the slab
    const device float* inputSlab  = inputMatrix  + (nxPadded * dimSizes.y * slabIdx);
          device float* outputSlab = outputMatrix + (dimSizes.x * nyPadded * slabIdx);

    dim3 tileIdx = {0, 0, 0};

    // Go over the tiles in the y dimension. Multiple tiles processed simultaneously are under each other.
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // Go over the tiles in the x dimension.
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over the rows in a tile and load data. Process only rows lying within the matrix.
        for (uint row = 0; (row < warpSize) && ((tileIdx.y * warpSize + row) < dimSizes.y); row++)
        {
          // Copy data into shared tile.
          const uint globalX = tileIdx.x * warpSize + threadIdx.x;
          const uint globalY = tileIdx.y * warpSize + row;

          // If the column is still within the matrix.
          if (globalX < dimSizes.x)
          {
            sharedTile[threadIdx.y][row][threadIdx.x] = inputSlab[globalY * nxPadded + globalX];
          }
        }// Load data
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Go over the rows in a transposed tile and store data. Process only rows lying within the matrix.
        for (uint row = 0; (row < warpSize) && ((tileIdx.x * warpSize + row) < dimSizes.x); row++)
        {
          // Copy data to output matrix.
          const uint globalY = tileIdx.x * warpSize + row;
          const uint globalX = tileIdx.y * warpSize + threadIdx.x;

          // If the row of the output matrix is still within the matrix.
          if (globalX < dimSizes.y)
          {
            outputSlab[globalY * nyPadded + globalX] = sharedTile[threadIdx.y][threadIdx.x][row];
          }
        }// Store data
        simdgroup_barrier(mem_flags::mem_threadgroup);

      }// for x dimension
    }// for y dimension
  }// slab
}// end of cudaTrasnposeReal3DMatrixXY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to transpose a 3D matrix in XZ planes of any dimension sizes.\n
 * Every CUDA block in a 1D grid transposes a few slabs.
 * Every CUDA block is composed of a 2D mesh of threads. The y dim gives the number of tiles processed
 * simultaneously. Each tile is processed by a single thread warp.
 * The shared memory is used to coalesce memory accesses and the shared memory padding is to eliminate bank
 * conflicts.
 * As a part of the transposition, the matrices can be padded to conform with cuFFT.
 *
 * In Metal, a warp is a SIMD group, see cudaTrasnposeReal3DMatrixXY.
 *
 * @tparam      padding      - Which matrices are padded.
 * @tparam      warpSize     - Set the warp size. Built in value cannot be used due to shared memory allocation.
 * @tparam      tilesAtOnce  - How many tiles to transpose at once (4 is default).
 *
 * @param [out] outputMatrix - Output matrix.
 * @param [in]  inputMatrix  - Input  matrix.
 * @param [in]  dimSizes     - Dimension sizes of the original matrix.
 *
 * @warning A blockDim.x has to of a warp size (typically 32) \n
 *          blockDim.y should be between 1 and 4 (four tiles at once).
 *          blockDim.y has to be equal with the tilesAtOnce parameter.  \n
 *          blockDim.z must stay 1. \n
 *          Grid has to be organized (N, 1 ,1).
 *
 */
template<TransposeCudaKernels::TransposePadding padding,
         int                                    warpSize,
         int                                    tilesAtOnce>
kernel void cudaTrasnposeReal3DMatrixXZ(device float*       outputMatrix [[buffer(2)]],
                                        const device float* inputMatrix  [[buffer(3)]],
                                        constant dim3&      dimSizes     [[buffer(4)]],
                                        TRANSPOSE_KERNEL_ARGS)
{
  // We transpose tilesAtOnce tiles of warp size^2 at the same time, +1 solves bank conflicts.
  threadgroup float sharedTile[tilesAtOnce][warpSize][ warpSize + 1];

  using TP = TransposeCudaKernels::TransposePadding;
  // Pad input and output dimension by 1 complex element if necessary
  const uint nxPadded = ((padding == TP::kInput)  || (padding == TP::kInputOutput))
                          ? 2 * (dimSizes.x / 2 + 1) : dimSizes.x;
  const uint nzPadded = ((padding == TP::kOutput) || (padding == TP::kInputOutput))
                          ? 2 * (dimSizes.z / 2 + 1) : dimSizes.z;

  const dim3 tileCount = {dimSizes.x / warpSize + 1, dimSizes.z / warpSize + 1, 1};

  // Run over all XZ slabs. Each CUDA block processes a single slab.
  for (auto row = blockIdx.x; row < dimSizes.y; row += gridDim.x )
  {
    dim3 tileIdx = {0, 0, 0};

    // Go over all all tiles in the XZ slab. Transpose multiple slabs at the same time (one per Z)
    for (tileIdx.y = threadIdx.y; tileIdx.y < tileCount.y; tileIdx.y += blockDim.y)
    {
      // Go over the tiles in the x dimension.
      for (tileIdx.x = 0; tileIdx.x < tileCount.x; tileIdx.x++)
      {
        // Go over the slabs in a tile and load data. Process only slabs lying within the matrix.
        for (uint slab = 0; (slab < warpSize) && ((tileIdx.y * warpSize + slab) < dimSizes.z); slab++)
        {
          // Copy data into shared tile.
          const uint globalX = tileIdx.x * warpSize + threadIdx.x;
          const uint globalY = tileIdx.y * warpSize + slab;

          if (globalX < dimSizes.x)
          {
            sharedTile[threadIdx.y][slab][threadIdx.x]
                  = inputMatrix[globalY * nxPadded * dimSizes.y + row * nxPadded + globalX];
          }
        }// Load data
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Go over the slabs in a transposed tile and store data. Process only slabs lying within the matrix.
        for (uint slab = 0; (slab < warpSize) && ((tileIdx.x * warpSize + slab) < dimSizes.x); slab++)
        {
          // Copy data to output matrix.
          const uint globalY = tileIdx.x * warpSize + slab;
          const uint globalX = tileIdx.y * warpSize + threadIdx.x;

          if (globalX < dimSizes.z)
          {
            outputMatrix[globalY * dimSizes.y * nzPadded + (row * nzPadded) + globalX]
                  = sharedTile[threadIdx.y][threadIdx.x][slab];
          }
        }// Store data
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }// for x dimension
    }// for y dimension
  }// slab
}// end of cudaTrasnposeReal3DMatrixXZ
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Explicit kernel instances ----------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Kernel arguments of the transpositions.
#define TRANSPOSE_ARGS device float*       outputMatrix [[buffer(2)]], \
                       const device float* inputMatrix  [[buffer(3)]], \
                       constant dim3&      dimSizes     [[buffer(4)]], \
                       TRANSPOSE_KERNEL_ARGS

/// Transpose a real 3D matrix in the X-Y direction, input matrix padded, output matrix compact.
template [[host_name("cudaTrasnposeReal3DMatrixXY_kInput")]]
kernel void cudaTrasnposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInput, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Y direction, input matrix compact, output matrix padded.
template [[host_name("cudaTrasnposeReal3DMatrixXY_kOutput")]]
kernel void cudaTrasnposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kOutput, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix compact.
template [[host_name("cudaTrasnposeReal3DMatrixXY_kNone")]]
kernel void cudaTrasnposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kNone, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Y direction, input and output matrix padded.
template [[host_name("cudaTrasnposeReal3DMatrixXY_kInputOutput")]]
kernel void cudaTrasnposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInputOutput, 32, 4>(TRANSPOSE_ARGS);

/// Transpose a real 3D matrix in the X-Z direction, input matrix padded, output matrix compact.
template [[host_name("cudaTrasnposeReal3DMatrixXZ_kInput")]]
kernel void cudaTrasnposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInput, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Z direction, input matrix compact, output matrix padded.
template [[host_name("cudaTrasnposeReal3DMatrixXZ_kOutput")]]
kernel void cudaTrasnposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kOutput, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Z direction, input and output matrix compact.
template [[host_name("cudaTrasnposeReal3DMatrixXZ_kNone")]]
kernel void cudaTrasnposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kNone, 32, 4>(TRANSPOSE_ARGS);
/// Transpose a real 3D matrix in the X-Z direction, input and output matrix padded.
template [[host_name("cudaTrasnposeReal3DMatrixXZ_kInputOutput")]]
kernel void cudaTrasnposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInputOutput, 32, 4>(TRANSPOSE_ARGS);
//----------------------------------------------------------------------------------------------------------------------
