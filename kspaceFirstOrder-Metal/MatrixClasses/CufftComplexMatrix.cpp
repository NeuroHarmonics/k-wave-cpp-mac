/**
 * @file      CufftComplexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the class implementing various and 1D FFTs using VkFFT with the
 *            Metal backend.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      09 August    2011, 13:10 (created) \n
 *            11 February  2020, 16:17 (revised)
 *
 * @copyright Copyright (C) 2011 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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
#include <stdexcept>

// VkFFT includes metal-cpp with its private implementation (NS_PRIVATE_IMPLEMENTATION and so on), so it must be
// included in this translation unit only.
#define VKFFT_BACKEND 5
#include <vkFFT.h>

#include <MatrixClasses/CufftComplexMatrix.h>
#include <MatrixClasses/TransposeCudaKernels.cuh>
#include <MatrixClasses/RealMatrix.h>
#include <Logger/Logger.h>
#include <KSpaceSolver/SolverCudaKernels.cuh>
#include <Utils/MetalContext.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @struct CufftComplexMatrix::FftPlan
 * @brief  VkFFT application and the buffer sizes it was created with.
 */
struct CufftComplexMatrix::FftPlan
{
  /// VkFFT application.
  VkFFTApplication application  = {};
  /// Size of the complex buffer in bytes.
  uint64_t         complexBytes = 0;
  /// Size of the real buffer in bytes.
  uint64_t         realBytes    = 0;
  /// Inverse (Complex-to-Real) transform?
  bool             inverse      = false;
  /// Out-of-place transform?
  bool             outOfPlace   = true;
};// end of FftPlan

CufftComplexMatrix::FftPlan* CufftComplexMatrix::sR2CFftPlanND = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sC2RFftPlanND = nullptr;

CufftComplexMatrix::FftPlan* CufftComplexMatrix::sR2CFftPlan1DX = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sR2CFftPlan1DY = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sR2CFftPlan1DZ = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sC2RFftPlan1DX = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sC2RFftPlan1DY = nullptr;
CufftComplexMatrix::FftPlan* CufftComplexMatrix::sC2RFftPlan1DZ = nullptr;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create an VkFFT plan for 2D/3D Real-to-Complex transform.
 */
void CufftComplexMatrix::createR2CFftPlanND(const DimensionSizes& inMatrixDims)
{
  const bool transformDims[3] = {true, true, Parameters::getInstance().isSimulation3D()};

  sR2CFftPlanND = createPlan(inMatrixDims, transformDims, true, false, kErrFmtCreateR2CFftPlanND);
}// end of createR2CFftPlanND
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 2D/3D Complex-to-Real transform.
 */
void CufftComplexMatrix::createC2RFftPlanND(const DimensionSizes& outMatrixDims)
{
  const bool transformDims[3] = {true, true, Parameters::getInstance().isSimulation3D()};

  sC2RFftPlanND = createPlan(outMatrixDims, transformDims, true, true, kErrFmtCreateC2RFftPlanND);
}// end of createC2RFftPlanND
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DX Real-to-Complex transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations. The transform is out-of-place, the y and z dimensions are batches.
 */
void CufftComplexMatrix::createR2CFftPlan1DX(const DimensionSizes& inMatrixDims)
{
  const bool transformDims[3] = {true, false, false};

  sR2CFftPlan1DX = createPlan(inMatrixDims, transformDims, true, false, kErrFmtCreateR2CFftPlan1DX);
}// end of createR2CFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DY Real-to-Complex transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations. The input matrix is transposed with every row padded, the FFT is done in-place.
 */
void CufftComplexMatrix::createR2CFftPlan1DY(const DimensionSizes& inMatrixDims)
{
  const bool transformDims[3] = {true, false, false};
  // Transposed dimensions
  const DimensionSizes fftDims(inMatrixDims.ny, inMatrixDims.nx, inMatrixDims.nz);

  sR2CFftPlan1DY = createPlan(fftDims, transformDims, false, false, kErrFmtCreateR2CFftPlan1DY);
}// end of createR2CFftPlan1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DZ Real-to-Complex transform. This routine throws en exception when called for 2D simulation.
 * The input matrix is transposed with every row padded, the FFT is done in-place.
 */
void CufftComplexMatrix::createR2CFftPlan1DZ(const DimensionSizes& inMatrixDims)
{
  if (Parameters::getInstance().isSimulation2D())
  {
    // Throw error when this routine is called for 2D simulations
    throw std::runtime_error(kErrFmtCannotCallR2CFftPlan1DZfor2D);
  }

  const bool transformDims[3] = {true, false, false};
  // Transposed dimensions
  const DimensionSizes fftDims(inMatrixDims.nz, inMatrixDims.ny, inMatrixDims.nx);

  sR2CFftPlan1DZ = createPlan(fftDims, transformDims, false, false, kErrFmtCreateR2CFftPlan1DZ);
}// end of createR2CFftPlan1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DX Complex-to-Real transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations. The transform is out-of-place, the y and z dimensions are batches.
 */
void CufftComplexMatrix::createC2RFftPlan1DX(const DimensionSizes& outMatrixDims)
{
  const bool transformDims[3] = {true, false, false};

  sC2RFftPlan1DX = createPlan(outMatrixDims, transformDims, true, true, kErrFmtCreateC2RFftPlan1DX);
}// end of createC2RFftPlan1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DY Complex-to-Real transform. Since nz == 1 in the 2D case, there's no need to modify this
 * routine for 2D simulations. The output matrix is transposed with every row padded, the FFT is done in-place.
 */
void CufftComplexMatrix::createC2RFftPlan1DY(const DimensionSizes& outMatrixDims)
{
  const bool transformDims[3] = {true, false, false};
  // Transposed dimensions
  const DimensionSizes fftDims(outMatrixDims.ny, outMatrixDims.nx, outMatrixDims.nz);

  sC2RFftPlan1DY = createPlan(fftDims, transformDims, false, true, kErrFmtCreateC2RFftPlan1DY);
}// end of createC2RFftPlan1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create VkFFT plan for 1DZ Complex-to-Real transform. This routine throws en exception when called for 2D simulation.
 * The output matrix is transposed with every row padded, the FFT is done in-place.
 */
void CufftComplexMatrix::createC2RFftPlan1DZ(const DimensionSizes& outMatrixDims)
{
  if (Parameters::getInstance().isSimulation2D())
  {
    // Throw error when this routine is called for 2D simulations
    throw std::runtime_error(kErrFmtCannotCallR2CFftPlan1DZfor2D);
  }

  const bool transformDims[3] = {true, false, false};
  // Transposed dimensions
  const DimensionSizes fftDims(outMatrixDims.nz, outMatrixDims.ny, outMatrixDims.nx);

  sC2RFftPlan1DZ = createPlan(fftDims, transformDims, false, true, kErrFmtCreateC2RFftPlan1DZ);
}// end of createC2RFftPlan1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destroy all static plans created by the application.
 */
void CufftComplexMatrix::destroyAllPlansAndStaticData()
{
  destroyPlan(sR2CFftPlanND);
  destroyPlan(sC2RFftPlanND);

  destroyPlan(sR2CFftPlan1DX);
  destroyPlan(sR2CFftPlan1DY);
  destroyPlan(sR2CFftPlan1DZ);

  destroyPlan(sC2RFftPlan1DX);
  destroyPlan(sC2RFftPlan1DY);
  destroyPlan(sC2RFftPlan1DZ);
}// end of destroyAllPlansAndStaticData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place (2D/3D) Real-to-Complex transform.
 */
void CufftComplexMatrix::computeR2CFftND(RealMatrix& inMatrix)
{
  executePlan(sR2CFftPlanND, inMatrix.getDeviceData(), mDeviceData, kErrFmtExecuteR2CFftPlanND);
}// end of computeR2CFftND
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place (2D/3D) Complex-to-Real transform.
 */
void CufftComplexMatrix::computeC2RFftND(RealMatrix& outMatrix)
{
  executePlan(sC2RFftPlanND, outMatrix.getDeviceData(), mDeviceData, kErrFmtExecuteC2RFftPlanND);
}// end of computeC2RFftND
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DX Real-to-Complex transform.
 */
void CufftComplexMatrix::computeR2CFft1DX(RealMatrix& inMatrix)
{
  executePlan(sR2CFftPlan1DX, inMatrix.getDeviceData(), mDeviceData, kErrFmtExecuteR2CFftPlan1DX);
}// end of computeR2CFft1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex transform. The matrix is first X<->Y transposed
 * followed by the 1D FFT. The matrix is left in the transposed format. \n
 *
 * As long as the blockSize.z == 1, the transposition works also for 2D case.
 */
void CufftComplexMatrix::computeR2CFft1DY(RealMatrix& inMatrix)
{
  /// Transpose a real 3D matrix in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

  TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kOutput>
                                              (mDeviceData,
                                               inMatrix.getDeviceData(),
                                               dimSizes);

  // Compute forward FFT. The FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  executePlan(sR2CFftPlan1DY, nullptr, mDeviceData, kErrFmtExecuteR2CFftPlan1DY);
}// end of computeR2CFft1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DZ Real-to-Complex transform. This routine throws en exception when called for 2D
 * simulation.
 */
void CufftComplexMatrix::computeR2CFft1DZ(RealMatrix& inMatrix)
{
  if (Parameters::getInstance().isSimulation3D())
  {
    /// Transpose a real 3D matrix in the X-Z direction
    dim3 dimSizes(static_cast<unsigned int>(inMatrix.getDimensionSizes().nx),
                  static_cast<unsigned int>(inMatrix.getDimensionSizes().ny),
                  static_cast<unsigned int>(inMatrix.getDimensionSizes().nz));

    TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kOutput>
                                                (mDeviceData,
                                                 inMatrix.getDeviceData(),
                                                 dimSizes);

    // Compute forward FFT. The FFT is calculated in-place (may be a bit slower than out-of-place, however
    // it does not request additional transfers and memory).
    executePlan(sR2CFftPlan1DZ, nullptr, mDeviceData, kErrFmtExecuteR2CFftPlan1DZ);
  }
  else
  {
    throwVkFFTException(VKFFT_ERROR_PLAN_NOT_INITIALIZED, kErrFmtExecuteR2CFftPlan1DZ);
  }
}// end of computeR2CFft1DZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DX Real-to-Complex transform.
 */
void CufftComplexMatrix::computeC2RFft1DX(RealMatrix& outMatrix)
{
  executePlan(sC2RFftPlan1DX, outMatrix.getDeviceData(), mDeviceData, kErrFmtExecuteC2RFftPlan1DX);
}// end of computeC2RFft1DX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer inverse out-of-place 1DY Real-to-Complex transform.
 * The matrix is taken in the transposed format and transposed at the end into a natural form. \n
 *
 * As long as the blockSize.z == 1, the transposition works also for 2D case.
 */
void CufftComplexMatrix::computeC2RFft1DY(RealMatrix& outMatrix)
{
  // Compute inverse FFT. The FFT is calculated in-place (may be a bit slower than out-of-place, however
  // it does not request additional transfers and memory).
  executePlan(sC2RFftPlan1DY, nullptr, mDeviceData, kErrFmtExecuteC2RFftPlan1DY);

  /// Transpose a real 3D matrix back in the X-Y direction
  dim3 dimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nx),
                static_cast<unsigned int>(outMatrix.getDimensionSizes().nz));

  TransposeCudaKernels::trasposeReal3DMatrixXY<TransposeCudaKernels::TransposePadding::kInput>
                                              (outMatrix.getDeviceData(),
                                               mDeviceData,
                                               dimSizes);
}// end of computeC2RFft1DY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Computer forward out-of-place 1DY Real-to-Complex transform. This routine throws en exception when called for 2D
 * simulation.
 */
void CufftComplexMatrix::computeC2RFft1DZ(RealMatrix& outMatrix)
{
  if (Parameters::getInstance().isSimulation3D())
  {
    // Compute inverse FFT. The FFT is calculated in-place (may be a bit slower than out-of-place, however
    // it does not request additional transfers and memory).
    executePlan(sC2RFftPlan1DZ, nullptr, mDeviceData, kErrFmtExecuteC2RFftPlan1DZ);

    /// Transpose a real 3D matrix in the Z<->X direction
    dim3 DimSizes(static_cast<unsigned int>(outMatrix.getDimensionSizes().nz),
                  static_cast<unsigned int>(outMatrix.getDimensionSizes().ny),
                  static_cast<unsigned int>(outMatrix.getDimensionSizes().nx));

    TransposeCudaKernels::trasposeReal3DMatrixXZ<TransposeCudaKernels::TransposePadding::kInput>
                                                (outMatrix.getDeviceData(),
                                                 getDeviceData(),
                                                 DimSizes);
  }
  else
  {
    throwVkFFTException(VKFFT_ERROR_PLAN_NOT_INITIALIZED, kErrFmtExecuteC2RFftPlan1DZ);
  }
}// end of computeC2RFft1DZ
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Protected methods ------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Create a VkFFT plan.
 *
 * Out-of-place plans (as cufftPlan3d, cufftPlan2d and cufftPlanMany used out-of-place in CUDA) read the real data from
 * an unpadded buffer passed as the input buffer and write the complex data into this matrix. The inverse transform
 * returns the result to the real buffer and, as cuFFT, may overwrite the complex data.
 *
 * In-place plans (the 1D transforms in y and z after transposition) work on real data padded to 2 * (nx / 2 + 1)
 * elements per row, the layout cuFFT uses for in-place transforms.
 */
CufftComplexMatrix::FftPlan* CufftComplexMatrix::createPlan(const DimensionSizes& fftDims,
                                                            const bool            transformDims[3],
                                                            const bool            outOfPlace,
                                                            const bool            inverse,
                                                            const std::string&    transformTypeName)
{
  MetalContext& metalContext = MetalContext::getInstance();

  const uint64_t nx  = fftDims.nx;
  const uint64_t ny  = fftDims.ny;
  const uint64_t nz  = fftDims.nz;
  const uint64_t nxR = nx / 2 + 1;

  FftPlan* plan = new FftPlan();
  plan->inverse      = inverse;
  plan->outOfPlace   = outOfPlace;
  plan->complexBytes = nxR * ny * nz * sizeof(cuFloatComplex);
  plan->realBytes    = (outOfPlace) ? nx * ny * nz * sizeof(float) : plan->complexBytes;

  VkFFTConfiguration configuration = {};

  configuration.FFTdim  = (nz > 1) ? 3 : 2;
  configuration.size[0] = nx;
  configuration.size[1] = ny;
  configuration.size[2] = nz;

  // Dimensions which are not transformed are batches
  configuration.omitDimension[1] = (transformDims[1]) ? 0 : 1;
  if (configuration.FFTdim == 3)
  {
    configuration.omitDimension[2] = (transformDims[2]) ? 0 : 1;
  }

  configuration.performR2C = 1;
  // Unnormalized transforms as cuFFT
  configuration.normalize  = 0;
  // Only one direction per plan as in CUDA
  configuration.makeForwardPlanOnly = (inverse) ? 0 : 1;
  configuration.makeInversePlanOnly = (inverse) ? 1 : 0;

  // Strides of the complex data
  configuration.bufferStride[0] = nxR;
  configuration.bufferStride[1] = nxR * ny;
  configuration.bufferStride[2] = nxR * ny * nz;

  if (outOfPlace)
  {
    // The real data is in its own, unpadded buffer
    configuration.isInputFormatted           = 1;
    configuration.inverseReturnToInputBuffer = 1;
    configuration.inputBufferStride[0]       = nx;
    configuration.inputBufferStride[1]       = nx * ny;
    configuration.inputBufferStride[2]       = nx * ny * nz;
  }

  MTL::Device*       device = metalContext.getDevice();
  MTL::CommandQueue* queue  = metalContext.getCommandQueue();
  configuration.device = device;
  configuration.queue  = queue;

  // Temporary buffers for planning, VkFFT only uses their size and type
  MTL::Buffer* complexBuffer = device->newBuffer(plan->complexBytes, MTL::ResourceStorageModeShared);
  MTL::Buffer* realBuffer    = (outOfPlace) ? device->newBuffer(plan->realBytes, MTL::ResourceStorageModeShared)
                                            : nullptr;

  configuration.buffer     = &complexBuffer;
  configuration.bufferSize = &plan->complexBytes;
  if (outOfPlace)
  {
    configuration.inputBuffer     = &realBuffer;
    configuration.inputBufferSize = &plan->realBytes;
  }

  // Plan creation may run work on the queue, so all work recorded so far is submitted first
  metalContext.commit();

  // VkFFT reads whole cache lines (coalescedMemory bytes) along strided axes. With the default for Apple GPUs, it
  // splits long strided axes (e.g., y in 2D grids of 1536^2 and more) into two passes over memory. A smaller value
  // fits the axis into one pass, which makes these transforms 25 to 35 percent faster. The first value that does
  // every axis in one pass is used, otherwise the one with the fewest passes.
  VkFFTResult result = VKFFT_SUCCESS;
  uint64_t    bestCoalescedMemory = 0;
  uint64_t    bestPasses          = UINT64_MAX;

  for (const uint64_t coalescedMemory : {0, 16, 8})
  {
    VkFFTApplication application = {};
    configuration.coalescedMemory = coalescedMemory;

    result = initializeVkFFT(&application, configuration);
    if (result != VKFFT_SUCCESS)
    {
      break;
    }

    const VkFFTPlan* fftPlan = (inverse) ? application.localFFTPlan_inverse : application.localFFTPlan;
    uint64_t passes = 0;
    for (uint64_t i = 0; i < configuration.FFTdim; i++)
    {
      passes += (configuration.omitDimension[i]) ? 0 : fftPlan->numAxisUploads[i];
    }
    deleteVkFFT(&application);

    if (passes < bestPasses)
    {
      bestPasses          = passes;
      bestCoalescedMemory = coalescedMemory;
    }
    if (passes <= configuration.FFTdim)
    {
      break;
    }
  }

  if (result == VKFFT_SUCCESS)
  {
    configuration.coalescedMemory = bestCoalescedMemory;
    result = initializeVkFFT(&plan->application, configuration);
  }

  complexBuffer->release();
  if (realBuffer)
  {
    realBuffer->release();
  }

  if (result != VKFFT_SUCCESS)
  {
    delete plan;
    throwVkFFTException(result, transformTypeName);
  }

  return plan;
}// end of createPlan
//----------------------------------------------------------------------------------------------------------------------

/**
 * Append the transform to the GPU work. The FFT is recorded into the compute encoder used by all kernels, so it is
 * ordered with them the same way as cuFFT calls on the default stream.
 */
void CufftComplexMatrix::executePlan(FftPlan*           plan,
                                     const float*       realData,
                                     float*             complexData,
                                     const std::string& transformTypeName)
{
  if (!plan)
  {
    throwVkFFTException(VKFFT_ERROR_PLAN_NOT_INITIALIZED, transformTypeName);
  }

  MetalContext& metalContext = MetalContext::getInstance();

  // Matrices always start at the beginning of their buffer. VkFFT can take offsets (specifyOffsetsAtLaunch), but
  // version 1.3.4 generates invalid Metal code with them.
  size_t       complexOffset = 0;
  MTL::Buffer* complexBuffer = metalContext.findBuffer(complexData, complexOffset);

  size_t       realOffset = 0;
  MTL::Buffer* realBuffer = (plan->outOfPlace) ? metalContext.findBuffer(realData, realOffset) : nullptr;

  if ((complexOffset != 0) || (realOffset != 0))
  {
    throwVkFFTException(VKFFT_ERROR_EMPTY_buffer, transformTypeName);
  }

  VkFFTLaunchParams launchParams = {};
  launchParams.commandEncoder = metalContext.getEncoder();
  launchParams.commandBuffer  = metalContext.getCommandBuffer();
  launchParams.buffer         = &complexBuffer;
  if (plan->outOfPlace)
  {
    launchParams.inputBuffer = &realBuffer;
  }

  const VkFFTResult result = VkFFTAppend(&plan->application, (plan->inverse) ? 1 : -1, &launchParams);
  if (result != VKFFT_SUCCESS)
  {
    throwVkFFTException(result, transformTypeName);
  }

  metalContext.profile("VkFFT " + transformTypeName);
}// end of executePlan
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destroy a plan.
 */
void CufftComplexMatrix::destroyPlan(FftPlan*& plan)
{
  if (plan)
  {
    // The GPU may still use the plan
    MetalContext::getInstance().synchronize();

    deleteVkFFT(&plan->application);
    delete plan;
    plan = nullptr;
  }
}// end of destroyPlan
//----------------------------------------------------------------------------------------------------------------------

/**
 * Throw VkFFT exception.
 */
void CufftComplexMatrix::throwVkFFTException(const int          vkfftError,
                                             const std::string& transformTypeName)
{
  throw std::runtime_error(Logger::formatMessage(kErrFmtVkFFTError, vkfftError, transformTypeName.c_str()));
}// end of throwVkFFTException
//----------------------------------------------------------------------------------------------------------------------
