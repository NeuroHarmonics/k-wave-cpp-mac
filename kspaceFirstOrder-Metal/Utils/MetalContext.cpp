/**
 * @file      MetalContext.cpp
 *
 * @brief     The implementation file of the class holding the Metal device, command queue, compute pipelines and
 *            the buffers.
 *
 * @version   kspaceFirstOrder 3.6
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

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <Utils/MetalContext.h>
#include <Containers/MatrixContainer.h>
#include <Logger/Logger.h>

/// Source of all kernels, generated from the .metal files by the Makefile.
extern const char kMetalKernelsSource[];

//--------------------------------------------------------------------------------------------------------------------//
//---------------------------------------------------- Constants -----------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Number of threads in a thread group of 1D kernels (CUDA block size).
static constexpr size_t kThreadGroupSize1D = 256;

/// Error raised by a command buffer, set by the completion handler.
static std::atomic<bool> sGpuErrorFlag{false};
/// Description of the first error raised by a command buffer.
static std::string       sGpuErrorMessage;
/// Mutex protecting the error message.
static std::mutex        sGpuErrorMutex;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Get the instance of the singleton.
 */
MetalContext& MetalContext::getInstance()
{
  static MetalContext instance;
  return instance;
}// end of getInstance
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the number of Metal devices in the system.
 */
int MetalContext::getDeviceCount()
{
  NS::Array* devices = MTL::CopyAllDevices();
  const int  nDevices = (devices) ? int(devices->count()) : 0;
  if (devices)
  {
    devices->release();
  }
  return nDevices;
}// end of getDeviceCount
//----------------------------------------------------------------------------------------------------------------------

/**
 * Create the device and the command queue and compile the kernels.
 */
void MetalContext::init(const int deviceIdx)
{
  if (isInitialized())
  {
    return;
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  if (deviceIdx < 0)
  {
    mDevice = MTL::CreateSystemDefaultDevice();
  }
  else
  {
    NS::Array* devices = MTL::CopyAllDevices();
    if (devices && (NS::UInteger(deviceIdx) < devices->count()))
    {
      mDevice = static_cast<MTL::Device*>(devices->object(deviceIdx))->retain();
    }
    if (devices)
    {
      devices->release();
    }
  }

  // The device pointers stored in the matrix container need Metal 3 (macOS 13 or newer)
  if ((!mDevice) || (!mDevice->supportsFamily(MTL::GPUFamilyMetal3)))
  {
    pool->release();
    throw std::runtime_error(kErrFmtNoMetalDevice);
  }

  mCommandQueue    = mDevice->newCommandQueue();
  mConstantsBuffer = mDevice->newBuffer(kConstantsBufferSize, MTL::ResourceStorageModeShared);
  mContainerBuffer = mDevice->newBuffer(MatrixContainer::getMatrixIdxCount() * sizeof(uint64_t),
                                        MTL::ResourceStorageModeShared);
  std::memset(mConstantsBuffer->contents(), 0, mConstantsBuffer->length());
  std::memset(mContainerBuffer->contents(), 0, mContainerBuffer->length());

  pool->release();

  const char* profile = std::getenv("KWAVE_METAL_PROFILE");
  mProfilingGpu = (profile != nullptr) && (std::string(profile) == "gpu");
  mProfiling    = (profile != nullptr) && !mProfilingGpu;

  compileKernels();
}// end of init
//----------------------------------------------------------------------------------------------------------------------

/**
 * Release all Metal objects.
 */
void MetalContext::reset()
{
  if (!isInitialized())
  {
    return;
  }

  // Finish all work, errors are not reported any more
  commit();
  if (mLastCommitted)
  {
    mLastCommitted->waitUntilCompleted();
    mLastCommitted->release();
    mLastCommitted = nullptr;
  }

  if (mProfilingGpu && (mGpuLastEnd > mGpuFirstStart))
  {
    fprintf(stderr, "\nGPU busy %.3f s of %.3f s between the first and the last command buffer (%.1f%%)\n",
            mGpuBusy, mGpuLastEnd - mGpuFirstStart, 100.0 * mGpuBusy / (mGpuLastEnd - mGpuFirstStart));
  }

  // Print the GPU times by kernel, the longest first
  if (mProfiling && !mProfile.empty())
  {
    std::vector<std::pair<std::string, std::pair<double, size_t>>> times(mProfile.begin(), mProfile.end());
    std::sort(times.begin(), times.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });

    double total = 0.0;
    for (const auto& t : times)
    {
      total += t.second.first;
    }

    fprintf(stderr, "\nGPU time by kernel (KWAVE_METAL_PROFILE)\n");
    fprintf(stderr, "%-58s %8s %10s %10s %6s\n", "Kernel", "Calls", "Total ms", "ms/call", "%");
    for (const auto& t : times)
    {
      fprintf(stderr, "%-58.58s %8zu %10.1f %10.4f %6.1f\n", t.first.c_str(), t.second.second, 1e3 * t.second.first,
              1e3 * t.second.first / t.second.second, 100.0 * t.second.first / total);
    }
    fprintf(stderr, "%-58s %8s %10.1f\n", "Total", "", 1e3 * total);
    mProfile.clear();
  }

  for (auto& pipeline : mPipelines)
  {
    pipeline.second->release();
  }
  mPipelines.clear();

  for (auto& buffer : mBuffers)
  {
    buffer.second->release();
  }
  mBuffers.clear();

  mConstantsBuffer->release();
  mContainerBuffer->release();
  mLibrary->release();
  mCommandQueue->release();
  mDevice->release();

  mConstantsBuffer = nullptr;
  mContainerBuffer = nullptr;
  mLibrary         = nullptr;
  mCommandQueue    = nullptr;
  mDevice          = nullptr;
}// end of reset
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the name of the GPU.
 */
std::string MetalContext::getDeviceName() const
{
  return (mDevice) ? mDevice->name()->utf8String() : "";
}// end of getDeviceName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the size of memory the GPU can use without affecting performance.
 */
size_t MetalContext::getRecommendedMaxWorkingSetSize() const
{
  return (mDevice) ? mDevice->recommendedMaxWorkingSetSize() : 0;
}// end of getRecommendedMaxWorkingSetSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the size of memory allocated by the GPU.
 */
size_t MetalContext::getCurrentAllocatedSize() const
{
  return (mDevice) ? mDevice->currentAllocatedSize() : 0;
}// end of getCurrentAllocatedSize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate a shared storage buffer.
 */
void* MetalContext::allocate(const size_t sizeInBytes, void*& hostData)
{
  // Metal does not allocate empty buffers
  MTL::Buffer* buffer = mDevice->newBuffer(std::max<size_t>(sizeInBytes, 16), MTL::ResourceStorageModeShared);
  if (!buffer)
  {
    hostData = nullptr;
    return nullptr;
  }

  hostData = buffer->contents();
  mBuffers[buffer->gpuAddress()] = buffer;

  // A new buffer has to be made resident in the current encoder as well
  if (mEncoder)
  {
    commit();
  }

  return reinterpret_cast<void*>(buffer->gpuAddress());
}// end of allocate
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free a buffer.
 */
void MetalContext::deallocate(void* deviceData)
{
  auto it = mBuffers.find(reinterpret_cast<uint64_t>(deviceData));
  if (it == mBuffers.end())
  {
    return;
  }

  // The GPU may still use the buffer
  synchronize();

  it->second->release();
  mBuffers.erase(it);
}// end of deallocate
//----------------------------------------------------------------------------------------------------------------------

/**
 * Find the buffer a device pointer points into.
 */
MTL::Buffer* MetalContext::findBuffer(const void* deviceData, size_t& offset) const
{
  const uint64_t address = reinterpret_cast<uint64_t>(deviceData);

  auto it = mBuffers.upper_bound(address);
  if (it != mBuffers.begin())
  {
    --it;
    if (address < it->first + it->second->length())
    {
      offset = size_t(address - it->first);
      return it->second;
    }
  }

  throw std::runtime_error(kErrFmtMetalBufferNotFound);
}// end of findBuffer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Fill device memory with a byte value on the GPU.
 */
void MetalContext::memset(void* deviceData, const int value, const size_t sizeInBytes)
{
  const uint32_t byte  = uint32_t(value) & 0xFF;
  const uint32_t word  = byte | (byte << 8) | (byte << 16) | (byte << 24);
  const size_t   nWords = sizeInBytes / sizeof(uint32_t);

  launchKernel("cudaMemset", nullptr, nWords, static_cast<uint32_t*>(deviceData), word, nWords);
}// end of memset
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy the device constants into the constant buffer bound to every kernel.
 */
void MetalContext::setDeviceConstants(const void* data, const size_t size)
{
  if (size > kConstantsBufferSize)
  {
    throw std::runtime_error(kErrFmtMetalBufferNotFound);
  }
  // The GPU may still read the old values
  synchronize();
  std::memcpy(mConstantsBuffer->contents(), data, size);
}// end of setDeviceConstants
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy the table of matrix device pointers into the buffer bound to every kernel.
 */
void MetalContext::setMatrixContainer(const void* data, const size_t size)
{
  if (size > mContainerBuffer->length())
  {
    throw std::runtime_error(kErrFmtMetalBufferNotFound);
  }
  // The GPU may still read the old values
  synchronize();
  std::memcpy(mContainerBuffer->contents(), data, size);
}// end of setMatrixContainer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the compute encoder all GPU work is recorded into.
 */
MTL::ComputeCommandEncoder* MetalContext::getEncoder()
{
  if (!mEncoder)
  {
    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

    mCommandBuffer = mCommandQueue->commandBuffer()->retain();
    mCommandBuffer->addCompletedHandler([this](MTL::CommandBuffer* commandBuffer)
    {
      if (mProfilingGpu)
      {
        std::lock_guard<std::mutex> lock(sGpuErrorMutex);
        mGpuBusy      += commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime();
        mGpuFirstStart = (mGpuFirstStart == 0.0) ? commandBuffer->GPUStartTime() : mGpuFirstStart;
        mGpuLastEnd    = std::max(mGpuLastEnd, commandBuffer->GPUEndTime());
      }
      if (commandBuffer->status() == MTL::CommandBufferStatusError)
      {
        std::lock_guard<std::mutex> lock(sGpuErrorMutex);
        if (!sGpuErrorFlag)
        {
          sGpuErrorMessage = (commandBuffer->error())
                                ? commandBuffer->error()->localizedDescription()->utf8String() : "";
          sGpuErrorFlag = true;
        }
      }
    });

    // Serial dispatch, so each kernel sees the results of the previous one
    mEncoder = mCommandBuffer->computeCommandEncoder()->retain();

    // Matrices are reached through the matrix container, so all buffers have to be resident
    std::vector<const MTL::Resource*> resources;
    resources.reserve(mBuffers.size());
    for (const auto& buffer : mBuffers)
    {
      resources.push_back(buffer.second);
    }
    if (!resources.empty())
    {
      mEncoder->useResources(resources.data(), resources.size(), MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }

    pool->release();
  }

  return mEncoder;
}// end of getEncoder
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the command buffer the encoder belongs to.
 */
MTL::CommandBuffer* MetalContext::getCommandBuffer()
{
  getEncoder();
  return mCommandBuffer;
}// end of getCommandBuffer
//----------------------------------------------------------------------------------------------------------------------

/**
 * Submit the GPU work recorded so far without waiting for it.
 */
void MetalContext::commit()
{
  if (!mEncoder)
  {
    return;
  }

  mEncoder->endEncoding();
  mEncoder->release();
  mEncoder = nullptr;

  mCommandBuffer->commit();

  if (mLastCommitted)
  {
    mLastCommitted->release();
  }
  mLastCommitted = mCommandBuffer;
  mCommandBuffer = nullptr;
}// end of commit
//----------------------------------------------------------------------------------------------------------------------

/**
 * Submit the GPU work recorded so far and wait until all submitted work has finished.
 */
void MetalContext::synchronize()
{
  commit();

  // Command buffers complete in the order they were committed
  if (mLastCommitted)
  {
    waitForCommandBuffer(mLastCommitted);
  }
}// end of synchronize
//----------------------------------------------------------------------------------------------------------------------

/**
 * Record an event after the work submitted so far.
 */
void MetalContext::recordEvent(Event& event)
{
  commit();

  destroyEvent(event);
  if (mLastCommitted)
  {
    event = mLastCommitted->retain();
  }
}// end of recordEvent
//----------------------------------------------------------------------------------------------------------------------

/**
 * Wait until the work before the event has finished.
 */
void MetalContext::synchronizeEvent(Event& event)
{
  // Keep the GPU busy with the work recorded since the event
  commit();

  if (event)
  {
    waitForCommandBuffer(event);
  }
}// end of synchronizeEvent
//----------------------------------------------------------------------------------------------------------------------

/**
 * Profiling point, called after each kernel and FFT.
 */
void MetalContext::profile(const std::string& name)
{
  if (!mProfiling || !mEncoder)
  {
    return;
  }

  commit();
  mLastCommitted->waitUntilCompleted();

  auto& entry = mProfile[name];
  entry.first  += mLastCommitted->GPUEndTime() - mLastCommitted->GPUStartTime();
  entry.second += 1;
}// end of profile
//----------------------------------------------------------------------------------------------------------------------

/**
 * Destroy an event.
 */
void MetalContext::destroyEvent(Event& event)
{
  if (event)
  {
    event->release();
    event = nullptr;
  }
}// end of destroyEvent
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Compile the kernels from the source embedded in the binary.
 */
void MetalContext::compileKernels()
{
  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  MTL::CompileOptions* options = MTL::CompileOptions::alloc()->init();
  // Keep IEEE behaviour, the kernels are memory bound so fast math does not help
  options->setFastMathEnabled(false);
  options->setLanguageVersion(MTL::LanguageVersion3_0);

  // The kernels check their copy of the matrix indices against the host
  NS::Dictionary* macros = NS::Dictionary::dictionary(NS::Number::number(uint64_t(MatrixContainer::getMatrixIdxCount())),
                                                      NS::String::string("MATRIX_IDX_COUNT", NS::UTF8StringEncoding));
  options->setPreprocessorMacros(macros);

  NS::Error* error = nullptr;
  mLibrary = mDevice->newLibrary(NS::String::string(kMetalKernelsSource, NS::UTF8StringEncoding), options, &error);
  options->release();

  if (!mLibrary)
  {
    const std::string message = (error) ? error->localizedDescription()->utf8String() : "";
    pool->release();
    throw std::runtime_error(Logger::formatMessage(kErrFmtMetalCompilation, message.c_str()));
  }

  pool->release();
}// end of compileKernels
//----------------------------------------------------------------------------------------------------------------------

/**
 * Get the pipeline for a kernel, create it if necessary.
 */
MTL::ComputePipelineState* MetalContext::getPipeline(const std::string& kernelName, const KernelFlags* flags)
{
  std::string key = kernelName;
  if (flags)
  {
    key += "_" + std::to_string(flags->simulationDimension) + std::to_string(flags->rho0ScalarFlag)
         + std::to_string(flags->bOnAScalarFlag) + std::to_string(flags->c0ScalarFlag)
         + std::to_string(flags->alphaCoefScalarFlag);
  }

  auto it = mPipelines.find(key);
  if (it != mPipelines.end())
  {
    return it->second;
  }

  NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();

  NS::Error*     error    = nullptr;
  NS::String*    name     = NS::String::string(kernelName.c_str(), NS::UTF8StringEncoding);

  // All kernels see the function constants (Utils/CudaUtils.metal), so they need values even if they do not use them
  const KernelFlags defaultFlags = {0, false, false, false, false};
  const KernelFlags& values      = (flags) ? *flags : defaultFlags;

  // Function constant indices are given in Utils/CudaUtils.metal
  MTL::FunctionConstantValues* constantValues = MTL::FunctionConstantValues::alloc()->init();
  constantValues->setConstantValue(&values.simulationDimension, MTL::DataTypeInt,  NS::UInteger(0));
  constantValues->setConstantValue(&values.rho0ScalarFlag,      MTL::DataTypeBool, NS::UInteger(1));
  constantValues->setConstantValue(&values.bOnAScalarFlag,      MTL::DataTypeBool, NS::UInteger(2));
  constantValues->setConstantValue(&values.c0ScalarFlag,        MTL::DataTypeBool, NS::UInteger(3));
  constantValues->setConstantValue(&values.alphaCoefScalarFlag, MTL::DataTypeBool, NS::UInteger(4));

  MTL::Function* function = mLibrary->newFunction(name, constantValues, &error);
  constantValues->release();

  MTL::ComputePipelineState* pipeline = nullptr;
  if (function)
  {
    pipeline = mDevice->newComputePipelineState(function, &error);
    function->release();
  }

  if (!pipeline)
  {
    const std::string message = (error) ? error->localizedDescription()->utf8String() : "";
    pool->release();
    throw std::runtime_error(Logger::formatMessage(kErrFmtMetalKernel, kernelName.c_str(), message.c_str()));
  }

  pool->release();

  mPipelines[key] = pipeline;
  return pipeline;
}// end of getPipeline
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set the pipeline and bind the device constants and the matrix container.
 */
void MetalContext::beginKernel(const std::string& kernelName, const KernelFlags* flags)
{
  mCurrentPipeline   = getPipeline(kernelName, flags);
  mCurrentKernelName = kernelName;

  MTL::ComputeCommandEncoder* encoder = getEncoder();
  encoder->setComputePipelineState(mCurrentPipeline);
  encoder->setBuffer(mConstantsBuffer, 0, 0);
  encoder->setBuffer(mContainerBuffer, 0, 1);
}// end of beginKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Bind a device pointer.
 */
void MetalContext::setBufferArgument(const size_t index, const void* deviceData)
{
  if (!deviceData)
  {
    mEncoder->setBuffer(nullptr, 0, index);
    return;
  }

  size_t       offset = 0;
  MTL::Buffer* buffer = findBuffer(deviceData, offset);
  mEncoder->setBuffer(buffer, offset, index);
}// end of setBufferArgument
//----------------------------------------------------------------------------------------------------------------------

/**
 * Bind a value.
 */
void MetalContext::setBytesArgument(const size_t index, const void* data, const size_t size)
{
  mEncoder->setBytes(data, size, index);
}// end of setBytesArgument
//----------------------------------------------------------------------------------------------------------------------

/**
 * Dispatch the kernel over nThreads threads.
 */
void MetalContext::dispatchKernel(const size_t nThreads)
{
  if (nThreads == 0)
  {
    return;
  }

  const size_t threadGroupSize = std::min<size_t>(kThreadGroupSize1D,
                                                  mCurrentPipeline->maxTotalThreadsPerThreadgroup());

  mEncoder->dispatchThreads(MTL::Size(nThreads, 1, 1), MTL::Size(threadGroupSize, 1, 1));
  profile(mCurrentKernelName);
}// end of dispatchKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Dispatch the kernel over a grid of thread groups.
 */
void MetalContext::dispatchKernel(const dim3& gridSize, const dim3& blockSize)
{
  mEncoder->dispatchThreadgroups(MTL::Size(gridSize.x, gridSize.y, gridSize.z),
                                 MTL::Size(blockSize.x, blockSize.y, blockSize.z));
  profile(mCurrentKernelName);
}// end of dispatchKernel
//----------------------------------------------------------------------------------------------------------------------

/**
 * Wait for a command buffer and check its status.
 */
void MetalContext::waitForCommandBuffer(MTL::CommandBuffer* commandBuffer)
{
  commandBuffer->waitUntilCompleted();

  if (sGpuErrorFlag)
  {
    std::lock_guard<std::mutex> lock(sGpuErrorMutex);
    throw std::runtime_error(Logger::formatMessage(kErrFmtMetalExecution, sGpuErrorMessage.c_str()));
  }
}// end of waitForCommandBuffer
//----------------------------------------------------------------------------------------------------------------------
