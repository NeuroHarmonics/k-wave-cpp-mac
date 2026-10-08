/**
 * @file      MetalContext.h
 *
 * @brief     The header file of the class holding the Metal device, command queue, compute pipelines and the
 *            buffers. It takes the place of the CUDA runtime in the Metal port (device selection, memory allocation,
 *            kernel launches on the default stream, events and synchronization).
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

#ifndef METAL_CONTEXT_H
#define METAL_CONTEXT_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <Utils/CudaTypes.h>

namespace MTL
{
  class Buffer;
  class CommandBuffer;
  class CommandQueue;
  class ComputeCommandEncoder;
  class ComputePipelineState;
  class Device;
  class Library;
}

/**
 * @class   MetalContext
 * @brief   Singleton holding the Metal device, the command queue, the compiled kernels and all GPU buffers.
 *
 * @details All GPU work goes into one serial compute encoder of the current command buffer, so it executes in the
 *          order it was submitted, like the CUDA default stream. The command buffer is committed when an event is
 *          recorded, when the CPU has to wait for the GPU, or when the solver ends a time step.
 *
 *          Memory is allocated as shared storage buffers, which both the CPU and the GPU can access. Device pointers
 *          are the GPU addresses of the buffers (MTL::Buffer::gpuAddress), so they can be stored in tables read by
 *          the kernels, offset by pointer arithmetic and passed to kernels, but not dereferenced on the CPU, exactly
 *          like CUDA device pointers.
 */
class MetalContext
{
  public:
    /**
     * @struct KernelFlags
     * @brief  Values of the function constants the solver kernels are specialized with. They correspond to the
     *         template parameters of the CUDA kernels.
     */
    struct KernelFlags
    {
      /// Simulation dimension (0 for 2D, 1 for 3D).
      int  simulationDimension;
      /// Is density homogeneous?
      bool rho0ScalarFlag;
      /// Is nonlinearity homogeneous?
      bool bOnAScalarFlag;
      /// Is sound speed homogeneous?
      bool c0ScalarFlag;
      /// Is absorption homogeneous?
      bool alphaCoefScalarFlag;
    };

    /// Event recorded into the GPU work stream (a committed command buffer).
    using Event = MTL::CommandBuffer*;

    /// Get the instance of the singleton.
    static MetalContext& getInstance();
    /// Get the number of Metal devices in the system.
    static int getDeviceCount();

    /// Copy constructor not allowed.
    MetalContext(const MetalContext&) = delete;
    /// Operator = not allowed.
    MetalContext& operator=(const MetalContext&) = delete;

    /**
     * @brief  Create the device and the command queue and compile the kernels.
     * @param  [in] deviceIdx - Index of the device in MTLCopyAllDevices, the system default device if negative.
     * @throw  std::runtime_error - If there is no Metal 3 device or the kernels cannot be compiled.
     */
    void init(const int deviceIdx = -1);
    /// Release all Metal objects. The buffers must have been freed before.
    void reset();
    /// Is the device created?
    bool isInitialized() const { return mDevice != nullptr; }

    /// Get the name of the GPU.
    std::string getDeviceName() const;
    /// Get the size of memory the GPU can use without affecting performance.
    size_t getRecommendedMaxWorkingSetSize() const;
    /// Get the size of memory allocated by the GPU.
    size_t getCurrentAllocatedSize() const;

    /**
     * @brief  Allocate a shared storage buffer.
     * @param  [in] sizeInBytes - Size of the buffer.
     * @param  [out] hostData   - Pointer to the buffer for the CPU.
     * @return Device pointer to the buffer (GPU address), nullptr if the allocation failed.
     */
    void* allocate(const size_t sizeInBytes, void*& hostData);
    /**
     * @brief Free a buffer.
     * @param [in] deviceData - Device pointer returned by allocate.
     */
    void  deallocate(void* deviceData);
    /**
     * @brief  Find the buffer a device pointer points into.
     * @param  [in]  deviceData - Device pointer.
     * @param  [out] offset     - Offset of the pointer from the start of the buffer in bytes.
     * @return The buffer.
     * @throw  std::runtime_error - If the pointer is not inside any buffer.
     */
    MTL::Buffer* findBuffer(const void* deviceData, size_t& offset) const;
    /**
     * @brief Fill device memory with a byte value on the GPU, the counterpart of cudaMemset.
     * @param [in] deviceData  - Device pointer, aligned to 4 bytes.
     * @param [in] value       - Byte value.
     * @param [in] sizeInBytes - Size of the memory to fill, a multiple of 4 bytes.
     */
    void memset(void* deviceData, const int value, const size_t sizeInBytes);

    /**
     * @brief Copy the device constants into the constant buffer bound to every kernel.
     * @param [in] data - Device constants.
     * @param [in] size - Size of the structure.
     */
    void setDeviceConstants(const void* data, const size_t size);
    /**
     * @brief Copy the table of matrix device pointers into the buffer bound to every kernel.
     * @param [in] data - Table of device pointers.
     * @param [in] size - Size of the table.
     */
    void setMatrixContainer(const void* data, const size_t size);

    /**
     * @brief Launch a 1D kernel over nThreads threads, the counterpart of kernel<<<grid, block>>>(args...).
     *
     * Pointer arguments must be device pointers and are bound as buffers, other arguments are copied by value.
     * They are bound in order from buffer index 2 (0 holds the device constants and 1 the matrix container).
     *
     * @param [in] kernelName - Name of the kernel function.
     * @param [in] flags      - Function constants for the solver kernels, nullptr for other kernels.
     * @param [in] nThreads   - Number of threads to launch.
     * @param [in] args       - Kernel arguments.
     */
    template<typename... Args>
    void launchKernel(const std::string& kernelName,
                      const KernelFlags* flags,
                      const size_t       nThreads,
                      const Args&...     args)
    {
      beginKernel(kernelName, flags);
      setKernelArguments(2, args...);
      dispatchKernel(nThreads);
    }

    /**
     * @brief Launch a kernel over a grid of thread groups, the counterpart of kernel<<<gridSize, blockSize>>>(args...).
     *
     * @param [in] kernelName - Name of the kernel function.
     * @param [in] gridSize   - Number of thread groups.
     * @param [in] blockSize  - Number of threads in a thread group.
     * @param [in] args       - Kernel arguments.
     */
    template<typename... Args>
    void launchKernel(const std::string& kernelName,
                      const dim3&        gridSize,
                      const dim3&        blockSize,
                      const Args&...     args)
    {
      beginKernel(kernelName, nullptr);
      setKernelArguments(2, args...);
      dispatchKernel(gridSize, blockSize);
    }

    /**
     * @brief  Get the compute encoder all GPU work is recorded into. A new command buffer and encoder are created if
     *         needed.
     * @return Compute encoder.
     */
    MTL::ComputeCommandEncoder* getEncoder();
    /// Get the command buffer the encoder belongs to.
    MTL::CommandBuffer*         getCommandBuffer();
    /// Get the device.
    MTL::Device*                getDevice()       { return mDevice; }
    /// Get the command queue.
    MTL::CommandQueue*          getCommandQueue() { return mCommandQueue; }

    /// Submit the GPU work recorded so far without waiting for it.
    void commit();
    /**
     * @brief Submit the GPU work recorded so far and wait until all submitted work has finished, the counterpart of
     *        cudaDeviceSynchronize.
     * @throw std::runtime_error - If the GPU reported an error.
     */
    void synchronize();

    /**
     * @brief Record an event after the work submitted so far, the counterpart of cudaEventRecord.
     * @param [in, out] event - Event to record.
     */
    void recordEvent(Event& event);
    /**
     * @brief Wait until the work before the event has finished, the counterpart of cudaEventSynchronize.
     *        Work recorded since is submitted first, so the GPU keeps running while the CPU continues.
     * @param [in] event - Event to wait for.
     * @throw std::runtime_error - If the GPU reported an error.
     */
    void synchronizeEvent(Event& event);
    /**
     * @brief Destroy an event.
     * @param [in, out] event - Event to destroy.
     */
    void destroyEvent(Event& event);

    /**
     * @brief Profiling point, called after each kernel and FFT. If the environment variable KWAVE_METAL_PROFILE is
     *        set to 1, the work recorded so far is run on its own and its GPU time is added to the given name. The
     *        times are printed by reset(). Profiling serializes the GPU work, so the total run time gets longer.
     *        With KWAVE_METAL_PROFILE=gpu, the work is not split and reset() prints how long the GPU was busy.
     * @param [in] name - Name the GPU time is added to.
     */
    void profile(const std::string& name);

  private:
    /// Size of the buffer for the device constants.
    static constexpr size_t kConstantsBufferSize = 256;

    /// Constructor.
    MetalContext() = default;
    /// Destructor.
    ~MetalContext() = default;

    /// Compile the kernels from the source embedded in the binary.
    void compileKernels();
    /// Get the pipeline for a kernel, create it if necessary.
    MTL::ComputePipelineState* getPipeline(const std::string& kernelName, const KernelFlags* flags);

    /// Set the pipeline and bind the device constants and the matrix container.
    void beginKernel(const std::string& kernelName, const KernelFlags* flags);
    /// Bind a device pointer.
    void setBufferArgument(const size_t index, const void* deviceData);
    /// Bind a value.
    void setBytesArgument(const size_t index, const void* data, const size_t size);
    /// Dispatch the kernel over nThreads threads.
    void dispatchKernel(const size_t nThreads);
    /// Dispatch the kernel over a grid of thread groups.
    void dispatchKernel(const dim3& gridSize, const dim3& blockSize);
    /// Wait for a command buffer and check its status.
    void waitForCommandBuffer(MTL::CommandBuffer* commandBuffer);

    /// End of the argument list.
    void setKernelArguments(const size_t) {}

    /// Bind the kernel arguments one by one.
    template<typename T, typename... Rest>
    void setKernelArguments(const size_t index, const T& arg, const Rest&... rest)
    {
      if constexpr (std::is_pointer_v<T>)
      {
        setBufferArgument(index, arg);
      }
      else
      {
        static_assert(std::is_trivially_copyable_v<T>, "Kernel arguments must be pointers or plain values");
        setBytesArgument(index, &arg, sizeof(T));
      }
      setKernelArguments(index + 1, rest...);
    }

    /// Metal device.
    MTL::Device*                mDevice           = nullptr;
    /// Command queue.
    MTL::CommandQueue*          mCommandQueue     = nullptr;
    /// Compiled kernels.
    MTL::Library*               mLibrary          = nullptr;
    /// Command buffer being recorded.
    MTL::CommandBuffer*         mCommandBuffer    = nullptr;
    /// Compute encoder being recorded.
    MTL::ComputeCommandEncoder* mEncoder          = nullptr;
    /// Last committed command buffer.
    MTL::CommandBuffer*         mLastCommitted    = nullptr;
    /// Pipeline being dispatched.
    MTL::ComputePipelineState*  mCurrentPipeline  = nullptr;
    /// Name of the kernel being dispatched.
    std::string                 mCurrentKernelName;

    /// Is profiling by kernel enabled (KWAVE_METAL_PROFILE=1)?
    bool                        mProfiling        = false;
    /// Is profiling of the GPU busy time enabled (KWAVE_METAL_PROFILE=gpu)?
    bool                        mProfilingGpu     = false;
    /// GPU busy time, start of the first and end of the last command buffer in seconds (KWAVE_METAL_PROFILE=gpu).
    double                      mGpuBusy = 0.0, mGpuFirstStart = 0.0, mGpuLastEnd = 0.0;
    /// GPU time in seconds and number of calls by kernel or FFT name.
    std::map<std::string, std::pair<double, size_t>> mProfile;

    /// Buffer with the device constants.
    MTL::Buffer*                mConstantsBuffer  = nullptr;
    /// Buffer with the matrix container.
    MTL::Buffer*                mContainerBuffer  = nullptr;

    /// Allocated buffers ordered by their GPU address.
    std::map<uint64_t, MTL::Buffer*>                  mBuffers;
    /// Compute pipelines by kernel name and flags.
    std::map<std::string, MTL::ComputePipelineState*> mPipelines;
};// end of MetalContext
//----------------------------------------------------------------------------------------------------------------------

#endif /* METAL_CONTEXT_H */
