/**
 * @file      BaseFloatMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the base class for single precisions floating point numbers (floats).
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 July      2011, 12:13 (created) \n
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

#include <MatrixClasses/BaseFloatMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Utils/MetalContext.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
BaseFloatMatrix::BaseFloatMatrix()
  : BaseMatrix(),
    mDimensionSizes(),
    mSize(0),
    mCapacity(0),
    mHostData(nullptr),
    mDeviceData(nullptr)
{

}// end of BaseFloatMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements in parallel. \n
 * Also work as the first touch strategy on NUMA machines.
 */
void BaseFloatMatrix::zeroMatrix()
{
  #pragma omp parallel for schedule (static)
  for (size_t i = 0; i < mCapacity; i++)
  {
    mHostData[i] = 0.0f;
  }
}// end of zeroMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements on device.
 */
void BaseFloatMatrix::zeroDeviceMatrix()
{
  MetalContext::getInstance().memset(mDeviceData, 0, mCapacity * sizeof(float));
}// end of zeroDeviceMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Calculate matrix = scalar / matrix.
 */
void BaseFloatMatrix::scalarDividedBy(const float scalar)
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < mCapacity; i++)
  {
    mHostData[i] = scalar / mHostData[i];
  }
}// end of scalarDividedBy
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from host -> device (CPU -> GPU).
 * The host and device data share the same memory, so there is nothing to copy.
 */
void BaseFloatMatrix::copyToDevice()
{

}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from device -> host (GPU -> CPU).
 * The host and device data share the same memory, so it is enough to wait until the GPU has finished.
 */
void BaseFloatMatrix::copyFromDevice()
{
  MetalContext::getInstance().synchronize();
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the capacity. \n
 * The memory is a Metal buffer shared by the CPU and the GPU (page aligned) and zeroed.
 */
void BaseFloatMatrix::allocateMemory()
{
  // Size of memory to allocate
  size_t sizeInBytes = mCapacity * sizeof(float);

  // Allocate memory shared by the CPU and the GPU
  void* hostData = nullptr;
  mDeviceData = static_cast<float*>(MetalContext::getInstance().allocate(sizeInBytes, hostData));
  mHostData   = static_cast<float*>(hostData);

  if ((!mDeviceData) || (!mHostData))
  {
    throw std::bad_alloc();
  }
  // This has to be done for simulations based on input sources
  zeroMatrix();
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseFloatMatrix::freeMemory()
{
  // Free memory shared by the CPU and the GPU
  if (mDeviceData)
  {
    MetalContext::getInstance().deallocate(mDeviceData);
  }
  mHostData   = nullptr;
  mDeviceData = nullptr;
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
