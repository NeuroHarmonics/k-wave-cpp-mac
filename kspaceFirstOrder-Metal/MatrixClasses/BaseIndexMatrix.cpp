/**
 * @file      BaseIndexMatrix.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file containing the base class for index matrices (based on the size_t datatype).
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      26 July      2011, 14:17 (created) \n
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

#include <MatrixClasses/BaseIndexMatrix.h>
#include <Utils/DimensionSizes.h>
#include <Utils/MetalContext.h>
#include <Logger/Logger.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
BaseIndexMatrix::BaseIndexMatrix()
  : BaseMatrix(),
    mDimensionSizes(),
    mSize(0),
    mCapacity(0),
    mHostData(nullptr),
    mDeviceData(nullptr)
{

}// end of BaseIndexMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Zero all allocated elements.
 */
void BaseIndexMatrix::zeroMatrix()
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < mCapacity; i++)
  {
    mHostData[i] = size_t(0);
  }
}// end of zeroMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from host -> device (CPU -> GPU).
 * The host and device data share the same memory, so there is nothing to copy.
 */
void BaseIndexMatrix::copyToDevice()
{

}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Copy data from device-> host (GPU -> CPU).
 * The host and device data share the same memory, so it is enough to wait until the GPU has finished.
 */
void BaseIndexMatrix::copyFromDevice()
{
  MetalContext::getInstance().synchronize();
}// end of copyFromDevice
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Memory allocation based on the capacity. \n
 *
 * The memory is a Metal buffer shared by the CPU and the GPU (page aligned).
 */
void BaseIndexMatrix::allocateMemory()
{
  // Size of memory to allocate
  size_t sizeInBytes = mCapacity * sizeof(size_t);

  // Allocate memory shared by the CPU and the GPU
  void* hostData = nullptr;
  mDeviceData = static_cast<size_t*>(MetalContext::getInstance().allocate(sizeInBytes, hostData));
  mHostData   = static_cast<size_t*>(hostData);

  if ((!mDeviceData) || (!mHostData))
  {
    throw std::bad_alloc();
  }
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory.
 */
void BaseIndexMatrix::freeMemory()
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
//------------------------------------------------ Private methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
