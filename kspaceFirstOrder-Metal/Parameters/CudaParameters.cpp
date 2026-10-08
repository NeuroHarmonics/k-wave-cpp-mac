/**
 * @file      CudaParameters.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The header file for the class for setting CUDA kernel parameters.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      12 November  2015, 16:49 (created) \n
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

#include <cstring>
#include <stdexcept>

#include <Parameters/CudaParameters.h>
#include <Parameters/CudaDeviceConstants.cuh>
#include <Parameters/Parameters.h>

#include <Logger/Logger.h>

#include <Utils/MetalContext.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Default constructor.
 */
CudaParameters::CudaParameters()
  : mDeviceIdx(kDefaultDeviceIdx),
    mSolverTransposeBlockSize(kUndefinedSize),
    mSolverTransposeGirdSize(kUndefinedSize)
{

}// end of CudaParameters
//----------------------------------------------------------------------------------------------------------------------

/**
 * Return the name of device used.
 */
std::string CudaParameters::getDeviceName() const
{
  const std::string deviceName = MetalContext::getInstance().getDeviceName();
  if (deviceName == "")
  {
    return kOutFmtDeviceNameUnknown;
  }
  return deviceName;
}// end of getDeviceName
//----------------------------------------------------------------------------------------------------------------------

/**
 * Select cuda device for execution. If no device is specified, the first free is chosen. The routine also checks
 * whether the CUDA runtime and driver version match and whether the GPU is supported by the code. If there is no
 * free device is present, the code terminates with a runtime error.
 */
void CudaParameters::selectDevice(const int deviceIdx)
{
  // Check the device index if the user provided one
  const int nDevices = MetalContext::getDeviceCount();
  if ((deviceIdx != kDefaultDeviceIdx) && ((deviceIdx > nDevices - 1) || (deviceIdx < 0)))
  {
    throw std::runtime_error(Logger::formatMessage(kErrFmtBadDeviceIndex, deviceIdx, nDevices - 1));
  }

  // Create the device and compile the kernels, the system default device is used if no index is given.
  MetalContext::getInstance().init(deviceIdx);

  mDeviceIdx = (deviceIdx == kDefaultDeviceIdx) ? 0 : deviceIdx;
}// end of selectDevice
//----------------------------------------------------------------------------------------------------------------------

/**
 * Set the configuration of the transposition kernels. They work by processing tiles of 32x32 by 4 SIMD groups, every
 * thread group is responsible for a few 2D slabs. All other kernels are launched with one thread per element
 * (MetalContext::launchKernel), so they need no configuration.
 */
void CudaParameters::setKernelConfiguration()
{
  // Transposition kernels, each thread group (one SIMD group per tile) transposes a few slabs.
  mSolverTransposeBlockSize = dim3(32, 4 , 1);
  // Grid size for the transposition kernels, enough thread groups to fill the largest Apple GPUs.
  mSolverTransposeGirdSize = dim3(1024, 1, 1);
}// end of setKernelConfiguration
//----------------------------------------------------------------------------------------------------------------------

/**
 * Upload useful simulation constants into device constant memory.
 */
void CudaParameters::setUpDeviceConstants() const
{
  CudaDeviceConstants constantsToTransfer;

  Parameters& params = Parameters::getInstance();
  DimensionSizes fullDimSizes    = params.getFullDimensionSizes();
  DimensionSizes reducedDimSizes = params.getReducedDimensionSizes();


  // Set values for constant memory
  constantsToTransfer.simulationDimension = params.getSimulationDimension();

  constantsToTransfer.nx  = static_cast<unsigned int>(fullDimSizes.nx);
  constantsToTransfer.ny  = static_cast<unsigned int>(fullDimSizes.ny);
  constantsToTransfer.nz  = static_cast<unsigned int>(fullDimSizes.nz);
  constantsToTransfer.nElements = static_cast<unsigned int>(fullDimSizes.nElements());

  constantsToTransfer.nxComplex = static_cast<unsigned int>(reducedDimSizes.nx);
  constantsToTransfer.nyComplex = static_cast<unsigned int>(reducedDimSizes.ny);
  constantsToTransfer.nzComplex = static_cast<unsigned int>(reducedDimSizes.nz);
  constantsToTransfer.nElementsComplex = static_cast<unsigned int>(reducedDimSizes.nElements());

  constantsToTransfer.fftDivider  = 1.0f / fullDimSizes.nElements();
  constantsToTransfer.fftDividerX = 1.0f / fullDimSizes.nx;
  constantsToTransfer.fftDividerY = 1.0f / fullDimSizes.ny;
  constantsToTransfer.fftDividerZ = 1.0f / fullDimSizes.nz;

  constantsToTransfer.dt      = params.getDt();
  constantsToTransfer.dtBy2   = params.getDt() * 2.0f;
  constantsToTransfer.c2      = params.getC2Scalar();

  constantsToTransfer.rho0      = params.getRho0Scalar();
  constantsToTransfer.dtRho0    = params.getRho0Scalar() * params.getDt();
  constantsToTransfer.dtRho0Sgx = params.getDtRho0SgxScalar();
  constantsToTransfer.dtRho0Sgy = params.getDtRho0SgyScalar(),
  constantsToTransfer.dtRho0Sgz = params.getDtRho0SgzScalar(),

  constantsToTransfer.bOnA      = params.getBOnAScalar();
  constantsToTransfer.absorbTau = params.getAbsorbTauScalar();
  constantsToTransfer.absorbEta = params.getAbsorbEtaScalar();

  // Source masks
  constantsToTransfer.presureSourceSize = static_cast<unsigned int>(params.getPressureSourceIndexSize());
  constantsToTransfer.presureSourceMode = params.getPressureSourceMode();
  constantsToTransfer.presureSourceMany = static_cast<unsigned int>(params.getPressureSourceMany());

  constantsToTransfer.velocitySourceSize = static_cast<unsigned int>(params.getVelocitySourceIndexSize());
  constantsToTransfer.velocitySourceMode = params.getVelocitySourceMode();
  constantsToTransfer.velocitySourceMany = static_cast<unsigned int>(params.getVelocitySourceMany());

  constantsToTransfer.copyToDevice();
}// end of setUpDeviceConstants
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------ Protected methods -------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Private methods --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//
