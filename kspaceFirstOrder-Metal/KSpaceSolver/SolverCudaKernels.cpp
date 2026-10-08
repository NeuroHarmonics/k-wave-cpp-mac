/**
 * @file      SolverCudaKernels.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation of the interface to all cuda kernels used in the GPU implementation of the k-space
 *            solver. The kernels are in SolverCudaKernels.metal and launched through MetalContext.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      11 March     2013, 13:10 (created) \n
 *            11 February  2020, 16:14 (revised)
 *
 * @copyright Copyright (C) 2013 - 2020 SC\@FIT Research Group, Brno University of Technology, Brno, CZ.
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

#include <KSpaceSolver/SolverCudaKernels.cuh>
#include <Logger/Logger.h>
#include <Utils/MetalContext.h>

/// Shortcut for matrix id datatype.
using MI = MatrixContainer::MatrixIdx;
/// Shortcut for Simulation dimension datatype.
using SD = Parameters::SimulationDimension;

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Global methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @brief  Get the number of elements of the real matrices, the number of threads for most kernels.
 * @return Number of elements.
 */
inline size_t getElementCount()
{
  return Parameters::getInstance().getFullDimensionSizes().nElements();
}// end of getElementCount
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief  Get the number of elements of the complex matrices, the number of threads for kernels in k-space.
 * @return Number of elements.
 */
inline size_t getElementCountComplex()
{
  return Parameters::getInstance().getReducedDimensionSizes().nElements();
}// end of getElementCountComplex
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Launch a solver kernel over nThreads threads, specialized with the template parameters of SolverCudaKernels
 *        (the function constants of the Metal kernels).
 *
 * @param [in] kernelName - Name of the kernel.
 * @param [in] nThreads   - Number of threads, one per element.
 * @param [in] args       - Kernel arguments, device pointers or values.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag,
         typename... Args>
inline void launchKernelWithFlags(const char* kernelName, const size_t nThreads, const Args&... args)
{
  static constexpr MetalContext::KernelFlags flags = {int(simulationDimension),
                                                      rho0ScalarFlag,
                                                      bOnAScalarFlag,
                                                      c0ScalarFlag,
                                                      alphaCoefScalarFlag};

  MetalContext::getInstance().launchKernel(kernelName, &flags, nThreads, args...);
}// end of launchKernelWithFlags
//----------------------------------------------------------------------------------------------------------------------

/// Launch a solver kernel with the template parameters of the SolverCudaKernels class it is called from.
#define launchSolverKernel(...) launchKernelWithFlags<simulationDimension, \
                                                      rho0ScalarFlag,      \
                                                      bOnAScalarFlag,      \
                                                      c0ScalarFlag,        \
                                                      alphaCoefScalarFlag>(__VA_ARGS__)

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 *  Interface to kernel to compute spectral part of pressure gradient in between FFTs.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureGradient()
{
  launchSolverKernel("cudaComputePressureGradient",
                     getElementCountComplex());
}// end of computePressureGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to the cuda kernel computing new values for particle velocity on a uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityUniform()
{
  launchSolverKernel("cudaComputeVelocityUniform",
                     getElementCount());
}// end of computeVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to compute acoustic velocity for homogenous medium and nonuniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityHomogeneousNonuniform()
{
  launchSolverKernel("cudaComputeVelocityHomogeneousNonuniform",
                     getElementCount());
}// end of computeVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the x axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityShiftInX()
{
  launchSolverKernel("cudaComputeVelocityShiftInX",
                     getElementCountComplex());
 }// end of computeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the y axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                          rho0ScalarFlag,
                          bOnAScalarFlag,
                          c0ScalarFlag,
                          alphaCoefScalarFlag>::
computeVelocityShiftInY()
{
  const DimensionSizes& dims = Parameters::getInstance().getFullDimensionSizes();

  launchSolverKernel("cudaComputeVelocityShiftInY",
                     dims.nx * (dims.ny / 2 + 1) * dims.nz);
}// end of ComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute the velocity shift in Fourier space over the z axis.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityShiftInZ()
{
  const DimensionSizes& dims = Parameters::getInstance().getFullDimensionSizes();

  launchSolverKernel("cudaComputeVelocityShiftInZ",
                     dims.nx * dims.ny * (dims.nz / 2 + 1));
}// end of computeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Compute spatial part of the velocity gradient in between FFTs on uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityGradient()
{
  launchSolverKernel("cudaComputeVelocityGradient",
                     getElementCountComplex());
}// end of computeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to shift gradient of acoustic velocity on non-uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeVelocityGradientShiftNonuniform()
{
  launchSolverKernel("cudaComputeVelocityGradientShiftNonuniform",
                     getElementCount());
}// end of computeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute acoustic density for non-linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeDensityNonlinear()
{
  launchSolverKernel("cudaComputeDensityNonlinear",
                     getElementCount());
}// end of computeDensityNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute acoustic density for linear case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeDensityLinear()
{
  launchSolverKernel("cudaComputeDensityLinear",
                     getElementCount());
}// end of computeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms for new pressure in non-linear lossless case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureNonlinearLossless()
{
  launchSolverKernel("cudaSumPressureNonlinearLossless",
                     getElementCount());
}// end of sumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute three temporary sums in the new pressure formula in non-linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureTermsNonlinearPowerLaw(RealMatrix& densitySum,
                                      RealMatrix& nonlinearTerm,
                                      RealMatrix& velocityGradientSum)
{
  launchSolverKernel("cudaComputePressureTermsNonlinearPowerLaw",
                     getElementCount(),
                     densitySum.getDeviceData(),
                     nonlinearTerm.getDeviceData(),
                     velocityGradientSum.getDeviceData());
}// end of computePressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to computes absorbing term with abosrbNabla1 and  absorbNabla2.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeAbsorbtionTerm()
{
  launchSolverKernel("cudaComputeAbsorbtionTerm",
                     getElementCountComplex());
}// end of computeAbsorbtionTerm
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel Sum sub-terms to compute new pressure in non-linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureTermsNonlinearPowerLaw(const RealMatrix& nonlinearTerm,
                                  const RealMatrix& absorbTauTerm,
                                  const RealMatrix& absorbEtaTerm)
{
  launchSolverKernel("cudaSumPressureTermsNonlinearPowerlaw",
                     getElementCount(),
                     nonlinearTerm.getDeviceData(),
                     absorbTauTerm.getDeviceData(),
                     absorbEtaTerm.getDeviceData());
}// end of sumPressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in  nonlinear stokes case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureNonlinearStokes()
{
  launchSolverKernel("cudaSumPressureNonlinearStokes",
                     getElementCount());
}// end of sumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms for new pressure in linear lossless case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                          rho0ScalarFlag,
                          bOnAScalarFlag,
                          c0ScalarFlag,
                          alphaCoefScalarFlag>::
sumPressureLinearLossless()
{
  launchSolverKernel("cudaSumPressureLinearLossless",
                     getElementCount());
}// end of sumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute two temporary sums in the new pressure formula for linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computePressureTermsLinearPowerLaw(RealMatrix& densitySum,
                                   RealMatrix& velocityGradientSum)
{
  launchSolverKernel("cudaComputePressureTermsLinearPowerLaw",
                     getElementCount(),
                     densitySum.getDeviceData(),
                     velocityGradientSum.getDeviceData());
}// end of computePressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in linear power law case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureTermsLinearPowerLaw(const RealMatrix& absorbTauTerm,
                               const RealMatrix& absorbEtaTerm,
                               const RealMatrix& densitySum)
{
  launchSolverKernel("cudaSumPressureTermsLinearPowerLaw",
                     getElementCount(),
                     absorbTauTerm.getDeviceData(),
                     absorbEtaTerm.getDeviceData(),
                     densitySum.getDeviceData());
}// end of sumPressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to sum sub-terms to compute new pressure in linear stokes case.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
sumPressureLinearStokes()
{
  launchSolverKernel("cudaSumPressureLinearStokes",
                     getElementCount());
}// end of sumPressureTermsLinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add in pressure source (to acoustic density).
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addPressureSource(const MatrixContainer& container)
{
  const int sourceSize = int(container.getMatrix<IndexMatrix>(MI::kPressureSourceIndex).size());

  launchSolverKernel("cudaAddPressureSource",
                     sourceSize,
                     Parameters::getInstance().getTimeIndex());
}// end of addPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add transducer data source to velocity x component.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addTransducerSource(const MatrixContainer& container)
{
  // Cuda only supports 32bits anyway
  const int sourceSize = int(container.getMatrix<IndexMatrix>(MI::kVelocitySourceIndex).size());

  launchSolverKernel("cudaAddTransducerSource",
                     sourceSize,
                     Parameters::getInstance().getTimeIndex());
}// end of AddTransducerSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to add in velocity source terms.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addVelocitySource(RealMatrix&        velocity,
                  const RealMatrix&  velocitySourceInput,
                  const IndexMatrix& velocitySourceIndex)
{
  const int sourceSize = static_cast<int>(velocitySourceIndex.size());

  launchSolverKernel("cudaAddVelocitySource",
                     sourceSize,
                     velocity.getDeviceData(),
                     velocitySourceInput.getDeviceData(),
                     velocitySourceIndex.getDeviceData(),
                     Parameters::getInstance().getTimeIndex());
}// end of addVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add scaled pressure source to acoustic density.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addPressureScaledSource(const RealMatrix& scaledSource)
{
  launchSolverKernel("cudaAddPressureScaledSource",
                     getElementCount(),
                     scaledSource.getDeviceData());
}// end of AddPressureScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to add scaled pressure source to acoustic density.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addVelocityScaledSource(RealMatrix&       velocity,
                        const RealMatrix& scaledSource)
{
  launchSolverKernel("cudaAddVelocityScaledSource",
                     getElementCount(),
                     velocity.getDeviceData(),
                     scaledSource.getDeviceData());
}// end of AddVelocityScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to insert source signal into scaling matrix.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
insertSourceIntoScalingMatrix(RealMatrix&        scaledSource,
                              const RealMatrix&  sourceInput,
                              const IndexMatrix& sourceIndex,
                              const size_t       manyFlag)
{
  const int sourceSize = static_cast<int>(sourceIndex.size());

  if (manyFlag == 0)
  { // Multiple signals
    launchSolverKernel("cudaInsertSourceIntoScalingMatrix_false",
                       sourceSize,
                       scaledSource.getDeviceData(),
                       sourceInput.getDeviceData(),
                       sourceIndex.getDeviceData(),
                       sourceIndex.size(),
                       Parameters::getInstance().getTimeIndex());
  }
  else
  { // Single signal
    launchSolverKernel("cudaInsertSourceIntoScalingMatrix_true",
                       sourceSize,
                       scaledSource.getDeviceData(),
                       sourceInput.getDeviceData(),
                       sourceIndex.getDeviceData(),
                       sourceIndex.size(),
                       Parameters::getInstance().getTimeIndex());
  }
}// end of insertSourceIntoScalingMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to kernel to compute source gradient.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeSourceGradient(CufftComplexMatrix& sourceSpectrum,
                      const RealMatrix&   sourceKappa)
{
  launchSolverKernel("cudaComputeSourceGradient",
                     getElementCountComplex(),
                     sourceSpectrum.getComplexDeviceData(),
                     sourceKappa.getDeviceData());
}// end of computeSourceGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface for kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
addInitialPressureSource()
{
  launchSolverKernel("cudaAddInitialPressureSource",
                     getElementCount());
}// end of addInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface kernel to compute acoustic velocity for initial pressure problem uniform grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeInitialVelocityUniform()
{
  launchSolverKernel("cudaComputeInitialVelocityUniform",
                     getElementCount());
}// end of computeInitialVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Interface to cuda kernel to compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform
 * grid.
 */
template<SD   simulationDimension,
         bool rho0ScalarFlag,
         bool bOnAScalarFlag,
         bool c0ScalarFlag,
         bool alphaCoefScalarFlag>
void SolverCudaKernels<simulationDimension,
                       rho0ScalarFlag,
                       bOnAScalarFlag,
                       c0ScalarFlag,
                       alphaCoefScalarFlag>::
computeInitialVelocityHomogeneousNonuniform()
{
  launchSolverKernel("cudaComputeInitialVelocityHomogeneousNonuniform",
                     getElementCount());
}// end of computeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Initialization ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

// The class is instantiated after all members have been defined, so they are instantiated as well.

template class SolverCudaKernels<SD::k3D,  true,  true, true,  true>;
template class SolverCudaKernels<SD::k3D,  true,  true, true,  false>;
template class SolverCudaKernels<SD::k3D,  true,  true, false, true>;
template class SolverCudaKernels<SD::k3D,  true,  true, false, false>;

template class SolverCudaKernels<SD::k3D,  true,  false, true,  true>;
template class SolverCudaKernels<SD::k3D,  true,  false, true,  false>;
template class SolverCudaKernels<SD::k3D,  true,  false, false, true>;
template class SolverCudaKernels<SD::k3D,  true,  false, false, false>;

template class SolverCudaKernels<SD::k3D,  false,  true, true,  true>;
template class SolverCudaKernels<SD::k3D,  false,  true, true,  false>;
template class SolverCudaKernels<SD::k3D,  false,  true, false, true>;
template class SolverCudaKernels<SD::k3D,  false,  true, false, false>;

template class SolverCudaKernels<SD::k3D,  false,  false, true,  true>;
template class SolverCudaKernels<SD::k3D,  false,  false, true,  false>;
template class SolverCudaKernels<SD::k3D,  false,  false, false, true>;
template class SolverCudaKernels<SD::k3D,  false,  false, false, false>;


template class SolverCudaKernels<SD::k2D,  true,  true, true,  true>;
template class SolverCudaKernels<SD::k2D,  true,  true, true,  false>;
template class SolverCudaKernels<SD::k2D,  true,  true, false, true>;
template class SolverCudaKernels<SD::k2D,  true,  true, false, false>;

template class SolverCudaKernels<SD::k2D,  true,  false, true,  true>;
template class SolverCudaKernels<SD::k2D,  true,  false, true,  false>;
template class SolverCudaKernels<SD::k2D,  true,  false, false, true>;
template class SolverCudaKernels<SD::k2D,  true,  false, false, false>;

template class SolverCudaKernels<SD::k2D,  false,  true, true,  true>;
template class SolverCudaKernels<SD::k2D,  false,  true, true,  false>;
template class SolverCudaKernels<SD::k2D,  false,  true, false, true>;
template class SolverCudaKernels<SD::k2D,  false,  true, false, false>;

template class SolverCudaKernels<SD::k2D,  false,  false, true,  true>;
template class SolverCudaKernels<SD::k2D,  false,  false, true,  false>;
template class SolverCudaKernels<SD::k2D,  false,  false, false, true>;
template class SolverCudaKernels<SD::k2D,  false,  false, false, false>;
