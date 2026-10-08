/**
 * @file      SolverCudaKernels.metal
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation containing all cuda kernels used in the GPU implementation of the k-space solver,
 *            ported to Metal. The host interface is in SolverCudaKernels.cpp.
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

// The template parameters of the CUDA kernels (simulationDimension, rho0ScalarFlag, bOnAScalarFlag, c0ScalarFlag and
// alphaCoefScalarFlag) are function constants declared in Utils/CudaUtils.metal.

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public routines --------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Cuda kernel to compute spectral part of pressure gradient in between FFTs.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 *  bsxfun(@times, ddx_k_shift_pos, kappa .* p_k);
 * \endcode
 */
kernel void cudaComputePressureGradient(KERNEL_ARGS)
{
  const device cuFloatComplex* ddxKShiftPos = getComplexData(MI::kDdxKShiftPosR);
  const device cuFloatComplex* ddyKShiftPos = getComplexData(MI::kDdyKShiftPos);
  const device cuFloatComplex* ddzKShiftPos = getComplexData(MI::kDdzKShiftPos);

  const device float*  kappa = getRealData(MI::kKappa);

  device cuFloatComplex* ifftX = getComplexData(MI::kTempCufftX);
  device cuFloatComplex* ifftY = getComplexData(MI::kTempCufftY);
  device cuFloatComplex* ifftZ = getComplexData(MI::kTempCufftZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplexCoords(cudaDeviceConstants, i);

    const cuFloatComplex eKappa = ifftX[i] * kappa[i];

    ifftX[i] = cuCmulf(eKappa, ddxKShiftPos[coords.x]);
    ifftY[i] = cuCmulf(eKappa, ddyKShiftPos[coords.y]);
    if (simulationDimension == SD::k3D)
    {
      ifftZ[i] = cuCmulf(eKappa, ddzKShiftPos[coords.z]);
    }
  }
}// end of cudaComputePressureGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic velocity for a uniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 *<b> Matlab code: </b> \code
 *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx) - dt .* rho0_sgx_inv .* real(ifftX);
 *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) - dt .* rho0_sgy_inv .* real(ifftY);
 *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz) - dt .* rho0_sgz_inv .* real(ifftZ);
 * \endcode
 */
kernel void cudaComputeVelocityUniform(KERNEL_ARGS)
{
  const device float* dtRho0SgxMatrix = getRealData(MI::kDtRho0Sgx);
  const device float* dtRho0SgyMatrix = getRealData(MI::kDtRho0Sgy);
  const device float* dtRho0SgzMatrix = getRealData(MI::kDtRho0Sgz);

  const device float* ifftX = getRealData(MI::kTemp1RealND);
  const device float* ifftY = getRealData(MI::kTemp2RealND);
  const device float* ifftZ = getRealData(MI::kTemp3RealND);

  const device float* pmlX  = getRealData(MI::kPmlXSgx);
  const device float* pmlY  = getRealData(MI::kPmlYSgy);
  const device float* pmlZ  = getRealData(MI::kPmlZSgz);

  device float* uxSgx = getRealData(MI::kUxSgx);
  device float* uySgy = getRealData(MI::kUySgy);
  device float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    const float dtRho0Sgx  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgx : dtRho0SgxMatrix[i];
    const float dtRho0Sgy  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgy : dtRho0SgyMatrix[i];

    uxSgx[i] = (uxSgx[i] * pmlX[coords.x] - cudaDeviceConstants.fftDivider * ifftX[i] * dtRho0Sgx) * pmlX[coords.x];
    uySgy[i] = (uySgy[i] * pmlY[coords.y] - cudaDeviceConstants.fftDivider * ifftY[i] * dtRho0Sgy) * pmlY[coords.y];

    if (simulationDimension == SD::k3D)
    {
      const float dtRho0Sgz = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0Sgz : dtRho0SgzMatrix[i];

      uzSgz[i] = (uzSgz[i] * pmlZ[coords.z] - cudaDeviceConstants.fftDivider * ifftZ[i] * dtRho0Sgz) * pmlZ[coords.z];
    }
  }
}// end of cudaComputeVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to Compute acoustic velocity for homogenous medium and nonuniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = bsxfun(@times, pml_x_sgx, bsxfun(@times, pml_x_sgx, ux_sgx)  ...
 *                  - dt .* rho0_sgx_inv .* dxudxnSgx.* real(ifftX));
 *  uy_sgy = bsxfun(@times, pml_y_sgy, bsxfun(@times, pml_y_sgy, uy_sgy) ...
 *                  - dt .* rho0_sgy_inv .* dyudynSgy.* real(ifftY);
 *  uz_sgz = bsxfun(@times, pml_z_sgz, bsxfun(@times, pml_z_sgz, uz_sgz)
 *                  - dt .* rho0_sgz_inv .* dzudznSgz.* real(ifftZ);
 *\endcode
 */
kernel void cudaComputeVelocityHomogeneousNonuniform(KERNEL_ARGS)
{
  const float dividerX = cudaDeviceConstants.dtRho0Sgx * cudaDeviceConstants.fftDivider;
  const float dividerY = cudaDeviceConstants.dtRho0Sgy * cudaDeviceConstants.fftDivider;;
  const float dividerZ = (simulationDimension == SD::k3D)
                            ? cudaDeviceConstants.dtRho0Sgz * cudaDeviceConstants.fftDivider : 1.0f;

  const device float* dxudxnSgx = getRealData(MI::kDxudxnSgx);
  const device float* dyudynSgy = getRealData(MI::kDyudynSgy);
  const device float* dzudznSgz = getRealData(MI::kDzudznSgz);

  const device float* ifftX = getRealData(MI::kTemp1RealND);
  const device float* ifftY = getRealData(MI::kTemp2RealND);
  const device float* ifftZ = getRealData(MI::kTemp3RealND);

  const device float* pmlX  = getRealData(MI::kPmlXSgx);
  const device float* pmlY  = getRealData(MI::kPmlYSgy);
  const device float* pmlZ = getRealData(MI::kPmlZSgz);

  device float* uxSgx = getRealData(MI::kUxSgx);
  device float* uySgy = getRealData(MI::kUySgy);
  device float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    uxSgx[i] = (uxSgx[i] * pmlX[coords.x] - dividerX * dxudxnSgx[coords.x] * ifftX[i]) * pmlX[coords.x];
    uySgy[i] = (uySgy[i] * pmlY[coords.y] - dividerY * dyudynSgy[coords.y] * ifftY[i]) * pmlY[coords.y];

    if (simulationDimension == SD::k3D)
    {
      uzSgz[i] = (uzSgz[i] * pmlZ[coords.z] - dividerZ * dzudznSgz[coords.z] * ifftZ[i]) * pmlZ[coords.z];
    }
  }// for
}// end of cudaComputeVelocityHomogeneouosNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in the x direction.
 */
kernel void cudaComputeVelocityShiftInX(KERNEL_ARGS)
{
  const device cuFloatComplex* xShiftNegR = getComplexData(MI::kXShiftNegR);

  device cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const auto  x = i % cudaDeviceConstants.nxComplex;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], xShiftNegR[x]) * cudaDeviceConstants.fftDividerX;

    // The shift makes the Nyquist frequency imaginary. cuFFT ignores the imaginary part there, VkFFT does not.
    if ((cudaDeviceConstants.nx % 2 == 0) && (x == cudaDeviceConstants.nx / 2))
    {
      cufftShiftTemp[i].y = 0.0f;
    }
  }
}// end of cudaComputeVelocityShiftInX
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in Y. The matrix is XY transposed.
 */
kernel void cudaComputeVelocityShiftInY(KERNEL_ARGS)
{
  const auto nyR       = cudaDeviceConstants.ny / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * nyR * cudaDeviceConstants.nz;

  const device cuFloatComplex* yShiftNegR = getComplexData(MI::kYShiftNegR);

  device cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // Rotated dimensions
    const auto  y = i % nyR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], yShiftNegR[y]) * cudaDeviceConstants.fftDividerY;

    // The shift makes the Nyquist frequency imaginary. cuFFT ignores the imaginary part there, VkFFT does not.
    if ((cudaDeviceConstants.ny % 2 == 0) && (y == cudaDeviceConstants.ny / 2))
    {
      cufftShiftTemp[i].y = 0.0f;
    }
  }
}// end of cudaComputeVelocityShiftInY
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute velocity shift in Z. The matrix is XZ transposed.
 *
 */
kernel void cudaComputeVelocityShiftInZ(KERNEL_ARGS)
{
  const auto nzR       = cudaDeviceConstants.nz / 2 + 1;
  const auto nElements = cudaDeviceConstants.nx * cudaDeviceConstants.ny * nzR;

  const device cuFloatComplex* zShiftNegR = getComplexData(MI::kZShiftNegR);

  device cuFloatComplex* cufftShiftTemp   = getComplexData(MI::kTempCufftShift);

  for (auto i = getIndex(); i < nElements; i += getStride())
  {
    // Rotated dimensions
    const auto  z = i % nzR;

    cufftShiftTemp[i] = cuCmulf(cufftShiftTemp[i], zShiftNegR[z]) * cudaDeviceConstants.fftDividerZ;

    // The shift makes the Nyquist frequency imaginary. cuFFT ignores the imaginary part there, VkFFT does not.
    if ((cudaDeviceConstants.nz % 2 == 0) && (z == cudaDeviceConstants.nz / 2))
    {
      cufftShiftTemp[i].y = 0.0f;
    }
  }
}// end of cudaComputeVelocityShiftInZ
//----------------------------------------------------------------------------------------------------------------------

/**
 * Kernel to compute spatial part of the velocity gradient in between FFTs on uniform grid.
 * Complex numbers are passed as float2 structures.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  bsxfun(@times, ddx_k_shift_neg, kappa .* fftn(ux_sgx));
 *  bsxfun(@times, ddy_k_shift_neg, kappa .* fftn(uy_sgy));
 *  bsxfun(@times, ddz_k_shift_neg, kappa .* fftn(uz_sgz));
 * \endcode
 */
kernel void cudaComputeVelocityGradient(KERNEL_ARGS)
{
  const device cuFloatComplex* ddxKShiftNeg = getComplexData(MI::kDdxKShiftNegR);
  const device cuFloatComplex* ddyKShiftNeg = getComplexData(MI::kDdyKShiftNeg);
  const device cuFloatComplex* ddzKShiftNeg = getComplexData(MI::kDdzKShiftNeg);

  const device float* kappa   = getRealData(MI::kKappa);

  device cuFloatComplex* fftX = getComplexData(MI::kTempCufftX);
  device cuFloatComplex* fftY = getComplexData(MI::kTempCufftY);
  device cuFloatComplex* fftZ = getComplexData(MI::kTempCufftZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    const dim3 coords = getComplexCoords(cudaDeviceConstants, i);

    const float eKappa = kappa[i] * cudaDeviceConstants.fftDivider;

    fftX[i] = cuCmulf(fftX[i] * eKappa, ddxKShiftNeg[coords.x]);
    fftY[i] = cuCmulf(fftY[i] * eKappa, ddyKShiftNeg[coords.y]);

    if (simulationDimension == SD::k3D)
    {
      fftZ[i] = cuCmulf(fftZ[i] * eKappa, ddzKShiftNeg[coords.z]);
    }
  }// for
}// end of cudaComputeVelocityGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to shift gradient of acoustic velocity on non-uniform grid.
 * @tparam simulationDimension - Dimensionality of the simulation.
 */
kernel void cudaComputeVelocityGradientShiftNonuniform(KERNEL_ARGS)
{
  const device float* duxdxn = getRealData(MI::kDxudxn);
  const device float* duydyn = getRealData(MI::kDyudyn);
  const device float* duzdzn = getRealData(MI::kDzudzn);

  device float* duxdx = getRealData(MI::kDuxdx);
  device float* duydy = getRealData(MI::kDuydy);
  device float* duzdz = getRealData(MI::kDuzdz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    duxdx[i] *= duxdxn[coords.x];
    duydy[i] *= duydyn[coords.y];

    if (simulationDimension == SD::k3D)
    {
      duzdz[i] *= duzdzn[coords.z];
    }
  }
}// end of cudaComputeVelocityGradientShiftNonuniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic density for non-linear case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  rho0_plus_rho = 2 .* (rhox + rhoy + rhoz) + rho0;
 *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0_plus_rho .* duxdx);
 *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0_plus_rho .* duydy);
 *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0_plus_rho .* duzdz);
 * \endcode
 */
kernel void cudaComputeDensityNonlinear(KERNEL_ARGS)
{
  const device float* pmlX       = getRealData(MI::kPmlX);
  const device float* pmlY       = getRealData(MI::kPmlY);
  const device float* pmlZ       = getRealData(MI::kPmlZ);

  const device float* duxdx      = getRealData(MI::kDuxdx);
  const device float* duydy      = getRealData(MI::kDuydy);
  const device float* duzdz      = getRealData(MI::kDuzdz);

  const device float* rho0Matrix = getRealData(MI::kRho0);

  device float* rhoX = getRealData(MI::kRhoX);
  device float* rhoY = getRealData(MI::kRhoY);
  device float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    const float rho0      = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    // 3D and 2D summation
    const float sumRhos   = (simulationDimension == SD::k3D) ? (rhoX[i] + rhoY[i] + rhoZ[i])
                                                             : (rhoX[i] + rhoY[i]);

    const float sumRhosDt = (2.0f * sumRhos + rho0) * cudaDeviceConstants.dt;

    rhoX[i] = pmlX[coords.x] * ((pmlX[coords.x] * rhoX[i]) - sumRhosDt * duxdx[i]);
    rhoY[i] = pmlY[coords.y] * ((pmlY[coords.y] * rhoY[i]) - sumRhosDt * duydy[i]);

    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = pmlZ[coords.z] * ((pmlZ[coords.z] * rhoZ[i]) - sumRhosDt * duzdz[i]);
    }
  }
}//end of cudaComputeDensityNonlinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic density for linear case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  rhox = bsxfun(@times, pml_x, bsxfun(@times, pml_x, rhox) - dt .* rho0 .* duxdx);
 *  rhoy = bsxfun(@times, pml_y, bsxfun(@times, pml_y, rhoy) - dt .* rho0 .* duydy);
 *  rhoz = bsxfun(@times, pml_z, bsxfun(@times, pml_z, rhoz) - dt .* rho0 .* duzdz);
 * \endcode
 */
kernel void cudaComputeDensityLinear(KERNEL_ARGS)
{
  const device float* pmlX       = getRealData(MI::kPmlX);
  const device float* pmlY       = getRealData(MI::kPmlY);
  const device float* pmlZ       = getRealData(MI::kPmlZ);

  const device float* duxdx      = getRealData(MI::kDuxdx);
  const device float* duydy      = getRealData(MI::kDuydy);
  const device float* duzdz      = getRealData(MI::kDuzdz);

  const device float* rho0Matrix = getRealData(MI::kRho0);

  device float* rhoX = getRealData(MI::kRhoX);
  device float* rhoY = getRealData(MI::kRhoY);
  device float* rhoZ = getRealData(MI::kRhoZ);


  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    const float dtRho0  = (rho0ScalarFlag) ? cudaDeviceConstants.dtRho0 : cudaDeviceConstants.dt * rho0Matrix[i];

    rhoX[i] = pmlX[coords.x] * (pmlX[coords.x] * rhoX[i] - dtRho0 * duxdx[i]);
    rhoY[i] = pmlY[coords.y] * (pmlY[coords.y] * rhoY[i] - dtRho0 * duydy[i]);

    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = pmlZ[coords.z] * (pmlZ[coords.z] * rhoZ[i] - dtRho0 * duzdz[i]);
    }
  }
}// end of cudaComputeDensityLinear
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sums sub-terms for new pressure in non-linear lossless case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam isRho0Scalar        - Is density homogeneous?
 * @tparam isBOnAScalar        - Is nonlinearity homogeneous?
 * @tparam isC2Scalar          - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear adiabatic equation of state
 *  p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
 * \endcode
 */
kernel void cudaSumPressureNonlinearLossless(KERNEL_ARGS)
{
  const device float* rhoX = getRealData(MI::kRhoX);
  const device float* rhoY = getRealData(MI::kRhoY);
  const device float* rhoZ = getRealData(MI::kRhoZ);

  const device float* rho0Matrix = getRealData(MI::kRho0);
  const device float* bOnAMatrix = getRealData(MI::kBOnA);
  const device float* c2Matrix   = getRealData(MI::kC2);

  device float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    const float bOnA = (bOnAScalarFlag) ? cudaDeviceConstants.bOnA : bOnAMatrix[i];
    const float c2   = (c0ScalarFlag)   ? cudaDeviceConstants.c2   : c2Matrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                          : rhoX[i] + rhoY[i];

    p[i] = c2 * (rhoSum + (bOnA * (rhoSum * rhoSum) / (2.0f * rho0)));
  }
}// end of cudaSumPressureNonlinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute three temporary sums in the new pressure formula in non-linear power law case.
 *
 * @tparam simulationDimension      - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag           - Is rho0 a scalar value (homogeneous)?
 * @tparam bOnAScalarFlag           - Is B on A homogeneous?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
 * @param [out] nonlinearTerm       - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz);
 * @param [out] velocityGradientSum - rho0* (duxdx + duydy + duzdz);
 */
kernel void cudaComputePressureTermsNonlinearPowerLaw(device float* densitySum          [[buffer(2)]],
                                                      device float* nonlinearTerm       [[buffer(3)]],
                                                      device float* velocityGradientSum [[buffer(4)]],
                                                      KERNEL_ARGS)
{
  const device float* rhoX       = getRealData(MI::kRhoX);
  const device float* rhoY       = getRealData(MI::kRhoY);
  const device float* rhoZ       = getRealData(MI::kRhoZ);

  const device float* duxdx      = getRealData(MI::kDuxdx);
  const device float* duydy      = getRealData(MI::kDuydy);
  const device float* duzdz      = getRealData(MI::kDuzdz);

  const device float* rho0Matrix = getRealData(MI::kRho0);
  const device float* bOnAMatrix = getRealData(MI::kBOnA);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0 = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];
    const float bonA = (bOnAScalarFlag) ? cudaDeviceConstants.bOnA : bOnAMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? (rhoX[i] + rhoY[i] + rhoZ[i])
                                                          : (rhoX[i] + rhoY[i]);

    const float duSum  = (simulationDimension == SD::k3D) ? (duxdx[i] + duydy[i] + duzdz[i])
                                                          : (duxdx[i] + duydy[i]);

    densitySum[i]          = rhoSum;
    nonlinearTerm[i]       = ((bonA * rhoSum * rhoSum) / (2.0f * rho0)) + rhoSum;
    velocityGradientSum[i] = rho0 * duSum;
  }
}// end of cudaComputePressureTermsNonlinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute absorbing term with abosrbNabla1 and  absorbNabla2.
 *
 * <b>Matlab code:</b> \code
 *  fftPart1 = absorbNabla1 .* fftPart1;
 *  fftPart2 = absorbNabla2 .* fftPart2;
 * \endcode
 */
kernel void cudaComputeAbsorbtionTerm(KERNEL_ARGS)
{

  const device float* absorbNabla1 = getRealData(MI::kAbsorbNabla1);
  const device float* absorbNabla2 = getRealData(MI::kAbsorbNabla2);

  device cuFloatComplex* fftPart1  = getComplexData(MI::kTempCufftX);
  device cuFloatComplex* fftPart2  = getComplexData(MI::kTempCufftY);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    fftPart1[i] *= absorbNabla1[i];
    fftPart2[i] *= absorbNabla2[i];
  }
}// end of computeAbsorbtionTerm
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in non-linear power law case.
 *
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?

 * @param [in] nonlinearTerm   - Nonlinear term
 * @param [in] absorbTauTerm   - Absorb tau term from the pressure eq.
 * @param [in] absorbEtaTerm   - BonA + rho ^2 / 2 rho0  + (rhox_sgx + rhoy_sgy + rhoz_sgz);
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear absorbing equation of state
 *  p = c0.^2 .* (...
 *                nonlinearTerm ...
 *                + absorb_tau .* absorbTauTerm...
 *                - absorb_eta .* absorbEtaTerm...
 *                );
 * \endcode
 */
kernel void cudaSumPressureTermsNonlinearPowerlaw(const device float* nonlinearTerm [[buffer(2)]],
                                                  const device float* absorbTauTerm [[buffer(3)]],
                                                  const device float* absorbEtaTerm [[buffer(4)]],
                                                  KERNEL_ARGS)
{
  const device float* c2Matrix        = getRealData(MI::kC2);
  const device float* absorbTauMatrix = getRealData(MI::kAbsorbTau);
  const device float* absorbEtaMatrix = getRealData(MI::kAbsorbEta);

  device float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];
    const float absorbEta = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbEta : absorbEtaMatrix[i];

    p[i] = c2 * (nonlinearTerm[i] + (cudaDeviceConstants.fftDivider *
                                     ((absorbTauTerm[i] * absorbTau) - (absorbEtaTerm[i] * absorbEta))));
  }
}// end of cudaSumPressureTermsNonlinearPowerlaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sum sub-terms to compute new pressure in  nonlinear stokes case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam isC2Scalar          - Is sound speed homogeneous?
 * @tparam isBOnAScalar        - Is nonlinearity homogeneous?
 * @tparam isRho0Scalar        - Is density homogeneous?
 * @tparam isAbsorbTauScalar   - Is absorption homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  p = c0.^2 .* ( ...
 *      (rhox + rhoy + rhoz) ...
 *       + absorb_tau .* rho0 .* (duxdx + duydy) ...
 *       + medium.BonA .* (rhox + rhoy).^2 ./ (2 .* rho0));
 * \endcode
 */
kernel void cudaSumPressureNonlinearStokes(KERNEL_ARGS)
{
  const device float* rhoX  = getRealData(MI::kRhoX);
  const device float* rhoY  = getRealData(MI::kRhoY);
  const device float* rhoZ  = getRealData(MI::kRhoZ);

  const device float* duxdx = getRealData(MI::kDuxdx);
  const device float* duydy = getRealData(MI::kDuydy);
  const device float* duzdz = getRealData(MI::kDuzdz);

  const device float* rho0Matrix      = getRealData(MI::kRho0);
  const device float* bOnAMatrix      = getRealData(MI::kBOnA);
  const device float* c2Matrix        = getRealData(MI::kC2);
  const device float* absorbTauMatrix = getRealData(MI::kAbsorbTau);

  device float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0      = (rho0ScalarFlag)      ? cudaDeviceConstants.rho0      : rho0Matrix[i];
    const float bOnA      = (bOnAScalarFlag)      ? cudaDeviceConstants.bOnA      : bOnAMatrix[i];
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i]  + rhoY[i]  + rhoZ[i]  : rhoX[i]  + rhoY[i];
    const float duSum  = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i] : duxdx[i] + duydy[i];

    p[i] = c2 * (rhoSum + absorbTau * rho0 * duSum + ((bOnA * rhoSum * rhoSum) / (2.0f * rho0)));
  }
}// end of cudaSumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * @brief Cuda kernel to sum sub-terms for new pressure in linear lossless case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam c0ScalarFlag        - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a linear adiabatic equation of state
 *  p = c0.^2 .* (rhox + rhoy + rhoz);
 * \endcode
 */
kernel void cudaSumPressureLinearLossless(KERNEL_ARGS)
{
  const device float* c2Matrix  = getRealData(MI::kC2);

  const device float* rhoX = getRealData(MI::kRhoX);
  const device float* rhoY = getRealData(MI::kRhoY);
  const device float* rhoZ = getRealData(MI::kRhoZ);

  device float* p  = getRealData(MI::kP);

  for (auto  i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2         = (c0ScalarFlag) ? cudaDeviceConstants.c2 : c2Matrix[i];

    const float sumDensity = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                              : rhoX[i] + rhoY[i];
    p[i] = c2 * sumDensity;
  }
}// end of cudaSumPressureLinearLossless
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute two temporary sums in the new pressure formula for linear power law case.
 *
 * @tparam simulationDimension      - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag           - Is density homogeneous?
 *
 * @param [out] densitySum          - rhox_sgx + rhoy_sgy + rhoz_sgz;
 * @param [out] velocityGradientSum - rho0 * (duxdx + duydy + duzdz);
 */
kernel void cudaComputePressureTermsLinearPowerLaw(device float* densitySum          [[buffer(2)]],
                                                   device float* velocityGradientSum [[buffer(3)]],
                                                   KERNEL_ARGS)
{
  const device float* rhoX = getRealData(MI::kRhoX);
  const device float* rhoY = getRealData(MI::kRhoY);
  const device float* rhoZ = getRealData(MI::kRhoZ);

  const device float* duxdx = getRealData(MI::kDuxdx);
  const device float* duydy = getRealData(MI::kDuydy);
  const device float* duzdz = getRealData(MI::kDuzdz);

  const device float* rho0Matrix = getRealData(MI::kRho0);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0  = (rho0ScalarFlag) ? cudaDeviceConstants.rho0 : rho0Matrix[i];

    densitySum[i]     = (simulationDimension == SD::k3D) ? rhoX[i] + rhoY[i] + rhoZ[i]
                                                         : rhoX[i] + rhoY[i];
    const float duSum = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i]
                                                         : duxdx[i] + duydy[i];

    velocityGradientSum[i] = rho0 * duSum;
  }
}// end of cudaComputePressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in linear power law case.
 *
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
 *
 * @param [in] absorbTauTerm   - Absorb tau term from the pressure eq.
 * @param [in] absorbEtaTerm   - Absorb tau term from the pressure eq.
 * @param [in] densitySum      - Sum of acoustic density.
 *
 * <b>Matlab code:</b> \code
 *  % calculate p using a nonlinear absorbing equation of state
 *  p = c0.^2 .* (...
 *                densitySum
 *                + absorb_tau .* absorbTauTerm...
 *                - absorb_eta .* absorbEtaTerm...
 *                );
 * \endcode
 */
kernel void cudaSumPressureTermsLinearPowerLaw(const device float* absorbTauTerm [[buffer(2)]],
                                               const device float* absorbEtaTerm [[buffer(3)]],
                                               const device float* densitySum    [[buffer(4)]],
                                               KERNEL_ARGS)
{
  const device float* c2Matrix        = getRealData(MI::kC2);
  const device float* absorbTauMatrix = getRealData(MI::kAbsorbTau);
  const device float* absorbEtaMatrix = getRealData(MI::kAbsorbEta);

  device float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];
    const float absorbEta = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbEta : absorbEtaMatrix[i];

    p[i] = c2 * (densitySum[i] + (cudaDeviceConstants.fftDivider *
                (absorbTauTerm[i] * absorbTau - absorbEtaTerm[i] * absorbEta)));
  }
}// end of cudaSumPressureTermsLinearPowerLaw
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to sum sub-terms to compute new pressure in linear stokes case.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Is density homogeneous?
 * @tparam c0ScalarFlag        - Is sound speed homogeneous?
 * @tparam alphaCoefScalarFlag - Is absorption homogeneous?
 *
 * <b>Matlab code:</b> \code
 *  p = c0.^2 .* (rhox + rhoy + rhoz + medium.BonA .* (rhox + rhoy + rhoz).^2 ./ (2 .* rho0));
 * \endcode
 */
kernel void cudaSumPressureLinearStokes(KERNEL_ARGS)
{
  const device float* rhoX  = getRealData(MI::kRhoX);
  const device float* rhoY  = getRealData(MI::kRhoY);
  const device float* rhoZ  = getRealData(MI::kRhoZ);

  const device float* duxdx = getRealData(MI::kDuxdx);
  const device float* duydy = getRealData(MI::kDuydy);
  const device float* duzdz = getRealData(MI::kDuzdz);

  const device float* rho0Matrix      = getRealData(MI::kRho0);
  const device float* c2Matrix        = getRealData(MI::kC2);
  const device float* absorbTauMatrix = getRealData(MI::kAbsorbTau);

  device float* p = getRealData(MI::kP);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float rho0      = (rho0ScalarFlag)      ? cudaDeviceConstants.rho0      : rho0Matrix[i];
    const float c2        = (c0ScalarFlag)        ? cudaDeviceConstants.c2        : c2Matrix[i];
    const float absorbTau = (alphaCoefScalarFlag) ? cudaDeviceConstants.absorbTau : absorbTauMatrix[i];

    const float rhoSum = (simulationDimension == SD::k3D) ? rhoX[i]  + rhoY[i]  + rhoZ[i]  : rhoX[i]  + rhoY[i];
    const float duSum  = (simulationDimension == SD::k3D) ? duxdx[i] + duydy[i] + duzdz[i] : duxdx[i] + duydy[i];

     p[i] = c2 * (rhoSum + absorbTau * rho0 * duSum);
  }
}// end of cudaSumPressureNonlinearStokes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add pressure source to acoustic density.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @param [in] timeIndex       - Actual time step.
 */
kernel void cudaAddPressureSource(constant size_t& timeIndex [[buffer(2)]],
                                  KERNEL_ARGS)
{
  const device float*  pressureSourceInput = getRealData(MI::kPressureSourceInput);
  const device size_t* pressureSourceIndex = getIndexData(MI::kPressureSourceIndex);

  device float* rhoX = getRealData(MI::kRhoX);
  device float* rhoY = getRealData(MI::kRhoY);
  device float* rhoZ = getRealData(MI::kRhoZ);

  // Set 1D or 2D step for source
  const auto index2D = (cudaDeviceConstants.presureSourceMany == 0)
                          ? timeIndex : timeIndex * cudaDeviceConstants.presureSourceSize;

  // Different pressure sources
  switch (cudaDeviceConstants.presureSourceMode)
  {
    case Parameters::SourceMode::kDirichlet:
    {
      if (cudaDeviceConstants.presureSourceMany == 0)
      { // Single signal
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D];
          }
        }
      }
      else
      { // Multiple signals
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          rhoY[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] = pressureSourceInput[index2D + i];
          }
        }
      }
      break;
    }// Dirichlet

    case Parameters::SourceMode::kAdditiveNoCorrection:
    {
      if (cudaDeviceConstants.presureSourceMany == 0)
      { // Single signal
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D];
          }
        }
      }
      else
      { // Multiple signals
        for (auto i = getIndex(); i < cudaDeviceConstants.presureSourceSize; i += getStride())
        {
          rhoX[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          rhoY[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          if (simulationDimension == SD::k3D)
          {
            rhoZ[pressureSourceIndex[i]] += pressureSourceInput[index2D + i];
          }
        }
      }
      break;
    }
    default:
    {
      break;
    }
  }// end switch
}// end of cudaAddPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add transducer data source to velocity x component.
 *
 * @param [in] timeIndex - Actual time step.
 */
kernel void cudaAddTransducerSource(constant size_t& timeIndex [[buffer(2)]],
                                    KERNEL_ARGS)
{
  const device size_t* velocitySourceIndex   = getIndexData(MI::kVelocitySourceIndex);
  const device size_t* delayMask             = getIndexData(MI::kDelayMask);
  const device float*  transducerSourceInput = getRealData(MI::kTransducerSourceInput);

  device float* uxSgx = getRealData(MI::kUxSgx);

  for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
  {
    uxSgx[velocitySourceIndex[i]] += transducerSourceInput[delayMask[i] + timeIndex];
  }
}// end of cudaAddTransducerSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add in velocity source terms.
 *
 * @param [in, out] velocity            - velocity matrix to update.
 * @param [in]      velocitySourceInput - Source input to add.
 * @param [in]      velocitySourceIndex - Index matrix.
 * @param [in]      timeIndex           - Actual time step.
 */
kernel void cudaAddVelocitySource(device float*        velocity            [[buffer(2)]],
                                  const device float*  velocitySourceInput [[buffer(3)]],
                                  const device size_t* velocitySourceIndex [[buffer(4)]],
                                  constant size_t&     timeIndex           [[buffer(5)]],
                                  KERNEL_ARGS)
{
  // Set 1D or 2D step for source
  const auto index2D = (cudaDeviceConstants.velocitySourceMany == 0)
                          ? timeIndex : timeIndex * cudaDeviceConstants.velocitySourceSize;

  if (cudaDeviceConstants.velocitySourceMode == Parameters::SourceMode::kDirichlet)
  {
    for (auto i = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] = (cudaDeviceConstants.velocitySourceMany == 0)
                                            ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }// for
  }// end of Dirichlet

  if (cudaDeviceConstants.velocitySourceMode == Parameters::SourceMode::kAdditiveNoCorrection)
  {
    for (auto i  = getIndex(); i < cudaDeviceConstants.velocitySourceSize; i += getStride())
    {
      velocity[velocitySourceIndex[i]] += (cudaDeviceConstants.velocitySourceMany == 0)
                                             ? velocitySourceInput[index2D] : velocitySourceInput[index2D + i];
    }
  }// end additive
}// end of cudaAddVelocitySource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add scaled pressure source to acoustic density.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @param  [in] scaledSource   - Scaled source.
 */
kernel void cudaAddPressureScaledSource(const device float* scaledSource [[buffer(2)]],
                                        KERNEL_ARGS)
{
  device float* rhoX = getRealData(MI::kRhoX);
  device float* rhoY = getRealData(MI::kRhoY);
  device float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const float eScaledSource = scaledSource[i];
    rhoX[i] += eScaledSource;
    rhoY[i] += eScaledSource;
    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] += eScaledSource;
    }
  }
}// end of cudaAddPressureScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add scaled pressure source to acoustic density.
 *
 * @param [in, out] velocity     - Velocity matrix to update.
 * @param [in]      scaledSource - Scaled source.
 */
kernel void cudaAddVelocityScaledSource(device float*       velocity     [[buffer(2)]],
                                        const device float* scaledSource [[buffer(3)]],
                                        KERNEL_ARGS)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    velocity[i] += scaledSource[i];
  }
}// end of cudaAddVelocityScaledSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add pressure source to acoustic density.
 *
 * @tparam isMany            - Is the many flag set
 *
 * @param [out] scaledSource - Temporary matrix to insert source before scaling.
 * @param [in]  sourceInput  - Source input signal.
 * @param [in]  sourceIndex  - Source geometry.
 * @param [in]  sourceSize   - Size of the source.
 * @param [in]  timeIndex    - Actual time step.
 */
template<bool isMany>
kernel void cudaInsertSourceIntoScalingMatrix(device float*        scaledSource [[buffer(2)]],
                                              const device float*  sourceInput  [[buffer(3)]],
                                              const device size_t* sourceIndex  [[buffer(4)]],
                                              constant size_t&     sourceSize   [[buffer(5)]],
                                              constant size_t&     timeIndex    [[buffer(6)]],
                                              KERNEL_ARGS)
{
  // Set 1D or 2D step for source
  const auto index2D = (isMany) ? timeIndex * sourceSize : timeIndex;

  // Different pressure sources
  if (isMany)
  { // Multiple signals
    for (auto i = getIndex(); i < sourceSize; i += getStride())
    {
      scaledSource[sourceIndex[i]] = sourceInput[index2D + i];
    }
  }
  else
  { // Single signal
    for (auto i = getIndex(); i < sourceSize; i += getStride())
    {
      scaledSource[sourceIndex[i]] = sourceInput[index2D];
    }
  }
}// end of cudaInsertSourceIntoScalingMatrix
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute source gradient.
 *
 * @param [in, out] sourceSpectrum  - Source spectrum.
 * @param [in]      sourceKappa     - Source kappa.
 */
kernel void cudaComputeSourceGradient(device cuFloatComplex* sourceSpectrum [[buffer(2)]],
                                      const device float*    sourceKappa    [[buffer(3)]],
                                      KERNEL_ARGS)
{
  for (auto i = getIndex(); i < cudaDeviceConstants.nElementsComplex; i += getStride())
  {
    sourceSpectrum[i] *= sourceKappa[i] * cudaDeviceConstants.fftDivider;
  }
}// end of cudaComputeSourceGradient
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to add initial pressure initialPerssureSource into p, rhoX, rhoY, rhoZ.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam c0ScalarFlag        - Is sound speed homogenous?
 *
 * <b>Matlab code:</b> \code
 *  % add the initial pressure to rho as a mass source (3D code)
 *  p = source.p0;
 *  rhox = source.p0 ./ (3 .* c.^2);
 *  rhoy = source.p0 ./ (3 .* c.^2);
 *  rhoz = source.p0 ./ (3 .* c.^2);
 * \endcode
 */
kernel void cudaAddInitialPressureSource(KERNEL_ARGS)
{
  const float dimScalingFactor = (simulationDimension == SD::k3D) ? 3.0f : 2.0f;

  const device float* sourceInput = getRealData(MI::kInitialPressureSourceInput);
  const device float* c2          = getRealData(MI::kC2);

  device float* p    = getRealData(MI::kP);
  device float* rhoX = getRealData(MI::kRhoX);
  device float* rhoY = getRealData(MI::kRhoY);
  device float* rhoZ = getRealData(MI::kRhoZ);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    p[i] = sourceInput[i];

    const float tmp = (c0ScalarFlag) ? sourceInput[i] / (dimScalingFactor * cudaDeviceConstants.c2)
                                     : sourceInput[i] / (dimScalingFactor * c2[i]);

    rhoX[i] = tmp;
    rhoY[i] = tmp;
    if (simulationDimension == SD::k3D)
    {
      rhoZ[i] = tmp;
    }
  }
}// end of cudaAddInitialPressureSource
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel compute acoustic velocity for initial pressure problem.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 * @tparam rho0ScalarFlag      - Homogenous or heterogenous medium.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = dt ./ rho0_sgx .* ifft(ux_sgx);
 *  uy_sgy = dt ./ rho0_sgy .* ifft(uy_sgy);
 *  uz_sgz = dt ./ rho0_sgz .* ifft(uz_sgz);
 * \endcode
 */
kernel void cudaComputeInitialVelocityUniform(KERNEL_ARGS)
{
  device float* uxSgx = getRealData(MI::kUxSgx);
  device float* uySgy = getRealData(MI::kUySgy);
  device float* uzSgz = getRealData(MI::kUzSgz);

  if (rho0ScalarFlag)
  {
    const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
    const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
    const float dividerZ = (simulationDimension == SD::k3D)
                              ? cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz : 1.0f;

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dividerX;
      uySgy[i] *= dividerY;
      if (simulationDimension == SD::k3D)
      {
        uzSgz[i] *= dividerZ;
      }
    }
  }
  else
  { // heterogeneous
    const float divider = cudaDeviceConstants.fftDivider * 0.5f;

    const device float* dtRho0Sgx = getRealData(MI::kDtRho0Sgx);
    const device float* dtRho0Sgy = getRealData(MI::kDtRho0Sgy);
    const device float* dtRho0Sgz = getRealData(MI::kDtRho0Sgz);

    for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
    {
      uxSgx[i] *= dtRho0Sgx[i] * divider;
      uySgy[i] *= dtRho0Sgy[i] * divider;
      if (simulationDimension == SD::k3D)
      {
        uzSgz[i] *= dtRho0Sgz[i] * divider;
      }
    }
  }
}// end of cudaComputeInitialVelocityUniform
//----------------------------------------------------------------------------------------------------------------------

/**
 * Cuda kernel to compute acoustic velocity for initial pressure problem, homogenous medium, non-uniform grid.
 *
 * @tparam simulationDimension - Dimensionality of the simulation.
 *
 * <b> Matlab code: </b> \code
 *  ux_sgx = dt ./ rho0_sgx .* dxudxn_sgx .* ifft(ux_sgx);
 *  uy_sgy = dt ./ rho0_sgy .* dyudxn_sgy .* ifft(uy_sgy);
 *  uz_sgz = dt ./ rho0_sgz .* dzudzn_sgz .* ifft(uz_sgz);
 * \endcode
 */
kernel void cudaComputeInitialVelocityHomogeneousNonuniform(KERNEL_ARGS)
{
  const float dividerX = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgx;
  const float dividerY = cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgy;
  const float dividerZ = (simulationDimension == SD::k3D)
                            ? cudaDeviceConstants.fftDivider * 0.5f * cudaDeviceConstants.dtRho0Sgz : 1.0f;

  const device float* dxudxnSgx = getRealData(MI::kDxudxnSgx);
  const device float* dyudynSgy = getRealData(MI::kDyudynSgy);
  const device float* dzudznSgz = getRealData(MI::kDzudznSgz);

  device float* uxSgx = getRealData(MI::kUxSgx);
  device float* uySgy = getRealData(MI::kUySgy);
  device float* uzSgz = getRealData(MI::kUzSgz);

  for (auto i = getIndex(); i < cudaDeviceConstants.nElements; i += getStride())
  {
    const dim3 coords = getRealCoords(cudaDeviceConstants, i);

    uxSgx[i] *= dividerX * dxudxnSgx[coords.x];
    uySgy[i] *= dividerY * dyudynSgy[coords.y];

    if (simulationDimension == SD::k3D)
    {
      uzSgz[i] *= dividerZ * dzudznSgz[coords.z];
    }
  }
}// end of cudaComputeInitialVelocityHomogeneousNonuniform
//----------------------------------------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------- Explicit kernel instances ----------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/// Insert source signal into scaling matrix, single signal.
template [[host_name("cudaInsertSourceIntoScalingMatrix_false")]]
kernel void cudaInsertSourceIntoScalingMatrix<false>(device float*        scaledSource [[buffer(2)]],
                                                     const device float*  sourceInput  [[buffer(3)]],
                                                     const device size_t* sourceIndex  [[buffer(4)]],
                                                     constant size_t&     sourceSize   [[buffer(5)]],
                                                     constant size_t&     timeIndex    [[buffer(6)]],
                                                     KERNEL_ARGS);
/// Insert source signal into scaling matrix, multiple signals.
template [[host_name("cudaInsertSourceIntoScalingMatrix_true")]]
kernel void cudaInsertSourceIntoScalingMatrix<true>(device float*        scaledSource [[buffer(2)]],
                                                    const device float*  sourceInput  [[buffer(3)]],
                                                    const device size_t* sourceIndex  [[buffer(4)]],
                                                    constant size_t&     sourceSize   [[buffer(5)]],
                                                    constant size_t&     timeIndex    [[buffer(6)]],
                                                    KERNEL_ARGS);
//----------------------------------------------------------------------------------------------------------------------
