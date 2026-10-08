/**
 * @file      CudaTypes.h
 *
 * @brief     Host side replacements of the CUDA vector types used in the interfaces of the kernels (dim3 and
 *            cuFloatComplex). The layouts match the structures used by the Metal kernels (Utils/CudaUtils.metal).
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

#ifndef CUDA_TYPES_H
#define CUDA_TYPES_H

/**
 * @struct dim3
 * @brief  Three unsigned integers, used for grid and block sizes and for 3D coordinates.
 */
struct dim3
{
  /// Default constructor, same defaults as CUDA.
  constexpr dim3(const unsigned int x = 1, const unsigned int y = 1, const unsigned int z = 1)
    : x(x), y(y), z(z)
  {}

  /// x component.
  unsigned int x;
  /// y component.
  unsigned int y;
  /// z component.
  unsigned int z;
};// end of dim3

static_assert(sizeof(dim3) == 12, "dim3 must match the layout used by the kernels");

/**
 * @struct cuFloatComplex
 * @brief  Single precision complex number, the layout of float2.
 */
struct alignas(8) cuFloatComplex
{
  /// Real part.
  float x;
  /// Imaginary part.
  float y;
};// end of cuFloatComplex

#endif /* CUDA_TYPES_H */
