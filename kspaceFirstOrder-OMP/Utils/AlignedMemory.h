/**
 * @file      AlignedMemory.h
 *
 * @brief     Portable aligned memory allocation. On x86 the _mm_malloc and _mm_free intrinsics are used, on other
 *            architectures (e.g., Apple silicon) they are emulated using posix_memalign and free.
 *
 * @version   kspaceFirstOrder 2.17
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

#ifndef ALIGNED_MEMORY_H
#define ALIGNED_MEMORY_H

#if (defined(__x86_64__) || defined(_M_X64))
  #include <immintrin.h>
#else
  #include <cstdlib>

  /**
   * @brief  Allocate aligned memory.
   * @param  [in] size      - Number of bytes to allocate.
   * @param  [in] alignment - Alignment in bytes (power of two, multiple of sizeof(void*)).
   * @return Pointer to the allocated memory, nullptr if the allocation failed.
   */
  inline void* _mm_malloc(size_t size, size_t alignment)
  {
    void* ptr = nullptr;
    return (posix_memalign(&ptr, alignment, size) == 0) ? ptr : nullptr;
  }// end of _mm_malloc

  /**
   * @brief Free memory allocated by _mm_malloc.
   * @param [in] ptr - Pointer to the memory.
   */
  inline void _mm_free(void* ptr)
  {
    free(ptr);
  }// end of _mm_free
#endif

#endif /* ALIGNED_MEMORY_H */
