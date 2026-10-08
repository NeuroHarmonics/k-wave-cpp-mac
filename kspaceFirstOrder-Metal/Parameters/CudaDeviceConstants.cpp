/**
 * @file      CudaDeviceConstants.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     The implementation file for the class for storing constants residing in GPU constant memory.
 *
 * @version   kspaceFirstOrder 3.6
 *
 * @date      17 February  2016, 10:53 (created) \n
 *            11 February  2020, 16:21 (revised)
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

#include <Parameters/CudaDeviceConstants.cuh>
#include <Logger/Logger.h>
#include <Utils/MetalContext.h>

//--------------------------------------------------------------------------------------------------------------------//
//------------------------------------------------- Public methods ---------------------------------------------------//
//--------------------------------------------------------------------------------------------------------------------//

/**
 * Copy the structure with simulation constants to the buffer bound to every kernel as cudaDeviceConstants.
 */
void CudaDeviceConstants::copyToDevice()
{
  MetalContext::getInstance().setDeviceConstants(this, sizeof(CudaDeviceConstants));
}// end of copyToDevice
//----------------------------------------------------------------------------------------------------------------------
