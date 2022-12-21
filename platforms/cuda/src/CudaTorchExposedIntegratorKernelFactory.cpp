
/*---------------------------------------------------------------------------------------------------
 * Copyright (c) 2022 Samuel C. Musson
 *
 * openmmtorchplugin is free software ;
 * you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation ;
 * either version 2 of the License, or (at your option) any later version.
 * openmmtorchplugin is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY ;
 * without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License along with openmmtorchplugin ;
 * if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
*-------------------------------------------------------------------------------- */
#include <exception>

#include "CudaTorchExposedIntegratorKernelFactory.h"
#include "CudaTorchExposedIntegratorKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace TorchExposedIntegratorPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaTorchExposedIntegratorKernelFactory* factory = new CudaTorchExposedIntegratorKernelFactory();
        platform.registerKernelFactory(IntegrateTorchExposedStepKernel::Name(), factory);
    }
    catch (std::exception ex) {
    }
}

extern "C" OPENMM_EXPORT void registerTorchExposedIntegratorCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaTorchExposedIntegratorKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == IntegrateTorchExposedStepKernel::Name())
        return new CudaIntegrateTorchExposedStepKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
