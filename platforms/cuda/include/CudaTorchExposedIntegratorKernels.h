#ifndef CUDA_TORCHEXPOSEDINTEGRATOR_KERNELS_H_
#define CUDA_TORCHEXPOSEDINTEGRATOR_KERNELS_H_

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

#include "TorchExposedIntegratorKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace TorchExposedIntegratorPlugin {


class CudaIntegrateTorchExposedStepKernel : public IntegrateTorchExposedStepKernel {
public:
    CudaIntegrateTorchExposedStepKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
	    IntegrateTorchExposedStepKernel(name, platform), cu(cu){
	    }
    ~CudaIntegrateTorchExposedStepKernel();
    void initialize(const OpenMM::System& system, const TorchExposedIntegrator& integrator);
    //void execute(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator);
    void executePSet(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int in, int numParticles, int offset);
    void executePGet(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int out, int numParticles, int offset);
    double computeKineticEnergy(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator);
private:

    OpenMM::CudaContext& cu;
    CUfunction setInputsKernel, getForcesKernel;
};
} // namespace TorchExposedIntegratorPlugin

#endif /*CUDA_TORCHEXPOSEDINTEGRATOR_KERNELS_H_*/
