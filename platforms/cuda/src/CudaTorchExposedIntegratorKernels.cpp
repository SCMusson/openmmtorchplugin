
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
#include "CudaTorchExposedIntegratorKernels.h"
#include "CudaTorchExposedIntegratorKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/common/ContextSelector.h"

using namespace TorchExposedIntegratorPlugin;
using namespace OpenMM;
using namespace std;






CudaIntegrateTorchExposedStepKernel::~CudaIntegrateTorchExposedStepKernel(){
}

void CudaIntegrateTorchExposedStepKernel::initialize(const System& system, const TorchExposedIntegrator& integrator){
    cu.getPlatformData().initializeContexts(system);
    //cu.setAsCurrent();
    ContextSelector selector(cu);
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchExposedIntegratorKernelSources::torchExposedIntegrator, defines, "");
    setInputsKernel = cu.getKernel(program, "setInputs");
    getForcesKernel = cu.getKernel(program, "getForces");
}

void CudaIntegrateTorchExposedStepKernel::executePSet(ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int in, int numParticles, int offset){
    float * ptr = reinterpret_cast<float*>(in+(4*3*offset*numParticles));
    ContextSelector selector(cu);
    void* setArgs[] = {&ptr, &cu.getPosq().getDevicePointer(),&cu.getAtomIndexArray().getDevicePointer(), &numParticles};
    cu.executeKernel(setInputsKernel, setArgs, numParticles);
}
void CudaIntegrateTorchExposedStepKernel::executePGet(ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int out, int numParticles, int offset){
    float * ptr = reinterpret_cast<float*>(out+(4*3*offset*numParticles));
    ContextSelector selector(cu);
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    float scale = 1.0/(double) 0x100000000LL;
    void* outArgs[] = {&ptr, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &scale};
    cu.executeKernel(getForcesKernel, outArgs, numParticles);
}
double CudaIntegrateTorchExposedStepKernel::computeKineticEnergy(ContextImpl& context, const TorchExposedIntegrator& integrator){
    return 0.0;
}





