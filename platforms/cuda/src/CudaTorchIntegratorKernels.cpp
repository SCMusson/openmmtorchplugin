
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
#include "CudaTorchIntegratorKernels.h"
#include "CudaTorchIntegratorKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/common/ContextSelector.h"

using namespace TorchIntegratorPlugin;
using namespace OpenMM;
using namespace std;






CudaIntegrateMyStepKernel::~CudaIntegrateMyStepKernel(){
    cu.setAsCurrent();
}

void CudaIntegrateMyStepKernel::initialize(const System& system, const MyIntegrator& integrator){
    cu.getPlatformData().initializeContexts(system);
    cu.setAsCurrent();
    cu.getIntegrationUtilities().initRandomNumberGenerator(integrator.getRandomNumberSeed());
    map<string, string> defines;
    CUmodule program = cu.createModule(CudaTorchIntegratorKernelSources::torchIntegrator, defines);
    setInputsKernel = cu.getKernel(program, "setInputs");
    getForcesKernel = cu.getKernel(program, "getForces");
}

void CudaIntegrateMyStepKernel::execute(ContextImpl&, const MyIntegrator& integrator){
   return;
}

void CudaIntegrateMyStepKernel::executePSet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int in, int numParticles, int offset){
    //cout << "Here in CudaIntegrateMyStepKernel::executePSet " << endl;

    float * ptr = reinterpret_cast<float*>(in+(4*3*offset*numParticles));
    /*
    torch::Tensor input = torch::from_blob(ptr, {numParticles*3}, torch::kCUDA);//, torch::TensorOptions().dtype(torch::kFloat));
    //vector<Vec3>& posData = extractPositions(context);
    torch::Tensor cpu_input = input.to(torch::kCPU);
    float suminput = 0.0;
    for (int i =0; i<numParticles*3; i++) {
        suminput+=*cpu_input[i].abs().data_ptr<float>();
    }
    cout << "suminput: " << suminput<< endl;
    //torch::Tensor _input = input.to(torch::kFloat64);
    CUdeviceptr __input = (CUdeviceptr)input.data_ptr<float>();
    //{ //surrounded by brackets for some reason 
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    //CUresult  result;
    */
    ContextSelector selector(cu);
    
    //result = cuMemcpy(cu.getPosq().getDevicePointer(), __input, numParticles*4*4);
    //cudaMemcpy2D((float*)cu.getPosq().getDevicePointer(), 4*4, input.data_ptr<float>(), 4*4, 4, numParticles, cudaMemcpyDeviceToDevice);
    //cout << "result: " << result << endl; 
    //o
    //////CUresult cuMemcpy ( CUdeviceptr dst, CUdeviceptr src, size_t ByteCount )
    //void* setArgs[] = {&_input, &cu.getPosq().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
    //
    void* setArgs[] = {&ptr, &cu.getPosq().getDevicePointer(),&cu.getAtomIndexArray().getDevicePointer(), &numParticles};
    cu.executeKernel(setInputsKernel, setArgs, numParticles);
    
    //CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
    
    //}// see above
	//int numParticles = posData.size();
}
void CudaIntegrateMyStepKernel::executePGet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int out, int numParticles, int offset){
    float * ptr = reinterpret_cast<float*>(out+(4*3*offset*numParticles));
    ContextSelector selector(cu);
    //torch::Tensor output = torch::from_blob(ptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
    //{
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    float scale = 1.0/(double) 0x100000000LL;
    void* outArgs[] = {&ptr, &cu.getForce().getDevicePointer(), &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms, &scale};
    cu.executeKernel(getForcesKernel, outArgs, numParticles);

    //CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
    //cuCtxSynchronize()
    //}
}
double CudaIntegrateMyStepKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator){
   return 0.0;
}





