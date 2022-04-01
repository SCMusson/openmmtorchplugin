/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

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





