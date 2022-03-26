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

class CudaExampleForceInfo : public CudaForceInfo {
public:
    CudaExampleForceInfo(const ExampleForce& force) : force(force) {
    }
    int getNumParticleGroups() {
        return force.getNumBonds();
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        int particle1, particle2;
        double length, k;
        force.getBondParameters(index, particle1, particle2, length, k);
        particles.resize(2);
        particles[0] = particle1;
        particles[1] = particle2;
    }
    bool areGroupsIdentical(int group1, int group2) {
        int particle1, particle2;
        double length1, length2, k1, k2;
        force.getBondParameters(group1, particle1, particle2, length1, k1);
        force.getBondParameters(group2, particle1, particle2, length2, k2);
        return (length1 == length2 && k1 == k2);
    }
private:
    const ExampleForce& force;
};

CudaCalcExampleForceKernel::~CudaCalcExampleForceKernel() {
    cu.setAsCurrent();
    if (params != NULL)
        delete params;
}

void CudaCalcExampleForceKernel::initialize(const System& system, const ExampleForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    numBonds = endIndex-startIndex;
    if (numBonds == 0)
        return;
    vector<vector<int> > atoms(numBonds, vector<int>(2));
    params = CudaArray::create<float2>(cu, numBonds, "bondParams");
    vector<float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        double length, k;
        force.getBondParameters(startIndex+i, atoms[i][0], atoms[i][1], length, k);
        paramVector[i] = make_float2((float) length, (float) k);
    }
    params->upload(paramVector);
    map<string, string> replacements;
    replacements["PARAMS"] = cu.getBondedUtilities().addArgument(params->getDevicePointer(), "float2");
    cu.getBondedUtilities().addInteraction(atoms, cu.replaceStrings(CudaTorchIntegratorKernelSources::exampleForce, replacements), force.getForceGroup());
    cu.addForce(new CudaExampleForceInfo(force));
}

double CudaCalcExampleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    return 0.0;
}

void CudaCalcExampleForceKernel::copyParametersToContext(ContextImpl& context, const ExampleForce& force) {
    cu.setAsCurrent();
    int numContexts = cu.getPlatformData().contexts.size();
    int startIndex = cu.getContextIndex()*force.getNumBonds()/numContexts;
    int endIndex = (cu.getContextIndex()+1)*force.getNumBonds()/numContexts;
    if (numBonds != endIndex-startIndex)
        throw OpenMMException("updateParametersInContext: The number of bonds has changed");
    if (numBonds == 0)
        return;
    
    // Record the per-bond parameters.
    
    vector<float2> paramVector(numBonds);
    for (int i = 0; i < numBonds; i++) {
        int atom1, atom2;
        double length, k;
        force.getBondParameters(startIndex+i, atom1, atom2, length, k);
        paramVector[i] = make_float2((float) length, (float) k);
    }
    params->upload(paramVector);
    
    // Mark that the current reordering may be invalid.
    
    cu.invalidateMolecules();
}

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

void CudaIntegrateMyStepKernel::executePSet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int in, int numParticles){
    //cout << "Here in CudaIntegrateMyStepKernel::executePSet " << endl;

    float * ptr = reinterpret_cast<float*>(in);
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
    
    void* setArgs[] = {&ptr, &cu.getPosq().getDevicePointer(), &numParticles};
    cu.executeKernel(setInputsKernel, setArgs, numParticles);
    
    //CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
    
    //}// see above
	//int numParticles = posData.size();
}
void CudaIntegrateMyStepKernel::executePGet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int out, int numParticles){
    float * ptr = reinterpret_cast<float*>(out);
    //torch::Tensor output = torch::from_blob(ptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
    //{
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    void* outArgs[] = {&ptr, &cu.getForce().getDevicePointer(), &numParticles, &paddedNumAtoms};
    cu.executeKernel(getForcesKernel, outArgs, numParticles);
    //CHECK_RESULT(cuCtxSynchronize(), "Error synchronizing CUDA context");
    //}
}
double CudaIntegrateMyStepKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator){
   return 0.0;
}





