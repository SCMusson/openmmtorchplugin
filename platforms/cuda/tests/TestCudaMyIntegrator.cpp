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

/**
 * This tests the Reference implementation of ExampleForce.
 */

#include "MyIntegrator.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/HarmonicBondForce.h"
#include <cmath>
#include <iostream>
#include <vector>
#include "torch/torch.h"
using namespace TorchIntegratorPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerTorchIntegratorCudaKernelFactories();

void testIntegrator() {
    // Create a chain of particles connected by bonds.
    cout << "Start testIntegration" << endl;
    const int numBonds = 10;
    const int numParticles = numBonds+1;
    System system;
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(i, 0.1*i, -0.3*i);
    }
    HarmonicBondForce* force = new HarmonicBondForce();
    system.addForce(force);
    for (int i = 0; i < numBonds; i++)
        force->addBond(i, i+1, 1.0+sin(0.8*i), cos(0.3*i));
    
    // Compute the forces and energy.

    MyIntegrator integ(1.0, 1.0,1.0);

    Platform& platform = Platform::getPlatformByName("CUDA");
    Context context(system, integ, platform);
    context.setPositions(positions);
    State statebefore = context.getState(State::Energy | State::Forces | State::Positions);
    //torch::Tensor input = torch::randn({numParticles*3}, torch::TensorOptions().device(torch::kCUDA,1));
    auto input = torch::randn({numParticles*3,}, torch::TensorOptions().device(torch::kCUDA,0));

    unsigned long int ptr = reinterpret_cast<unsigned long int>(input.data_ptr<float>());
    integ.torchset(ptr, numParticles);
    State stateset = context.getState(State::Energy | State::Forces | State::Positions);
    torch::Tensor cpu_input = input.to(torch::kCPU);
    double diffpos = 0.0;
    double sumposstate = 0.0;
    double sumposinput = 0.0;
    cout << "here?" <<endl; 
    for (int i = 0; i < numParticles; i++) {
	for (int j = 0; j < 3; j++) {
	    diffpos += abs(stateset.getPositions()[i][j]) - *cpu_input[3*i+j].abs().data_ptr<float>();
            sumposstate += abs(stateset.getPositions()[i][j]);
	    sumposinput += *cpu_input[3*i+j].abs().data_ptr<float>();
	}
    }
    cout << "diff pos: " << diffpos << endl;
    cout << "sumposstate: " << sumposstate << endl;
    cout << "sumposinput: " << sumposinput << endl;

    ASSERT_EQUAL_TOL(diffpos, 0.0, 1e-5);
    ASSERT_EQUAL_TOL(sumposstate, sumposinput, 1e-5);

    //update
    integ.torchupdate();

    //Torchget
    torch::Tensor output = torch::zeros({numParticles*3}, torch::TensorOptions().device(torch::kCUDA));
    unsigned long int fptr = reinterpret_cast<unsigned long int>(output.data_ptr<float>());

    integ.torchget(fptr, numParticles);

    State stateget = context.getState(State::Energy | State::Forces | State::Positions);
    torch::Tensor cpu_output = output.to(torch::kCPU);

    double diffforce = 0.0;
    double sumforcestate = 0.0;
    double sumforceoutput = 0.0;
    double scale = 1.0/(double) 0x100000000LL;
    for (int i = 0; i < numParticles; i++){
	cout << statebefore.getForces()[i][0] << " <- " << stateget.getForces()[i][0] << " <- " << *cpu_output[3*i].data_ptr<float>()*scale << endl;
        for (int j = 0; j < 3; j++){
	    diffforce += abs(stateget.getForces()[i][j]) - *cpu_output[3*i+j].abs().data_ptr<float>();
	    sumforcestate += abs(stateget.getForces()[i][j]);
	    sumforceoutput += *cpu_output[3*i+j].abs().data_ptr<float>();
	}
    }
    cout << "diff force: " << diffforce << endl;
    cout << "sumforcestate: " << sumforcestate << endl;
    cout << "sumforceoutptu: " << sumforceoutput << endl;
    /*
    ASSERT_EQUAL_TOL(diffforce, 0.0, 1e-5);
    ASSERT_EQUAL_TOL(sumforcestate, sumforceoutput, 1e-5);
    */
}

int main(int argc, char* argv[]) {
    try {
        registerTorchIntegratorCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testIntegrator();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
