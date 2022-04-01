/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
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

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "MyIntegrator.h"
//#include "openmm/LangevinIntegrator.h"
//#include "openmm/SimTKOpenMMRealType.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
//#include "openmm/library/
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <chrono>
#define KILO    	(1e3)
#define BOLTZMANN	(1.380649e-23)
#define AVOGADRO        (6.02214076e23)
#define RGAS    	(BOLTZMANN*AVOGADRO)
#define BOLTZ           (RGAS/KILO)
using namespace TorchIntegratorPlugin;
using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;


extern "C" OPENMM_EXPORT void registerTorchIntegratorCudaKernelFactories();

void testIntegrator() {
    // Create a chain of particles connected by bonds.
    cout << "Start testIntegration cuda" << endl;
    const int numBonds = 10;
    const int numParticles = numBonds+1;
    const int batchSize = 5;
    Platform& platform = Platform::getPlatformByName("CUDA");
    System system;
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i] = Vec3(i, 0.1*i, -0.3*i);
    }
    HarmonicBondForce* force = new HarmonicBondForce();
    system.addForce(force);
    for (int i = 0; i < numBonds; i++)
        force->addBond(i, i+1, 1.5, 0.8);
    
    // Compute the forces and energy.
    MyIntegrator integ(1.0, 1.0,1.0);

    Context context(system, integ, platform);
    context.setPositions(positions);
    State statebefore = context.getState(State::Energy | State::Forces | State::Positions);
    auto input = torch::randn({batchSize*numParticles*3,}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));
    auto output = torch::zeros({batchSize*numParticles*3,}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat));

    unsigned long int ptr = reinterpret_cast<unsigned long int>(input.data_ptr<float>());
    unsigned long int fptr = reinterpret_cast<unsigned long int>(output.data_ptr<float>());
    integ.torchMultiStructure(ptr, fptr, numParticles, batchSize);
    torch::Tensor cpu_input = input.to(torch::kCPU);
    torch::Tensor cpu_output = output.to(torch::kCPU);
    State stateset = context.getState(State::Energy | State::Forces | State:: Positions);
    vector<Vec3> force_vector(numParticles*batchSize);
    for (int batch = 0; batch < batchSize; batch++){
	vector<Vec3> bond_vector(numBonds);
	vector<double> bond_scalar(numBonds);
	vector<double> force_scalar(numBonds);
	int boffset = batch*numParticles*3;
        for (int bond = 0; bond < numBonds; bond++){
	    for (int dim = 0; dim < 3; dim++){
	    bond_vector[bond][dim] = *cpu_input[3+bond*3+dim+boffset].data_ptr<float>()-*cpu_input[bond*3+dim+boffset].data_ptr<float>();
	    }
	    bond_scalar[bond] = sqrt(bond_vector[bond].dot(bond_vector[bond]));
	    force_scalar[bond] = -0.8*(bond_scalar[bond]-1.5);
	    force_vector[bond+numParticles*batch] += -(bond_vector[bond]/bond_scalar[bond])*force_scalar[bond];
	    force_vector[bond+1+numParticles*batch] += (bond_vector[bond]/bond_scalar[bond])*force_scalar[bond];

	}
    }
 //   cout << cpu_output << " is the same as " << force_vector << endl;
    for (int batch = 0; batch < batchSize; batch++){
        for (int atom = 0; atom < numParticles; atom++){
	    Vec3 force_vec;
	    for (int dim = 0; dim < 3; dim++){
		force_vec[dim] = *cpu_output[batch*numParticles*3+atom*3+dim].data_ptr<float>();
	    }
	    ASSERT_EQUAL_VEC(force_vector[batch*numParticles+atom],force_vec,1e-5);
	}
    }
}


int main(int argc, char* argv[]) {
    try {
        registerTorchIntegratorCudaKernelFactories();
        if (argc > 1)
            Platform::getPlatformByName("CUDA").setPropertyDefaultValue("CudaPrecision", string(argv[1]));
        testIntegrator();
        //testIntegrator();
    }
    catch(const std::exception& e) {
        std::cout << "exception: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "Done" << std::endl;
    return 0;
}
