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


extern "C" OPENMM_EXPORT void registerTorchIntegratorReferenceKernelFactories();
Platform& platform = Platform::getPlatformByName("Reference");

void testIntegrator() {
    // Create a chain of particles connected by bonds.
    cout << "Start testIntegrationCPU" << endl;
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
    cout << "make integrator" << endl;
    MyIntegrator integ(1.0, 1.0,1.0);

    Context context(system, integ, platform);
    context.setPositions(positions);
    State statebefore = context.getState(State::Energy | State::Forces | State::Positions);
    auto input = torch::randn({numParticles*3,}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64));

    unsigned long int ptr = reinterpret_cast<unsigned long int>(input.data_ptr<double>());
    integ.torchset(ptr, numParticles);
    State stateset = context.getState(State::Energy | State::Forces | State::Positions);
    torch::Tensor cpu_input = input.to(torch::kCPU);
    double diffpos = 0.0;
    double sumposstate = 0.0;
    double sumposinput = 0.0;
    cout << "here?" <<endl; 
    for (int i = 0; i < numParticles; i++) {
	for (int j = 0; j < 3; j++) {
	    diffpos += abs(stateset.getPositions()[i][j]) - *cpu_input[3*i+j].abs().data_ptr<double>();
            sumposstate += abs(stateset.getPositions()[i][j]);
	    sumposinput += *cpu_input[3*i+j].abs().data_ptr<double>();
	}
    }
    cout << "diff pos: " << diffpos << endl;
    cout << "sumposstate: " << sumposstate << endl;
    cout << "sumposinput: " << sumposinput << endl;

    ASSERT_EQUAL_TOL(0.0, diffpos, 1e-5);
    ASSERT_EQUAL_TOL(sumposstate, sumposinput, 1e-5);

    //update
    integ.torchupdate();

    //Torchget
    torch::Tensor output = torch::zeros({numParticles*3}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64));
    unsigned long int fptr = reinterpret_cast<unsigned long int>(output.data_ptr<double>());

    integ.torchget(fptr, numParticles);

    State stateget = context.getState(State::Energy | State::Forces | State::Positions);
    torch::Tensor cpu_output = output.to(torch::kCPU);

    double diffforce = 0.0;
    double sumforcestate = 0.0;
    double sumforceoutput = 0.0;
    //double scale = 1.0/(double) 0x100000000LL;
    double scale = 1.0;///(double) 0x100000000LL;
    for (int i = 0; i < numParticles; i++){
	//cout <<"0: "  << statebefore.getForces()[i][0] << " <- " << stateget.getForces()[i][0] << " <- " << *cpu_output[3*i].data_ptr<double>() << " scale " << *cpu_output[3*i].data_ptr<double>()*scale << endl;
	//cout <<"1: "  << statebefore.getForces()[i][1] << " <- " << stateget.getForces()[i][1] << " <- " << *cpu_output[3*i+1].data_ptr<double>() << endl;
	//cout <<"2: "  << statebefore.getForces()[i][2] << " <- " << stateget.getForces()[i][2] << " <- " << *cpu_output[3*i+2].data_ptr<double>() << endl;
        for (int j = 0; j < 3; j++){
	    diffforce += abs(stateget.getForces()[i][j]) - *cpu_output[3*i+j].abs().data_ptr<double>();
	    sumforcestate += abs(stateget.getForces()[i][j]);
	    sumforceoutput += *cpu_output[3*i+j].abs().data_ptr<double>();
	}
    }
    cout << "diff force: " << diffforce << endl;
    cout << "sumforcestate: " << sumforcestate << endl;
    cout << "sumforceoutptu: " << sumforceoutput << endl;
    ASSERT_EQUAL_TOL(0.0, diffforce, 1e-5);
    ASSERT_EQUAL_TOL(sumforcestate, sumforceoutput, 1e-5);
}




void testSingleBond(){
    System system;
    system.addParticle(1.0);
    system.addParticle(1.0);
    MyIntegrator integ(0, 0.1,0.1);
    HarmonicBondForce* forceField = new HarmonicBondForce();
    forceField->addBond(0, 1, 1.5, 0.8);
    system.addForce(forceField);
    Context context(system, integ, platform);
    vector<Vec3> pos(2);
    pos[0] = Vec3(0, 2, 0);
    pos[1] = Vec3(0, 0, 0);
    int numParticles = 2;
    context.setPositions(pos);
    State state = context.getState(State::Positions | State::Velocities | State::Forces);
    {

        const vector<Vec3>& forces = state.getForces();
	
	//cout << 0.8*(1.5-(pos[0][1]-pos[1][1])) << endl;
	//cout << 0.8*(-1.5-(pos[1][1]-pos[0][1])) << endl;
	
	ASSERT_EQUAL_VEC(Vec3(0, 0.8*(1.5-(pos[0][1]-pos[1][1])), 0), forces[0], 1e-5);
	ASSERT_EQUAL_VEC(Vec3(0, 0.8*(-1.5-(pos[1][1]-pos[0][1])), 0), forces[1], 1e-5);
    }
    for (int i = 0; i < 10; i++){
    auto input = torch::randn({numParticles*3,}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64));
    unsigned long int ptr = reinterpret_cast<unsigned long int>(input.data_ptr<double>());
    integ.torchset(ptr, numParticles);
    State stateset = context.getState(State::Energy | State::Forces | State::Positions);
    torch::Tensor cpu_input = input.to(torch::kCPU);

    {
	Vec3 dvec;
	for (int i = 0; i < 3; i++){
	dvec[i]=*cpu_input[3+i].data_ptr<double>()-*cpu_input[i].data_ptr<double>();
	}
	double dsca = sqrt(dvec.dot(dvec));
	double force = - 0.8*(dsca-1.5);
	Vec3 force0 = -(dvec/dsca)*force;
	Vec3 force1 = (dvec/dsca)*force;
	vector<Vec3> forces  = stateset.getForces();
	/*
	cout << dvec << " and " << dsca << " or " << force<< endl;
	cout << "Force0: " << force0 << "  " << forces[0] <<endl;
	cout << "Force1: " << force1 << "  " << forces[0] <<endl;
	*/
	ASSERT_EQUAL_VEC(force0, forces[0], 1e-5);
	ASSERT_EQUAL_VEC(force1, forces[1], 1e-5);

    }
    }

}




int main() {
    try {
        registerTorchIntegratorReferenceKernelFactories();
	testIntegrator();
	testSingleBond();
        //initializeTests();
        //runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
