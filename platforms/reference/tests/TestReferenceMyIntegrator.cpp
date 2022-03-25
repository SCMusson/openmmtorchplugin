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

void customtest() {
    System system;
    system.addParticle(2.0);
    system.addParticle(2.0);
    MyIntegrator integrator(0, 0.1, 0.01);
    HarmonicBondForce* forceField = new HarmonicBondForce();
    forceField->addBond(0, 1, 1.5, 1);
    system.addForce(forceField);
    Context context(system, integrator, platform);
    auto start = chrono::high_resolution_clock::now();
    vector<Vec3> positions(2);
    positions[0] = Vec3(-1, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    context.setPositions(positions);
    State state1 = context.getState(State::Positions | State::Velocities | State::Forces);
    auto stop = chrono::high_resolution_clock::now();
    torch::Tensor input = torch::randn({2*3});
    torch::Tensor output;
    auto start1 = chrono::high_resolution_clock::now();
    
    
    
    
    //integrator.torchstep(5, input, output);
    
    
    
    
    
    auto stop1 = chrono::high_resolution_clock::now();
    State state2 = context.getState(State::Positions | State::Velocities | State::Forces);
    int numParticles = 6;
    double force1 = 0.0;
    double force2 = 0.0;
    double force3 = 0.0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            force1 += abs(state1.getForces()[i][j]);
            force2 += abs(state2.getForces()[i][j]);
            force3 += *output[i][j].abs().data_ptr<double>();
            //forcediff = forcediff + abs(state1.getForces()[i][j]-state2.getForces()[i][j]);
            //cout << forcediff<<endl;
            //cout << i <<" : " << input[j+3*i] << output[i][j] << state2.getPositions()[i][j]<< "Forces"<<state1.getForces()[i][j]<<state2.getForces()[i][j]<<endl;
        }
    }       
    double forcediff = abs(force1-force2);
    //cout << force1 << "  "<< force2 << "  "<< force3 <<"  " << forcediff <<endl;
    bool test = (forcediff>TOL*100); //should be none zero
    //cout << test <<endl;//forcediff<<endl;
    ASSERT_EQUAL(test,true); //Low Random chance of it being smaller by accident
    ASSERT_EQUAL_TOL(force2, force3, 1e-10)
    //ASSERT(false)
    auto duration = chrono::duration_cast<chrono::nanoseconds>(stop-start);
    auto duration1 = chrono::duration_cast<chrono::nanoseconds>(stop1-start1);
    //cout<<duration.count() << "  " << duration1.count() << endl;
}
int main() {
    try {
        registerTorchIntegratorReferenceKernelFactories();
	//customtest();
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
