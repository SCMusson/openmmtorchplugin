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

#include "ReferenceTorchIntegratorKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
//#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"

using namespace TorchIntegratorPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}
static ReferenceConstraints& extractConstraints(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->constraints;
}
static double computeShiftedKineticEnergy(ContextImpl& context, vector<double>& masses, double timeShift) {
    int numParticles = context.getSystem().getNumParticles();
    vector<Vec3> shiftedVel(numParticles);
    context.computeShiftedVelocities(timeShift, shiftedVel);
    double energy = 0.0;
    for (int i = 0; i < numParticles; ++i)
        if (masses[i] > 0)
            energy += masses[i]*(shiftedVel[i].dot(shiftedVel[i]));
    return 0.5*energy;
}



ReferenceIntegrateMyStepKernel::~ReferenceIntegrateMyStepKernel() {
    if (dynamics)
        delete dynamics;
}

void ReferenceIntegrateMyStepKernel::initialize(const System& system, const MyIntegrator& integrator) {
    int numParticles = system.getNumParticles();
    masses.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
        masses[i] = system.getParticleMass(i);
    SimTKOpenMMUtilities::setRandomNumberSeed((unsigned int) integrator.getRandomNumberSeed());
}

void ReferenceIntegrateMyStepKernel::execute(ContextImpl& context, const MyIntegrator& integrator) {
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double stepSize = integrator.getStepSize();
    vector<Vec3>& posData = extractPositions(context);
    vector<Vec3>& velData = extractVelocities(context);
    vector<Vec3>& forceData = extractForces(context);
    if (dynamics == 0 || temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Recreate the computation objects with the new parameters.
        
        if (dynamics)
            delete dynamics;
        dynamics = new ReferenceStochasticDynamics(
                context.getSystem().getNumParticles(), 
                stepSize, 
                friction, 
                temperature);
        dynamics->setReferenceConstraintAlgorithm(&extractConstraints(context));
        prevTemp = temperature;
        prevFriction = friction;
        prevStepSize = stepSize;
    }
    dynamics->update(context.getSystem(), posData, velData, forceData, masses, integrator.getConstraintTolerance());
    data.time += stepSize;
    data.stepCount++;
}
void ReferenceIntegrateMyStepKernel::executePSet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int in, int numParticles) {
    double * ptr = reinterpret_cast<double*>(in);
    torch::Tensor input = torch::from_blob(ptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat));
    vector<Vec3>& posData = extractPositions(context);
    torch::Tensor _input = input.to(torch::kFloat64);
    double* __input = _input.data_ptr<double>();
	//int numParticles = posData.size();
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = __input[3*i];
        posData[i][1] = __input[3*i+1];
        posData[i][2] = __input[3*i+2];
    }
}

void ReferenceIntegrateMyStepKernel::executePGet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int out, int numParticles) {
    double * ptr = reinterpret_cast<double*>(out);
    torch::Tensor output = torch::from_blob(ptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
    vector<Vec3>& ForceData = extractForces(context);
    torch::Tensor fdata = torch::from_blob(ForceData.data(), {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
    //std::memcpy(fdata.data_ptr<double>(), output.data_ptr<double>(), 3*numParticles);
    std::memcpy(output.data_ptr<double>(), fdata.data_ptr<double>(), 8*3*numParticles);
    //torch::TensorAccessor<double, 2> f_a = output.accessor<double, 2>();
    //f_a[0][0]+=5.0;
}
double ReferenceIntegrateMyStepKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator) {
    return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}


