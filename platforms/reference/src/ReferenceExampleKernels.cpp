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

#include "ReferenceExampleKernels.h"
#include "ExampleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
//#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"

using namespace ExamplePlugin;
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

void ReferenceCalcExampleForceKernel::initialize(const System& system, const ExampleForce& force) {
    // Initialize bond parameters.
    
    int numBonds = force.getNumBonds();
    particle1.resize(numBonds);
    particle2.resize(numBonds);
    length.resize(numBonds);
    k.resize(numBonds);
    for (int i = 0; i < numBonds; i++)
        force.getBondParameters(i, particle1[i], particle2[i], length[i], k[i]);
}

double ReferenceCalcExampleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    int numBonds = particle1.size();
    double energy = 0;
    
    // Compute the interactions.
    
    for (int i = 0; i < numBonds; i++) {
        int p1 = particle1[i];
        int p2 = particle2[i];
        RealVec delta = pos[p1]-pos[p2];
        RealOpenMM r2 = delta.dot(delta);
        RealOpenMM r = sqrt(r2);
        RealOpenMM dr = (r-length[i]);
        RealOpenMM dr2 = dr*dr;
        energy += k[i]*dr2*dr2;
        RealOpenMM dEdR = 4*k[i]*dr2*dr;
        dEdR = (r > 0) ? (dEdR/r) : 0;
        force[p1] -= delta*dEdR;
        force[p2] += delta*dEdR;
    }
    return energy;
}

void ReferenceCalcExampleForceKernel::copyParametersToContext(ContextImpl& context, const ExampleForce& force) {
    if (force.getNumBonds() != particle1.size())
        throw OpenMMException("updateParametersInContext: The number of Example bonds has changed");
    for (int i = 0; i < force.getNumBonds(); i++) {
        int p1, p2;
        force.getBondParameters(i, p1, p2, length[i], k[i]);
        if (p1 != particle1[i] || p2 != particle2[i])
            throw OpenMMException("updateParametersInContext: A particle index has changed");
    }
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
void ReferenceIntegrateMyStepKernel::executeSet(ContextImpl& context, const MyIntegrator& integrator, torch::Tensor& input) {
    vector<Vec3>& posData = extractPositions(context);
    torch::Tensor _input = input.to(torch::kFloat64);
    double* __input = _input.data_ptr<double>();
    int numParticles = posData.size();
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = __input[3*i+0];
        posData[i][1] = __input[3*i+1];
        posData[i][2] = __input[3*i+2];
    }
}

void ReferenceIntegrateMyStepKernel::executeGet(ContextImpl& context, const MyIntegrator& integrator, torch::Tensor& input) {
    vector<Vec3>& ForceData = extractForces(context);
    int numParticles = ForceData.size();
    input = torch::from_blob(ForceData.data(), {numParticles, 3}, torch::TensorOptions().dtype(torch::kFloat64));
}

double ReferenceIntegrateMyStepKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator) {
    return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}

/* ###################torch stuff#####################
 * ###################################################
 * ###################################################
 * ###################################################
 */


void setPositions(ContextImpl& context, const std::vector<Vec3>& positions) {
    int numParticles = context.getSystem().getNumParticles();
    vector<Vec3>& posData = extractPositions(context);
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = positions[i][0];
        posData[i][1] = positions[i][1];
        posData[i][2] = positions[i][2];
    }
}
ReferenceIntegrateTorchSetKernel::~ReferenceIntegrateTorchSetKernel() {
    if (dynamics)
        delete dynamics;
}

void ReferenceIntegrateTorchSetKernel::initialize(const System& system, const MyIntegrator& integrator) {
    int numParticles = system.getNumParticles();
    masses.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
        masses[i] = system.getParticleMass(i);
    SimTKOpenMMUtilities::setRandomNumberSeed((unsigned int) integrator.getRandomNumberSeed());
}

void ReferenceIntegrateTorchSetKernel::execute(ContextImpl& context, const MyIntegrator& integrator, torch::Tensor& input) {

    //double temperature = integrator.getTemperature();
    //double friction = integrator.getFriction();
    //double stepSize = integrator.getStepSize();
    //vector<Vec3>& posData = extractPositions(context);
    //vector<Vec3>& velData = extractVelocities(context);
    //vector<Vec3>& forceData = extractForces(context);
    //if (dynamics == 0 || temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Recreate the computation objects with the new parameters.
        
    //    if (dynamics)
    //        delete dynamics;
    //    dynamics = new ReferenceStochasticDynamics(
    //            context.getSystem().getNumParticles(), 
    //            stepSize, 
    //            friction, 
    //            temperature);
    //    dynamics->setReferenceConstraintAlgorithm(&extractConstraints(context));
    //    prevTemp = temperature;
    //    prevFriction = friction;
    //    prevStepSize = stepSize;
    //}
    //dynamics->update(context.getSystem(), posData, velData, forceData, masses, integrator.getConstraintTolerance());
    //data.time += stepSize;
    //data.stepCount++;
    
    //setPositions(ContextImpl& context, const std::vector<Vec3>& positions) {
    vector<Vec3>& posData = extractPositions(context);
    torch::Tensor _input = input.to(torch::kFloat64);
    double* __input = _input.data_ptr<double>();
    int numParticles = posData.size();
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = __input[3*i+0];
        posData[i][1] = __input[3*i+1];
        posData[i][2] = __input[3*i+2];
    }

    //setPositions(context, posData);
    //torch::Tensor forceTensor;
    //vector<Vec3>& forceData = extractForces(context);
}

double ReferenceIntegrateTorchSetKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator) {

    return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}

ReferenceIntegrateTorchGetKernel::~ReferenceIntegrateTorchGetKernel() {
    if (dynamics)
        delete dynamics;
}

void ReferenceIntegrateTorchGetKernel::initialize(const System& system, const MyIntegrator& integrator) {
    int numParticles = system.getNumParticles();
    masses.resize(numParticles);
    for (int i = 0; i < numParticles; ++i)
        masses[i] = system.getParticleMass(i);
    SimTKOpenMMUtilities::setRandomNumberSeed((unsigned int) integrator.getRandomNumberSeed());
}

void ReferenceIntegrateTorchGetKernel::execute(ContextImpl& context, const MyIntegrator& integrator, torch::Tensor& input) {

    //double temperature = integrator.getTemperature();
    //double friction = integrator.getFriction();
    //double stepSize = integrator.getStepSize();
    //vector<Vec3>& posData = extractPositions(context);
    //vector<Vec3>& velData = extractVelocities(context);
    //vector<Vec3>& forceData = extractForces(context);
    //if (dynamics == 0 || temperature != prevTemp || friction != prevFriction || stepSize != prevStepSize) {
        // Recreate the computation objects with the new parameters.
        
    //    if (dynamics)
    //        delete dynamics;
    //    dynamics = new ReferenceStochasticDynamics(
    //            context.getSystem().getNumParticles(), 
    //            stepSize, 
    //            friction, 
    //            temperature);
    //    dynamics->setReferenceConstraintAlgorithm(&extractConstraints(context));
    //    prevTemp = temperature;
    //    prevFriction = friction;
    //    prevStepSize = stepSize;
    //}
    //dynamics->update(context.getSystem(), posData, velData, forceData, masses, integrator.getConstraintTolerance());
    //data.time += stepSize;
    //data.stepCount++;
    
    //setPositions(ContextImpl& context, const std::vector<Vec3>& positions) {
    vector<Vec3>& posData = extractPositions(context);
    int numParticles = posData.size();
    input = torch::from_blob(posData.data(), {numParticles, 3}, torch::TensorOptions().dtype(torch::kFloat64));
    //torch::Tensor _input = input.to(torch::kFloat64);
    /*
    double* __input = _input.data_ptr<double>();
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = __input[3*i+0];
        posData[i][1] = __input[3*i+1];
        posData[i][2] = __input[3*i+2];
    }
     */
    //setPositions(context, posData);
    //torch::Tensor forceTensor;
    //vector<Vec3>& forceData = extractForces(context);
}

double ReferenceIntegrateTorchGetKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator) {

    return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}



