
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
#include "ReferenceTorchIntegratorKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
//#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMUtilities.h"
#include "torch/torch.h"

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
    /*int numParticles = context.getSystem().getNumParticles();
    vector<Vec3> shiftedVel(numParticles);
    context.computeShiftedVelocities(timeShift, shiftedVel);
    double energy = 0.0;
    for (int i = 0; i < numParticles; ++i)
        if (masses[i] > 0)
            energy += masses[i]*(shiftedVel[i].dot(shiftedVel[i]));
    return 0.5*energy;
    */
    return 0.0;
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
void ReferenceIntegrateMyStepKernel::executePSet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int positions_in, int numParticles, int offset) {
    double * ptr = reinterpret_cast<double*>(positions_in+(8*3*offset*numParticles));
    torch::Tensor input = torch::from_blob(ptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
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

void ReferenceIntegrateMyStepKernel::executePGet(ContextImpl& context, const MyIntegrator& integrator, unsigned long int forces_out, int numParticles, int offset) {
    double * fptr = reinterpret_cast<double*>(forces_out+(8*3*offset*numParticles));
    //torch::Tensor output = torch::from_blob(fptr, {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64));
    vector<Vec3>& ForceData = extractForces(context);
    torch::Tensor fdata = torch::from_blob(ForceData.data(), {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    //std::memcpy(fdata.data_ptr<double>(), output.data_ptr<double>(), 3*numParticles);
    //std::memcpy(output.data_ptr<double>(), fdata.data_ptr<double>(), 8*3*numParticles);
    //torch::Tensor forces_out;
    //torch::Scalar scale = 1.0/(double) 0x100000000LL;
    //torch::Tensor forces_out = torch::mul(torch::from_blob(ForceData.data(), {numParticles*3}, torch::TensorOptions().dtype(torch::kFloat64)).clone(), scale);
    //fdata = torch::mul(fdata, scale);
    //std::memcpy(output.data_ptr<double>(), fdata.data_ptr<double>(), 8*3*numParticles);
    std::memcpy(fptr, fdata.data_ptr<double>(), 8*3*numParticles);
    //torch::TensorAccessor<double, 2> f_a = output.accessor<double, 2>();
    //f_a[0][0]+=5.0;
}
double ReferenceIntegrateMyStepKernel::computeKineticEnergy(ContextImpl& context, const MyIntegrator& integrator) {
    return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}


