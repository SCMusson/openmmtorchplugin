
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
#include "ReferenceTorchExposedIntegratorKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
//#include "openmm/reference/ReferencePlatform.h"
//#include "openmm/reference/SimTKOpenMMUtilities.h"
//#include "torch/torch.h"

using namespace TorchExposedIntegratorPlugin;
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
    //int numParticles = context.getSystem().getNumParticles();
    //vector<Vec3> shiftedVel(numParticles);
    //context.computeShiftedVelocities(timeShift, shiftedVel);
    //double energy = 0.0;
    //for (int i = 0; i < numParticles; ++i)
    //    if (masses[i] > 0)
    //        energy += masses[i]*(shiftedVel[i].dot(shiftedVel[i]));
    //return 0.5*energy;
    return 0.0;
    }



ReferenceIntegrateTorchExposedStepKernel::~ReferenceIntegrateTorchExposedStepKernel() {
}

void ReferenceIntegrateTorchExposedStepKernel::initialize(const System& system, const TorchExposedIntegrator& integrator) {
    //int numParticles = system.getNumParticles();
}

void ReferenceIntegrateTorchExposedStepKernel::executePSet(ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int positions_in, int numParticles, int offset) {
    double * ptr = reinterpret_cast<double*>(positions_in+(8*3*offset*numParticles));
    vector<Vec3>& posData = extractPositions(context);
    for (int i = 0; i < numParticles; ++i) {
        posData[i][0] = ptr[3*i];
        posData[i][1] = ptr[3*i+1];
        posData[i][2] = ptr[3*i+2];
    }
}

void ReferenceIntegrateTorchExposedStepKernel::executePGet(ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int forces_out, int numParticles, int offset) {
    double * fptr = reinterpret_cast<double*>(forces_out+(8*3*offset*numParticles));
    vector<Vec3>& ForceData = extractForces(context);
    for (int i = 0; i < numParticles; ++i) {
        ptr[3*i] = ForceData[i][0];
        ptr[3*i+1] = ForceData[i][1];
        ptr[3*i+2] = ForceData[i][2];
        
    }
}
double ReferenceIntegrateTorchExposedStepKernel::computeKineticEnergy(ContextImpl& context, const TorchExposedIntegrator& integrator) {
    return 0.0;    
//return computeShiftedKineticEnergy(context, masses, 0.5*integrator.getStepSize());
}


