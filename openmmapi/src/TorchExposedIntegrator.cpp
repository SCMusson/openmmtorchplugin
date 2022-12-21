
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
#include "TorchExposedIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "TorchExposedIntegratorKernels.h"
#include <string>

#include <iostream>

using namespace TorchExposedIntegratorPlugin;
using std::string;
using std::vector;

TorchExposedIntegrator::TorchExposedIntegrator() {
}

void TorchExposedIntegrator::initialize(OpenMM::ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMM::OpenMMException("This Integrator is already bound to a context");
    context = &contextRef;
    owner = &contextRef.getOwner();
    std::cout << "nothing else" << std::endl;
    kernel = context->getPlatform().createKernel(IntegrateTorchExposedStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateTorchExposedStepKernel>().initialize(contextRef.getSystem(), *this);
}

void TorchExposedIntegrator::cleanup() {
    kernel = OpenMM::Kernel();
}

vector<string> TorchExposedIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateTorchExposedStepKernel::Name());
    return names;
}

double TorchExposedIntegrator::computeKineticEnergy() {
    return 0.0;
	//return kernel.getAs<IntegrateTorchExposedStepKernel>().computeKineticEnergy(*context, *this);
}

void TorchExposedIntegrator::step(int steps) {
}

void TorchExposedIntegrator::torchset(unsigned long positions_in, int numParticles){
    kernel.getAs<IntegrateTorchExposedStepKernel>().executePSet(*context, *this, positions_in, numParticles);
}


void TorchExposedIntegrator::torchget(unsigned long positions_in, int numParticles){
    kernel.getAs<IntegrateTorchExposedStepKernel>().executePGet(*context, *this, positions_in, numParticles);
}

void TorchExposedIntegrator::torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, int numParticles, int batch_size){
    for (int i = 0; i < batch_size; i++) {
        kernel.getAs<IntegrateTorchExposedStepKernel>().executePSet(*context, *this, positions_in, numParticles, i);
	context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
	kernel.getAs<IntegrateTorchExposedStepKernel>().executePGet(*context, *this, forces_out, numParticles, i);
    }
}
void TorchExposedIntegrator::torchMultiStructureE(unsigned long int positions_in, unsigned long int forces_out, unsigned long int energy_out, int numParticles, int batch_size){
    double * eptr = reinterpret_cast<double*>(energy_out);
    for (int i = 0; i < batch_size; i++) {
        kernel.getAs<IntegrateTorchExposedStepKernel>().executePSet(*context, *this, positions_in, numParticles, i);
	eptr[i] = context->calcForcesAndEnergy(true, true, getIntegrationForceGroups());
	kernel.getAs<IntegrateTorchExposedStepKernel>().executePGet(*context, *this, forces_out, numParticles, i);
    }
}

void TorchExposedIntegrator::torchupdate(){
    context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());

}
