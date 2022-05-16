
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
#include "MyIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "TorchIntegratorKernels.h"
#include <string>

using namespace TorchIntegratorPlugin;
using std::string;
using std::vector;

MyIntegrator::MyIntegrator(double temperature, double frictionCoeff, double stepSize) {
    setTemperature(temperature);
    setFriction(frictionCoeff);
    setStepSize(stepSize);
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);
}

void MyIntegrator::initialize(OpenMM::ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMM::OpenMMException("This Integrator is already bound to a context");
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateMyStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateMyStepKernel>().initialize(contextRef.getSystem(), *this);
}

void MyIntegrator::setTemperature(double temp) {
    return; // no need
}

void MyIntegrator::setFriction(double coeff) {

}

void MyIntegrator::cleanup() {
    kernel = OpenMM::Kernel();
}

vector<string> MyIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateMyStepKernel::Name());
    return names;
}

double MyIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateMyStepKernel>().computeKineticEnergy(*context, *this);
}

void MyIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMM::OpenMMException("This Integrator is not bound to a context!");  
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
        kernel.getAs<IntegrateMyStepKernel>().execute(*context, *this);
    }
}


void MyIntegrator::torchset(unsigned long positions_in, int numParticles){
    kernel.getAs<IntegrateMyStepKernel>().executePSet(*context, *this, positions_in, numParticles);
}


void MyIntegrator::torchget(unsigned long positions_in, int numParticles){
    kernel.getAs<IntegrateMyStepKernel>().executePGet(*context, *this, positions_in, numParticles);
}

void MyIntegrator::torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, int numParticles, int batch_size){
    for (int i = 0; i < batch_size; i++) {
        kernel.getAs<IntegrateMyStepKernel>().executePSet(*context, *this, positions_in, numParticles, i);
	context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
	kernel.getAs<IntegrateMyStepKernel>().executePGet(*context, *this, forces_out, numParticles, i);
    }
}
void MyIntegrator::torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, unsigned long int energy_out, int numParticles, int batch_size){
    double * eptr = reinterpret_cast<double*>(energy_out);
    for (int i = 0; i < batch_size; i++) {
        kernel.getAs<IntegrateMyStepKernel>().executePSet(*context, *this, positions_in, numParticles, i);
	eptr[i] = context->calcForcesAndEnergy(true, true, getIntegrationForceGroups());
	kernel.getAs<IntegrateMyStepKernel>().executePGet(*context, *this, forces_out, numParticles, i);
    }
}

void MyIntegrator::torchupdate(){
    context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());

}

