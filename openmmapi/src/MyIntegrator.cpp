/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 *
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
    //kernelset = context->getPlatform().createKernel(IntegrateTorchSetKernel::Name(), contextRef);
    //kernelget = context->getPlatform().createKernel(IntegrateTorchGetKernel::Name(), contextRef);

    kernel.getAs<IntegrateMyStepKernel>().initialize(contextRef.getSystem(), *this);
    //kernelset.getAs<IntegrateTorchSetKernel>().initialize(contextRef.getSystem(), *this);
    //kernelget.getAs<IntegrateTorchGetKernel>().initialize(contextRef.getSystem(), *this);
}

void MyIntegrator::setTemperature(double temp) {
    if (temp < 0)
        throw OpenMM::OpenMMException("Temperature cannot be negative");
    temperature = temp;
}

void MyIntegrator::setFriction(double coeff) {
    if (coeff < 0)
        throw OpenMM::OpenMMException("Friction cannot be negative");
    friction = coeff;
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

void MyIntegrator::torchstep(int steps, torch::Tensor& input, torch::Tensor& output){
    context->updateContextState();
    //context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
    kernel.getAs<IntegrateMyStepKernel>().executeSet(*context, *this, input);

        
    context->updateContextState();
    context->calcForcesAndEnergy(true, false, getIntegrationForceGroups());
    kernel.getAs<IntegrateMyStepKernel>().executeGet(*context, *this, output);
    //
}

void MyIntegrator::torchset(unsigned long in, int numParticles){
    kernel.getAs<IntegrateMyStepKernel>().executePSet(*context, *this, in, numParticles);
}


void MyIntegrator::torchget(unsigned long in, int numParticles){
    kernel.getAs<IntegrateMyStepKernel>().executePGet(*context, *this, in, numParticles);
}


