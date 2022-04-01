#ifndef REFERENCE_TORCHINTEGRATOR_KERNELS_H_
#define REFERENCE_TORCHINTEGRATOR_KERNELS_H_

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

#include "TorchIntegratorKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/ReferenceStochasticDynamics.h"
#include "openmm/Platform.h"
#include <vector>
namespace TorchIntegratorPlugin {


/**
 * This kernel is invoked by MyIntegrator to take one time step.
 */
class ReferenceIntegrateMyStepKernel : public IntegrateMyStepKernel {
public:
    ReferenceIntegrateMyStepKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ReferencePlatform::PlatformData& data) : IntegrateMyStepKernel(name, platform),
        data(data), dynamics(0) {
    }
    ~ReferenceIntegrateMyStepKernel();
    /**
     * Initialize the kernel, setting up the particle masses.
     * 
     * @param system     the System this kernel will be applied to
     * @param integrator the MyIntegrator this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const MyIntegrator& integrator);
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the MyIntegrator this kernel is being used for
     */
    void execute(OpenMM::ContextImpl& context, const MyIntegrator& integrator);
    void executePSet(OpenMM::ContextImpl& context, const MyIntegrator& integrator, unsigned long int positions_in, int numParticles, int offset);
    void executePGet(OpenMM::ContextImpl& context, const MyIntegrator& integrator, unsigned long int forces_out, int numParticles, int offset);
    
    /**
     * Compute the kinetic energy.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the MyIntegrator this kernel is being used for
     */

    double computeKineticEnergy(OpenMM::ContextImpl& context, const MyIntegrator& integrator);
private:
    OpenMM::ReferencePlatform::PlatformData& data;
    OpenMM::ReferenceStochasticDynamics* dynamics;
    std::vector<double> masses;
    double prevTemp, prevFriction, prevStepSize;
};

} // namespace TorchIntegratorPlugin

#endif /*REFERENCE_TORCHINTEGRATOR_KERNELS_H_*/
