#ifndef REFERENCE_TORCHINTEGRATOR_KERNELS_H_
#define REFERENCE_TORCHINTEGRATOR_KERNELS_H_


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
