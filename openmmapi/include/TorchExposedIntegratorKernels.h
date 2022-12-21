#ifndef TORCHEXPOSEDINTEGRATOR_KERNELS_H_
#define TORCHEXPOSEDINTEGRATOR_KERNELS_H_
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
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace TorchExposedIntegratorPlugin {

class IntegrateTorchExposedStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "IntegrateTorchExposedStep";
    }
    IntegrateTorchExposedStepKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param integrator the TorchExposedIntegrator this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const TorchExposedIntegrator& integrator) = 0;
    /**
     * Execute the kernel.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the TorchExposedIntegrator this kernel is being used for
     */

    /*
    virtual void execute(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator) = 0;
    */
    virtual void executePSet(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int positions_in, int numParticles, int offset = 0) = 0;
    virtual void executePGet(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator, unsigned long int out_forces, int numParticles, int offset = 0) = 0;
    /**
     * Compute the kinetic energy.
     * 
     * @param context    the context in which to execute this kernel
     * @param integrator the TorchExposedIntegrator this kernel is being used for
     */
    virtual double computeKineticEnergy(OpenMM::ContextImpl& context, const TorchExposedIntegrator& integrator) = 0;
};

}

#endif /*TORCHEXPOSEDINTEGRATOR_KERNELS_H_*/
