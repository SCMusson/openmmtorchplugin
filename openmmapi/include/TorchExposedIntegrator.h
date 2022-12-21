#ifndef OPENMM_TORCHEXPOSEDINTEGRATOR_H_
#define OPENMM_TORCHEXPOSEDINTEGRATOR_H_
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

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "internal/windowsExportTorchExposedIntegrator.h"
namespace TorchExposedIntegratorPlugin {

class OPENMM_EXPORT_TORCHEXPOSEDINTEGRATOR TorchExposedIntegrator : public OpenMM::Integrator {
public:
    /**
     * Create a TorchExposedIntegrator, OpenMM::Integrator.
     * 
     */
    TorchExposedIntegrator();
    /**
     * Advance a simulation through time by taking a series of time steps.
     * 
     * @param steps   the number of time steps to take
     */
    void step(int steps);
    /*
     * My Extra methods, everything else can kinda be ignored
     *
     */
    void torchset(unsigned long int positions_in, int numParticles);
    void torchget(unsigned long int forces_out, int numParticles);
    void torchupdate();
    void torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, int numParticles, int batch_size);
    void torchMultiStructureE(unsigned long int positions_in, unsigned long int forces_out, unsigned long int energy_out, int numParticles, int batch_size);
protected:
    /**
     * This will be called by the Context when it is created.  It informs the Integrator
     * of what context it will be integrating, and gives it a chance to do any necessary initialization.
     * It will also get called again if the application calls reinitialize() on the Context.
     */
    void initialize(OpenMM::ContextImpl& context);
    /**
     * This will be called by the Context when it is destroyed to let the Integrator do any necessary
     * cleanup.  It will also get called again if the application calls reinitialize() on the Context.
     */
    void cleanup();
    /**
     * Get the names of all Kernels used by this Integrator.
     */
    std::vector<std::string> getKernelNames();
    /**
     * Compute the kinetic energy of the system at the current time.
     */
    double computeKineticEnergy();
private:
    OpenMM::Kernel kernel;
};

} 

#endif /*OPENMM_TORCHEXPOSEDINTEGRATOR_H_*/
