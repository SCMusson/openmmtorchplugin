#ifndef OPENMM_LANGEVININTEGRATOR_H_
#define OPENMM_LANGEVININTEGRATOR_H_

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

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "internal/windowsExportTorchIntegrator.h"
//#include "torch/torch.h"
#include <stdint.h>
//namespace OpenMM {
namespace TorchIntegratorPlugin {

/**
 * This was formerly LangevinIntegrator.
 */

class OPENMM_EXPORT_TORCHINTEGRATOR MyIntegrator : public OpenMM::Integrator {
public:
    /**
     * Create a MyIntegrator, with a bunch of excess methods to satisfy virtual methods in OpenMM::Integrator.
     * 
     * @param temperature    the temperature of the heat bath (in Kelvin)
     * @param frictionCoeff  the friction coefficient which couples the system to the heat bath (in inverse picoseconds)
     * @param stepSize       the step size with which to integrate the system (in picoseconds)
     */
    MyIntegrator(double temperature, double frictionCoeff, double stepSize);
    /**
     * Get the temperature of the heat bath (in Kelvin).
     *
     * @return the temperature of the heat bath, measured in Kelvin
     */
    double getTemperature() const {
        return temperature;
    }
    /**
     * Set the temperature of the heat bath (in Kelvin).
     *
     * @param temp    the temperature of the heat bath, measured in Kelvin
     */
    void setTemperature(double temp);
    /**
     * Get the friction coefficient which determines how strongly the system is coupled to
     * the heat bath (in inverse ps).
     *
     * @return the friction coefficient, measured in 1/ps
     */
    double getFriction() const {
        return friction;
    }
    /**
     * Set the friction coefficient which determines how strongly the system is coupled to
     * the heat bath (in inverse ps).
     *
     * @param coeff    the friction coefficient, measured in 1/ps
     */
    void setFriction(double coeff);
    /**
     * Get the random number seed.  See setRandomNumberSeed() for details.
     */
    int getRandomNumberSeed() const {
        return randomNumberSeed;
    }
    /**
     * Set the random number seed.  The precise meaning of this parameter is undefined, and is left up
     * to each Platform to interpret in an appropriate way.  It is guaranteed that if two simulations
     * are run with different random number seeds, the sequence of random forces will be different.  On
     * the other hand, no guarantees are made about the behavior of simulations that use the same seed.
     * In particular, Platforms are permitted to use non-deterministic algorithms which produce different
     * results on successive runs, even if those runs were initialized identically.
     *
     * If seed is set to 0 (which is the default value assigned), a unique seed is chosen when a Context
     * is created from this Force. This is done to ensure that each Context receives unique random seeds
     * without you needing to set them explicitly.
     */
    void setRandomNumberSeed(int seed) {
        randomNumberSeed = seed;
    }
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
    void torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, int numParticles, int batch_size);
    void torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, unsigned long int energy_out, int numParticles, int batch_size);
    void torchupdate();
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
    /**
     * Get the time interval by which velocities are offset from positions.  This is used to
     * adjust velocities when setVelocitiesToTemperature() is called on a Context.
     */
    double getVelocityTimeOffset() const {
        return getStepSize()/2;
    }
private:
    double temperature, friction;
    int randomNumberSeed;
    OpenMM::Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_LANGEVININTEGRATOR_H_*/
