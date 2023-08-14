%module torchexposedintegratorplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"





/*
 * The following lines are needed to handle std::vector.
 * Similar lines may be needed for vectors of vectors or
 * for other STL types like maps.
 */

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "TorchExposedIntegrator.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Add units to function outputs.
*/

/*
 * Convert C++ exceptions to Python exceptions.
*/

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}


namespace TorchExposedIntegratorPlugin {


class TorchExposedIntegrator : public OpenMM::Integrator {
public:
    TorchExposedIntegrator();
    void step(int steps);


    void torchset(unsigned long int in, int numParticles);
    void torchget(unsigned long int out, int numParticles);
    void torchupdate();
    void torchMultiStructure(unsigned long int positions_in, unsigned long int forces_out, int numParticles, int batch_size);
    void torchMultiStructureE(unsigned long int positions_in, unsigned long int forces_out, unsigned long int energy_out, int numParticles, int batch_size);
};



}

