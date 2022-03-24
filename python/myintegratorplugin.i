%module myintegratorplugin

%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
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
#include "ExampleForce.h"
#include "MyIntegrator.h"
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


namespace ExamplePlugin {


class MyIntegrator : public OpenMM::Integrator {
public:
    MyIntegrator(double temperature, double frictionCoeff, double stepSize);

    double getTemperature() const;

    void setTemperature(double temp);

    double getFriction() const;

    void setFriction(double coeff);

    int getRandomNumberSeed() const;

    void setRandomNumberSeed(int seed);

    void step(int steps);

    %apply torch::Tensor& OUTPUT {torch::Tensor& output};
    void torchstep(int steps, torch::Tensor& input, torch::Tensor& output);
    %clear torch::Tensor& output;
/*
    %extend {
        static ExamplePlugin::MyIntegrator& cast(OpenMM::Integrator& integrator) {
            return dynamic_cast<ExamplePlugin::MyIntegrator&>(integrator);
        }

        static bool isinstance(OpenMM::Integrator& integrator){
            return (dynamic_cast<ExamplePlugin::MyIntegrator*>(&integrator) != NULL);
        }
    }
*/
};



}

