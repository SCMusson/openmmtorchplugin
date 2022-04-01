
import torch
import unittest
from parameterized import parameterized
import os
import sys
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from IPython import embed
from torchintegratorplugin import MyIntegrator
from torch.utils.cpp_extension import load

#getpointer = load(name='torch_extension', sources=['torch_extension.cpp'])

class TestMyIntegrator(unittest.TestCase):
    def test_reference(self,):
        data_dir = os.path.join(os.path.abspath(os.path.split(__file__)[0]), 'app', 'data')
        pdb = PDBFile(os.path.join(data_dir, 'test.pdb'))
        forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=1*nanometer, constraints=HBonds, ewaldErrorTolerance=1e-4)
        platform = Platform.getPlatformByName('Reference')
        integrator = MyIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator, platform)
        n_particles = simulation.context.getSystem().getNumParticles()
        input = np.random.randn(n_particles, 3)
        simulation.context.setPositions(input)
        output =  simulation.context.getState(getPositions=True).getPositions(asNumpy=True)

        tinput = torch.randn(output.shape, dtype=torch.double)
        integrator.torchset(tinput.data_ptr(), tinput.shape[0])
        testpositions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        assert np.allclose(tinput, testpositions)

        integrator.torchupdate()

        toutput = torch.zeros(output.shape, dtype=torch.double)
        integrator.torchget(toutput.data_ptr(), toutput.shape[0])
        foutput = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value
        assert np.allclose(toutput, foutput)


    def test_cuda(self,):
        data_dir = os.path.join(os.path.abspath(os.path.split(__file__)[0]), 'app', 'data')
        pdb = PDBFile(os.path.join(data_dir, 'test.pdb'))
        forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=1*nanometer, constraints=HBonds, ewaldErrorTolerance=1e-4)
        platform = Platform.getPlatformByName('CUDA')
        integrator = MyIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator, platform)
        n_particles = simulation.context.getSystem().getNumParticles()
        platform.setPropertyValue(simulation.context, 'Precision', 'single')
        input = np.random.randn(n_particles, 3)
        #simulation.context.setPeriodicBoxVectors(Vec3(10, 0, 0),Vec3(0, 10, 0),Vec3(0,0,10))
        simulation.context.setPositions(input)
        output =  simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        pbvectors = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True)._value
        tinput = torch.randn(output.shape, device=torch.device('cuda'), dtype=torch.float)
        integrator.torchset(tinput.data_ptr(), tinput.shape[0])
        testpositions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
        for i, atom in enumerate(testpositions):
            for j, x in enumerate(atom):
                cpu_input = tinput.cpu().numpy()
                if np.isclose(x, cpu_input[i][j]):
                    pass
                elif np.isclose(abs(x-cpu_input[i][j]), pbvectors[j][j]):
                    pass
                else:
                    raise ValueError(f'atom {i} should be {cpu_input[i]} but state returned {atom}')

        integrator.torchupdate()

        toutput = torch.zeros(output.shape, device=torch.device('cuda'), dtype=torch.float)
        integrator.torchget(toutput.data_ptr(), toutput.shape[0])
        foutput = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value
        assert np.allclose(toutput.cpu().numpy(), foutput, rtol=1e-1)#where does ti fail?
        assert np.allclose(toutput.cpu().numpy(), foutput, rtol=1e-2)

        noise = torch.randn(output.shape, device=torch.device('cuda'), dtype=torch.float)
        tinput = torch.zeros(output.shape, device=torch.device('cuda'), dtype=torch.float)
        for i,atom in enumerate(pdb.positions):
            for j,dim in enumerate(atom):
                tinput[i][j] = dim._value

        toutput = torch.zeros(output.shape, device=torch.device('cuda'), dtype=torch.float)
        #integrator.torchMultiStructure(tinput.data_ptr(), toutput.data_ptr(), n_particles, batch_size)
        energy = 1e10 # arbritrary big number
        for i in range(10000):
            integrator.torchset(tinput.data_ptr(), tinput.shape[0])
            integrator.torchupdate()
            integrator.torchget(toutput.data_ptr(), toutput.shape[0])
            if np.sqrt(i) % 1 == 0.0:
                new_energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
                if new_energy<energy/100:
                    break
            tinput+=(0.0001/toutput.max())*toutput
        else:
            raise ValueError(f"Starting energy was: {energy} but only managed to get to {new_energy}")

if __name__=='__main__':
    unittest.main()
