
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

    def test_cuda_multi(self,):
        print('multi')
        batch_size = 8
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
        '''
        tinput = torch.randn((batch_size,*output.shape), device=torch.device('cuda'), dtype=torch.float)
        toutput = torch.zeros((batch_size, *output.shape), device=torch.device('cuda'), dtype=torch.float)
        tenergy = torch.zeros((batch_size,), device=torch.device('cpu'), dtype=torch.double)
        #integrator.torchMultiStructure(tinput.data_ptr(), toutput.data_ptr(), n_particles, batch_size)
        integrator.torchMultiStructure(tinput.data_ptr(), toutput.data_ptr(), tenergy.data_ptr(), n_particles, batch_size)
        stateforces = np.zeros(toutput.shape)
        for i in range(batch_size):
            ti = tinput[i]
            integrator.torchset(ti.data_ptr(), ti.shape[0])
            integrator.torchupdate()
            stateforces[i] = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value
        '''
        noise = torch.randn((batch_size,*output.shape), device=torch.device('cuda'), dtype=torch.float)
        tinput = torch.zeros((batch_size,*output.shape), device=torch.device('cuda'), dtype=torch.float)
        #tinput = torch.zeros(output.shape, device=torch.device('cuda'), dtype=torch.float)
        tenergy = torch.zeros((batch_size,), device=torch.device('cpu'), dtype=torch.double)
        for i,atom in enumerate(pdb.positions):
            for j,dim in enumerate(atom):
                tinput[:,i,j] = dim._value
        tinput+=0.02*noise
        energy = 0
        toutput = torch.zeros((batch_size, *output.shape), device=torch.device('cuda'), dtype=torch.float)
        for i in range(10000):
            integrator.torchMultiStructure(tinput.data_ptr(), toutput.data_ptr(), tenergy.data_ptr(), n_particles, batch_size)
            if i==0:
                energy = tenergy.mean()
                assert energy>-50000
            elif np.sqrt(i) % 1 == 0.0:
                print(energy)
                new_energy = tenergy.mean()
                print(new_energy)
                if new_energy<-100000:
                    break
            tinput+=(0.0001/toutput.max())*toutput
        else:
            raise ValueError(f"Starting energy was: {energy} but only managed to get to {new_energy}")
        return
        assert np.allclose(toutput.cpu().numpy(), stateforces, rtol=1e-1)#where does ti fail?
        assert np.allclose(toutput.cpu().numpy(), stateforces, rtol=1e-2)
        assert np.allclose(toutput.cpu().numpy(), stateforces, rtol=1e-3)
if __name__=='__main__':
    unittest.main()
