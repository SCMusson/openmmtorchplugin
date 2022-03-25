
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

tinput = torch.randn(output.shape)
integrator.torchset(tinput.data_ptr(), tinput.shape[0])
testpositions = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)._value
assert np.allclose(tinput, testpositions)

integrator.torchupdate()

toutput = torch.zeros(output.shape, dtype=torch.double)
integrator.torchget(toutput.data_ptr(), toutput.shape[0])
foutput = simulation.context.getState(getForces=True).getForces(asNumpy=True)._value
assert np.allclose(toutput, foutput)

#embed(header='end')
