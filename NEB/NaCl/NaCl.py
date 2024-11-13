# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from ase.io import read, write
from ase import atoms
from ase.calculators import cp2k
import numpy as np
import matplotlib.pyplot as plt

NaCl_crystal = read('NaCl.pdb')

cp2k_input = """
&FORCE_EVAL
  METHOD Fist
  &MM
    &FORCEFIELD
      &CHARGE
        ATOM Na
        CHARGE +1.000
      &END CHARGE
      &CHARGE
        ATOM Cl
        CHARGE -1.000
      &END CHARGE
      &NONBONDED
        &BMHFT
          map_atoms NA NA
          atoms NA NA
          RCUT 10.0
        &END BMHFT
        &BMHFT
          map_atoms NA CL
          atoms NA CL
          RCUT 10.0
        &END BMHFT
        &BMHFT
          map_atoms CL CL
          atoms CL CL
          RCUT 10.0
        &END BMHFT
      &END NONBONDED
    &END FORCEFIELD
    &POISSON
      &EWALD
        EWALD_TYPE spme
        ALPHA .5
        GMAX 40
        O_SPLINE 6
      &END EWALD
    &END POISSON
  &END MM
  &SUBSYS
    &CELL
      ABC 22.480 22.480 22.480
    &END CELL
    &TOPOLOGY
      &GENERATE
         BONDLENGTH_MAX [bohr] 6.5
      &END
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL
&MOTION
  &MD
    ENSEMBLE NVT
    STEPS 10
    TIMESTEP 2.5
    TEMPERATURE 300.0
    &THERMOSTAT
      &NOSE
        LENGTH 3
        YOSHIDA 3
        TIMECON 1000
        MTS 2
      &END NOSE
      &PRINT
        &ENERGY
        &END
      &END
    &END
  &END MD
&END MOTION
"""

with cp2k.CP2K(
    inp=cp2k_input,
    basis_set=None,
    basis_set_file=None,
    max_scf=None,
    cutoff=None,
    force_eval_method=None,
    potential_file=None,
    poisson_solver=None,
    pseudo_potential=None,
    stress_tensor=False,
    xc=None,
    label='NaCl') as calc:
        calc.command="cp2k_shell.psmp"
        energy = calc.get_potential_energy(atoms=NaCl_crystal)
        print(NaCl_crystal, energy)
        
        vac1 = NaCl_crystal[[atom.index for atom in NaCl_crystal if atom.index != 1]]
        energy = calc.get_potential_energy(atoms=vac1)
        print(vac1, energy)
                
        NaCl_crystal.positions[[1, 3]] = NaCl_crystal.positions[[3, 1]]
        vac2 = NaCl_crystal[[atom.index for atom in NaCl_crystal if atom.index != 1]]
        energy = calc.get_potential_energy(atoms=vac2)
        print(vac2, energy)


# %env OMP_NUM_THREADS=1

# +
from ase.constraints import FixAtoms
from ase.io import read
from ase.mep import NEB
from ase.optimize.fire import FIRE as QuasiNewton

# Read the previous configurations
initial = vac1
final = vac2

for ends in initial, final:
    ends.calc =cp2k.CP2K(
    inp=cp2k_input,
    basis_set=None,
    basis_set_file=None,
    max_scf=None,
    cutoff=None,
    force_eval_method=None,
    potential_file=None,
    poisson_solver=None,
    pseudo_potential=None,
    stress_tensor=False,
    xc=None,
    label='NaCl_vac')
    ends.calc.command="mpirun -np 2 cp2k_shell.psmp"
    #config.set_constraint(constraint)



# +
relax = QuasiNewton(initial)
relax.run(fmax=0.05)
print('initial state:', initial.calc.get_potential_energy())
write('initial.traj', initial)

relax = QuasiNewton(final)
relax.run(fmax=0.05)
print('final state:', final.calc.get_potential_energy())
write('final.traj', final)

# +
#  Make 9 images (note the use of copy)
configs = [initial.copy() for i in range(20)] + [final]

# As before, fix the Cu atoms
#constraint = FixAtoms(mask=[atom.symbol != 'N' for atom in initial])

for config in configs:
    config.calc =cp2k.CP2K(
    inp=cp2k_input,
    #basis_set=None,
    #basis_set_file=None,
    #max_scf=None,
    #cutoff=None,
    force_eval_method=None,
    #potential_file=None,
    #poisson_solver=None,
    #pseudo_potential=None,
    #stress_tensor=False,
    #xc=None,
    label='NaCl_vac')
    config.calc.command="cp2k_shell.psmp"
    #config.set_constraint(constraint)

# Make the NEB object, interpolate to guess the intermediate steps
band = NEB(configs, )
band.interpolate()

relax = QuasiNewton(band, trajectory='A2B.traj')

# Do the calculation
relax.run()

# Compare intermediate steps to initial energy
e0 = configs[0].get_potential_energy()
for config in configs:
    d = config[-2].position - config[-1].position
    print(np.linalg.norm(d), config.get_potential_energy() - e0)
# -

e0 = configs[0].get_potential_energy()
distance, energy = [], []
tot_distance = 0
for config in configs:
    d = config[-2].position - config[-1].position
    tot_distance += np.linalg.norm(d)
    distance.append(tot_distance)
    energy.append(config.get_potential_energy() - e0)
plt.plot(distance, energy)
plt.scatter(distance, energy)

import ase
for i in range(1, 9):
    write("image_{}.xyz".format(i), configs[i])
