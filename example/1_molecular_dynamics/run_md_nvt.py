"""Example for running a molecular dynamics simulation.
"""
import ase.build

import aml

device = "cpu"
# device = "cuda"  # !!! uncomment this line to run on GPU !!!

model = aml.load_iap("../pretrained_model/painn_water_molecule_best.ckpt").to(device)
calc = aml.simulations.AMLCalculator(
    model,
    neighborlist_backend="matscipy",  # Faster neighborlist backend than ase
    neighborlist_skin=0.2,  # Only compute neighbors again when atoms move more than 0.2 Angstrom
)
atoms = ase.build.molecule("H2O")
atoms.calc = calc  # attach the calculator to the atoms object

md = aml.simulations.MolecularDynamics(
    atoms=atoms,
    timestep=0.5,  # in fs
    temperature=300,  # in K
    ensemble="nvt_langevin",  # NVT ensemble with langevin dynamics
    ensemble_params={"friction": 0.01},  # friction in 1/fs for langevin dynamics
    log_file="water_md.log",
    log_interval=10,
    trajectory="water_md.xyz",
    trajectory_interval=5,
)
md.run(n_steps=100)
