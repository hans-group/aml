# Performing Monte-carlo simulations to find stable adsorption configuration

## Task description

The objective of this task is to find thermodynamically stable structures of Pt(111)/O<sub>ads</sub> at various coverage of O using machine-learned potential and Monte-Carlo simulations.

> Of course this can be more efficiently done with cluster expansion. This is just for demonstration and tutorial purpose.

The workflow is:
- Generate training dataset spanning configurational space of Pt(111)/O<sub>ads</sub> system
- Train the potential
- Run MC

## Data description

`Pt111_O_ads.db` contains 53 structures of 6-layer Pt(111) surface with various configuration of adsorbed O on hollow sites. Unlike previous examples, the structures are unrelaxed and corresponding energies are from DFT-relaxed structures. So the traingng task is **"Initial structure to relaxed energy"**. This is similar to `IS2RE` task in open catalyst project(OCP). This limits the degree of freedom to only occupation state, not the random displacements from lattice sites, then simplifying the learning task. (So 53 training structures are enough)

The configurations are generated using `icet`, up to 5 times of surface primitive cell. The vacany atoms are represented as "X".

## Training the model

Run `python train.py -c config.yaml`.