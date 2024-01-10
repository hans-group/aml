# Atomistic Machine Learning (AML)

Python package for atomistic machine learning. Currently focuses on interatomic potential (IAP), but not limited to.

See `tutorials/` for how-to-start guide. In `examples/` directory there are example codes for the most common tasks.

## Features

- Implementations of well-established IAP models. See below for the list of models.
- Tightly integrated with pytorch-lightning
- Interface to `ase` calculator and customized molecular simulation codes
  - Geometry optimization
  - Molecular dynamics (NVE, NVT, NPT)
  - Metadynamics
  - Monte-carlo simulations (NVT)

## Installation

- Install pytorch, pytorch_geometric & pytorch_lightning
  - Also pytorch_scatter and pytorch_sparse is required
  - Follow the official docs: [[PyTorch]](https://pytorch.org/get-started/locally/) [[PyTorch Geometric]](https://pytorch-geometric.readthedocs.io/en/2.3.1/install/installation.html)
- Clone this repo (git clone ...)
- Run `pip install .`


## Implemented models
Model codes are adopted from the sources below and modified:
  - BPNN [1]: Custom implementation
  - SchNet [2]: [pytorch-geometric code](https://github.com/pyg-team/pytorch_geometric)
  - PaiNN [3]: [Official Github repository](https://github.com/atomistic-machine-learning/schnetpack)
  - Equivariant Transformer [4]: [TorchMD-net Github repository](https://github.com/torchmd/torchmd-net)
  - Gemnet [5]: [open catalyst project (OCP)](https://github.com/Open-Catalyst-Project/ocp)
  - NequIP [6]: [Official Github repository](https://github.com/mir-group/nequip)
  - Allegro [7]: [Official Github repository](https://github.com/mir-group/allegro)
  - MACE [8]: [Official Github repository](https://github.com/ACEsuit/mace)
  - EquiFormerV2 [9]: [Official Github repository](https://github.com/atomicarchitects/equiformer_v2)

## References

[1] Behler J, Parrinello M. Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. Phys Rev Lett 2007 Apr;98:146401.

[2] Schütt KT, Kindermans PJ, Sauceda HE, Chmiela S, Tkatchenko A, Müller KR, SchNet: A continuous-filter convolutional
neural network for modeling quantum interactions; 2017.

[3] Schütt KT, Unke OT, Gastegger M, Equivariant message passing for the prediction of tensorial properties and molecular
spectra; 2021.

[4] Thölke P, Fabritiis GD, TorchMD-NET: Equivariant Transformers for Neural Network based Molecular Potentials; 2022.

[5] Gasteiger J, Becker F, Günnemann S, GemNet: Universal Directional Graph Neural Networks for Molecules; 2022.


[6] Batzner, S., Musaelian, A., Sun, L. et al. E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials. Nat Commun 13, 2453 (2022).


[7] Musaelian, A., Batzner, S., Johansson, A. et al. Learning local equivariant representations for large-scale atomistic dynamics. Nat Commun 14, 579 (2023).

[8] Batatia I, Kovács DP, Simm GNC, Ortner C, Csányi G, MACE: Higher Order Equivariant Message Passing Neural Networks
for Fast and Accurate Force Fields; 2023.

[9] Liao YL, Wood B, Das A, Smidt T, EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Repre-
sentations; 2023.

