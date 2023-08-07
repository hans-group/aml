# Atomistic Machine Learning (AML)

Python package for atomistic machine learning. Currently focuses on interatomic potential (IAP).

## Features

- Contains implementations of well-established IAP models (schnet, nequip, gemnet, ...)
- Tightly integrated with pytorch-lightning
- Interface to `ase` calculator for molecular simulations
- Easy customization

## Installation

- Install pytorch, pytorch_geometric & pytorch_lightning
  - Also pytorch_scatter and pytorch_sparse is required
- Clone this repo (git clone ...)
- Run `pip install .`

## Credits

- Overall structure of project, especially idea of global registry, is inspired by [MMF](https://github.com/facebookresearch/mmf).
- Implementation of each model borrows codes from:
  - SchNet: [pytorch-geometric](https://github.com/pyg-team/pytorch_geometric)
  - PaiNN: [schnetpack](https://github.com/atomistic-machine-learning/schnetpack)
  - Gemnet: [open catalyst project](https://github.com/Open-Catalyst-Project/ocp)
  - NequIP: Official implementation by authors - [nequip](https://github.com/mir-group/nequip)
  - Allegro: Official implementation by authors - [allegro](https://github.com/mir-group/allegro)
  - MACE: Official implementation by authors - [mace](https://github.com/ACEsuit/mace)
  - BPNN: Custom implementation by repo author
