Introduction to AML
===================

`aml` is a python tookit for atomistic machine learning.


Basic concept 
-------------

Interatomic potential (IAP) is a function of atomic structure which predicts the potential energy of given structure.
In general, it can be written as following functional form:

.. math::
    E = f(Z_1, Z_2, \dots, Z_n, \mathbf{r}_1, \mathbf{r}_2, \dots, \mathbf{r}_n;\theta_1, \theta_2, \dots, \theta_m)

where :math:`Z_i \in \mathbb{Z}` is the atomic number of atom :math:`i`, and :math:`\mathbf{r}_i \in \mathbb{R}^3` is the position of atom :math:`i`, and :math:`\theta_i \in \mathbb{R}` are the parameters of the potential.
Many properties depending on the potential energy can be calculated by IAP. For example, force acting on an atom can be calculated by the first derivative of the potential energy with respect to the atomic positions:

.. math::
    F = -\nabla_\mathbf{r} E = \left[-\frac{\partial E}{\partial r_x}, -\frac{\partial E}{\partial r_y}, -\frac{\partial E}{\partial r_z}\right]

However, finding such high-dimensional relationship is not trivial.

**Classical IAPs** use hand-crafted geometric features(ex. bond length, angle, torsion, etc.) constructed from structure and apply speicific functional forms(ex. polynomial, harmonic potential, etc.) to predict the potential energy.
The parameters are optimized by fitting the potential energy to the reference data(ex. experimental data, DFT data, etc.). These are efficient, but often fail to capture the complex behavior of the system such as chemical reaction.

**Neural IAPs** utilize data-driven approach to learn the potential energy from the reference data. They use neural networks to approximate the functional form of the potential energy. 
The parameters of the neural network are optimized by fitting the potential energy to the reference data, usually quamtum chemical calculations such as DFT. 
Neural IAP can achieve QM-level accuacy with highly reduced cost.

The way of structural representation and architecture of neural network varies depending on individual models.
Popular Behler-Parrinello neural network(BPNN) uses hand-crafted representation of local environment of each atom as input of neural network and compute atomic energy with it.
More recent models such as SchNet uses graph convolutional network to automatically learn the representation of atom.

This package provides modular implementations of popular neural IAP models and general tools for training, evaluating, and using them.
The project is based on ``pytorch`` and tightly interfaces with ``pytorch_lightning`` for training and ``ase`` for atomistic simulation.

Implemented models
------------------
- Behler-Parrinello neural network (BPNN)
- SchNet
- PaiNN
- Gemnet_T
- NequIP
- MACE

Installation
------------

First, install ``pytorch`` and ``pytorch_geometric`` depending on your system.
Then, install ``neural_iap`` with ``pip``:

.. code-block:: bash

    pip install git+https://github.com/mjhong0708/neural_iap.git

If you want to modify or develop the package, clone the repository and install it with ``pip -e``:

.. code-block:: bash

    git clone git+https://github.com/mjhong0708/neural_iap.git
    cd neural_iap
    pip install -e ".[dev]"
    