Installation
============


Dependencies
------------

These packages should be installed before installing this package with pip.

- ``python>=3.10``
- ``torch>=2.0.1``
- ``torch_geometric>=2.3``
- ``torch_scatter``
- ``torch_sparse``
- ``torch_cluster``

See `pytorch-geometric docs <https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html>`_  and `PyTorch docs <https://pytorch.org/get-started/locally>`_ 
for more details on how to install these packages.

After installing the above packages, install this package with pip.

Install (user)
--------------
.. code-block:: bash

    git clone git+https://github.com/hans-group/aml.git
    cd aml
    pip install -e "."

The option ``-e`` is not necessary, but it is recommended to use it to install the package in the editable mode.
Then you can update the package by pulling the latest version from the repository.

Install (developer)
-------------------
.. code-block:: bash

    git clone git+https://github.com/hans-group/aml.git
    cd neural_iap
    pip install -e ".[dev]"

This installs recommended packages for developers. (flake8, pytest, black, etc.)
    