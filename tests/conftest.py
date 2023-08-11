import ase.build
import pytest
from torch_geometric.data import Batch

from aml.data.data_structure import AtomsGraph

# ========================== Data ==========================


@pytest.fixture(scope="session", autouse=True)
def water_molecule():
    atoms = ase.build.molecule("H2O")
    graph = AtomsGraph.from_ase(atoms, neighborlist_cutoff=5.0)
    return graph


@pytest.fixture(scope="session", autouse=True)
def si_bulk():
    atoms = ase.build.bulk("Si", cubic=True)
    graph = AtomsGraph.from_ase(atoms, neighborlist_cutoff=5.0)
    return graph


@pytest.fixture(scope="session", autouse=True)
def batch():
    atoms_1 = ase.build.bulk("Si", cubic=True)
    atoms_1[0].symbol = "Ge"
    atoms_2 = ase.build.bulk("Pt", cubic=True).repeat([2, 2, 2])
    atoms_2[0].symbol = "Ni"
    graph_1 = AtomsGraph.from_ase(atoms_1, neighborlist_cutoff=5.0)
    graph_2 = AtomsGraph.from_ase(atoms_2, neighborlist_cutoff=5.0)
    batch = Batch.from_data_list([graph_1, graph_2])
    return batch
