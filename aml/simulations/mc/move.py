from abc import abstractmethod

import numpy as np
from ase import Atoms
from ase_extension.neighborlist import neighbor_list

from .utils import find_fixatoms_constraint


class RandomMove:
    """Base class for single step of random move."""

    def __init__(self):
        self.previous = None

    def __call__(self, atoms: Atoms) -> Atoms:
        self.previous = atoms
        tmp = atoms.copy()
        tmp.calc = atoms.calc
        moved_atoms = self.move(tmp)
        return moved_atoms

    @staticmethod
    def _get_movable_idx(atoms):
        all_idx = np.arange(len(atoms))
        fixatom = find_fixatoms_constraint(atoms)
        fixed_idx = fixatom.index if fixatom is not None else np.array([])
        candidate_idx = np.setdiff1d(all_idx, fixed_idx)
        return candidate_idx

    @abstractmethod
    def move(self, atoms: Atoms) -> Atoms:
        pass

    def restore(self):
        return self.previous


class Swap(RandomMove):
    def move(self, atoms: Atoms) -> Atoms:
        movable_idx = self._get_movable_idx(atoms)
        if len(movable_idx) == 0:
            raise RuntimeError("No move available")
        idx_i = np.random.choice(movable_idx)

        species = atoms.get_atomic_numbers()
        elem_i = species[idx_i]
        elem_mask = species[movable_idx] != elem_i  # Mask for different species
        candidate_idx = movable_idx[elem_mask]

        if len(candidate_idx) == 0:
            raise RuntimeError("No move available")
        idx_j = np.random.choice(candidate_idx)
        elem_j = species[idx_j]

        # Do swap
        atoms[idx_i].number = elem_j
        atoms[idx_j].number = elem_i
        return atoms


class NeighborSwap(RandomMove):
    def __init__(self, cutoff: float = 5.0):
        super().__init__()
        self.cutoff = cutoff

    def move(self, atoms: Atoms) -> Atoms:
        movable_idx = self._get_movable_idx(atoms)
        if len(movable_idx) == 0:
            raise RuntimeError("No move available")
        neigh_i, neigh_j = neighbor_list("ij", atoms, cutoff=self.cutoff, self_interaction=False)
        neigh_i = neigh_i.astype(np.int64)
        neigh_j = neigh_j.astype(np.int64)
        idx_i = np.random.choice(movable_idx)
        neigh_ij = neigh_j[neigh_i == idx_i]

        species = atoms.get_atomic_numbers()
        elem_i = species[idx_i]
        elem_mask = species[movable_idx] != elem_i  # Mask for different species
        candidate_idx = movable_idx[elem_mask]
        candidate_idx = np.intersect1d(candidate_idx, neigh_ij)
        if len(candidate_idx) == 0:
            return atoms
        idx_j = np.random.choice(candidate_idx)
        elem_j = species[idx_j]

        # Do swap
        atoms[idx_i].number = elem_j
        atoms[idx_j].number = elem_i
        return atoms
