import os
from abc import abstractmethod

import numpy as np
from ase import Atoms, units
from ase.neighborlist import neighbor_list

from aml.simulations.simulation import Simulation
from aml.simulations.temperature_strategy import ConstantTemperature, TemperatureStrategy

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


class CanonicalSwapMonteCarlo(Simulation):
    """Base class for performing Metropolis Monte-Carlo simulations in NVT ensemble."""

    def __init__(
        self,
        atoms: Atoms,
        swap_mode: str = "neighbor",  # "neighbor" or "full"
        swap_cutoff: float = 5.0,
        temperature: TemperatureStrategy | float = 300.0,
        log_file: os.PathLike | None = None,
        log_interval: int = 1,
        trajectory: os.PathLike | None = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
    ):
        if swap_mode == "neighbor":
            self.move_fn = NeighborSwap(cutoff=swap_cutoff)
        elif swap_mode == "full":
            self.move_fn = Swap()
        else:
            raise ValueError(f"swap_mode={swap_mode} is not supported.")
        if isinstance(temperature, (float, int)):
            temperature = ConstantTemperature(temperature)
        self.temperature = temperature
        super().__init__(atoms, log_file, log_interval, trajectory, trajectory_interval, append_trajectory)

    def compute_probabilty(self, atoms_i, atoms_j):
        E_i = atoms_i.get_potential_energy()
        E_j = atoms_j.get_potential_energy()
        dE = E_j - E_i
        T = self.temperature()

        if dE <= 0:
            return 1.0
        elif T <= 1e-16:
            return 0.0
        p = np.exp(-dE / (units.kB * T))
        return p

    def _make_log_entry(self):
        log_entry = {
            "PE [eV/atom]": self.atoms.get_potential_energy() / len(self.atoms),
            "Swap probability": self._prob,
            "Accepted": self._accepted,
            "T [K]": self.temperature.curr_temperature(),
        }
        return log_entry

    def step(self) -> bool:
        candidate = self.move_fn(self.atoms)
        prob = self.compute_probabilty(self.atoms, candidate)
        self._prob = prob
        if np.random.rand() < prob:
            self.atoms = candidate
            self._accepted = True
        else:
            self._accepted = False
        return False

    def run(self, n_steps: int) -> None:
        if self.temperature.n_steps is not None and self.temperature.n_steps != n_steps:
            raise ValueError(f"n_steps={n_steps} does not match temperature.n_steps={self.temperature.n_steps}")
        return super().run(n_steps)
