import os

import numpy as np
from ase import Atoms, units

from aml.simulations.simulation import Simulation
from aml.simulations.temperature_strategy import ConstantTemperature, TemperatureStrategy

from .move import NeighborSwap, Swap



class GrandCanonicalSingleMonteCarlo(Simulation):
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


# TODO: Implement later for multi-component system
class GrandCanonicalSwapMonteCarlo(Simulation):
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
