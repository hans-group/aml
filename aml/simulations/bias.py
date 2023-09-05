import warnings
from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from ase_extension.geometry import _ext


class BiasPotential(ABC):
    @abstractmethod
    def _get_bias_energy_and_force(self, atoms):
        pass

    def adjust_forces(self, atoms, forces):
        _, F_bias = self._get_bias_energy_and_force(atoms)
        forces += F_bias

    def adjust_potential_energy(self, atoms):
        E_bias, _ = self._get_bias_energy_and_force(atoms)
        return E_bias

    def adjust_positions(self, atoms, new):
        pass

    def get_removed_dof(self, atoms):
        return 0


class RMSDBiasPotential(BiasPotential):
    def __init__(self, reference_points: list[Atoms], k, alpha, kappa):
        """
        Args:
            reference_points: list of reference points
            k: pushing intensity
            alpha: width of the gaussian
            kappa: damping factor
        """
        self.reference_points = reference_points
        self.step_count = np.zeros(len(reference_points))
        self.k = k
        self.alpha = alpha
        self.kappa = kappa

    def _update_reference(self, R):
        self.reference_points.append(R)
        self.step_count += 1
        self.step_count = np.append(self.step_count, 0)

    def _remove_oldest_reference(self):
        self.reference_points.pop(0)
        self.step_count = np.delete(self.step_count, 0)

    def _get_bias_energy_and_force(self, atoms):
        if not self.reference_points:
            return 0, 0
        R = atoms.get_positions()
        E = 0
        F = np.zeros_like(R)
        k = self.k * R.shape[0]  # k is per atom
        for atoms_ref, step in zip(self.reference_points, self.step_count, strict=True):
            R_ref = atoms_ref.get_positions()
            rmsd, rmsd_grad, *_ = _ext.compute_minimum_rmsd(R, R_ref, True)
            damping_factor = 2 / (1 + np.exp(-self.kappa * step)) - 1
            if damping_factor > 0:
                dE = k * np.exp(-self.alpha * rmsd) * damping_factor
                dF = -k * np.exp(-self.alpha * rmsd) * (-self.alpha * rmsd_grad) * damping_factor
                if np.isnan(dF).any():
                    warnings.warn(
                        "NaN in bias force, possibly due to duplicate structure. Zeroing force.", stacklevel=1
                    )
                    dF = np.zeros_like(dF)
                E += dE
                F += dF
        return E, F
