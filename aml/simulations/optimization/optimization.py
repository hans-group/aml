from os import PathLike

import numpy as np
from ase import Atoms
from ase.constraints import ExpCellFilter
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, GPMin, LBFGSLineSearch, QuasiNewton

from aml.common.utils import log_and_print
from aml.simulations.simulation import Simulation


def get_max_force(force):
    return np.sqrt(force**2).sum(axis=1).max()


ase_optimizer_map = {
    "lbfgs": LBFGS,
    "lbfgs_linesearch": LBFGSLineSearch,
    "bfgs": BFGS,
    "bfgs_linesearch": BFGSLineSearch,
    "fire": FIRE,
    "gpmin": GPMin,
    "quasi_newton": QuasiNewton,
}


class GeometryOptimization(Simulation):
    def __init__(
        self,
        atoms: Atoms,
        algorithm: str = "quasi_newton",
        optimizer_config: dict | None = None,
        optimize_cell: bool = False,
        max_force: float = 0.02,
        log_file: PathLike | None = None,
        log_interval: int = 1,
        append_log: bool = False,
        trajectory: PathLike | None = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
        *,
        strain_mask: list[bool] | None = None,
        constant_volume: bool = False,
    ):
        """Class to perform geometry optimization on an ASE Atoms object.
        Wraps ASE optimizers and provides logging and trajectory writing.

        Args:
            atoms (Atoms): ASE Atoms object to optimize.
            algorithm (str, optional): ASE optimizer to use. Defaults to "quasi_newton".
            optimizer_config (dict, optional): Configuration for the optimizer. Defaults to None.
            optimize_cell (bool, optional): Whether to optimize the cell. Defaults to False.
            max_force (float, optional): Maximum force allowed. Defaults to 0.02.
            log_file (PathLike, optional): File to write log to. Defaults to None.
            log_interval (int, optional): Interval to write log. Defaults to 1.
            trajectory (PathLike, optional): File to write trajectory to. Defaults to None.
            trajectory_interval (int, optional): Interval to write trajectory. Defaults to 1.
            append_trajectory (bool, optional): Whether to append to trajectory file. Defaults to False.
            strain_mask (list[bool], optional): The components of strains to ignore (if False).
                The components are [aa, bb, cc, ab, bc, ca]. Defaults to None, which means no strain is ignored.
            constant_volume (bool, optional): Whether to keep the volume constant. Defaults to False.
        """
        super().__init__(atoms, log_file, log_interval, append_log, trajectory, trajectory_interval, append_trajectory)
        if optimize_cell:
            self._objective = ExpCellFilter(self.atoms, mask=strain_mask, constant_volume=constant_volume)
        else:
            self._objective = self.atoms

        self._energy = self.atoms.get_potential_energy()
        self._fmax = get_max_force(self._objective.get_forces())
        self.max_force = max_force
        self.algorithm = algorithm
        self.optimizer_config = optimizer_config or {}
        self.optimize_cell = optimize_cell
        self.strain_mask = strain_mask
        self.constant_volume = constant_volume

        self.optimizer = ase_optimizer_map[self.algorithm](self._objective, **self.optimizer_config)

        self._converged = False

    def _make_log_entry(self) -> dict[str, str | int | float | bool]:
        log_entry = {
            "PE [eV]": float(self._energy),
            "PE [eV/atom]": float(self._energy / len(self.atoms)),
            "F_max [eV/A]": float(self._fmax),
        }
        return log_entry

    def step(self) -> None:
        self.optimizer.step()
        self._energy = self.atoms.get_potential_energy()
        self._fmax = get_max_force(self._objective.get_forces())
        if self._fmax < self.max_force:
            self._converged = True
        return self._converged

    def run(self, max_steps: int = 50) -> None:
        log_and_print(f"Inital energy: {self._energy:.6f}", self.log_file)
        log_and_print(f"Inital fmax: {self._fmax:.6f}", self.log_file)
        if self.trajectory is not None:
            with self.traj_writer as w:
                w.write(self.atoms)
        super().run(max_steps)
        if not self._converged:
            log_and_print("Geometry optimization did not converge.", self.log_file)
            return False
        return True
