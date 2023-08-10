from os import PathLike

import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS, FIRE, LBFGS, BFGSLineSearch, GPMin, LBFGSLineSearch, QuasiNewton

from aml.common.utils import log_and_print
from aml.data.data_structure import AtomsGraph
from aml.simulations.ase_interface import AMLCalculator
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
        algorithm: str = "torch_lbfgs",
        optimizer_config: dict | None = None,
        max_force: float = 0.02,
        log_file: PathLike | None = None,
        log_interval: int = 1,
        trajectory: PathLike | None = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
    ):
        super().__init__(atoms, log_file, log_interval, trajectory, trajectory_interval, append_trajectory)
        self._energy = atoms.get_potential_energy()
        self._fmax = get_max_force(atoms.get_forces())
        self.max_force = max_force
        self.algorithm = algorithm
        self.optimizer_config = optimizer_config or {}
        self.converged = False

        if self.algorithm == "torch_lbfgs":
            self._setup_torch_opt(self.optimizer_config)
        else:
            self.optimizer = ase_optimizer_map[self.algorithm](self.atoms, **self.optimizer_config)

    def _setup_torch_opt(self, optimizer_config):
        if not isinstance(self.atoms.calc, AMLCalculator):
            raise ValueError("For 'torch_lbfgs' algorithm, calculator must be AMLCalculator.")
        self.model = self.atoms.calc.model

        device = self.model.parameters().__next__().device
        self.data = AtomsGraph.from_ase(self.atoms, self.model.cutoff).to(device)
        self.data["pos"].requires_grad = True
        # Default values
        lr = optimizer_config.pop("lr", 1.0)
        max_iter = optimizer_config.pop("max_iter", 15)
        line_search_fn = optimizer_config.pop("line_search_fn", None)
        self.optimizer = torch.optim.LBFGS(
            (self.data["pos"],), lr=lr, max_iter=max_iter, line_search_fn=line_search_fn, **optimizer_config
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

        def closure():
            self.optimizer.zero_grad()
            energy = self.model(self.data)["energy"]
            energy.backward()
            return energy

        self.closure = closure

    def _make_log_entry(self) -> dict[str, str | int | float | bool]:
        log_entry = {
            "Step": self._step,
            "PE [eV]": float(self._energy),
            "PE [eV/atom]": float(self._energy / len(self.atoms)),
            "F_max [eV/A]": float(self._fmax),
        }
        return log_entry

    def _torch_step(self):
        self.optimizer.step(self.closure)
        output = self.model(self.data)
        energy = output["energy"].detach().cpu().numpy().squeeze()
        forces = output["force"].detach().cpu().numpy().squeeze()
        fmax = get_max_force(forces)
        self.scheduler.step(fmax)
        self._energy = energy
        self._fmax = fmax
        self.atoms.set_positions(self.data["pos"].detach().cpu().numpy())

    def _ase_step(self):
        self.optimizer.step()
        self._energy = self.atoms.get_potential_energy()
        self._fmax = get_max_force(self.atoms.get_forces())

    def step(self) -> None:
        if self.algorithm == "torch_lbfgs":
            self._torch_step()
        else:
            self._ase_step()
        if self._fmax < self.max_force:
            self.converged = True
        return self.converged

    def run(self, max_steps: int = 50) -> None:
        if self.algorithm == "torch_lbfgs":
            self.model.train()
        log_and_print(f"Inital energy: {self._energy:.6f}", self.log_file)
        log_and_print(f"Inital fmax: {self._fmax:.6f}", self.log_file)
        if self.trajectory is not None:
            with self.traj_writer as w:
                w.write(self.atoms)
        super().run(max_steps)
        if self.algorithm == "torch_lbfgs":
            self.model.eval()
        if not self.converged:
            log_and_print("Geometry optimization did not converge.", self.log_file)
            return False
        return True
