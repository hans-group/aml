import os
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import Batch

from aml.common.utils import log_and_print
from aml.data.data_structure import AtomsGraph
from aml.simulations.ase_interface import AMLCalculator
from aml.simulations.io import ASETrajWriter, XYZTrajectoryWriter
from aml.simulations.simulation import Simulation


def get_max_force(force):  # force
    return np.sqrt(force**2).sum(axis=1).max()


def add_index_to_filename(path: PathLike, index: int) -> PathLike:
    path = Path(path)
    return path.parent / f"{path.stem}_{index}{path.suffix}"


class BatchedGeometryOptimization(Simulation):
    def __init__(
        self,
        atoms_list: list[Atoms],
        calc: AMLCalculator,
        optimizer_config: dict | None = None,
        max_force: float = 0.05,
        log_file: PathLike | None = None,
        log_interval: int = 1,
        trajectory: PathLike | None = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
    ):
        self.atoms_list = atoms_list

        self.log_file = log_file
        if log_file is not None:
            if os.path.exists(log_file):
                os.remove(log_file)
        self.log_interval = log_interval
        self.trajectory = trajectory
        self.trajectory_interval = trajectory_interval
        self.optimizer_config = optimizer_config or {}

        if trajectory is not None:
            if str(trajectory).endswith(".xyz"):
                self.traj_writer = []
                for i, _ in enumerate(atoms_list):
                    self.traj_writer.append(
                        XYZTrajectoryWriter(add_index_to_filename(trajectory, i), append_trajectory)
                    )
            elif str(trajectory).endswith(".traj"):
                self.traj_writer = []
                for i, _ in enumerate(atoms_list):
                    self.traj_writer.append(ASETrajWriter(add_index_to_filename(trajectory, i), append_trajectory))
            else:
                raise ValueError("Unsupported trajectory format. Use .xyz or .traj.")
        else:
            self.traj_writer = None

        self._step = 1
        self._energy = None
        self._fmax = None
        self.max_force = max_force

        self.model = calc.model
        device = self.model.parameters().__next__().device
        data_list = [AtomsGraph.from_ase(atoms, calc.r_cut) for atoms in atoms_list]
        self.batch = Batch.from_data_list(data_list).to(device)
        self.batch.pos.requires_grad = True

        lr = self.optimizer_config.pop("lr", 1.0)
        max_iter = self.optimizer_config.pop("max_iter", 15)
        line_search_fn = self.optimizer_config.pop("line_search_fn", None)
        self.optimizer = torch.optim.LBFGS((self.batch.pos,), lr=lr, max_iter=max_iter, line_search_fn=line_search_fn)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)

        def closure():
            self.optimizer.zero_grad()
            energy = self.model(self.batch.to_dict())["energy"]
            energy_sum = energy.sum()
            energy_sum.backward()
            return energy_sum

        self.closure = closure
        self.converged = False

    def _make_log_entry(self) -> dict[str, str | int | float | bool]:
        log_entry = {
            "Step": self._step,
            "PE [eV]": self._energy,
            "F_max [eV/A]": float(self._fmax),
        }
        return log_entry

    def step(self) -> bool:
        self.optimizer.step(self.closure)
        output = self.model(self.batch.to_dict())
        self._energy = output["energy"].detach().cpu().numpy()
        self._fmax = get_max_force(output["force"].detach().cpu().numpy())
        self.scheduler.step(self._fmax)
        self.converged = self._fmax < self.max_force
        datalist = self.batch.to_data_list()
        for i, data in enumerate(datalist):
            self.atoms_list[i].set_positions(data.pos.detach().cpu().numpy())
        return self.converged

    def run(self, max_steps: int = 50) -> None:
        initial_out = self.model(self.batch.to_dict())
        self._energy = initial_out["energy"].detach().cpu().numpy()
        self._fmax = get_max_force(initial_out["force"].detach().cpu().numpy())
        with np.printoptions(precision=4, suppress=True):
            self.model.train()
            log_and_print(f"Inital energy: {self._energy}", self.log_file)
            log_and_print(f"Inital fmax: {self._fmax:.6f}", self.log_file)
            if self.trajectory is not None:
                for i, writer in enumerate(self.traj_writer):
                    with writer as w:
                        w.write(self.atoms_list[i])

            for n in range(1, max_steps + 1):
                converged = self.step()
                if n % self.log_interval == 0:
                    self._log()
                if self.trajectory is not None and n % self.trajectory_interval == 0:
                    for i, writer in enumerate(self.traj_writer):
                        with writer as w:
                            w.write(self.atoms_list[i])
                if converged:
                    break
                self._step += 1
            self.model.eval()
            if not self.converged:
                log_and_print("Geometry optimization did not converge.", self.log_file)
        return self.converged
