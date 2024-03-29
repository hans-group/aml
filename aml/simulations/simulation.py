import os
from abc import ABC, abstractmethod
from datetime import datetime
from os import PathLike

from ase import Atoms

from aml.common.utils import log_and_print
from aml.simulations.io import ASETrajWriter, XYZTrajectoryWriter


class Simulation(ABC):
    def __init__(
        self,
        atoms: Atoms,
        log_file: PathLike | None = None,
        log_interval: int = 1,
        append_log: bool = False,
        trajectory: PathLike | None = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
        store_trajectory: bool = False,
    ):
        self.atoms = atoms
        if atoms.calc is None:
            raise ValueError("Atoms object must have a calculator.")
        self.log_file = log_file
        if log_file is not None:
            if os.path.exists(log_file) and not append_log:
                os.remove(log_file)
        self.log_interval = log_interval
        self.trajectory = trajectory
        self.trajectory_interval = trajectory_interval
        self.store_trajectory = store_trajectory
        if trajectory is not None:
            if str(trajectory).endswith(".xyz"):
                self.traj_writer = XYZTrajectoryWriter(trajectory, append_trajectory)
            elif str(trajectory).endswith(".traj"):
                self.traj_writer = ASETrajWriter(trajectory, append_trajectory)
            else:
                raise ValueError("Unsupported trajectory format. Use .xyz or .traj.")
        else:
            self.traj_writer = None
        self._step = 1
        self.traj = []

    @abstractmethod
    def _make_log_entry(self) -> dict[str, str | int | float | bool]:
        pass

    def _log(self):
        log_entry = self._make_log_entry()
        curr_time = datetime.now().strftime("%H:%M:%S")
        log_str = f"[{curr_time}] Step: {self._step} "
        for key, val in log_entry.items():
            if isinstance(val, float):
                log_str += f"{key}: {val:.4f} "
            else:
                log_str += f"{key}: {val} "
        log_and_print(log_str, self.log_file)

    @abstractmethod
    def step(self) -> bool:
        """A single step of the simulation. Returns True if the simulation should stop."""
        pass

    def run(self, n_steps: int) -> None:
        for n in range(1, n_steps + 1):
            stop = self.step()
            if n % self.log_interval == 0:
                self._log()
            if self.trajectory is not None and n % self.trajectory_interval == 0:
                with self.traj_writer as w:
                    w.write(self.atoms)
            if self.store_trajectory and n % self.trajectory_interval == 0:
                atoms = self.atoms.copy()
                calc = self.atoms.calc
                atoms.calc = calc
                self.traj.append(self.atoms.copy())
            if stop:
                break
            self._step += 1
