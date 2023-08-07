import os
from abc import ABC, abstractmethod
from os import PathLike

import ase.io
from ase import Atoms


class TrajectoryWriter(ABC):
    def __init__(self, filename: PathLike, append: bool = False):
        self.filename = filename
        self.append = append
        if not append and os.path.exists(filename):
            os.remove(filename)

    @abstractmethod
    def write(self, atoms: Atoms) -> None:
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class XYZTrajectoryWriter(TrajectoryWriter):
    def __init__(self, filename: PathLike, append: bool = False):
        super().__init__(filename, append)
        self.fileobj = None

    def write(self, atoms: Atoms) -> None:
        ase.io.write(self.fileobj, atoms, format="extxyz")

    def __enter__(self):
        self.fileobj = open(self.filename, "a")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fileobj.close()


class ASETrajWriter(TrajectoryWriter):
    def __init__(self, filename: PathLike, append: bool = False):
        super().__init__(filename, append)
        self.trajectory = ase.io.Trajectory(filename, "a")

    def write(self, atoms: Atoms) -> None:
        self.trajectory.write(atoms)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
