import os
from collections.abc import Sequence
from typing import Dict, Literal

import ase.data
import ase.io
import ase.neighborlist
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch_geometric.data import Data

from .data_structure import AtomsGraph


@functional_datapipe("read_ase")
class ASEFileReader(IterDataPipe):
    """
    Read atoms from ase readable file.
    Multiple files can be read by providing a list of filenames.

    Args:
        dp (IterDataPipe): Iterable data pipe that yields str or (str, str) where the first str is the filename and the
            second str is the index of the atoms in the file. If the second str is not provided,
            all atoms in the file will be read.
    """

    def __init__(self, dp: IterDataPipe):
        self.dp = dp

    def __iter__(self):
        for item in self.dp:
            if isinstance(item, str):
                filename, index = item, ":"
            elif isinstance(item, Sequence):
                if len(item) != 2:
                    raise ValueError("Input datapipe must yield str or (str, str), but got {}".format(item))
                filename, index = item
            if not isinstance(filename, (str, os.PathLike)):
                raise TypeError("Input datapipe must yield str or os.PathLike, but got {}".format(type(filename)))
            for atoms in ase.io.iread(filename, index):
                yield atoms


@functional_datapipe("atoms_to_graph")
class AtomsGraphParser(IterDataPipe):
    """
    Convert ase.Atoms to AtomsGraph.

    Args:
        dp (IterDataPipe): Iterable data pipe that yields ase.Atoms.
        read_properties (bool): Whether to read properties (energy, ...) from ase.Atoms. Default: True.

    """

    default_float_dtype = torch.get_default_dtype()

    def __init__(self, dp: IterDataPipe, read_properties: bool = True):
        self.dp = dp
        self.read_properties = read_properties

    def __iter__(self):
        for atoms in self.dp:
            if not isinstance(atoms, ase.Atoms):
                raise TypeError("Input datapipe must yield ase.Atoms, but got {}".format(type(atoms)))
            # do something with atoms
            yield AtomsGraph.from_ase(atoms, read_properties=self.read_properties)


@functional_datapipe("build_neighbor_list")
class NeighborListBuilder(IterDataPipe):
    """Neighbor list builder.

    Args:
        dp (IterDataPipe): Iterable data pipe that yields AtomsGraph.
        cutoff (float): Cutoff radius for neighbor list.
        self_interaction (bool): Whether to include self interaction. Default: False.
        backend (str): Backend for neighbor list builder. Default: "ase".
    """

    def __init__(
        self,
        dp: IterDataPipe,
        cutoff: float,
        self_interaction: bool = False,
        backend: Literal["ase", "torch", "matscipy"] = "ase",
        **kwargs,
    ):
        self.dp = dp
        self.cutoff = cutoff
        self.self_interaction = self_interaction
        self.backend = backend
        self.kwargs = kwargs

    def __iter__(self):
        for data in self.dp:
            if not isinstance(data, AtomsGraph):
                raise TypeError("Input datapipe must yield AtomsGraph, but got {}".format(type(data)))
            if data.cell.shape[0] != 1:
                raise ValueError("Does not support batched data")

            data.build_neighborlist(
                self.cutoff, self_interaction=self.self_interaction, backend=self.backend, **self.kwargs
            )
            yield data


@functional_datapipe("standardize_property")
class PropertyStandardizer(IterDataPipe):
    def __init__(self, dp: IterDataPipe, mean: float, std: float, target: str):
        """Standardize property in data.

        Args:
            dp (IterDataPipe): Iterable data pipe that yields torch_geometric.data.Data.
            mean (float): Mean of the property.
            std (float): Standard deviation of the property.
            target (str): Property to be standardized.

        """
        self.dp = dp
        self.mean = mean
        self.std = std
        self.target = target

    def __iter__(self):
        for data in self.dp:
            if not isinstance(data, Data):
                raise TypeError("Input datapipe must yield torch_geometric.data.Data, but got {}".format(type(data)))
            if self.target not in data:
                raise ValueError("Property {} not found in data".format(self.target))
            data[self.target] = (data[self.target] - self.mean) / self.std
            yield data


@functional_datapipe("unstandardize_property")
class PropertyUnStandardizer(IterDataPipe):
    def __init__(self, dp: IterDataPipe, mean: float, std: float, target: str):
        """Unstandardize property in data.

        Args:
            dp (IterDataPipe): Iterable data pipe that yields torch_geometric.data.Data.
            mean (float): Mean of the property.
            std (float): Standard deviation of the property.
            target (str): Property to be unstandardized.

        """
        self.dp = dp
        self.mean = mean
        self.std = std
        self.target = target

    def __iter__(self):
        for data in self.dp:
            if not isinstance(data, Data):
                raise TypeError("Input datapipe must yield torch_geometric.data.Data, but got {}".format(type(data)))
            if self.target not in data:
                raise ValueError("Property {} not found in data".format(self.target))
            data[self.target] = data[self.target] * self.std + self.mean
            yield data


@functional_datapipe("subtract_atomref")
class SubtractAtomref(IterDataPipe):
    """Subtract atomic energy from total energy.

    Args:
        dp (IterDataPipe): Iterable data pipe that yields torch_geometric.data.Data.
        atomic_energies (Dict[str, float]): Atomic energies.
    """

    def __init__(self, dp: IterDataPipe, atomic_energies: Dict[str, float]):
        self.dp = dp
        self.atomic_energies = atomic_energies

    def __iter__(self):
        for data in self.dp:
            if not isinstance(data, Data):
                raise TypeError("Input datapipe must yield torch_geometric.data.Data, but got {}".format(type(data)))
            if data.cell.shape[0] != 1:
                raise ValueError("Does not support batched data")

            elems = data.elems.numpy()
            symbols = [ase.data.chemical_symbols[elem] for elem in elems]
            for symbol in symbols:
                if symbol not in self.atomic_energies:
                    raise ValueError("Atomic energy for {} not found".format(symbol))
                data["energy"] -= self.atomic_energies[symbol]

            yield data
