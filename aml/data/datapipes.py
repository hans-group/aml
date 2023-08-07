import os
from collections.abc import Sequence
from typing import Dict

import ase.data
import ase.io
import ase.neighborlist
import torch
from torch.utils.data import IterDataPipe, functional_datapipe
from torch_geometric.data import Data

from .data_structure import AtomsGraph


@functional_datapipe("read_ase")
class ASEFileReader(IterDataPipe):
    def __init__(self, dp):
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
    default_float_dtype = torch.get_default_dtype()

    def __init__(self, dp):
        self.dp = dp

    def __iter__(self):
        for atoms in self.dp:
            if not isinstance(atoms, ase.Atoms):
                raise TypeError("Input datapipe must yield ase.Atoms, but got {}".format(type(atoms)))
            # do something with atoms
            yield AtomsGraph.from_ase(atoms)


@functional_datapipe("build_neighbor_list")
class NeighborListBuilder(IterDataPipe):
    def __init__(self, dp, cutoff, self_interaction=False, backend="ase", **kwargs):
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
    def __init__(self, dp, mean, std, target):
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
    def __init__(self, dp, mean, std, target):
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
    def __init__(self, dp, atomic_energies: Dict[str, float]):
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
