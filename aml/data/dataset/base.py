from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Sequence, Tuple

import ase.data
import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import InMemoryDataset as PyGInMemoryDataset
from torch_geometric.data.data import BaseData
from torchdata.datapipes.iter import IterDataPipe
from tqdm import tqdm
from typing_extensions import Self

from aml.common.registry import registry
from aml.common.utils import Configurable
from aml.data.utils import write_lmdb_dataset


@registry.register_dataset("base_graph_dataset")
class BaseDataset(Dataset, Configurable, ABC):
    """Abstract base class for graph dataset."""

    def subset(self, size: int | float, seed: int = 0, return_idx=False) -> Self:
        """Create a subset of the dataset with `n` elements."""
        size = _determine_size(self, size)
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return self[indices[:size]], indices[:size]
        return self[indices[:size]]

    def split(self, size: int | float, seed: int = 0, return_idx=False) -> Tuple[Self, Self]:
        """Split the dataset into two subsets of `n` and `len(self) - n` elements."""
        size = _determine_size(self, size)
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return (self[indices[:size]], self[indices[size:]]), (indices[:size], indices[size:])
        return self[indices[:size]], self[indices[size:]]

    def train_val_test_split(
        self,
        train_size: int | float,
        val_size: int | float,
        seed: int = 0,
        return_idx=False,
    ) -> Tuple[Self, Self, Self]:
        train_size = _determine_size(self, train_size)
        val_size = _determine_size(self, val_size)

        if return_idx:
            (train_dataset, rest_dataset), (train_idx, _) = self.split(train_size, seed, return_idx)
            (val_dataset, test_dataset), (val_idx, test_idx) = rest_dataset.split(val_size, seed, return_idx)
            return (train_dataset, val_dataset, test_dataset), (train_idx, val_idx, test_idx)

        train_dataset, rest_dataset = self.split(train_size, seed)
        val_dataset, test_dataset = rest_dataset.split(val_size, seed)
        return train_dataset, val_dataset, test_dataset

    def write_lmdb(
        self,
        path: str,
        map_size: int = 1099511627776 * 2,
        prog_bar: bool = True,
        compress: bool = False,
    ):
        """Write dataset to LMDB database.

        Args:
            path (str): Path to LMDB database.
            map_size (int, optional): Size of memory mapping. Defaults to 1099511627776*2.
            prog_bar (bool, optional): Whether to show progress bar. Defaults to True.
            compress (bool, optional): Whether to compress data by zlib. Defaults to False.
        """
        data_iter = tqdm(self, total=len(self)) if prog_bar else self
        dataset_config = self.get_config()
        metadata = {"parent_dataset": dataset_config, "map_size": map_size, "compress": compress}
        write_lmdb_dataset(data_iter, path, map_size=map_size, compress=compress, metadata=metadata)

    @property
    @abstractmethod
    def avg_num_neighbors(self):
        """Compute average number of neighbors per atom."""

    # Statistics
    def avg_atomic_property(self, key: str) -> dict[str, float]:
        """Compute average atomic property.
        Only makes sense if the property is extensive scalar property. ex) total energy

        Args:
            dataset (GraphDatasetMixin):
            key (str): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            dict[str, float]: _description_
        """
        sample = self[0]
        _check_contain_key(sample, "elems")
        _check_contain_key(sample, key)
        _check_scalar_key(sample, key)

        data_list = [data for data in self]
        elems = np.concatenate([data.elems for data in data_list])
        species = np.unique(elems)
        n_structures = len(self)
        n_elems = len(species)

        A = np.zeros((n_structures, n_elems))
        B = np.zeros((n_structures,))
        for i, data in enumerate(data_list):
            B[i] = data[key].squeeze().item()
            for j, elem in enumerate(species):
                A[i, j] = np.count_nonzero(data.elems == elem)

        try:
            solution = np.linalg.lstsq(A, B, rcond=None)[0]
            atomic_properties = {}
            for i, elem in enumerate(species):
                symbol = ase.data.chemical_symbols[elem]
                atomic_properties[symbol] = solution[i]
        except np.linalg.LinAlgError:
            atomic_properties = {}
            for elem in species:
                symbol = ase.data.chemical_symbols[elem]
                atomic_properties[symbol] = 0.0
        return atomic_properties

    def get_statistics(self, key, per_atom=False, per_species=False, reduce="mean"):
        if reduce == "mean":
            reduce_fn = torch.mean
        elif reduce == "std":
            reduce_fn = torch.std
        elif reduce == "rms":

            def reduce_fn(x):
                return torch.sqrt(torch.mean(x**2))

        sample = self[0]
        data_list = [data for data in self]
        _check_contain_key(sample, key)
        if per_atom and per_species:
            raise ValueError("per_atom and per_species cannot be True at the same time.")
        if per_atom:
            _check_contain_key(sample, "n_atoms")
            _check_scalar_key(sample, key)
        if per_species:
            _check_atomic_property_key(sample, key)
            elems = torch.cat([data.elems for data in data_list], dim=0)
            species = torch.unique(elems)
            values = {k.item(): [] for k in species}
            for data in data_list:
                elems = data.elems
                for s in species:
                    values[s.item()].append(data[key][elems == s])
            values = {k: torch.cat(v, dim=0) for k, v in values.items()}
            return {ase.data.chemical_symbols[k]: reduce_fn(v).item() for k, v in values.items()}
        else:
            values = []
            for data in data_list:
                x = data[key]
                if x.ndim == 0:
                    x = x.unsqueeze(0)
                if per_atom:
                    x = x / data.n_atoms
                values.append(x)
            values = torch.cat(values, dim=0)
            return reduce_fn(values)

    def get_config(self):
        config = {}
        config["@name"] = self.__class__.name
        config.update(super().get_config())
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = deepcopy(config)
        name = config.pop("@name", None)
        if cls.__name__ == "BaseDataset":
            if name is None:
                raise ValueError("Cannot initialize BaseDataset from config. Please specify the name of the model.")
            model_class = registry.get_dataset_class(name)
        else:
            if name is not None and hasattr(cls, "name") and cls.name != name:
                raise ValueError("The name in the config is different from the class name.")
            model_class = cls
        return super().from_config(config, actual_cls=model_class)


@registry.register_dataset("in_memory_dataset")
class InMemoryDataset(BaseDataset, PyGInMemoryDataset):
    def __init__(self, data: IterDataPipe | Sequence[BaseData] = None):
        PyGInMemoryDataset.__init__(self)
        if data is None:
            self.data, self.slices = None, None
        else:
            data = [d for d in data]
            self.data, self.slices = self.collate(data)

    def save(self, path):
        config = self.get_config()
        torch.save((config, self._data, self.slices), path)

    @classmethod
    def load(cls, path):
        config, data, slices = torch.load(path)
        dataset = cls()
        dataset.data, dataset.slices = data, slices
        for k, v in config.items():
            setattr(dataset, k, v)
        return dataset

    def get_config(self):
        if self.__class__.__name__ == "InMemoryDataset":
            return {}
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict):
        if cls.__name__ == "InMemoryDataset":
            raise RuntimeError("InMemoryDataset cannot be deserialized.")
        return super().from_config(config)

    @property
    def avg_num_neighbors(self):
        return self._data.edge_index.size(1) / self._data.pos.size(0)


def _determine_size(dataset, size):
    if isinstance(size, float):
        size = int(len(dataset) * size)
    elif isinstance(size, int):
        size = size
    else:
        raise TypeError(f"size must be int or float, not {type(size)}")
    if size > len(dataset):
        raise ValueError(f"size must be less than or equal to the length of dataset, {len(dataset)}, but got {size}")
    return size


def _check_scalar_key(data, key):
    sample_value = data[key].squeeze()
    if not sample_value.ndim == 0:
        raise ValueError(f"Value of '{key}' must be a scalar.")


def _check_atomic_property_key(data, key):
    sample_value = data[key]
    if sample_value.size(0) != data.num_nodes:
        raise ValueError(f"Value of '{key}' must have the same length as the number of nodes.")


def _check_contain_key(data, key):
    if key not in data:
        raise ValueError(f"Data must contain '{key}' field.")
