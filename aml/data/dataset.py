from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm
from typing_extensions import Self

from . import datapipes  # noqa: F401
from .data_structure import AtomsGraph

IndexType = slice | torch.Tensor | np.ndarray | Sequence


class SimpleDataset(InMemoryDataset):
    def __init__(self, dp=None):
        super().__init__()
        if dp is not None:
            data_list = [d for d in tqdm(dp)]
            self.data, self.slices = self.collate(data_list)
        else:
            self.data, self.slices = None, None

    def save(self, path):
        torch.save((self._data, self.slices), path)

    @classmethod
    def load(cls, path):
        dataset = cls()
        dataset.data, dataset.slices = torch.load(path)
        return dataset

    @property
    def avg_num_neighbors(self):
        return self._data.edge_index.size(1) / self._data.pos.size(0)


class ASEDataset(InMemoryDataset):
    def __init__(
        self,
        data_source: str | List[str],
        index: str | List[str] = ":",
        neighborlist_cutoff: float = 5.0,
        neighborlist_backend: str = "ase",
        progress_bar: bool = True,
        atomref_energies: Dict[str, float] | None = None,
        build: bool = True,
        cache_path: str | None = None,
    ):
        super().__init__()
        data_source = _maybe_listify(data_source)
        index = _maybe_listify(index)
        self.__args = (data_source, index, neighborlist_cutoff, neighborlist_backend, progress_bar, atomref_energies)
        self.data_source = data_source
        self.index = index
        self.neighborlist_cutoff = neighborlist_cutoff
        self.neighborlist_backend = neighborlist_backend
        self.progress_bar = progress_bar
        self.atomref_energies = atomref_energies

        # sanity check
        if len(self.index) == 1:
            self.index = self.index * len(self.data_source)
        if len(self.data_source) != len(self.index):
            raise ValueError("Length of data_source and index must be same.")

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                args, data, slices = torch.load(cache_path)
                # Check args
                if args != self.__args:
                    raise ValueError("args mismatch.")
                self.data, self.slices = data, slices
        else:
            # build data pipeline
            if build:
                dp = IterableWrapper(self.data_source).zip(IterableWrapper(self.index))
                dp = dp.read_ase()
                dp = dp.atoms_to_graph()
                if atomref_energies is not None:
                    dp = dp.subtract_atomref(atomref_energies)
                dp = dp.build_neighbor_list(cutoff=self.neighborlist_cutoff, backend=self.neighborlist_backend)

                dp_iter = tqdm(dp) if self.progress_bar else dp
                data_list = [d for d in dp_iter]
                self.data, self.slices = self.collate(data_list)
            else:
                self.data, self.slices = None, None

            if cache_path is not None:
                torch.save((self.__args, self.data, self.slices), cache_path)

    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union[Self, AtomsGraph]:
        item = super().__getitem__(idx)
        if isinstance(item, AtomsGraph):
            item.batch = torch.zeros_like(item.elems, dtype=torch.long, device=item.pos.device)
        return item

    def save(self, path):
        torch.save((self.__args, self.data, self.slices), path)

    @classmethod
    def load(cls, path):
        args, data, slices = torch.load(path)
        dataset = cls(*args, build=False)
        dataset.data, dataset.slices = data, slices
        return dataset

    def to_ase(self):
        return [atoms.to_ase() for atoms in self]

    def split(self, n: int, seed: int = 0, return_idx=False) -> Tuple[Self, Self]:
        """Split the dataset into two subsets of `n` and `len(self) - n` elements."""
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return (self[indices[:n]], self[indices[n:]]), (indices[:n], indices[n:])
        return self[indices[:n]], self[indices[n:]]

    def subset(self, n: int, seed: int = 0, return_idx=False) -> Self:
        """Create a subset of the dataset with `n` elements."""
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return self[indices[:n]], indices[:n]
        return self[indices[:n]]

    def train_val_test_split(self, train_size, val_size, seed: int = 0, return_idx=False) -> Tuple[Self, Self, Self]:
        num_data = len(self)

        if isinstance(train_size, float):
            train_size = int(train_size * num_data)

        if isinstance(val_size, float):
            val_size = int(val_size * num_data)

        if train_size + val_size > num_data:
            raise ValueError("train_size and val_size are too large.")

        if return_idx:
            (train_dataset, rest_dataset), (train_idx, _) = self.split(train_size, seed, return_idx)
            (val_dataset, test_dataset), (val_idx, test_idx) = rest_dataset.split(val_size, seed, return_idx)
            return (train_dataset, val_dataset, test_dataset), (train_idx, val_idx, test_idx)

        train_dataset, rest_dataset = self.split(train_size, seed)
        val_dataset, test_dataset = rest_dataset.split(val_size, seed)
        return train_dataset, val_dataset, test_dataset

    @property
    def avg_num_neighbors(self):
        return self._data.edge_index.size(1) / self._data.pos.size(0)

    def get_config(self):
        return {
            "data_source": self.data_source,
            "index": self.index,
            "neighborlist_cutoff": self.neighborlist_cutoff,
            "neighborlist_backend": self.neighborlist_backend,
            "progress_bar": self.progress_bar,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _maybe_listify(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]
