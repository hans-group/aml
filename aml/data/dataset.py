import bisect
import pickle
import warnings
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, InMemoryDataset
from torchdata.datapipes.iter import IterableWrapper
from tqdm import tqdm
from typing_extensions import Self

from . import datapipes  # noqa: F401
from .data_structure import AtomsGraph
from .utils import is_pkl, maybe_list

T_co = TypeVar("T_co", covariant=True)
IndexType = TypeVar("IndexType", int, slice, list, tuple, np.ndarray, np.integer, torch.Tensor)


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
        data_source = maybe_list(data_source)
        index = maybe_list(index)
        self.__args = (data_source, index, neighborlist_cutoff, neighborlist_backend, progress_bar, atomref_energies)
        self.data_source = data_source
        self.index = index
        self.neighborlist_cutoff = neighborlist_cutoff
        self.neighborlist_backend = neighborlist_backend
        self.progress_bar = progress_bar
        if atomref_energies is not None:
            warnings.warn(
                "atomref_energies is deprecated and the values will be ignored",
                DeprecationWarning,
                stacklevel=1,
            )
        self.atomref_energies = None

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
        torch.save((self.__args, self._data, self.slices), path)

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


class LMDBDataset(Dataset[T_co]):
    """Dataset using LMDB memory-mapped db.
    This dataset is designed for large dataset that cannot be loaded into memory.
    Expects db to store {idx: data} pairs, where idx is integer and data is a pickled object.
    The data could be compressed by zlib or not.

    Args:
        db_path (str | list[str]): Path to LMDB database
        collate_data (bool, optional): Whether to collate data into a Batch. Defaults to False.
    """

    def __init__(self, db_path: str | list[str], collate_data: bool = False):
        super().__init__()
        self.db_path = maybe_list(db_path)
        self.collate_data = collate_data

        self.envs = [self.connect_db(path) for path in self.db_path]
        self.db_lengths = [self.get_db_length(env) for env in self.envs]
        self.cumsum_db_lengths = np.cumsum(self.db_lengths).tolist()
        self.db_indices = [list(range(length)) for length in self.db_lengths]

    @staticmethod
    def connect_db(lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        env = lmdb.open(
            str(lmdb_path), subdir=False, readonly=True, lock=False, readahead=True, meminit=False, max_readers=1
        )
        return env

    def close_db(self) -> None:
        if not self.db_path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

    @staticmethod
    def get_db_length(env: lmdb.Environment) -> int:
        txn = env.begin()
        if (length_pkl := txn.get("length".encode("ascii"))) is not None:
            length = pickle.loads(length_pkl)
            assert isinstance(length, int)
        else:
            length = env.stat()["entries"]
        if txn.get("metadata".encode("ascii")) is not None:
            length -= 1
        return length

    def __len__(self):
        return sum(self.db_lengths)

    def __getitem__(self, idx: IndexType) -> T_co:
        def _get_single(idx):
            # Determine which db to use
            env_idx = bisect.bisect(self.cumsum_db_lengths, idx)
            elem_idx = idx - self.cumsum_db_lengths[env_idx - 1] if env_idx > 0 else idx
            txn = self.envs[env_idx].begin()
            # Get data (compressed by zlib or not)
            pkl = txn.get(str(elem_idx).encode("ascii"))
            if not is_pkl(pkl):
                pkl = zlib.decompress(pkl)
            data = pickle.loads(pkl)
            return data

        if isinstance(idx, (list, tuple)):
            datalist = [_get_single(i) for i in idx]
            if self.collate_data:
                return Batch.from_data_list(datalist)
            return datalist
        elif isinstance(idx, slice):
            start = idx.start or 0
            stop = idx.stop or len(self)
            step = idx.step or 1
            indices = list(range(start, stop, step))
            return self[indices]
        elif isinstance(idx, (np.ndarray, torch.Tensor)):
            if idx.shape == ():
                return _get_single(idx.item())
            else:
                return self[idx.tolist()]
        elif isinstance(idx, (int, np.integer)):
            return _get_single(int(idx))
        else:
            raise TypeError(f"Invalid argument type {type(idx)}")

    def get_batch(self, idx: IndexType) -> Batch:
        """Get all values"""
        datalist = self[idx]
        if isinstance(datalist, list):
            return Batch.from_data_list(datalist)
        return datalist

    def __repr__(self):
        return f"LMDBDataset({len(self)})"
