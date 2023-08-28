import bisect
import pickle
import warnings
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import lmdb
import numpy as np
import torch
from ase import Atoms
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset, IndexType
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from tqdm import tqdm
from typing_extensions import Self

from . import datapipes  # noqa: F401
from .data_structure import AtomsGraph
from .utils import is_pkl, maybe_list

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
Images = List[Atoms]


def read_ase_datapipe(data_source: List[str] | List[Images], index: List[str]) -> IterDataPipe:
    """ASE file reader datapipe.

    Args:
        data_source (str | list[str]): Path to ASE database.
        index (str | list[str]): Index of ASE database.
    """
    if len(index) == 1:
        index = index * len(data_source)
    if len(data_source) != len(index):
        raise ValueError("Length of data_source and index must be same.")
    if isinstance(data_source[0], str):
        dp = IterableWrapper(data_source).zip(IterableWrapper(index))
        dp = dp.read_ase()
    elif isinstance(data_source[0], list) and isinstance(data_source[0][0], Atoms):
        images = []
        for images_, index_ in zip(data_source, index, strict=True):
            if index_ == ":":
                images.extend(images_)
            else:
                images.extend(images_[index_])
        dp = IterableWrapper(images)
    else:
        raise TypeError("data_source must be str or list of ASE atoms.")
    return dp


def a2g_datapipe(
    atoms_dp: IterDataPipe, neighborlist_cutoff: float, neighborlist_backend: str, read_properties: bool
) -> IterDataPipe:
    """Atoms to graph datapipe.

    Args:
        atoms_dp (IterDataPipe): Data pipeline that yields ASE atoms.
        neighborlist_cutoff (float): Cutoff radius for neighborlist.
        neighborlist_backend (str): Backend for neighborlist computation.
    """
    dp = atoms_dp.atoms_to_graph(read_properties=read_properties)
    dp = dp.build_neighbor_list(cutoff=neighborlist_cutoff, backend=neighborlist_backend)
    return dp


class GraphDatasetMixin(ABC):
    """Mixin class for graph dataset."""

    @property
    @abstractmethod
    def avg_num_neighbors(self):
        """Compute average number of neighbors per atom."""

    def subset(self, n: int, seed: int = 0, return_idx=False) -> Self:
        """Create a subset of the dataset with `n` elements."""
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return self[indices[:n]], indices[:n]
        return self[indices[:n]]

    def split(self, n: int, seed: int = 0, return_idx=False) -> Tuple[Self, Self]:
        """Split the dataset into two subsets of `n` and `len(self) - n` elements."""
        indices = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        if return_idx:
            return (self[indices[:n]], self[indices[n:]]), (indices[:n], indices[n:])
        return self[indices[:n]], self[indices[n:]]

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
        db = lmdb.open(path, map_size=map_size, subdir=False, meminit=False, map_async=True)
        dataiter = tqdm(self, total=len(self)) if prog_bar else self
        for i, data in enumerate(dataiter):
            pkl = pickle.dumps(data.clone().contiguous())
            if compress:
                pkl = zlib.compress(pkl)
            txn = db.begin(write=True)
            txn.put(str(i).encode("ascii"), pkl)
            txn.commit()
        db.sync()
        db.close()


class SimpleDataset(InMemoryDataset, GraphDatasetMixin):
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


class ASEDataset(InMemoryDataset, GraphDatasetMixin):
    def __init__(
        self,
        data_source: str | List[str] | Images | List[Images],
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
        # Additional listify for raw images input
        if isinstance(data_source, list) and isinstance(data_source[0], Atoms):
            data_source = [data_source]
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
                dp = read_ase_datapipe(self.data_source, self.index)
                dp = a2g_datapipe(dp, self.neighborlist_cutoff, self.neighborlist_backend, read_properties=True)
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


class LMDBDataset(Dataset, GraphDatasetMixin):
    """Dataset using LMDB memory-mapped db.
    This dataset is designed for large dataset that cannot be loaded into memory.
    Expects db to store {idx: data} pairs, where idx is integer and data is a pickled object.
    The data could be compressed by zlib or not.

    Args:
        db_path (str | list[str]): Path to LMDB database
        collate_data (bool, optional): Whether to collate data into a Batch. Defaults to False.
    """

    def __init__(self, db_path: str | list[str]):
        super().__init__()
        self.db_path = maybe_list(db_path)

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
        for env in self.envs:
            env.close()

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

    def len(self) -> int:
        return sum(self.db_lengths)

    def get(self, idx: int) -> BaseData:
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

    @property
    def avg_num_neighbors(self):
        avg_num_neighbors = 0
        for i in range(len(self)):
            data = self.get(i)
            avg_num_neighbors += data.edge_index.size(1) / data.pos.size(0) / len(self)
        return avg_num_neighbors
