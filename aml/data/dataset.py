import bisect
import pickle
import warnings
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union

import ase.data
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


class GraphDatasetMixin(ABC):
    """Mixin class for graph dataset."""

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

    @property
    @abstractmethod
    def avg_num_neighbors(self):
        """Compute average number of neighbors per atom."""

    @abstractmethod
    def __getitem__(self, idx: Union[int, np.integer, IndexType]) -> Union[Self, BaseData]:
        """Get item from dataset."""

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
