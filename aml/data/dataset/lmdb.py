import bisect
import pickle
import zlib
from pathlib import Path

import lmdb
import numpy as np
from torch_geometric.data.data import BaseData
from torch_geometric.transforms import BaseTransform

from aml.common.registry import registry
from aml.data.utils import is_pkl, maybe_list

from .base import BaseDataset


@registry.register_dataset("lmdb_dataset")
class LMDBDataset(BaseDataset):
    """Dataset using LMDB memory-mapped db.
    This dataset is designed for large dataset that cannot be loaded into memory.
    Expects db to store {idx: data} pairs, where idx is integer and data is a pickled object.
    The data could be compressed by zlib or not.

    Args:
        db_path (str | list[str]): Path to LMDB database
        collate_data (bool, optional): Whether to collate data into a Batch. Defaults to False.
    """

    def __init__(self, db_path: str | list[str], transform: BaseTransform | None = None):
        super().__init__()
        self.db_path = maybe_list(db_path)

        self.envs = [self.connect_db(path) for path in self.db_path]
        self.db_lengths = [self.get_db_length(env) for env in self.envs]
        self.cumsum_db_lengths = np.cumsum(self.db_lengths).tolist()
        self.db_indices = [list(range(length)) for length in self.db_lengths]
        self.transform = transform

    @staticmethod
    def connect_db(lmdb_path: Path | None = None) -> lmdb.Environment:
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

    @property
    def metadata(self):
        txn = self.envs[0].begin()
        metadata = txn.get("metadata".encode("ascii"))
        if metadata is not None:
            return pickle.loads(metadata)
