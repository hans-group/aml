import pickle
import zlib
from typing import Iterable, TypeVar

import torch
import lmdb
from ase import Atoms
from ase.constraints import FixAtoms
from torch_geometric.data import Data

from aml.typing import DataDict
from . import keys as K

T = TypeVar("T")


def maybe_list(x: T | list[T]) -> list[T]:
    """Listify x if it is not a list.
    TODO: Rewrite to use `typing.Sequence` instead of `list`.

    Args:
        x (T | list[T]): The input.

    Returns:
        list[T]: The listified input.
    """
    if isinstance(x, list):
        return x
    else:
        return [x]


def is_pkl(b: bytes) -> bool:
    """Check if the bytes is a pickle file by checking the first two bytes.

    Args:
        b (bytes): The bytes.

    Returns:
        bool: Whether the bytes is a pickle file.
    """
    first_two_bytes = b[:2]
    if first_two_bytes in (b"cc", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return True
    return False


def write_lmdb_dataset(
    data_iter: Iterable[Data],
    path: str,
    map_size: int = 1099511627776 * 2,
    compress: bool = False,
    metadata: dict | None = None,
):
    """Write dataset to LMDB database.

    Args:
        data_iter (Iterable): Iterable of data.
        path (str): Path to the database.
        map_size (int, optional): Size of memory mapping. Defaults to 1099511627776*2.
        compress (bool, optional): Whether to compress data by zlib. Defaults to False.
        metadata (dict, optional): Metadata to store in the database. Defaults to None.
    """
    db = lmdb.open(path, map_size=map_size, subdir=False, meminit=False, map_async=True)

    with db.begin(write=True) as txn:
        for i, data in enumerate(data_iter):
            pkl = pickle.dumps(data.clone().contiguous())
            if compress:
                pkl = zlib.compress(pkl)

            txn.put(str(i).encode("ascii"), pkl)
    if metadata is not None:
        with db.begin(write=True) as txn:
            txn.put("metadata".encode("ascii"), pickle.dumps(metadata))

    db.sync()
    db.close()


def find_fixatoms_constraint(atoms: Atoms) -> FixAtoms | None:
    """If atoms as FixAtoms contraint, return it.
    Otherwise returns None.

    Args:
        atoms(Atoms): A Atoms object.

    Returns:
        FixAtoms | None
    """
    if not atoms.constraints:
        return None
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            return c
    return None


def compute_neighbor_vecs(data: DataDict) -> DataDict:
    """Compute the vectors between atoms and their neighbors (i->j)
    and store them in ``data[K.edge_vec]``.
    The ``data`` must contain ``data[K.pos]``, ``data[K.edge_index]``, ``data[K.edge_shift]``,
    This function should be called inside ``forward`` since the dependency of neighbor positions
    on atomic positions needs to be tracked by autograd in order to appropriately compute forces.

    Args:
        data (DataDict): The data dictionary.

    Returns:
        DataDict: The data dictionary with ``data[K.edge_vec]``.
    """
    batch = data[K.batch]
    pos = data[K.pos]
    edge_index = data[K.edge_index]  # neighbors
    edge_shift = data[K.edge_shift]  # shift vectors
    batch_size = int((batch.max() + 1).item())
    cell = data[K.cell] if "cell" in data else torch.zeros((batch_size, 3, 3)).to(pos.device)
    idx_i = edge_index[1]
    idx_j = edge_index[0]

    edge_batch = batch[idx_i]  # batch index for edges(neighbors)
    edge_vec = pos[idx_j] - pos[idx_i] + torch.einsum("ni,nij->nj", edge_shift, cell[edge_batch])
    data[K.edge_vec] = edge_vec
    return data

def get_batch_size(batch: DataDict) -> int:
    """Get the batch size of a data batch.

    Args:
        batch (DataDict): The data batch.

    Returns:
        int: The batch size.
    """
    if K.batch in batch:
        return batch[K.batch][-1] + 1
    return 1

