import pickle
from typing import TypeVar
import zlib

import lmdb

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


def maybe_list(x: T | list[T]) -> list[T]:
    if isinstance(x, list):
        return x
    else:
        return [x]


def is_pkl(b: bytes) -> bool:
    first_two_bytes = b[:2]
    if first_two_bytes in (b"cc", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        return True
    return False


def write_lmdb_dataset(
    data_iter,
    path: str,
    map_size: int = 1099511627776 * 2,
    compress: bool = False,
    metadata: dict | None = None,
):
    """Write dataset to LMDB database.

    Args:
        data_iter (Iterable): Iterable of data.
        map_size (int, optional): Size of memory mapping. Defaults to 1099511627776*2.
        compress (bool, optional): Whether to compress data by zlib. Defaults to False.
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
