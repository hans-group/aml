import functools
import inspect
import json
import warnings
from os import PathLike
from pathlib import Path

import ase.data
import numpy as np
import tomli
import torch
import yaml

from aml.data import keys as K
from aml.typing import DataDict, Tensor


def warn_unstable(cls_or_fn):
    """Warns that a class or function is unstable.

    Args:
        cls_or_fn: Class or function to warn about.
    """
    if inspect.isclass(cls_or_fn):
        name = cls_or_fn.__name__
        orig_init = cls_or_fn.__init__

        def __init__(self, *args, **kws):
            warnings.warn(f"{name} is unstable and may change in future versions.", UserWarning, stacklevel=1)
            orig_init(self, *args, **kws)  # Call the original __init__

        cls_or_fn.__init__ = __init__
        return cls_or_fn

    else:
        name = cls_or_fn.__qualname__

        @functools.wraps(cls_or_fn)
        def wrapper(*args, **kwargs):
            warnings.warn(f"{name} is unstable and may change in future versions.", UserWarning, stacklevel=1)

            return cls_or_fn(*args, **kwargs)

    return wrapper


def log_and_print(contents: str, filepath: PathLike = None, end="\n"):
    """Log and print the contents.

    Args:
        contents (str): The contents to log and print.
        filepath (PathLike): The path to the log file.

    Returns:
        None
    """
    if filepath is not None:
        with open(filepath, "a") as f:
            f.write(contents + end)
    print(contents, end=end)


def compute_average_E0s(dataset, stride: int | None = None) -> dict[str, float]:
    if stride is not None:
        dataset = dataset[::stride]
    # determine list of unique atomic numbers
    species = np.unique(dataset._data.elems)
    n_structures = len(dataset)
    n_elems = len(species)
    # compute average energy per atom for each element by lstsq
    A = np.zeros((n_structures, n_elems))
    B = np.zeros((n_structures,))

    for i, data in enumerate(dataset):
        B[i] = data.energy.squeeze().item()
        for j, elem in enumerate(species):
            A[i, j] = np.count_nonzero(data.elems == elem)
    try:
        E0s = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energies = {}
        for i, elem in enumerate(species):
            symbol = ase.data.chemical_symbols[elem]
            atomic_energies[symbol] = E0s[i]
    except np.linalg.LinAlgError:
        atomic_energies = {}
        for elem in species:
            symbol = ase.data.chemical_symbols[elem]
            atomic_energies[symbol] = 0.0
    return atomic_energies


def compute_force_rms_per_species(dataset, stride=None) -> dict[int, float]:
    if stride is not None:
        dataset = dataset[::stride]
    # determine list of unique atomic numbers
    species = torch.unique(dataset._data.elems)
    force_rms = {}
    for elem in species:
        force_elem = dataset._data.force[dataset._data.elems == elem]
        symbol = ase.data.chemical_symbols[elem.item()]
        force_rms[symbol] = torch.sqrt(torch.mean(torch.sum(force_elem**2, dim=-1)))
    return force_rms


def remove_unused_kwargs(func, kwargs):
    """Remove unused kwargs from a function call."""
    valid_args = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in valid_args}


def compute_neighbor_vecs(data: DataDict) -> DataDict:
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


def canocialize_species(species: Tensor | list[int] | list[str]) -> Tensor:
    if isinstance(species[0], str):
        species = [ase.data.atomic_numbers[s] for s in species]
        species = torch.as_tensor(species, dtype=torch.long)
    elif isinstance(species, list) and isinstance(species[0], int):
        species = torch.as_tensor(species, dtype=torch.long)
    elif isinstance(species, np.ndarray):
        species = torch.as_tensor(species, dtype=torch.long)
    return species


def get_batch_size(batch: DataDict) -> int:
    return batch[K.batch][-1] + 1


def load_config(filepath: PathLike) -> dict:
    """Load a config file.

    Args:
        filepath (PathLike): The path to the config file.

    Returns:
        dict: The loaded config.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".json":
        with open(filepath, "r") as f:
            config = json.load(f)
    elif filepath.suffix == ".yaml":
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    elif filepath.suffix == ".toml":
        with open(filepath, "rb") as f:
            config = tomli.load(f)
    else:
        raise ValueError(f"Invalid config file extension: {filepath.suffix}")
    return config
