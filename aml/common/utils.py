import functools
import inspect
import json
import warnings
from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from numbers import Number
from os import PathLike
from pathlib import Path

import ase.data
import numpy as np
import torch

from aml.data import keys as K
from aml.typing import DataDict, Tensor


def _get_init_args(cls):
    params = inspect.signature(cls.__init__).parameters
    args = list(params.keys())[1:]
    defaults = [params[arg].default for arg in args]
    return args, defaults


class Configurable:
    supported_val_types = (str, bool, Number, type(None))

    def get_config(self, param_name_map=None):
        param_name_map = param_name_map or {}
        args, _ = _get_init_args(self.__class__)

        def recursive_as_dict(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, str):
                return obj
            if isinstance(obj, (Sequence, Set)):
                return type(obj)([recursive_as_dict(it) for it in obj])
            if isinstance(obj, Mapping):
                return type(obj)({kk: recursive_as_dict(vv) for kk, vv in obj.items()})
            if hasattr(obj, "as_dict"):
                return obj.as_dict()
            if isinstance(obj, Configurable):
                return obj.get_config()
            return obj

        config = {}
        for arg in args:
            argname = param_name_map.get(arg, arg)
            try:
                val = getattr(self, argname)
            except AttributeError as e:
                raise ValueError(f"Argument {arg} should be present as attribute.") from e
            config[arg] = recursive_as_dict(val)

        return config

    @classmethod
    def from_config(cls, config, actual_cls=None):
        if actual_cls is None:
            actual_cls = cls
        config = deepcopy(config)
        args, defaults = _get_init_args(actual_cls)
        # Construct default config
        sanitized_config = {k: v for k, v in zip(args, defaults, strict=True) if v is not inspect.Parameter.empty}
        # Update the default config with the config file
        for k, v in config.items():
            if k in args:
                sanitized_config[k] = v
            else:
                warnings.warn(f"Argument {k} is not used in {actual_cls.__name__}.__init__(). Ignored.", stacklevel=1)

        init_signature = inspect.signature(actual_cls.__init__)
        types = {k: p.annotation for k, p in init_signature.parameters.items() if p.annotation != inspect._empty}
        # Convert the config to the correct type
        for k, v in types.items():
            if hasattr(v, "from_config"):
                c = config[k]
                sanitized_config[k] = v.from_config(c)
        return actual_cls(**sanitized_config)


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
        try:
            import yaml
        except ImportError as e:
            raise ImportError("Please install pyyaml to load yaml config files.") from e
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
    elif filepath.suffix == ".toml":
        try:
            import tomli
        except ImportError as e:
            raise ImportError("Please install tomli to load toml config files.") from e
        with open(filepath, "rb") as f:
            config = tomli.load(f)
    else:
        raise ValueError(f"Invalid config file extension: {filepath.suffix}")
    return config
