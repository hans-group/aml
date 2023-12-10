import os
import warnings
from pathlib import Path
from typing import Literal

import torch
from ase import Atoms
from torch_geometric.data.dataset import IndexType
from torch_geometric.transforms import BaseTransform
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe
from tqdm import tqdm
from typing_extensions import Self

from aml.common.registry import registry
from aml.data import datapipes  # noqa: F401
from aml.data.data_structure import AtomsGraph
from aml.data.utils import maybe_list

from .base import BaseDataset, InMemoryDataset

Images = list[Atoms]


@registry.register_dataset("ase_dataset")
class ASEDataset(InMemoryDataset, BaseDataset):
    """In-memory dataset from reading files with ASE.
    Creates neighbor list for each ``Atoms`` object.

    Args:
        data_source (str | list[str] | Images | list[Images]): Path to ASE-readable file(s) or (list of) images.
        index (str | list[str]): Index to read. The length of index must be same as data_source.
        neighborlist_cutoff (float, optional): Cutoff radius for neighborlist. Defaults to 5.0.
        neighborlist_backend (str, optional): Backend for neighborlist computation. Defaults to "ase".
            Options are "ase", "matscipy", and "torch".
            Usually "torch" is the fastest, but the speed is simliary to "matscipy" when PBC is used.
            "matscipy" is also fast, but it uses minimal image convention,
                so cannot be used for small cell.
            "ase" is the slowest, but it can be used on any system.
        progress_bar (bool, optional): Whether to show progress bar. Defaults to True.
        atomref_energies (dict[str, float], optional): Atomic reference energies. Defaults to None.
            Warning: This argument is deprecated and the values will be ignored.
    """

    def __init__(
        self,
        data_source: str | list[str] | Images | list[Images] = None,
        index: str | list[str] = ":",
        neighborlist_cutoff: float | None = None,
        neighborlist_backend: Literal["ase", "matscipy", "torch"] = "ase",
        progress_bar: bool = True,
        transform: BaseTransform = None,
        atomref_energies: dict[str, float] | None = None,  # Deprecated
    ):
        data_source = maybe_list(data_source)
        # Additional listify for raw images input
        if isinstance(data_source, list) and isinstance(data_source[0], Atoms):
            data_source = [data_source]
        for i in range(len(data_source)):
            if isinstance(data_source[i], (str, Path)):
                data_source[i] = os.path.abspath(data_source[i])

        index = maybe_list(index)
        self.data_source = data_source
        self.index = index
        self.neighborlist_cutoff = neighborlist_cutoff
        self.neighborlist_backend = neighborlist_backend
        self.progress_bar = progress_bar
        self.atomref_energies = atomref_energies

        if atomref_energies is not None:
            warnings.warn(
                "atomref_energies is deprecated and the values will be ignored", DeprecationWarning, stacklevel=1
            )
        self.atomref_energies = None

        # build data pipeline
        if all(d is None for d in data_source):
            dp = None
        else:
            dp = read_ase_datapipe(self.data_source, self.index)
            dp = a2g_datapipe(dp, self.neighborlist_cutoff, self.neighborlist_backend, read_properties=True)
            dp = tqdm(dp) if self.progress_bar else dp

        InMemoryDataset.__init__(self, dp)
        self.transform = transform

    def __getitem__(self, idx: IndexType) -> Self | AtomsGraph:
        item = InMemoryDataset.__getitem__(self, idx)
        if isinstance(item, AtomsGraph):
            item.batch = torch.zeros_like(item.elems, dtype=torch.long, device=item.pos.device)
        return item

    def to_ase(self) -> list[Atoms]:
        """Convert to ASE Atoms objects.

        Returns:
            list[Atoms]: List of ASE Atoms objects.
        """
        return [atoms.to_ase() for atoms in self]

    def get_config(self):
        config = super().get_config()
        if isinstance(self.data_source[0][0], Atoms):
            config["data_source"] = "@raw"
            config["index"] = None
        return config

    @classmethod
    def from_config(cls, config):
        if config["data_source"] == "@raw":
            raise ValueError("Cannot load from config when data_source is raw images.")
        return super().from_config(config)


def read_ase_datapipe(data_source: list[str] | list[Images], index: list[str]) -> IterDataPipe:
    """ASE file reader datapipe.

    Args:
        data_source (str | list[str]): Path to ASE database.
        index (str | list[str]): Index of ASE database.

    Returns:
        IterDataPipe: Data pipeline that yields ASE atoms.
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
        dp = IterableWrapper(images)  # noqa: F821
    else:
        raise TypeError("data_source must be str or list of ASE atoms.")
    return dp


def a2g_datapipe(
    atoms_dp: IterDataPipe,
    neighborlist_cutoff: float | None,
    neighborlist_backend: str,
    read_properties: bool,
) -> IterDataPipe:
    """Atoms to graph datapipe.

    Args:
        atoms_dp (IterDataPipe): Data pipeline that yields ASE atoms.
        neighborlist_cutoff (float): Cutoff radius for neighborlist.
        neighborlist_backend (str): Backend for neighborlist computation.

    Returns:
        IterDataPipe: Data pipeline that yields AtomsGraph.
    """
    dp = atoms_dp.atoms_to_graph(read_properties=read_properties)
    if neighborlist_cutoff is not None:
        dp = dp.build_neighbor_list(cutoff=neighborlist_cutoff, backend=neighborlist_backend)
    return dp
