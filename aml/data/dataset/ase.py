import warnings

import torch
from ase import Atoms
from torch_geometric.data.dataset import IndexType
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
    def __init__(
        self,
        data_source: str | list[str] | Images | list[Images] = None,
        index: str | list[str] = ":",
        neighborlist_cutoff: float = 5.0,
        neighborlist_backend: str = "ase",
        progress_bar: bool = True,
        atomref_energies: dict[str, float] | None = None,
    ):
        data_source = maybe_list(data_source)
        # Additional listify for raw images input
        if isinstance(data_source, list) and isinstance(data_source[0], Atoms):
            data_source = [data_source]
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

    def __getitem__(self, idx: IndexType) -> Self | AtomsGraph:
        item = InMemoryDataset.__getitem__(self, idx)
        if isinstance(item, AtomsGraph):
            item.batch = torch.zeros_like(item.elems, dtype=torch.long, device=item.pos.device)
        return item

    def to_ase(self):
        return [atoms.to_ase() for atoms in self]


def read_ase_datapipe(data_source: list[str] | list[Images], index: list[str]) -> IterDataPipe:
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
        dp = IterableWrapper(images)  # noqa: F821
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
