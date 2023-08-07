import warnings
from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import ase.neighborlist
import matscipy.neighbours
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data


def distance_to_face(x: np.ndarray, face_vec_1: np.ndarray, face_vec_2: np.ndarray) -> float:
    face_normal_vec = np.cross(face_vec_1, face_vec_2)
    face_normal_vec /= np.linalg.norm(face_normal_vec)
    return np.abs(np.dot(face_normal_vec, x))


def minimum_distance_to_cell(x: Tensor, cell: Tensor) -> float:
    """Compute the minimum distance between a point and a cell."""
    vec_a = cell[0]
    vec_b = cell[1]
    vec_c = cell[2]
    face_dist_ab = distance_to_face(x, vec_a, vec_b)
    face_dist_bc = distance_to_face(x, vec_b, vec_c)
    face_dist_ca = distance_to_face(x, vec_c, vec_a)
    face_dist_min = min(face_dist_ab, face_dist_bc, face_dist_ca)
    return face_dist_min


class NeighborList(NamedTuple):
    center_idx: Tensor
    neighbor_idx: Tensor
    offset: Tensor


class NeighborListBuilder(ABC):
    def __init__(self, cutoff: float, self_interaction: bool = False):
        self.cutoff = cutoff
        self.self_interaction = self_interaction

    @abstractmethod
    def build(self, atoms_graph: Data) -> NeighborList:
        """Build neighbor list for given atoms."""


class ASENeighborListBuilder(NeighborListBuilder):
    def build(self, atoms_graph: Data) -> NeighborList:
        if atoms_graph.volume() == 0:
            pbc = np.array([False, False, False])
        else:
            pbc = np.array([True, True, True])
        pos = atoms_graph.pos.numpy().astype(np.float64)
        cell = atoms_graph.cell.squeeze().numpy().astype(np.float64)
        elems = atoms_graph.elems.numpy().astype(np.int32)

        center_idx, neighbor_idx, offset = ase.neighborlist.primitive_neighbor_list(
            "ijS", pbc, cell, pos, self.cutoff, elems, self_interaction=self.self_interaction
        )
        return NeighborList(
            center_idx=torch.LongTensor(center_idx),
            neighbor_idx=torch.LongTensor(neighbor_idx),
            offset=torch.as_tensor(offset, dtype=torch.float),
        )


class MatscipyNeighborListBuilder(NeighborListBuilder):
    def build(self, atoms_graph: Data) -> NeighborList:
        if atoms_graph.volume() == 0:
            pbc = np.array([False, False, False])
        else:
            pbc = np.array([True, True, True])

        # matscipy.neighbours.neighbour_list fails for non-periodic systems
        pos = atoms_graph.pos.numpy().astype(np.float64)
        cell = atoms_graph.cell.squeeze().numpy().astype(np.float64)
        elems = atoms_graph.elems.numpy().astype(np.int32)
        if not pbc.all():
            # put atoms in a box with periodic boundary conditions
            rmin = np.min(pos, axis=0)
            rmax = np.max(pos, axis=0)
            celldim = np.max(rmax - rmin) + 2.5 * self.cutoff
            cell = np.eye(3) * celldim
        else:
            cell_center = np.sum(cell, axis=0) / 2
            min_cell_dist = minimum_distance_to_cell(cell_center, cell)
            if min_cell_dist < self.cutoff:
                warnings.warn(
                    "Cutoff is larger than the minimum distance to the cell. "
                    "It may break MIC and return wrong neighbor lists.",
                    stacklevel=1,
                )
        center_idx, neighbor_idx, offset = matscipy.neighbours.neighbour_list(
            "ijS", cutoff=self.cutoff, positions=pos, pbc=pbc, cell=cell, numbers=elems
        )
        # add self interaction as ase.neighborlist does
        if self.self_interaction:
            center_idx = np.concatenate([center_idx, np.arange(len(pos))])
            neighbor_idx = np.concatenate([neighbor_idx, np.arange(len(pos))])
            offset = np.concatenate([offset, np.zeros((len(pos), 3))])
            # sort by center_idx
            idx = np.argsort(center_idx)
            center_idx = center_idx[idx]
            neighbor_idx = neighbor_idx[idx]
            offset = offset[idx]

        return NeighborList(
            center_idx=torch.LongTensor(center_idx),
            neighbor_idx=torch.LongTensor(neighbor_idx),
            offset=torch.as_tensor(offset, dtype=torch.float),
        )


_neighborlistbuilder_cls_map = {"ase": ASENeighborListBuilder, "matscipy": MatscipyNeighborListBuilder}


def resolve_neighborlist_builder(neighborlist_backend: Union[str, object]) -> object:
    if isinstance(neighborlist_backend, str):
        return _neighborlistbuilder_cls_map[neighborlist_backend]
    return neighborlist_backend
