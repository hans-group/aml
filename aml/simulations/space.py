from abc import ABC, abstractmethod

import torch
from torch.func import vmap

from aml.typing import Tensor


def _get_free_indices(n: int) -> str:
    return "".join([chr(ord("a") + i) for i in range(n)])


@torch.jit.script
def cell_transform(cell: Tensor, R: Tensor) -> Tensor:
    free_indices = _get_free_indices(R.ndim - 1)
    left_indices = free_indices + "j"
    right_indices = free_indices + "i"
    return torch.einsum(f"ij,{left_indices}->{right_indices}", cell, R)


class Space(ABC):
    """Abstract class for space handler."""

    @abstractmethod
    def displacement(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        """Compute the displacement vector between two points.

        Args:
            r_1 (Tensor): First point.
            r_2 (Tensor): Second point.

        Returns:
            Tensor: Displacement vector.
        """
        pass

    def pairwise_displacements(self, R_1: Tensor, R_2: Tensor) -> Tensor:
        """Compute the pairwise displacements between two sets of points.

        Args:
            R_1 (Tensor): First set of points.
            R_2 (Tensor): Second set of points.

        Returns:
            Tensor: Pairwise displacements.
        """
        return vmap(self.displacement)(R_1, R_2)

    @abstractmethod
    def distance(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        """Compute the distance between two points.

        Args:
            r_1 (Tensor): First point.
            r_2 (Tensor): Second point.

        Returns:
            Tensor: Distance.
        """
        pass

    def pairwise_distances(self, R_1: Tensor, R_2: Tensor) -> Tensor:
        """Compute the pairwise distances between two sets of points.

        Args:
            R_1 (Tensor): First set of points.
            R_2 (Tensor): Second set of points.

        Returns:
            Tensor: Pairwise distances.
        """
        return vmap(self.distance)(R_1, R_2)

    @abstractmethod
    def shift(self, R: Tensor, dR: Tensor) -> Tensor:
        """Shift a set of points by a displacement vector.

        Args:
            R (Tensor): Set of points.
            dR (Tensor): Displacement vector.

        Returns:
            Tensor: Shifted set of points.
        """
        pass

    def brute_force_neighbor_list(self, R: Tensor, r_cut: float) -> tuple:
        """Brute force search"""
        idx_i = torch.arange(R.size(0), device=R.device).repeat_interleave(R.size(0))
        idx_j = torch.arange(R.size(0), device=R.device).repeat(R.size(0))
        R_i, R_j = R[idx_i], R[idx_j]
        d_ij = self.pairwise_distances(R_i, R_j)
        mask = torch.logical_and(d_ij <= r_cut, idx_i != idx_j)
        return idx_j[mask], idx_i[mask]


class FreeSpace(Space):
    def __init__(self):
        pass

    def displacement(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        return r_1 - r_2

    def distance(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        return torch.norm(self.displacement(r_1, r_2), dim=-1)

    def shift(self, R: Tensor, dR: Tensor) -> Tensor:
        return R + dR


class PeriodicSpace(Space):
    """Periodic space handler.
    Borrowed implementation details from jax_md (https://github.com/google/jax-md)
    and simplified for only 3D general periodic lattice.

    Args:
        cell (Tensor): 3x3 matrix representing the lattice vectors.
        fractional_coordinates (bool, optional): Whether the coordinates are in fractional or
            cartesian coordinates. Defaults to False.
    """

    def __init__(self, cell: Tensor, fractional_coordinates: bool = False):
        self.cell = cell
        self.inv_cell = torch.linalg.inv(cell)
        self.fractional_coordinates = fractional_coordinates

    def displacement(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        # Wrap into unit cube
        if not self.fractional_coordinates:
            r_1 = cell_transform(self.inv_cell, r_1)
            r_2 = cell_transform(self.inv_cell, r_2)

        dR = r_1 - r_2
        dR = torch.remainder(dR + 0.5, 1.0) - (0.5)
        dR = cell_transform(self.cell, dR)

        return dR

    def distance(self, r_1: Tensor, r_2: Tensor) -> Tensor:
        return torch.norm(self.displacement(r_1, r_2), dim=-1)

    def shift(self, r: Tensor, dr: Tensor, wrap: bool = True) -> Tensor:
        if not self.fractional_coordinates and not wrap:
            return r + dr

        dr = cell_transform(self.inv_cell, dr)
        if not self.fractional_coordinates:
            r = cell_transform(self.inv_cell, r)

        if wrap:
            r = torch.remainder(r + dr, 1.0)
        else:
            r = r + dr

        if not self.fractional_coordinates:
            r = cell_transform(self.cell, r)
        return r
