import torch

from aml.data.neighbor_list import NeighborListBuilder, resolve_neighborlist_builder
from aml.typing import DataDict

from .space import FreeSpace, PeriodicSpace, Space


class NeighborlistUpdater:
    def __init__(
        self,
        r_cut: float,
        self_interaction: bool = False,
        ref: DataDict | None = None,
        tol: float = 0.2,
        backend: str = "ase",
    ):
        self.r_cut = r_cut
        self.ref = ref
        self.tol = tol
        if ref.cell.abs().max() < 1e-6:
            self.space: Space = FreeSpace()
        else:
            self.space: Space = PeriodicSpace(ref.cell[0])

        neighborlist_builder_cls = resolve_neighborlist_builder(backend)
        self.neighborlist_builder: NeighborListBuilder = neighborlist_builder_cls(r_cut, self_interaction)
        self._apply_new_neighbors(self.ref)

    def _check_system_change(self, new: DataDict):
        new_pos = new.pos
        ref_pos = self.ref.pos
        if new_pos.shape != ref_pos.shape:
            return True
        new_cell = new.cell[0]
        ref_cell = self.ref.cell[0]
        if not torch.allclose(new_cell, ref_cell):
            return True
        return False

    def _check_pos_change(self, new: DataDict):
        new_pos = new.pos
        ref_pos = self.ref.pos
        dists = self.space.pairwise_distances(new_pos, ref_pos)
        if torch.any(dists > self.tol):
            return True
        return False

    def _apply_ref_neighbors(self, new: DataDict):
        """Take neighborlist from reference atoms and apply to new atoms.

        Args:
            new (DataDict): New atoms to apply neighborlist.
        """
        new.edge_index = self.ref.edge_index
        new.edge_shift = self.ref.edge_shift

    def _apply_new_neighbors(self, new: DataDict):
        """Build new neighborlist and apply to new atoms.

        Args:
            new (DataDict): New atoms to apply neighborlist.
        """
        device = new.pos.device
        center_idx, neigh_idx, edge_shift = self.neighborlist_builder.build(new.cpu())
        edge_index = torch.stack([neigh_idx, center_idx], dim=0)
        new.edge_index = edge_index
        new.edge_shift = edge_shift
        self.ref = new.to(device)

    def update(self, new: DataDict, verbose: bool = False):
        """Update neighborlist of given atoms.

        1) If reference atoms is None, build new neighborlist.
        2) If reference atoms exists and system is changed, build new neighborlist.
        3) If reference atoms exists and system is not changed, check distance change.
            3-1) If distance is changed more than `self.tol`, build new neighborlist.
            3-2) If distance is not changed, take neighborlist from reference atoms.

        Args:
            new (DataDict): New atoms to apply neighborlist.
        """
        # If new
        if self.ref is None:
            if verbose:
                print("Initialized neighborlist.")
            self._apply_new_neighbors(new)
            return
        # Check system change
        if self._check_system_change(new):
            if verbose:
                print("System changed. Rebuild neighborlist.")
            self._apply_new_neighbors(new)
            return
        # Check distance change
        if self._check_pos_change(new):
            if verbose:
                print("Distance changed. Rebuild neighborlist.")
            self._apply_new_neighbors(new)
            return
        self._apply_ref_neighbors(new)
