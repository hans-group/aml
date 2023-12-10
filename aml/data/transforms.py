from typing import Literal

from torch_geometric.transforms import BaseTransform

from aml.data.data_structure import AtomsGraph


class BuildNeighborList(BaseTransform):
    def __init__(
        self,
        cutoff: float,
        self_interaction: bool = False,
        backend: Literal["ase", "matscipy", "torch", "auto"] = "ase",
        override: bool = True,
        **kwargs,
    ):
        """
        Args:
            cutoff (float): Cutoff radius.
            self_interaction (bool, optional): Whether to include self interaction. Defaults to False.
            backend (str, optional): Backend to use. Defaults to "ase".
            override (bool, optional): Whether to override the existing edge_index. Defaults to True.
                If False, the transform will be skipped if edge_index already exists.
        """
        self.cutoff = cutoff
        self.self_interaction = self_interaction
        self.backend = backend
        self.kwargs = kwargs
        self.override = override

    def __call__(self, data: AtomsGraph) -> AtomsGraph:
        if not self.override and "edge_index" in data:
            return data
        data = data.clone()  # Avoid in-place modification
        data.build_neighborlist(
            self.cutoff, self_interaction=self.self_interaction, backend=self.backend, **self.kwargs
        )
        return data
