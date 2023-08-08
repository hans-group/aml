"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
from torch_geometric.utils import scatter
from torch_scatter import segment_coo, segment_csr
from torch_sparse import SparseTensor

from aml.common.registry import registry
from aml.data import keys as K
from aml.nn.gemnet.atom_update_block import OutputBlock
from aml.nn.gemnet.base_layers import Dense
from aml.nn.gemnet.efficient import EfficientInteractionDownProjection
from aml.nn.gemnet.embedding_block import AtomEmbedding, EdgeEmbedding
from aml.nn.gemnet.gemnet_utils import inner_product_normalized, mask_neighbors, ragged_range, repeat_blocks
from aml.nn.gemnet.interaction_block import InteractionBlockTripletsOnly
from aml.nn.gemnet.radial_basis import RadialBasis
from aml.nn.gemnet.spherical_basis import CircularBasisLayer

from .base import BaseEnergyModel


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = edge_index[1].new_ones(1).expand_as(edge_index[1])
    num_neighbors = segment_coo(ones, edge_index[1], dim_size=data[K.n_atoms].sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(data[K.n_atoms].shape[0] + 1, device=data[K.pos].device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(data[K.n_atoms], dim=0)
    neighbors = segment_csr(num_neighbors, image_indptr)
    return neighbors


@registry.register_energy_model("gemnet_t")
class GemNetT(BaseEnergyModel):
    """
    GemNet-T, triplets-only variant of GemNet

    Parameters
    ----------
        num_spherical: int
            Controls maximum frequency.
        num_radial: int
            Controls maximum frequency.
        num_blocks: int
            Number of building blocks to be stacked.

        emb_size_atom: int
            Embedding size of the atoms.
        emb_size_edge: int
            Embedding size of the edges.
        emb_size_trip: int
            (Down-projected) Embedding size in the triplet message passing block.
        emb_size_rbf: int
            Embedding size of the radial basis transformation.
        emb_size_cbf: int
            Embedding size of the circular basis transformation (one angle).
        emb_size_bil_trip: int
            Embedding size of the edge embeddings in the triplet-based message passing block after the bilinear layer.

        num_before_skip: int
            Number of residual blocks before the first skip connection.
        num_after_skip: int
            Number of residual blocks after the first skip connection.
        num_concat: int
            Number of residual blocks after the concatenation.
        num_atom: int
            Number of residual blocks in the atom embedding blocks.


        cutoff: float
            Embedding cutoff for interactomic directions in Angstrom.
        rbf: dict
            Name and hyperparameters of the radial basis function.
        envelope: dict
            Name and hyperparameters of the envelope function.
        cbf: dict
            Name and hyperparameters of the cosine basis function.
        extensive: bool
            Whether the output should be extensive (proportional to the number of atoms)
        output_init: str
            Initialization method for the final dense layer.
        activation: str
            Name of the activation function.

    """

    embedding_keys = [K.node_features, K.edge_features]

    def __init__(
        self,
        species: list[str],
        cutoff: float = 5.0,
        num_spherical: int = 7,
        num_radial: int = 64,
        num_blocks: int = 3,
        emb_size_atom: int = 128,
        emb_size_edge: int = 128,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        num_elements: int = 83,
    ):
        super().__init__(species, cutoff)
        assert num_blocks > 0, "Number of blocks must be greater than 0."
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_blocks = num_blocks
        self.emb_size_atom = emb_size_atom
        self.emb_size_edge = emb_size_edge
        self.emb_size_trip = emb_size_trip
        self.emb_size_rbf = emb_size_rbf
        self.emb_size_cbf = emb_size_cbf
        self.emb_size_bil_trip = emb_size_bil_trip
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_concat = num_concat
        self.num_atom = num_atom
        self.num_elements = num_elements

        output_init: str = "HeOrthogonal"
        activation: str = "swish"

        # ---------------------------------- Basis Functions ---------------------------------- #
        rbf: dict = {"name": "gaussian"}  # noqa
        envelope: dict = {"name": "polynomial", "exponent": 5}  # noqa
        cbf: dict = {"name": "spherical_harmonics"}  # noqa
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        # ------------------------------------------------------------------------------------- #

        # ------------------------------- Share Down Projections ------------------------------ #
        # Share down projection across all interaction blocks
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf)

        # Share the dense Layer of the atom embedding block accross the interaction blocks
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        # ------------------------------------------------------------------------------------- #

        # Embedding block
        self.atom_emb = AtomEmbedding(emb_size_atom, num_elements)
        self.edge_emb = EdgeEmbedding(emb_size_atom, num_radial, emb_size_edge, activation=activation)

        out_blocks = []
        int_blocks = []

        # Interaction Blocks
        interaction_block = InteractionBlockTripletsOnly  # GemNet-(d)T
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                )
            )

        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=1,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=False,
                    name=f"OutBlock_{i}",
                )
            )

        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)

        self.shared_parameters = [
            (self.mlp_rbf3.linear.weight, self.num_blocks),
            (self.mlp_cbf3.weight, self.num_blocks),
            (self.mlp_rbf_h.linear.weight, self.num_blocks),
            (self.mlp_rbf_out.linear.weight, self.num_blocks + 1),
        ]

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype)
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_dist_new = self.select_symmetric_edges(edge_dist, mask, edge_reorder_idx, False)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        data,
        edge_index,
        cell_offsets,
        neighbors,
        edge_dist,
        edge_vector,
        cutoff=None,
    ):
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff

            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]

        empty_image = neighbors == 0
        if torch.any(empty_image):
            raise ValueError("An image has no neighbors")
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(self, data):
        num_atoms = data[K.pos].size(0)
        edge_index = data[K.edge_index]
        distance_vec = data[K.edge_vec]
        D_st = torch.linalg.norm(distance_vec, dim=1)
        cell_offsets = data[K.edge_shift]
        neighbors = compute_neighbors(data, edge_index)

        # These vectors actually point in the opposite direction.
        # But we want to use col as idx_t for efficient aggregation.
        V_st = -distance_vec / D_st[:, None]

        # Mask interaction edges if required

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.select_edges(
            data=data,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            edge_dist=D_st,
            edge_vector=V_st,
            cutoff=self.cutoff,
        )

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(edge_index, cell_offsets, neighbors, D_st, V_st)

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(edge_index, num_atoms=num_atoms)

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

    def forward(self, data):
        atomic_numbers = data[K.elems].long()

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E
        energy_i = E_t.squeeze(-1)
        energy_i = self.species_energy_scale(data, energy_i)
        energy = scatter(energy_i, data[K.batch], dim=0, reduce="sum")
        data[K.node_features] = h
        data[K.edge_features] = m
        return energy
