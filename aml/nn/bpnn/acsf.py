from typing import Sequence, Tuple

import torch
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor
from tqdm import tqdm

from aml.common.sparse_utils import sp_getitem
from aml.common.utils import canocialize_species
from aml.data import keys as K
from aml.nn.cutoff import CosineCutoff
from aml.typing import DataDict, Species, Tensor


def _default_acsf_params():
    params = {
        "g2": {
            "etas": [0.01, 0.05, 1.1, 1.9, 2, 9],
            "offsets": [0],
        },
        "g4": {
            "etas": [0.01, 0.1, 0.5, 1.1, 1.5, 2.5],
            "zetas": [1, 2, 4, 8],
            "lambdas": [1, -1],
        },
    }
    return params


class ACSF(torch.nn.Module):
    def __init__(self, species: Species, params: dict | None = None, cutoff: float = 5.0):
        super().__init__()
        species = canocialize_species(species).sort().values
        self.params = params or _default_acsf_params()
        self.register_buffer("species", torch.as_tensor(species, dtype=torch.long))
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=torch.float32))

        self.symm_funcs = torch.nn.ModuleList()
        if "g2" in self.params:
            g2_params = self.params["g2"]
            self.symm_funcs.append(G2Func(species, cutoff, g2_params["etas"], g2_params["offsets"]))
        if "g4" in self.params:
            g4_params = self.params["g4"]
            self.symm_funcs.append(G4Func(species, cutoff, g4_params["etas"], g4_params["zetas"], g4_params["lambdas"]))
        if "g5" in self.params:
            g5_params = self.params["g5"]
            self.symm_funcs.append(G5Func(species, cutoff, g5_params["etas"], g5_params["zetas"], g5_params["lambdas"]))
        self.n_descriptors = sum([symm_func.n_descriptors for symm_func in self.symm_funcs])

        self.register_buffer("mean", torch.zeros(self.n_descriptors))
        self.register_buffer("std", torch.ones(self.n_descriptors))

    @torch.jit.ignore
    def fit_scales(self, dataloader: torch.utils.data.DataLoader):
        G = []
        batch = next(iter(dataloader))
        device = self.cutoff.device
        for batch in tqdm(dataloader, desc="Fitting scales", total=len(dataloader)):
            batch = batch.to(device)
            G.append(self(batch))
        G = torch.cat(G)
        mean = G.mean(dim=0)
        std = G.std(dim=0)
        # do not scale when std is zero
        mask = std.abs() < 1e-8
        std[mask] = 1.0
        self.mean = mean
        self.std = std

    def forward(self, data: DataDict) -> Tensor:
        pos = data[K.pos]
        cell = data[K.cell]
        z = data[K.elems]
        edge_index = data[K.edge_index]
        edge_shift = data[K.edge_shift]
        batch = data[K.batch]

        G = []
        for symm_func in self.symm_funcs:
            G.append(symm_func(pos, cell, z, edge_index, edge_shift, batch))
        G = torch.cat(G).t().contiguous()
        return (G - self.mean) / self.std


class SymFunc(torch.nn.Module):
    def __init__(self, species: Species, cutoff: float):
        super().__init__()
        self.register_buffer("species", torch.as_tensor(species, dtype=torch.long))
        self.register_buffer("cutoff", torch.as_tensor(cutoff, dtype=torch.float32))
        self.params = None
        self.cutoff_fn = CosineCutoff(cutoff)


class G2Func(SymFunc):
    def __init__(self, species: Species, cutoff: float, etas: Sequence[float], offsets: Sequence[float]):
        super().__init__(species, cutoff)
        self._etas = torch.as_tensor(etas, dtype=torch.float32)
        self._offsets = torch.as_tensor(offsets, dtype=torch.float32)
        self.params = torch.cartesian_prod(self._etas, self._offsets)
        self.n_descriptors = self.params.size(0) * self.species.size(0)
        etas, offsets = self.params.t().contiguous()
        self.register_buffer("etas", etas)
        self.register_buffer("offsets", offsets)

    def forward(self, pos, cell, z, edge_index, edge_shift, batch):
        j, i = edge_index[1], edge_index[0]  # j->i
        vec_ij_doublet = pos[j] - pos[i] + torch.einsum("ni,nij->nj", edge_shift, cell[batch[i]])
        r_ij_doublet = torch.linalg.norm(vec_ij_doublet, dim=-1)

        G2_ij = torch.exp(
            -self.etas[:, None] * (r_ij_doublet - self.offsets[:, None]) ** 2 / self.cutoff**2
        ) * self.cutoff_fn(r_ij_doublet)

        G2_i = []
        for sp in self.species:
            mask = z[j] == sp
            G2_i.append(scatter(G2_ij * mask, i, dim_size=pos.size(0), reduce="sum", dim=-1))
        G2_i = torch.cat(G2_i, dim=0)
        return G2_i


class G4Func(SymFunc):
    def __init__(
        self, species: Species, cutoff: float, etas: Sequence[float], zetas: Sequence[float], lambdas: Sequence[float]
    ):
        super().__init__(species, cutoff)
        _etas = torch.as_tensor(etas, dtype=torch.float32)
        _zetas = torch.as_tensor(zetas, dtype=torch.float32)
        _lambdas = torch.as_tensor(lambdas, dtype=torch.float32)

        self.params = torch.cartesian_prod(_etas, _zetas, _lambdas)
        self.n_descriptors = self.params.size(0) * torch.combinations(self.species, 2, True).size(0)
        etas, zetas, lambdas = self.params.t().contiguous()
        self.register_buffer("etas", etas)
        self.register_buffer("zetas", zetas)
        self.register_buffer("lambdas", lambdas)

    def _calculate_triplet_indices(
        self, pos: Tensor, edge_index: Tensor, edge_shift: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        j, i = edge_index[0], edge_index[1]  # j->i
        num_nodes = pos.size(0)
        value = torch.arange(j.size(0), device=i.device)
        adj_t = SparseTensor(row=i, col=j, value=value, sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = sp_getitem(adj_t, j)
        num_triplets = adj_t_row.set_value(None).storage.rowcount().to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = i.repeat_interleave(num_triplets)
        idx_j = j.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()

        # Remove triplets where i = k.
        edge_shift_i = edge_shift.repeat_interleave(num_triplets, dim=0)
        mask = idx_i != idx_k
        idx_i, idx_j, idx_k, edge_shift_i = idx_i[mask], idx_j[mask], idx_k[mask], edge_shift_i[mask]
        return idx_i, idx_j, idx_k, edge_shift_i

    def _calculate_geometric_factors(
        self,
        pos: Tensor,
        cell: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        idx_k: Tensor,
        edge_shift_i: Tensor,
        batch: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        pos_i = pos[idx_i]
        vec_ij = pos[idx_j] - pos_i + torch.einsum("ni,nij->nj", edge_shift_i, cell[batch[idx_i]])
        vec_ik = pos[idx_k] - pos_i + torch.einsum("ni,nik->nk", edge_shift_i, cell[batch[idx_i]])

        r_ij = torch.linalg.norm(vec_ij, dim=-1)
        r_ik = torch.linalg.norm(vec_ik, dim=-1)
        r_jk = torch.linalg.norm(vec_ik - vec_ij, dim=-1)
        cos_ijk = (vec_ij * vec_ik).sum(dim=-1) / (r_ij * r_ik + 1e-12)
        return r_ij, r_ik, r_jk, cos_ijk

    def forward(
        self,
        pos: Tensor,
        cell: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_shift: Tensor,
        batch: Tensor,
    ) -> Tensor:
        idx_i, idx_j, idx_k, edge_shift_i = self._calculate_triplet_indices(pos, edge_index, edge_shift)
        r_ij, r_ik, r_jk, cos_ijk = self._calculate_geometric_factors(
            pos, cell, idx_i, idx_j, idx_k, edge_shift_i, batch
        )

        G4_ijk = (
            torch.pow(2, 1 - self.zetas)[:, None]
            * torch.pow(1 + self.lambdas[:, None] * cos_ijk, self.zetas[:, None])
            * torch.exp(-self.etas[:, None] * (r_ij**2 + r_ik**2 + r_jk**2) / self.cutoff**2)
            * self.cutoff_fn(r_ij)
            * self.cutoff_fn(r_ik)
            * self.cutoff_fn(r_jk)
        )

        G4_i = []
        for sp in self.species:
            mask = torch.logical_and(z[idx_j] == sp, z[idx_k] == sp)
            G4_i.append(scatter(G4_ijk * mask, idx_i, dim_size=pos.size(0), reduce="sum", dim=-1) / 2)
        for sps in torch.combinations(self.species):
            sp_j, sp_k = sps[0], sps[1]
            mask_1 = torch.logical_and(z[idx_j] == sp_j, z[idx_k] == sp_k)
            mask_2 = torch.logical_and(z[idx_j] == sp_k, z[idx_k] == sp_j)
            mask = torch.logical_or(mask_1, mask_2)
            G4_i.append(scatter(G4_ijk * mask, idx_i, dim_size=pos.size(0), reduce="sum", dim=-1) / 2)
        G4_i = torch.cat(G4_i, dim=0)
        return G4_i


class G5Func(G4Func):
    def __init__(
        self, species: Species, cutoff: float, etas: Sequence[float], zetas: Sequence[float], lambdas: Sequence[float]
    ):
        super().__init__(species, cutoff, etas, zetas, lambdas)

    def forward(
        self,
        pos: Tensor,
        cell: Tensor,
        z: Tensor,
        edge_index: Tensor,
        edge_shift: Tensor,
        batch: Tensor,
    ) -> Tensor:
        idx_i, idx_j, idx_k, edge_shift_i = self._calculate_triplet_indices(pos, edge_index, edge_shift)
        r_ij, r_ik, r_jk, cos_ijk = self._calculate_geometric_factors(
            pos, cell, idx_i, idx_j, idx_k, edge_shift_i, batch
        )

        G5_ijk = (
            torch.pow(2, 1 - self.zetas)[:, None]
            * torch.pow(1 + self.lambdas[:, None] * cos_ijk, self.zetas[:, None])
            * torch.exp(-self.etas[:, None] * (r_ij**2 + r_ik**2) / self.cutoff**2)
            * self.cutoff_fn(r_ij)
            * self.cutoff_fn(r_ik)
        )

        G5_i = []
        for sp in self.species:
            mask = torch.logical_and(z[idx_j] == sp, z[idx_k] == sp)
            G5_i.append(scatter(G5_ijk * mask, idx_i, dim_size=pos.size(0), reduce="sum", dim=-1) / 2)
        for sps in torch.combinations(self.species):
            sp_j, sp_k = sps[0], sps[1]
            mask_1 = torch.logical_and(z[idx_j] == sp_j, z[idx_k] == sp_k)
            mask_2 = torch.logical_and(z[idx_j] == sp_k, z[idx_k] == sp_j)
            mask = torch.logical_or(mask_1, mask_2)
            G5_i.append(scatter(G5_ijk * mask, idx_i, dim_size=pos.size(0), reduce="sum", dim=-1) / 2)
        G5_i = torch.cat(G5_i, dim=0)
        return G5_i
