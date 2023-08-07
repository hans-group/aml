###########################################################################################
# Utilities
# Authors: Ilyes Batatia, Gregor Simm and David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import collections
from typing import List, Union

import numpy as np
import torch
import torch.nn
import torch.utils.data
from e3nn import o3


def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


# Based on e3nn

_TP = collections.namedtuple("_TP", "op, args")
_INPUT = collections.namedtuple("_INPUT", "tensor, start, stop")


def _wigner_nj(
    irrepss: List[o3.Irreps],
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim)
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim,
                        *(irreps.dim for irreps in irrepss_left),
                        irreps_right.dim,
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out), args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop))
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim
    return sorted(ret, key=lambda x: x[0])


def U_matrix_real(
    irreps_in: Union[str, o3.Irreps],
    irreps_out: Union[str, o3.Irreps],
    correlation: int,
    normalization: str = "component",
    filter_ir_mid=None,
    dtype=None,
):
    irreps_out = o3.Irreps(irreps_out)
    irrepss = [o3.Irreps(irreps_in)] * correlation
    if correlation == 4:
        filter_ir_mid = [
            (0, 1),
            (1, -1),
            (2, 1),
            (3, -1),
            (4, 1),
            (5, -1),
            (6, 1),
            (7, -1),
            (8, 1),
            (9, -1),
            (10, 1),
            (11, -1),
        ]
    wigners = _wigner_nj(irrepss, normalization, filter_ir_mid, dtype)
    current_ir = wigners[0][0]
    out = []
    stack = torch.tensor([])

    for ir, _, base_o3 in wigners:
        if ir in irreps_out and ir == current_ir:
            stack = torch.cat((stack, base_o3.squeeze().unsqueeze(-1)), dim=-1)
            last_ir = current_ir
        elif ir in irreps_out and ir != current_ir:
            if len(stack) != 0:
                out += [last_ir, stack]
            stack = base_o3.squeeze().unsqueeze(-1)
            current_ir, last_ir = ir, ir
        else:
            current_ir = ir
    out += [last_ir, stack]
    return out


# def compute_mean_std_atomic_inter_energy(
#     data_loader: torch.utils.data.DataLoader,
#     atomic_energies: np.ndarray,
# ) -> Tuple[float, float]:
#     atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

#     avg_atom_inter_es_list = []

#     for batch in data_loader:
#         node_e0 = atomic_energies_fn(batch.node_attrs)
#         graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
#         graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
#         avg_atom_inter_es_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }

#     avg_atom_inter_es = torch.cat(avg_atom_inter_es_list)  # [total_n_graphs]
#     mean = to_numpy(torch.mean(avg_atom_inter_es)).item()
#     std = to_numpy(torch.std(avg_atom_inter_es)).item()

#     return mean, std


# def compute_mean_rms_energy_forces(
#     data_loader: torch.utils.data.DataLoader,
#     atomic_energies: np.ndarray,
# ) -> Tuple[float, float]:
#     atomic_energies_fn = AtomicEnergiesBlock(atomic_energies=atomic_energies)

#     atom_energy_list = []
#     forces_list = []

#     for batch in data_loader:
#         node_e0 = atomic_energies_fn(batch.node_attrs)
#         graph_e0s = scatter_sum(src=node_e0, index=batch.batch, dim=-1, dim_size=batch.num_graphs)
#         graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
#         atom_energy_list.append((batch.energy - graph_e0s) / graph_sizes)  # {[n_graphs], }
#         forces_list.append(batch.forces)  # {[n_graphs*n_atoms,3], }

#     atom_energies = torch.cat(atom_energy_list, dim=0)  # [total_n_graphs]
#     forces = torch.cat(forces_list, dim=0)  # {[total_n_graphs*n_atoms,3], }

#     mean = to_numpy(torch.mean(atom_energies)).item()
#     rms = to_numpy(torch.sqrt(torch.mean(torch.square(forces)))).item()

#     return mean, rms


# def compute_avg_num_neighbors(data_loader: torch.utils.data.DataLoader) -> float:
#     num_neighbors = []

#     for batch in data_loader:
#         _, receivers = batch.edge_index
#         _, counts = torch.unique(receivers, return_counts=True)
#         num_neighbors.append(counts)

#     avg_num_neighbors = torch.mean(torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype()))
#     return to_numpy(avg_num_neighbors).item()
