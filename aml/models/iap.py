from copy import deepcopy

import torch
from ase import Atoms

from aml.common.utils import compute_neighbor_vecs, load_config
from aml.data import keys as K
from aml.data.data_structure import AtomsGraph
from aml.models.energy_models.base import BaseEnergyModel
from aml.nn.grad import ComputeGradient
from aml.typing import DataDict, OutputDict

from .base import BaseModel


class InterAtomicPotential(BaseModel):
    def __init__(
        self,
        energy_model: BaseEnergyModel,
        compute_force: bool = True,
        compute_stress: bool = False,
        compute_hessian: bool = False,
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.energy_model = energy_model
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self._compute_hessian = compute_hessian
        self.return_embeddings = return_embeddings
        self.cutoff = self.energy_model.get_cutoff()

        grad_input_keys = []
        self.require_grad_keys = []
        if self._compute_force:
            grad_input_keys.append(K.pos)
            self.require_grad_keys.append(K.pos)
        if self._compute_stress:
            grad_input_keys.append(K.edge_vec)

        self.compute_grad = ComputeGradient(input_keys=grad_input_keys, output_key=K.energy)

    @property
    def compute_force(self):
        return self._compute_force

    @compute_force.setter
    def compute_force(self, value):
        self._compute_force = value
        if value:
            if K.pos not in self.require_grad_keys:
                self.require_grad_keys.append(K.pos)
            if K.pos not in self.compute_grad.input_keys:
                self.compute_grad.input_keys.append(K.pos)
        else:
            if K.pos in self.require_grad_keys:
                self.require_grad_keys.remove(K.pos)
            if K.pos in self.compute_grad.input_keys:
                self.compute_grad.input_keys.remove(K.pos)

    @property
    def compute_stress(self):
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, value):
        self._compute_stress = value
        if value:
            if K.edge_vec not in self.require_grad_keys:
                self.require_grad_keys.append(K.edge_vec)
            if K.edge_vec not in self.compute_grad.input_keys:
                self.compute_grad.input_keys.append(K.edge_vec)
        else:
            if K.edge_vec in self.require_grad_keys:
                self.require_grad_keys.remove(K.edge_vec)
            if K.edge_vec in self.compute_grad.input_keys:
                self.compute_grad.input_keys.remove(K.edge_vec)

    @property
    def compute_hessian(self):
        return self._compute_hessian

    @compute_hessian.setter
    def compute_hessian(self, value):
        self._compute_hessian = value
        if value:
            self.compute_grad.second_order_required = True
        else:
            self.compute_grad.second_order_required = False

    @property
    def output_keys(self) -> tuple[str, ...]:
        keys = [K.energy]
        if self.compute_force:
            keys.append(K.force)
        if self.compute_stress:
            keys.append(K.stress)
        if self.compute_hessian:
            keys.append(K.hessian)
        return tuple(keys)

    def forward(self, data: DataDict) -> OutputDict:
        if K.pos in self.require_grad_keys:
            data[K.pos].requires_grad_(True)
        compute_neighbor_vecs(data)
        for key in self.require_grad_keys:
            if key != K.pos:
                data[key].requires_grad_(True)
        outputs = {}
        outputs[K.energy] = self.energy_model(data)
        grad_vals = self.compute_grad(data, outputs)

        if self.compute_force:
            outputs[K.force] = -grad_vals[K.pos]
            if self._compute_hessian:
                r = -outputs[K.force].view(-1)
                s = r.size(0)
                hessian = outputs[K.energy].new_zeros((s, s))
                for iatom in range(s):
                    tmp = torch.autograd.grad([r[iatom]], data[K.pos], retain_graph=iatom < s)[0]
                    if tmp is not None:
                        hessian[iatom] = tmp.view(-1)
                outputs[K.hessian] = hessian

        if self.compute_stress:
            engrad_ij = grad_vals[K.edge_vec]
            if engrad_ij is None:
                engrad_ij = torch.zeros_like(data[K.edge_vec])
            F_ij = -engrad_ij
            sts = []
            count_edge = 0
            count_node = 0
            batch_size = int(data[K.batch].max() + 1)
            for i in range(batch_size):
                batch = data[K.batch]
                num_nodes = 0
                edge_batch = batch[data[K.edge_index][1, :]]
                num_edges = (edge_batch == i).sum()
                cell = data[K.cell][i]
                volume = torch.det(cell)
                if volume < 1e-6:
                    raise RuntimeError("Volume of cell is too small or zero. Make sure that the system is periodic.")
                sts.append(
                    -1
                    * (
                        torch.matmul(
                            data[K.edge_vec][count_edge : count_edge + num_edges].T,  # noqa
                            F_ij[count_edge : count_edge + num_edges],  # noqa
                        )
                        / volume
                    )
                )
                count_edge = count_edge + num_edges
                num_nodes = (batch == i).sum()
                count_node = count_node + num_nodes

            outputs[K.stress] = torch.stack(sts)

        if self.return_embeddings:
            for key in self.energy_model.embedding_keys:
                outputs[key] = data[key]
        return outputs

    def forward_atoms(self, atoms: Atoms, neighborlist_backend="ase") -> OutputDict:
        data = AtomsGraph.from_ase(atoms, self.energy_model.cutoff, neighborlist_backend=neighborlist_backend)
        device = self.parameters().__next__().device
        data = data.to(device)
        outputs = self(data)
        return outputs

    def get_config(self) -> dict:
        config = {
            "energy_model": self.energy_model.get_config(),
            "compute_force": self.compute_force,
            "compute_stress": self.compute_stress,
            "compute_hessian": self.compute_hessian,
        }
        return config

    @classmethod
    def from_config(cls, config: dict | str) -> "InterAtomicPotential":
        if not isinstance(config, dict):
            config = load_config(config)
        config = deepcopy(config)
        energy_model = BaseEnergyModel.from_config(config["energy_model"])
        compute_force = config.get("compute_force", True)
        compute_stress = config.get("compute_stress", False)
        compute_hessian = config.get("compute_hessian", False)
        return cls(energy_model, compute_force, compute_stress, compute_hessian)

    @classmethod
    def load(cls, path: str) -> "InterAtomicPotential":
        if str(path).endswith(".ckpt"):
            ckpt = torch.load(path)
            hparams = ckpt["hyper_parameters"]
            model_config = hparams["model"]
            state_dict = ckpt["state_dict"]
            state_dict = {".".join(k.split(".")[1:]): v for k, v in state_dict.items()}
            model = cls.from_config(model_config)
            model.load_state_dict(state_dict)
            return model
        else:
            raise NotImplementedError("Currently only support .ckpt file")
