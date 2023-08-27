from copy import deepcopy
from typing import Literal

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from aml.data import keys as K
from aml.data.data_structure import AtomsGraph
from aml.models.iap import InterAtomicPotential

from .neighbors import NeighborlistUpdater


class AMLCalculator(Calculator):  # noqa: F821
    """ASE calculator interface for aml.

    Args:
        model (ForceFieldModel): Should be a instance of BaseForceFieldModel or jitted model.
        r_cut (float, optional): Cutoff radius. Defaults to None.
        device (str, optional): Device to run the model. Defaults to "cpu".
        neighborlist_backend (Literal["ase", "matscipy"], optional): Backend of neighborlist calculation.
            Defaults to "matscipy".

    Raises:
        ValueError: If wrong neighborlist_backend is specified.
    """

    implemented_properties = ("energy", "free_energy", "forces", "stress", "hessian")

    def __init__(
        self,
        model: InterAtomicPotential,
        compute_force: bool = True,
        compute_stress: bool = False,
        compute_hessian: bool = False,
        device: str | None = None,
        neighborlist_backend: Literal["ase", "matscipy"] = "ase",
        neighborlist_skin: float = 0.0,  # tolerance os pos change for neighborlist update
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model = deepcopy(model)
        if device is not None:
            self.model = self.model.to(device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()
        self._compute_force = compute_force
        self._compute_stress = compute_stress
        self._compute_hessian = compute_hessian
        self.model.compute_force = compute_force
        self.model.compute_stress = compute_stress
        self.model.compute_hessian = compute_hessian
        self.r_cut = model.energy_model.cutoff

        if neighborlist_backend not in ["ase", "matscipy"]:
            raise ValueError(f"Invalid neighborlist_backend: '{neighborlist_backend}'")

        self.neighborlist_backend = neighborlist_backend
        self.neighborlist_skin = neighborlist_skin
        self.neighborlist_updater = None
        self.kwargs = kwargs

    @property
    def compute_force(self):
        return self._compute_force

    @compute_force.setter
    def compute_force(self, value):
        self._compute_force = value
        self.model.compute_force = value

    @property
    def compute_stress(self):
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, value):
        self._compute_stress = value
        self.model.compute_stress = value

    @property
    def compute_hessian(self):
        return self._compute_hessian

    @compute_hessian.setter
    def compute_hessian(self, value):
        self._compute_hessian = value
        self.model.compute_hessian = value

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=list[str] | None,
        system_changes: list | None = None,
    ):  # noqa
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms, properties, system_changes)

        data = AtomsGraph.from_ase(atoms, None, read_properties=False).to(self.device)
        if self.neighborlist_updater is None:
            self.neighborlist_updater = NeighborlistUpdater(
                self.r_cut,
                self_interaction=False,
                ref=data,
                tol=self.neighborlist_skin,
                backend=self.neighborlist_backend,
            )
        self.neighborlist_updater.update(data, verbose=self.kwargs.get("verbose", False))

        output = self.model(data.to_dict())
        energy = output[K.energy].detach().cpu().item()
        if "energy" in properties:
            self.results.update(energy=energy, free_energy=energy)
        if "forces" in properties:
            if not self.compute_force:
                raise RuntimeError("Force calculation is not enabled.")
            self.results.update(forces=output[K.force].detach().cpu().numpy())
        if "stress" in properties:
            if not self.compute_stress:
                raise RuntimeError("Stress calculation is not enabled.")
            s = output[K.stress].detach().cpu().numpy().squeeze()
            self.results.update(stress=full_3x3_to_voigt_6_stress(s))
        if "hessian" in properties:
            if not self.compute_hessian:
                raise RuntimeError("Hessian calculation is not enabled.")
            self.results.update(hessian=output[K.hessian].detach().cpu().numpy())
