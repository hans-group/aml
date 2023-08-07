import ase.data
import torch

from aml.common.utils import canocialize_species
from aml.data import keys as K


class GlobalScaleShift(torch.nn.Module):
    """Scale and shift the energy.
    Caution: mean value is for per atom energy, not per molecule energy.

    Args:
        torch (_type_): _description_
    """

    def __init__(self, mean=0.0, std=1.0, key=K.energy):
        super().__init__()
        self.key = key
        self.register_buffer("scale", torch.tensor(std, dtype=torch.float))
        self.register_buffer("shift", torch.tensor(mean, dtype=torch.float))

    def forward(self, data, energy):
        energy = energy * self.scale + self.shift * data[K.n_atoms]
        return energy


class PerSpeciesScaleShift(torch.nn.Module):
    def __init__(
        self,
        species,
        key=K.atomic_energy,
        initial_scales: dict[str, float] | None = None,
        initial_shifts: dict[str, float] | None = None,
        trainable: bool = True,
    ):
        super().__init__()
        self.species = canocialize_species(species).sort()[0]
        self._trainable = trainable
        elem_lookup = torch.zeros(100, dtype=torch.long)
        elem_lookup[self.species] = torch.arange(len(self.species))
        self.register_buffer("elem_lookup", elem_lookup)
        self.key = key
        # Per-element scale and shifts
        self.scales = torch.nn.Parameter(torch.ones(len(self.species)), requires_grad=self.trainable)
        self.shifts = torch.nn.Parameter(torch.zeros(len(self.species)), requires_grad=self.trainable)
        if initial_scales is not None:
            scales = []
            for atomic_num in self.species:
                symbol = ase.data.chemical_symbols[atomic_num]
                scales.append(initial_scales[symbol])
            self.scales.data = torch.as_tensor(scales, dtype=torch.float32)
        if initial_shifts is not None:
            shifts = []
            for atomic_num in self.species:
                symbol = ase.data.chemical_symbols[atomic_num]
                shifts.append(initial_shifts[symbol])
            self.shifts.data = torch.as_tensor(shifts, dtype=torch.float32)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        self.scales.requires_grad = value
        self.shifts.requires_grad = value

    def forward(self, data, atomic_energy):
        species = data[K.elems]
        idx = self.elem_lookup[species]
        atomic_energy = atomic_energy * self.scales[idx] + self.shifts[idx]
        return atomic_energy
