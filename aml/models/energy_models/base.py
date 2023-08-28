import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Literal

import numpy as np
import torch
import torch.nn

from aml.common.registry import registry
from aml.common.utils import compute_average_E0s, compute_force_rms_per_species, load_config
from aml.data.dataset import GraphDatasetMixin
from aml.nn.scale import PerSpeciesScaleShift
from aml.typing import DataDict, Tensor


@registry.register_energy_model("base")
class BaseEnergyModel(torch.nn.Module, ABC):
    embedding_keys = []

    def __init__(self, species: list[str], cutoff: float = 5.0, *args, **kwargs):
        super().__init__()
        self.species = species
        self.cutoff = cutoff
        self.species_energy_scale = PerSpeciesScaleShift(species)
        self.embedding_keys = self.__class__.embedding_keys

    @torch.jit.ignore
    def fit_energy_scale(
        self,
        dataset,
        stride: int | None = None,
        use_avg_atomic_energy: bool = True,
        use_force_rms: bool = True,
    ):
        """Compute average energy and force RMS of the dataset and set the global energy scale accordingly.

        Args:
            dataset (_type_): Dataset to fit the global energy scale to.
        """
        if stride is not None:
            dataset = dataset[::stride]
        if use_avg_atomic_energy:
            energy_shifts = compute_average_E0s(dataset)
        else:
            all_energies = dataset._data.energy / dataset._data.n_atoms  # per atom
            energy_shifts = {s: all_energies.mean() for s in self.species}

        if use_force_rms:
            if "force" not in dataset._data:
                raise ValueError("Dataset does not contain forces.")
            energy_scales = compute_force_rms_per_species(dataset)
        else:
            warnings.warn(
                "Using energy std instead of force RMS for energy scaling."
                "Do not use this when the data is produced from LCAO calculations. (ex. ORCA, Gaussian)",
                stacklevel=1,
            )
            all_energies = dataset._data.energy  # not per atom for scales
            energy_scales = {s: all_energies.std() for s in self.species}

        return energy_shifts, energy_scales

    @torch.jit.ignore
    def initialize(
        self,
        energy_shift_mode: Literal["mean", "atomic_energies"] = "atomic_energies",
        energy_scale_mode: Literal["energy_std", "force_rms"] = "force_rms",
        energy_mean: Literal["auto"] | float | None = None,  # Must be per atom
        atomic_energies: Literal["auto"] | dict[str, float] | None = "auto",
        energy_scale: Literal["auto"] | float | dict[str, float] | None = "auto",
        trainable_scales: bool = True,
        dataset: GraphDatasetMixin | None = None,
        subset_size: int | float | None = None,
    ):
        """Initialize the model.
        This typically means setting up the energy scales.
        For equivariant models like NequIP and MACE, it would be neccesary to
        compute average number of neighbors.

        Args:
            dataset (_type_): Dataset to initialize the model with.
        """
        if dataset is not None and subset_size is not None:
            dataset = dataset.subset(subset_size)

        def _check_dict_species(d):
            if set(self.species) != set(d.keys()):
                raise ValueError("Keys of the dictionary must match the species.")

        _no_dataset_msg = (
            "Dataset must be provided when initializing the model "
            "if atomic_energies or energy_scale is set to 'auto'."
        )
        # Energy shift
        if energy_shift_mode == "mean":
            # Energy mean to shift
            if energy_mean == "auto":
                if dataset is None:
                    raise ValueError(_no_dataset_msg)
                _mean_val = dataset.get_statistics("energy", per_atom=True, reduce="mean")
            elif energy_mean is None:
                _mean_val = 0.0
            elif isinstance(energy_mean, (float, int, np.ndarray, torch.Tensor)):
                _mean_val = float(energy_mean)
            else:
                raise ValueError("Invalid value is given for energy_mean. Must be 'auto', float, or None.")
            per_species_energy_shifts = {s: _mean_val for s in self.species}

        elif energy_shift_mode == "atomic_energies":
            # Atomic energies to shift
            if atomic_energies == "auto":
                if dataset is None:
                    raise ValueError(_no_dataset_msg)
                atomic_energies = dataset.avg_atomic_property("energy")
            elif atomic_energies is None:
                atomic_energies = {s: 0.0 for s in self.species}
            elif isinstance(atomic_energies, dict):
                _check_dict_species(atomic_energies)
                atomic_energies = atomic_energies
            else:
                raise ValueError("Invalid value is given for atomic_energies. Must be 'auto', dictionart, or None.")
            per_species_energy_shifts = atomic_energies

        # Energy scale to multiply
        if energy_scale == "auto":
            if dataset is None:
                raise ValueError(_no_dataset_msg)
            if energy_scale_mode == "energy_std":
                _scale_val = dataset.get_statistics("energy", reduce="std")
                per_species_energy_scales = {s: _scale_val for s in self.species}
            elif energy_scale_mode == "force_rms":
                per_species_energy_scales = dataset.get_statistics("force", per_species=True, reduce="rms")
            else:
                raise ValueError("Invalid value is given for energy_scale_mode. Must be 'energy_std' or 'force_rms'.")
        elif energy_scale is None:
            per_species_energy_scales = {s: 1.0 for s in self.species}
        elif isinstance(energy_scale, (float, int, np.ndarray, torch.Tensor)):
            per_species_energy_scales = {s: float(energy_scale) for s in self.species}
        elif isinstance(energy_scale, dict):
            _check_dict_species(energy_scale)
            per_species_energy_scales = energy_scale
        else:
            raise ValueError("Invalid value is given for energy_scale. Must be 'auto', float, dict, or None.")

        self.species_energy_scale = PerSpeciesScaleShift(
            self.species,
            initial_scales=per_species_energy_scales,
            initial_shifts=per_species_energy_shifts,
            trainable=trainable_scales,
        )

    @abstractmethod
    def forward(self, data: DataDict) -> Tensor:
        pass

    @torch.jit.ignore
    def get_cutoff(self) -> float:
        return self.cutoff

    def get_config(self):
        config = {}
        config["@category"] = "energy_model"
        config["@name"] = self.__class__.name

        params = inspect.signature(self.__class__.__init__).parameters
        args = list(params.keys())[1:]
        # Get values of all arguments passed to the constructor
        values = [getattr(self, arg) for arg in args]
        # Create a dictionary mapping argument names to values
        config.update(dict(zip(args, values, strict=True)))
        return config

    @classmethod
    def from_config(cls, config: dict | str):
        if not isinstance(config, dict):
            config = load_config(config)
        config = deepcopy(config)
        name = config.pop("@name", None)
        config.pop("@category", None)
        if cls.__name__ == "BaseEnergyModel":
            if name is None:
                raise ValueError("Cannot initialize BaseEnergyModel from config. Please specify the name of the model.")
            model_class = registry.get_energy_model_class(name)
        else:
            model_class = cls
        return model_class(**config)

    @torch.jit.unused
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
