import inspect
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy

import torch.nn

from aml.common.registry import registry
from aml.common.utils import compute_average_E0s, compute_force_rms_per_species, load_config
from aml.nn.scale import PerSpeciesScaleShift


@registry.register_energy_model("base")
class BaseEnergyModel(torch.nn.Module, ABC):
    embedding_keys = []

    def __init__(self, species: list[str], cutoff: float = 5.0, *args, **kwargs):
        super().__init__()
        self.species = species
        self.cutoff = cutoff
        self.species_energy_scale = PerSpeciesScaleShift(species)

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

    def initialize(
        self,
        dataset,
        stride: int | None = None,
        use_avg_atomic_energy: bool = True,
        use_force_rms: bool = True,
    ):
        """Initialize the model.
        This typically means setting up the energy scales.
        For equivariant models like NequIP and MACE, it would be neccesary to
        compute average number of neighbors.

        Args:
            dataset (_type_): Dataset to initialize the model with.
        """
        energy_shifts, energy_scales = self.fit_energy_scale(dataset, stride, use_avg_atomic_energy, use_force_rms)
        self.species_energy_scale = PerSpeciesScaleShift(
            self.species, initial_scales=energy_scales, initial_shifts=energy_shifts
        )

    @abstractmethod
    def forward(self, data):
        pass

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

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
