from os import PathLike
from typing import Optional

from aml.simulations.bias import RMSDBiasPotential
from aml.simulations.md.molecular_dynamics import MolecularDynamics
from aml.simulations.temperature_strategy import TemperatureStrategy


class RMSDMetaDynamics(MolecularDynamics):
    def __init__(
        self,
        atoms,
        timestep: float,  # fs
        temperature: TemperatureStrategy | float = 300.0,  # K
        external_pressure: float = None,  # bar
        ensemble: str = "nvt_langevin",
        ensemble_params: dict = None,  # {"name": "...", ...},
        log_file: Optional[PathLike] = None,
        log_interval: int = 1,
        trajectory: Optional[PathLike] = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
        store_trajectory: bool = False,
        bias_scale: float = 0.5,  # eV/atom
        bias_width: float = 1.5,  # Angstrom
        bias_damping_fractor: float = 10.0,
        bias_update_frequency: int = 1000,
        maximum_num_bias: int = 10,
        verbose: bool = False,
    ):
        super().__init__(
            atoms,
            timestep=timestep,
            temperature=temperature,
            external_pressure=external_pressure,
            ensemble=ensemble,
            ensemble_params=ensemble_params,
            log_file=log_file,
            log_interval=log_interval,
            trajectory=trajectory,
            trajectory_interval=trajectory_interval,
            append_trajectory=append_trajectory,
            store_trajectory=store_trajectory,
        )
        self.bias_scale = bias_scale
        self.bias_width = bias_width
        self.bias_damping_fractor = bias_damping_fractor

        self.bias_potential = RMSDBiasPotential(
            reference_points=[], k=bias_scale, alpha=bias_width, kappa=bias_damping_fractor
        )
        self.bias_update_frequency = bias_update_frequency
        self.maximum_num_bias = maximum_num_bias

        current_constraints = self.atoms.constraints or []
        current_constraints.append(self.bias_potential)
        self.atoms.set_constraint(current_constraints)
        self.verbose = verbose

    def step(self) -> bool:
        if self._step % self.bias_update_frequency == 0:
            self.bias_potential._update_reference(self.atoms.copy())
            if self.verbose:
                print(
                    f"Step {self._step}: Updating bias potential, "
                    f"current {len(self.bias_potential.reference_points)} reference points"
                )
            if len(self.bias_potential.reference_points) > self.maximum_num_bias:
                self.bias_potential._remove_oldest_reference()
        self.dyn.run(1)
        if self.ensemble != "nve":
            self.dyn.set_temperature(temperature_K=self.temperature())

        return False
