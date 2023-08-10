from copy import deepcopy
from os import PathLike
from typing import Optional

from ase import units
from ase.md import Andersen, Langevin, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from aml.simulations.simulation import Simulation
from aml.simulations.temperature_strategy import ConstantTemperature, TemperatureStrategy

_ensemble_maps = {
    "nve": VelocityVerlet,
    "nvt_langevin": Langevin,
    "nvt_andersen": Andersen,
    "nvt_berendsen": NVTBerendsen,
    "npt_berendsen": NPTBerendsen,
    "npt_nosehoover": NPT,
}


class MolecularDynamics(Simulation):
    def __init__(
        self,
        atoms,
        timestep: float,  # fs
        temperature: TemperatureStrategy | float,  # K
        external_pressure: float = None,  # bar
        ensemble: str = "nvt_langevin",
        ensemble_params: dict = None,  # {"name": "...", ...},
        log_file: Optional[PathLike] = None,
        log_interval: int = 1,
        trajectory: Optional[PathLike] = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
    ):
        super().__init__(
            atoms,
            log_file,
            log_interval,
            trajectory,
            trajectory_interval,
            append_trajectory,
        )
        self.timestep = timestep
        if isinstance(temperature, float):
            temperature = ConstantTemperature(temperature)
        self.temperature = temperature
        self.initial_temperature = temperature.get_temperature(1)
        self.external_pressure = external_pressure
        self.ensemble = ensemble
        self.ensemble_params = ensemble_params
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.initial_temperature)
        self.dyn = self._build_dynamics()

    def _build_dynamics(self):
        if self.ensemble_params is None:
            self.ensemble_params = {}
        forbidden_keys = ["timestep", "temperature_K", "temperature", "pressure_au", "externalstress"]
        if any(val in self.ensemble_params for val in forbidden_keys):
            raise ValueError(f"Keys {forbidden_keys} are forbidden in ensemble_params.")

        kwargs = deepcopy(self.ensemble_params)
        if self.ensemble != "nve":
            kwargs["temperature_K"] = self.initial_temperature
        if self.ensemble == "npt_berendsen":
            kwargs["pressure_au"] = self.external_pressure
            if kwargs["pressure_au"] is not None:
                kwargs["pressure_au"] *= units.bar
        elif self.ensemble == "npt_nosehoover":
            kwargs["externalstress"] = self.external_pressure
            if kwargs["externalstress"] is not None:
                kwargs["externalstress"] *= units.bar
        dyn_class = _ensemble_maps[self.ensemble]
        dyn = dyn_class(self.atoms, timestep=self.timestep * units.fs, **kwargs)
        return dyn

    def _make_log_entry(self):
        log_entry = {
            "time [ps]": self.timestep * self._step / 1000,
            "PE [eV/atom]": self.atoms.get_potential_energy().squeeze().item() / len(self.atoms),
            "KE [eV/atom]": self.atoms.get_kinetic_energy().squeeze().item() / len(self.atoms),
            "T [K]": self.atoms.get_kinetic_energy().squeeze().item() / len(self.atoms) / (1.5 * units.kB),
            "T(target) [K]": self.temperature.curr_temperature(),
        }
        if "npt" in self.ensemble:
            log_entry["P(target) [bar]"] = self.external_pressure
            log_entry["V [A^3]"] = self.atoms.get_volume()
            log_entry["rho (g/cm3)"] = self.atoms.get_masses().sum() / self.atoms.get_volume() / units.mol * 1e24
        return log_entry

    def step(self) -> bool:
        self.dyn.run(1)
        if self.ensemble != "nve":
            self.dyn.set_temperature(temperature_K=self.temperature())
        return False

    def run(self, n_steps: int) -> None:
        if self.temperature.n_steps is not None and self.temperature.n_steps != n_steps:
            raise ValueError(f"n_steps={n_steps} does not match temperature.n_steps={self.temperature.n_steps}")
        return super().run(n_steps)
