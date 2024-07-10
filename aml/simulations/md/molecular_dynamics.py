from copy import deepcopy
from os import PathLike
from typing import Optional

from ase import units
from ase.md import Andersen, VelocityVerlet
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from aml.common.utils import log_and_print
from aml.simulations.md.ensembles import Langevin, NVTNoseHoover
from aml.simulations.simulation import Simulation
from aml.simulations.temperature_strategy import ConstantTemperature, TemperatureStrategy

_ensemble_maps = {
    "nve": VelocityVerlet,
    "nvt_langevin": Langevin, # It is Langevin integrator from ASE v3.22.1
    "nvt_andersen": Andersen,
    "nvt_berendsen": NVTBerendsen,
    "nvt_nosehoover": NVTNoseHoover,
    "npt_berendsen": NPTBerendsen,
    "npt_nosehoover": NPT,
}

_ensemble_default_params = {
    "nvt_langevin": {"friction": 0.02},
    "nvt_nosehoover": {"chain_length": 5, "chain_steps": 2, "sy_steps": 3, "tau": None},
}


class MolecularDynamics(Simulation):
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
        append_log: bool = False,
        trajectory: Optional[PathLike] = None,
        trajectory_interval: int = 1,
        append_trajectory: bool = False,
        store_trajectory: bool = False,
    ):
        super().__init__(
            atoms,
            log_file,
            log_interval,
            append_log,
            trajectory,
            trajectory_interval,
            append_trajectory,
            store_trajectory,
        )
        self.timestep = timestep
        if isinstance(temperature, (float, int)):
            temperature = ConstantTemperature(temperature)
        self.temperature = temperature
        self.initial_temperature = temperature.get_temperature(1)
        self.external_pressure = external_pressure
        self.ensemble = ensemble
        self.ensemble_params = ensemble_params
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.initial_temperature)
        self.dyn = self._build_dynamics()

    def _build_dynamics(self):
        default_kwargs = _ensemble_default_params.get(self.ensemble, {})
        kwargs = deepcopy(default_kwargs)
        if self.ensemble_params is not None:
            kwargs.update(self.ensemble_params)
        self.ensemble_params = deepcopy(kwargs)
        forbidden_keys = ["timestep", "temperature_K", "temperature", "pressure_au", "externalstress"]
        if any(val in kwargs for val in forbidden_keys):
            raise ValueError(f"Keys {forbidden_keys} are forbidden in ensemble_params.")
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
        for key, val in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = val
        dyn = dyn_class(self.atoms, timestep=self.timestep * units.fs, **kwargs)
        return dyn

    def _make_log_entry(self):
        PE = self.atoms.get_potential_energy() / len(self.atoms)
        KE = self.atoms.get_kinetic_energy() / len(self.atoms)
        log_entry = {
            "time [ps]": self.timestep * self._step / 1000,
            "PE [eV/atom]": PE,
            "KE [eV/atom]": KE,
            "TE [eV/atom]": PE + KE,
            "T [K]": self.atoms.get_kinetic_energy() / len(self.atoms) / (1.5 * units.kB),
            "T(target) [K]": self.temperature.curr_temperature(),
        }
        if "npt" in self.ensemble:
            log_entry["P(target) [bar]"] = self.external_pressure
            log_entry["V [A^3]"] = self.atoms.get_volume()
            log_entry["rho (g/cm3)"] = self.atoms.get_masses().sum() / self.atoms.get_volume() / units.mol * 1e24
        if self.ensemble == "nvt_nosehoover" and self.ensemble_params.get("debug", False):
            from jax_md import simulate

            log_entry["NHC_invariant"] = simulate.nvt_nose_hoover_invariant(
                self.dyn.energy_fn, self.dyn.state, self.dyn.kT
            ).item()
        return log_entry

    def step(self) -> bool:
        self.dyn.run(1)
        if self.ensemble != "nve":
            self.dyn.set_temperature(temperature_K=self.temperature())
        return False

    def print_info(self):
        header = (
            "========================================\n"
            "=======  Molecular Dynamics Run  =======\n"
            "========================================"
        )
        log_and_print(header, self.log_file)
        log_and_print(f"Ensemble type: {self.ensemble}", self.log_file)
        log_and_print("Parameters:", self.log_file)
        log_and_print(f"  Time step: {self.timestep:.2f} fs", self.log_file)
        log_and_print(f"  Initial temperature: {self.initial_temperature:.2f} K", self.log_file)
        temp_schedule = self.temperature.get_schedule(self.timestep)
        log_and_print(f"  Temperature schedule:\n{indent(temp_schedule, 4)}", self.log_file)
        if "npt" in self.ensemble:
            log_and_print(f"  External pressure: {self.external_pressure} eV/A^2", self.log_file)
        for name in self.ensemble_params:
            value = self.ensemble_params[name]
            if isinstance(value, float):
                value = round(value, 6)
            log_and_print(f"  {name}: {value}", self.log_file)

    def run(self, n_steps: int) -> None:
        self.print_info()
        log_and_print(f"Total simulation time: {(self.timestep * n_steps * 1e-3):.2f} ps", self.log_file)
        log_and_print("Starting simulation", self.log_file)
        if self.temperature.n_steps is not None and self.temperature.n_steps != n_steps:
            raise ValueError(f"n_steps={n_steps} does not match temperature.n_steps={self.temperature.n_steps}")
        return super().run(n_steps)


def indent(s, n=4):
    return "\n".join(" " * n + line for line in s.splitlines())
