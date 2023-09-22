from typing import Optional

import numpy as np
from ase import units
from ase.md.md import MolecularDynamics


class NVTNoseHoover(MolecularDynamics):
    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        chain_length: int = 5,
        chain_steps: int = 2,
        sy_steps: int = 3,
        tau: Optional[float] = None,
        random_seed=0,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
        **kwargs,
    ):
        del kwargs
        try:
            import jax

            jax.config.update("jax_enable_x64", True)
            import jax.numpy as jnp
            from jax_md import space

            from .nhc_utils import make_functional_calc, nvt_nose_hoover
        except ImportError as e:
            raise ImportError(
                "jax and jax-md is not installed. Please install it to use the NVTNoseHoover integrator."
            ) from e

        MolecularDynamics.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
        self.temperature_K = temperature_K
        self.kT = units.kB * self.temperature_K
        self.chain_length = chain_length
        self.chain_steps = chain_steps
        self.sy_steps = sy_steps
        self.tau = tau if tau is not None else timestep * 100

        # Simulation routines
        self.energy_fn, self.force_fn = make_functional_calc(atoms, atoms.calc)
        self.init_fn, self.apply_fn = nvt_nose_hoover(
            force_fn=self.force_fn,
            shift_fn=space.free()[1],
            dt=timestep,
            kT=self.kT,
            tau=tau,
        )
        self.state = self.init_fn(
            jax.random.PRNGKey(random_seed),
            jnp.array(atoms.get_positions()),
            jnp.array(atoms.get_masses()),
        )

    def step(self):
        self.state = self.apply_fn(self.state, kT=self.kT)
        self.atoms.set_positions(np.array(self.state.position))
        self.atoms.set_momenta(np.array(self.state.momentum))
        forces = np.array(self.state.force)
        return forces

    def set_temperature(self, temperature=None, temperature_K=None):
        self.kT = units.kB * self._process_temperature(temperature, temperature_K, "eV")
