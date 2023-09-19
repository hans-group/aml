import numpy as np
from ase import units
from ase.md.md import MolecularDynamics


class NoseHoover(MolecularDynamics):
    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        tau,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        MolecularDynamics.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

        # Initialize simulation parameters
        # convert units

        self.dt = timestep
        self.T = temperature_K * units.kB
        self.tau = tau
        # Q is chosen to be 6 N kT
        self.Natom = len(atoms)

        # no rotation or translation, so target kinetic energy
        # is 1/2 (3N - 6) kT
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T

        self.Q = (3.0 * self.Natom - 6) * self.T * (self.tau) ** 2
        self.zeta = 0.0

    def step(self):
        # get current acceleration and velocity:

        accel = self.atoms.get_forces(md=True) / self.atoms.get_masses().reshape(-1, 1)

        vel = self.atoms.get_velocities()

        # make full step in position
        x = self.atoms.get_positions() + vel * self.dt + (accel - self.zeta * vel) * (0.5 * self.dt**2)

        self.atoms.set_positions(x)

        # record current velocities
        KE_0 = self.atoms.get_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.dt * (accel - self.zeta * vel)

        self.atoms.set_velocities(vel_half)

        # make a full step in accelerations
        f = self.atoms.get_forces()
        accel = f / self.atoms.get_masses().reshape(-1, 1)

        # make a half step in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * (1 / self.Q) * (KE_0 - self.targeEkin)

        # make another halfstep in self.zeta
        self.zeta = self.zeta + 0.5 * self.dt * (1 / self.Q) * (self.atoms.get_kinetic_energy() - self.targeEkin)

        # make another half step in velocity
        vel = (self.atoms.get_velocities() + 0.5 * self.dt * accel) / (1 + 0.5 * self.dt * self.zeta)

        self.atoms.set_velocities(vel)

        return f

    def set_temperature(self, temperature=None, temperature_K=None):
        self.T = units.kB * self._process_temperature(temperature, temperature_K, "eV")
        self.updatevars()

    def set_timestep(self, timestep):
        self.dt = timestep
        self.updatevars()

    def updatevars(self):
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T
        self.Q = (3.0 * self.Natom - 6) * self.T * (self.tau) ** 2


class NoseHooverChain(NoseHoover):
    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        tau=20.0,
        num_chains=5,
        trajectory=None,
        logfile=None,
        loginterval=1,
        append_trajectory=False,
    ):
        NoseHoover.__init__(
            self,
            atoms=atoms,
            timestep=timestep,
            temperature_K=temperature_K,
            tau=tau,
            trajectory=trajectory,
            logfile=logfile,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )

        self.N_dof = 3.0 * self.Natom - 6
        q_0 = self.N_dof * self.T * self.tau**2
        q_n = self.T * self.tau**2
        self.num_chains = num_chains
        self.Q = 2 * np.array([q_0, *([q_n] * (num_chains - 1))])
        self.p_zeta = np.array([0.0] * num_chains)

    def get_time_derivatives(self):
        momenta = self.atoms.get_velocities() * self.atoms.get_masses().reshape(-1, 1)
        forces = self.atoms.get_forces()
        coupled_forces = self.p_zeta[0] * momenta / self.Q[0]

        accel = (forces - coupled_forces) / self.atoms.get_masses().reshape(-1, 1)

        current_ke = 0.5 * (np.power(momenta, 2) / self.atoms.get_masses().reshape(-1, 1)).sum()
        dpzeta_dt = np.zeros(shape=self.p_zeta.shape)
        dpzeta_dt[0] = 2 * (current_ke - self.targeEkin) - self.p_zeta[0] * self.p_zeta[1] / self.Q[1]
        dpzeta_dt[1:-1] = (np.power(self.p_zeta[:-2], 2) / self.Q[:-2] - self.T) - self.p_zeta[1:-1] * self.p_zeta[
            2:
        ] / self.Q[2:]
        dpzeta_dt[-1] = np.power(self.p_zeta[-2], 2) / self.Q[-2] - self.T

        return accel, dpzeta_dt

    def step(self):
        accel, dpzeta_dt = self.get_time_derivatives()
        # half step update for velocities and bath
        vel = self.atoms.get_velocities()
        vel += 0.5 * accel * self.dt
        self.atoms.set_velocities(vel)
        self.p_zeta += 0.5 * dpzeta_dt * self.dt

        # full step in coordinates
        new_positions = self.atoms.get_positions() + vel * self.dt
        self.atoms.set_positions(new_positions)

        accel, dpzeta_dt = self.get_time_derivatives()
        # half step update for velocities and bath
        vel = self.atoms.get_velocities()
        vel += 0.5 * accel * self.dt
        self.atoms.set_velocities(vel)
        self.p_zeta += 0.5 * dpzeta_dt * self.dt

    def updatevars(self):
        q_0 = self.N_dof * self.T * self.tau**2
        q_n = self.T * self.tau**2
        self.targeEkin = 0.5 * (3.0 * self.Natom - 6) * self.T
        self.Q = 2 * np.array([q_0, *([q_n] * (self.num_chains - 1))])
