import math

import numpy as np
import torch
from ase import Atoms, units
from scipy import stats

from aml.data import keys as K
from aml.data.data_structure import AtomsGraph
from aml.simulations.ase_interface import AMLCalculator

# conversion factor from eV/Å^2/amu to eV^2
eigenval_units_factor = units._hbar**2 * 1e10**2 / (units._e * units._amu)


class NormalModes:
    """Class to perform normal mode analysis on a given Atoms object.
    Code is copied from libatoms (https://github.com/libAtoms/workflow/blob/main/wfl/generate/normal_modes.py)
    and modified to work at general situations.
    """

    def __init__(self, atoms: Atoms, hessian: np.ndarray | None = None, hmass=None):
        """Allows to do normal mode-related operations.

        Args:
            atoms: Atoms object to be analysed.
            hessian: Hessian matrix in eV/Å^2. If not provided, it will be calculated using AMLCalculator.

        """
        self.atoms = atoms
        self.hessian = hessian  # eV/Å^2
        if self.hessian is None:
            self.calculate_hessian()
        self.num_at = len(self.atoms)
        self.num_nm = self.num_at * 3
        masses = self.atoms.get_masses()
        if hmass is not None:
            for i in range(len(atoms)):
                if atoms[i].symbol == "H":
                    masses[i] = hmass
        self.inverse_m = np.repeat(masses**-0.5, 3)

        # dynamical matrix's eigenvalues in eV/Å^2/amu
        self.eigenvalues = None

        # dynamical matrix's eigenvectors in Å * sqrt(amu)
        # normalised and orthogonal
        self.eigenvectors = np.zeros((3 * self.num_at, 3 * self.num_at))

        # normal mode displacements in Å
        self.modes = np.zeros((self.num_nm, self.num_at, 3))

        # normal mode frequencies in eV - square root of eigenvalue
        self.frequencies = np.zeros(self.num_nm)
        self.frequencies_cm = np.zeros(self.num_nm)
        self.derive_normal_mode_info()

    def calculate_hessian(self):
        if self.atoms.calc is None:
            raise RuntimeError("Atoms object has no calculator attached.")
        if not isinstance(self.atoms.calc, AMLCalculator):
            raise RuntimeError("AMLCalculator is required to calculate Hessian.")
        # Compute hessian
        energy_model = self.atoms.calc.model.energy_model
        data = AtomsGraph.from_ase(self.atoms, energy_model.cutoff).to(self.atoms.calc.device)
        data["pos"].requires_grad = True
        data.compute_edge_vecs()
        energy = energy_model(data.to_dict())
        engrad = torch.autograd.grad(
            [energy],
            [data["pos"]],
            torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
        )[0]
        r = engrad.view(-1)
        s = r.size(0)
        hessian = energy.new_zeros((s, s))
        for iatom in range(s):
            tmp = torch.autograd.grad([r[iatom]], [data[K.pos]], retain_graph=iatom < s)[0]
            if tmp is not None:
                hessian[iatom] = tmp.view(-1)
        self.hessian = hessian.detach().cpu().numpy().astype(np.float64)

    @staticmethod
    def freqs_to_evals(freqs):
        """Converts from frequencies (sqrt of eigenvalue) in eV to eigenvalues
        in eV/Å^2/amu. Negative frequencies are shorthand for imaginary
        frequencies."""

        evals = []
        for freq in freqs:
            if freq < 0:
                factor = -1
            else:
                factor = 1

            evals.append(factor * freq**2 / eigenval_units_factor)

        return np.array(evals)

    @staticmethod
    def evals_to_freqs(evals):
        """Converts from eigenvalues in eV/Å^2/amu to frequencies (square
        root of eigenvalue) in eV. Negative frequencies are shorthand for
        imaginary frequencies."""

        frequencies = []
        for eigenvalue in evals:
            freq = (eigenval_units_factor * eigenvalue.astype(complex)) ** 0.5

            if np.imag(freq) != 0:
                freq = -1 * np.imag(freq)
            else:
                freq = np.real(freq)

            frequencies.append(freq)

        return np.array(frequencies)

    @staticmethod
    def evecs_to_modes(evecs, masses=None, inverse_m=None):
        """converts from mass-weighted 3N-long eigenvector to 3N displacements
        in Cartesian coordinates"""

        assert masses is None or inverse_m is None

        if masses is not None:
            inverse_m = np.repeat(masses**-0.5, 3)

        n_free = len(evecs)
        modes = evecs * inverse_m
        # normalise before reshaping.
        # no way to select axis when dividing, so have to transpose,
        # normalise, transpose.
        norm = np.linalg.norm(modes.T, axis=0)
        modes = np.divide(modes.T, norm).T
        modes = modes.reshape(n_free, int(n_free / 3), 3)
        return modes

    @staticmethod
    def modes_to_evecs(modes, masses=None, inverse_m=None):
        """converts 3xN cartesian displacements to 1x3N mass-weighted
        eigenvectors"""

        assert masses is None or inverse_m is None

        if masses is not None:
            inverse_m = np.repeat(masses**-0.5, 3)

        n_free = len(inverse_m)
        eigenvectors = modes.reshape(n_free, n_free)
        eigenvectors /= inverse_m

        # normalise
        # no way to select axis when dividing,
        # so have to transpose, normalise, transpose.
        norm = np.linalg.norm(eigenvectors, axis=1)
        eigenvectors = np.divide(eigenvectors.T, norm).T

        return eigenvectors

    def summary(self):
        """Prints all vibrational frequencies."""

        print("---------------------\n")
        print("  #    meV     cm^-1\n")
        print("---------------------\n")
        for idx, en in enumerate(self.frequencies):
            if en < 0:
                c = " i"
                en = np.abs(en)
            else:
                c = "  "

            print(f"{idx:3d} {1000 * en:6.1f}{c} {en / units.invcm:7.1f}{c}")
        print("---------------------\n")

    def visualize_modes(
        self,
        normal_mode_numbers: int | list[int] | str = "all",
        temp: float = 300.0,
        nimages: int = 16,
    ):
        """Returns normal modes as trajectory.

        Args:
            normal_mode_numbers(int | list[int] | str): normal mode numbers
                to be visualised. If "all", all normal modes are visualised.
            temp(float): temperature in K
            nimages(int): number of images in trajectory

        Returns:
            list[ase.Atoms]: list of images in trajectory
        """

        if normal_mode_numbers == "all":
            normal_mode_numbers = np.arange(self.num_nm)
        elif isinstance(normal_mode_numbers, int):
            normal_mode_numbers = [normal_mode_numbers]

        all_modes = []
        for nm in normal_mode_numbers:
            images = []
            mode = self.modes[nm] * math.sqrt(units.kB * temp / abs(self.frequencies[nm]))
            for x in np.linspace(0, 2 * math.pi, nimages, endpoint=False):
                at = self.atoms.copy()
                at.positions += math.sin(x) * mode.reshape((self.num_at, 3))
                images.append(at)
            all_modes.append(images)
        return all_modes

    def sample_normal_modes(
        self,
        sample_size: int,
        temp: float | None = None,
        energies_for_modes: list[float] | None = None,
        normal_mode_numbers: int | list[int] | str = "all",
    ):
        """Randomly displace equilibrium structure's atoms along given
        normal modes so that normal mode energies follow Boltzmann
        distribution at a given temperature.

        Args:
            sample_size(int): how many randomly perturbed structures to return
            temp(float): temperature for the displacement (putting kT energy
                into each mode) alternative to `energies_for_modes`
            energies_for_modes(list[float]): list of energies (e.g. kT) for
                each of the normal modes for generating displacement
                magnitudes. Alternative to `temp`. Length either 3N - 6 if
                normal_mode_numbers == 'all' or matches
                len(normal_mode_numbers).
            normal_mode_numbers(int | list[int] | str): list of normal mode
                numbers to displace along. Alternatively if "all" is selected,
                all but first six (rotations and translations) normal modes
                are used.

        Returns:
            list[ase.Atoms]: list of perturbed structures
        """

        assert temp is None or energies_for_modes is None

        if isinstance(normal_mode_numbers, str):
            if normal_mode_numbers == "all":
                normal_mode_numbers = np.arange(6, self.num_at * 3)
        elif isinstance(normal_mode_numbers, int):
            normal_mode_numbers = [normal_mode_numbers]

        if energies_for_modes is not None:
            assert len(energies_for_modes) == len(normal_mode_numbers)
            if isinstance(energies_for_modes, list):
                energies_for_modes = np.array(energies_for_modes)

        elif temp is not None:
            energies_for_modes = np.array([units.kB * temp] * len(normal_mode_numbers))

        n = len(normal_mode_numbers)

        cov = np.eye(n) * energies_for_modes / self.eigenvalues[normal_mode_numbers]
        norm = stats.multivariate_normal(mean=np.zeros(n), cov=cov, allow_singular=True)

        alphas_list = norm.rvs(size=sample_size)
        if sample_size == 1:
            alphas_list = [alphas_list]

        sampled_configs = []
        for alphas in alphas_list:
            if len(normal_mode_numbers) == 1:
                alphas = [alphas]

            individual_displacements = np.array(
                [aa * evec for aa, evec in zip(alphas, self.eigenvectors[normal_mode_numbers], strict=False)]
            )

            mass_wt_displs = individual_displacements.sum(axis=0)
            displacements = mass_wt_displs * self.inverse_m
            displacements = displacements.reshape(len(self.atoms), 3)

            new_pos = self.atoms.positions.copy() + displacements
            symbols = self.atoms.symbols

            displaced_at = Atoms(symbols, positions=new_pos)

            energy = sum(
                [
                    aa**2 * eigenval / 2
                    for aa, eigenval in zip(alphas, self.eigenvalues[normal_mode_numbers], strict=False)
                ]
            )

            displaced_at.info["normal_mode_energy"] = energy

            if temp is not None:
                displaced_at.info["normal_mode_temperature"] = temp

            sampled_configs.append(displaced_at)
        return sampled_configs

    def derive_normal_mode_info(self):
        """Get normal mode information using numerical hessian

        Parameters
        ----------

        calculator: Calculator / (initializer, args, kwargs)
            ASE calculator or routine to call to create calculator
        parallel_hessian: bool, default=True
            whether to parallelize 6N calculations needed for approximating
            the Hessian.

        Returns
        -------
        """
        e_vals, e_vecs = np.linalg.eigh(np.array([self.inverse_m]).T * self.hessian * self.inverse_m)

        self.eigenvalues = e_vals
        self.eigenvectors = e_vecs.T
        self.frequencies = self.evals_to_freqs(self.eigenvalues)
        self.frequencies_cm = self.frequencies / units.invcm
        self.modes = self.evecs_to_modes(self.eigenvectors, inverse_m=self.inverse_m)
