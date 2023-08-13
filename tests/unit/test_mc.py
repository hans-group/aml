import ase.build
import numpy as np
from ase.constraints import FixAtoms

from aml.simulations.mc.move import Swap


def test_swap_nofix():
    atoms = ase.build.bulk("Pt", "fcc", cubic=True).repeat([3, 3, 3])
    idx = np.arange(len(atoms))
    np.random.shuffle(idx)
    for i in idx[:40]:
        atoms[i].symbol = "Ni"

    move_fn = Swap()
    prev = atoms
    for _ in range(10):
        new = move_fn(atoms)
        assert not np.allclose(new.numbers, prev.numbers)
        prev = atoms


def test_swap_fix():
    atoms = ase.build.bulk("Pt", "fcc", cubic=True).repeat([3, 3, 3])
    idx = np.arange(len(atoms))
    np.random.shuffle(idx)
    for i in idx[:40]:
        atoms[i].symbol = "Ni"

    np.random.shuffle(idx)
    atoms.set_constraint(FixAtoms(indices=idx[:25]))

    move_fn = Swap()
    prev = atoms
    for _ in range(10):
        new = move_fn(atoms)
        assert not np.allclose(new.numbers, prev.numbers)
        assert np.allclose(new.numbers[idx[:25]], prev.numbers[idx[:25]])  # fixed atoms did not move
        prev = atoms


def test_restore():
    atoms = ase.build.bulk("Pt", "fcc", cubic=True).repeat([3, 3, 3])
    idx = np.arange(len(atoms))
    np.random.shuffle(idx)
    for i in idx[:40]:
        atoms[i].symbol = "Ni"

    np.random.shuffle(idx)
    atoms.set_constraint(FixAtoms(indices=idx[:25]))

    move_fn = Swap()
    prev = atoms
    for _ in range(10):
        new = move_fn(atoms)
        assert not np.allclose(new.numbers, prev.numbers)
        assert np.allclose(new.numbers[idx[:25]], prev.numbers[idx[:25]])  # fixed atoms did not move
        assert move_fn.restore() == prev
        prev = atoms


test_swap_nofix()
test_swap_fix()
test_restore()
