from ase import Atoms
from ase.constraints import FixAtoms


def find_fixatoms_constraint(atoms: Atoms) -> FixAtoms | None:
    """If atoms as FixAtoms contraint, return it.
    Otherwise returns None.

    Args:
        atoms(Atoms): A Atoms object.

    Returns:
        FixAtoms | None
    """
    if not atoms.constraints:
        return None
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            return c
    return None
