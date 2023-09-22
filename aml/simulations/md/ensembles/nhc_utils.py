from typing import Callable, Optional, Tuple, TypeVar

import jax
import jax.numpy as jnp
from ase import Atoms
from jax_md import quantity, space, util
from jax_md.simulate import (
    NoseHooverChainFns,
    NVTNoseHooverState,
    canonicalize_mass,
    initialize_momenta,
    kinetic_energy,
    momentum_step,
    nose_hoover_chain,
)

Array = util.Array
f64 = util.f64

Box = space.Box

ShiftFn = space.ShiftFn

T = TypeVar("T")
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


@jax.jit
def position_step(state: T, dt: float, **kwargs) -> T:
    """Apply a single step of the time evolution operator for positions."""
    shift_fn = space.free()[1]
    if isinstance(shift_fn, Callable):
        shift_fn = jax.tree_map(lambda r: shift_fn, state.position)
    new_position = jax.tree_map(
        lambda s_fn, r, p, m: s_fn(r, dt * p / m, **kwargs), shift_fn, state.position, state.momentum, state.mass
    )
    return state.set(position=new_position)


momentum_step = jax.jit(momentum_step)


def velocity_verlet(force_fn: Callable[..., Array], shift_fn: ShiftFn, dt: float, state: T, **kwargs) -> T:
    """Apply a single step of velocity Verlet integration to a state."""
    dt = f64(dt)
    dt_2 = f64(dt / 2)

    state = momentum_step(state, dt_2)
    state = position_step(state, dt, **kwargs)
    state = state.set(force=force_fn(state.position, **kwargs))
    state = momentum_step(state, dt_2)

    return state


def nvt_nose_hoover(
    force_fn: Callable[..., Array],
    shift_fn: ShiftFn,
    dt: float,
    kT: float,
    chain_length: int = 5,
    chain_steps: int = 2,
    sy_steps: int = 3,
    tau: Optional[float] = None,
    **sim_kwargs,
) -> Simulator:
    dt = f64(dt)
    if tau is None:
        tau = dt * 100
    tau = f64(tau)

    thermostat = nose_hoover_chain(dt, chain_length, chain_steps, sy_steps, tau)
    thermostat = NoseHooverChainFns(
        initialize=thermostat.initialize,
        half_step=jax.jit(thermostat.half_step),
        update_mass=jax.jit(thermostat.update_mass),
    )

    def init_fn(key, R, mass=1.0, **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        dof = quantity.count_dof(R)

        state = NVTNoseHooverState(R, None, force_fn(R, **kwargs), mass, None)
        state = canonicalize_mass(state)
        state = initialize_momenta(state, key, _kT)
        KE = kinetic_energy(state)
        return state.set(chain=thermostat.initialize(dof, KE, _kT))

    def apply_fn(state, **kwargs):
        _kT = kT if "kT" not in kwargs else kwargs["kT"]

        chain = state.chain

        chain = thermostat.update_mass(chain, _kT)

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p)

        state = velocity_verlet(force_fn, shift_fn, dt, state, **kwargs)

        chain = chain.set(kinetic_energy=kinetic_energy(state))

        p, chain = thermostat.half_step(state.momentum, chain, _kT)
        state = state.set(momentum=p, chain=chain)

        return state

    return init_fn, apply_fn


def make_functional_calc(atoms, calc):
    def energy_fn(R, **kwargs):
        z = kwargs.get("z", atoms.get_atomic_numbers())
        cell = kwargs.get("cell", atoms.get_cell())
        sim_atoms = Atoms(numbers=z, positions=R, cell=cell, pbc=atoms.pbc)
        sim_atoms.calc = calc
        return sim_atoms.get_potential_energy()

    def force_fn(R, **kwargs):
        z = kwargs.get("z", atoms.get_atomic_numbers())
        cell = kwargs.get("cell", atoms.get_cell())
        sim_atoms = Atoms(numbers=z, positions=R, cell=cell, pbc=atoms.pbc)
        sim_atoms.calc = calc
        return jnp.asarray(sim_atoms.get_forces(md=kwargs.get("md", True)))

    return energy_fn, force_fn
