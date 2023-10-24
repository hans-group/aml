import argparse
import warnings
from numbers import Number
from typing import Union

import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.calculators.mixing import SumCalculator
from omegaconf import DictConfig

import aml
from aml.simulations.temperature_strategy import ConstantTemperature, LinearTemperature, combine_strategies


def common_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--write-default-config", action="store_true")
    args, unknown_args = parser.parse_known_args()
    return parser, args, unknown_args


def try_import(name, package=None):
    """
    Try importing an object from a package or from the global namespace.

    Parameters:
    - name: The name of the object/module you're trying to import.
    - package: The name of the package where the object/module resides (optional).

    Returns:
    - The imported object/module if it exists, or None otherwise.
    """
    try:
        if package:
            module = __import__(package, fromlist=[name])
            return getattr(module, name)
        else:
            return __import__(name)
    except (ImportError, AttributeError):
        return None


def resolve_device(device: str) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if device == "cpu":
        return torch.device(device)
    if device.startswith("cuda"):
        return torch.device(device)
    if device == "mps":
        return torch.device(device)
    raise ValueError(f"Unknown device: {device}")


def resolve_temperature_strategy(temperature: Union[Number, list]):
    if isinstance(temperature, Number):
        return temperature
    if isinstance(temperature, list):
        strategies = []
        for strategy in temperature:
            assert len(strategy) == 1
            name = next(iter(strategy.keys()))
            options = strategy[name]
            if name == "constant":
                strategies.append(ConstantTemperature(**options))
            elif name == "linear":
                strategies.append(LinearTemperature(**options))
            else:
                raise ValueError(f"Unknown temperature strategy: {name}")
        return combine_strategies(*strategies)


def _add_vdw_calc(calc: Calculator, config: DictConfig) -> Calculator:
    if not config:
        return calc
    vdw_kwargs = {"params_tweak": {"atm": config.three_body}}
    method = config.method.replace("-", "").replace("_", "").strip().lower()
    if method == "dftd3":
        vdw_cls = try_import("DFTD3", "dftd3.ase")
    elif method == "dftd4":
        vdw_cls = try_import("DFTD4", "dftd4.ase")
    else:
        raise ValueError(f"Unsupported method: {method}. Use dftd3 or dftd4.")

    if vdw_cls is None:
        warnings.warn(f"{method.lower()} is not installed. Ignoring...", stacklevel=1)
        return calc

    if method == "dftd3":
        functional = config.functional.upper()
        vdw_kwargs["params_tweak"]["method"] = functional
        vdw_kwargs["params_tweak"]["damping"] = config.damping
        vdw_kwargs["damping"] = config.damping
    vdw_kwargs["method"] = functional
    vdw_calc = vdw_cls(**vdw_kwargs)
    return SumCalculator([calc, vdw_calc])


def construct_calc(config: DictConfig, pbc: np.ndarray) -> Calculator:
    model = aml.load_iap(config.path)
    if not pbc.all():
        model.compute_stress = False
    device = resolve_device(config.device)
    calc = aml.simulations.AMLCalculator(model, device=device, **config.calc)
    if config.vdw:
        calc = _add_vdw_calc(calc, config.vdw)
    return calc
