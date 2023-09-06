from pathlib import Path

import torch
import yaml
from e3nn.util.jit import script as e3nn_script

from aml.models.iap import InterAtomicPotential

CONFIG_PATH = Path.home() / ".config/aml"


def load_pretrained_model(name: str) -> torch.nn.Module | torch.jit.ScriptModule:
    config_path = CONFIG_PATH / "models.yaml"
    if not config_path.exists():
        raise RuntimeError(f"Could not find config file at {config_path}.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model_path = CONFIG_PATH / config[name]["path"]
    if not model_path.exists():
        raise RuntimeError(f"Could not find model {name} at {model_path}.")
    if config[name]["type"] == "iap":
        model = load_iap(model_path)
    else:
        raise RuntimeError(f"Model type {config[name]['type']} not supported.")
    return model


def load_iap(path: str | Path) -> torch.nn.Module | torch.jit.ScriptModule:
    map_location = None if torch.cuda.is_available() else "cpu"
    if str(path).endswith(".ckpt"):
        return InterAtomicPotential.load(path)
    else:
        try:
            return torch.jit.load(path, map_location=map_location)
        except RuntimeError as err:
            raise RuntimeError(
                "Could not load model. Please make sure that the model was saved with torch.jit.save(model, path)"
            ) from err


def compile_iap(
    model: InterAtomicPotential,
    save_path: str | Path | None = None,
    *,
    compute_force: bool | None = None,
    compute_stress: bool | None = None,
    compute_hessian: bool | None = None,
) -> torch.jit.ScriptModule:
    if compute_force is not None:
        model.compute_force = compute_force
    if compute_stress is not None:
        model.compute_stress = compute_stress
    if compute_hessian is not None:
        model.compute_hessian = compute_hessian

    energy_model = model.energy_model
    if energy_model.name == "gemnet_t":
        raise ValueError("Gemnet is not supported by torch.jit.script")
    if energy_model.name in ("mace", "allegro", "nequip"):
        compiled_model = e3nn_script(model)
    else:
        compiled_model = torch.jit.script(model)
    if save_path is not None:
        torch.jit.save(compiled_model, save_path)
    return compiled_model
