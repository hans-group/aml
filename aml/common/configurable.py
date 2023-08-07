import inspect
import warnings
from copy import deepcopy

import torch

from aml.common.registry import registry


class Configurable:
    """Mixin class for configurable classes.
    Supports nested Configurable objects and torch jittable modules.
    Only collects the arguments in the __init__ method.
    Attributes are not considered.
    It has the same vulnerability as pickle: if the class definition is changed,
    the config file may not be able to load the class.
    In this case,explicitly call the class or set ignore_classdef to True.
    """

    supported_val_types = (str, int, float, bool, list, tuple, dict, set, None.__class__)

    @classmethod
    def _get_init_args(cls):
        params = inspect.signature(cls.__init__).parameters
        args = list(params.keys())[1:]
        defaults = [params[arg].default for arg in args]
        return args, defaults

    @torch.jit.ignore
    def get_config(self) -> dict:
        args, _ = self._get_init_args()
        config = {"@classdef": {"module": self.__class__.__module__, "name": self.__class__.__name__}}

        # Recursively get the config of the nested Configurable objects
        for arg in args:
            val = getattr(self, arg)
            if isinstance(val, Configurable):
                config[arg] = val.get_config()
            # case of nested configurable objects
            elif isinstance(val, (list, tuple, set)):
                config[arg] = []
                for v in val:
                    if isinstance(v, Configurable):
                        config[arg].append(v.get_config())
                    elif isinstance(v, self.supported_val_types):
                        config[arg].append(v)
                    else:
                        raise ValueError(f"Unsupported type {type(v)} for {arg}.")
                config[arg] = type(val)(config[arg])
            elif isinstance(val, dict):
                config[arg] = {}
                for k, v in val.items():
                    if isinstance(v, Configurable):
                        config[arg][k] = v.get_config()
                    elif isinstance(v, self.supported_val_types):
                        config[arg][k] = v
                    else:
                        raise ValueError(f"Unsupported type {type(v)} for {arg}.")
            elif isinstance(val, self.supported_val_types):
                config[arg] = val
            else:
                raise ValueError(f"Unsupported type {type(val)} for {arg}.")
        return config

    @classmethod
    def from_config(cls, config: dict, ignore_classdef: bool = False):
        """Initialize the model from a config file.

        Args:
            config (dict): The config file.
            ignore_classdef (bool, optional): Whether to ignore the class definition in the config file.
                Defaults to False.
        """
        # Validation of the config file
        config = deepcopy(config)
        classdef = config.pop("@classdef", None)

        if cls.__name__ == "Configurable":
            if classdef is None:
                raise ValueError(
                    "The class is being initialized from a config file, "
                    "but the config file does not contain class definition info."
                )
            # try to retreive the class definition from the config file
            if classdef is not None:
                module = classdef["module"]
                name = classdef["name"]
                cls = getattr(__import__(module, fromlist=[name]), name)
            else:
                # Try from registry
                category = config.pop("@category")
                name = config.pop("@name")
                cls = registry.construct_from_config(config, name, category)
        else:
            intrinsic_classdef = {"module": cls.__module__, "name": cls.__name__}
            if classdef is not None and classdef != intrinsic_classdef and not ignore_classdef:
                raise ValueError(
                    "The class is being initialized from a config, "
                    "but the config file contains class definition info that does not match the class definition."
                )

        # Main processing
        init_args, defaults = cls._get_init_args()
        args = dict(zip(init_args, defaults))
        config_filtered = {k: v for k, v in config.items() if k in init_args}

        if len(config_filtered.keys()) < len(config):
            warnings.warn(
                "Some config parameters are not used: {}".format(set(config) - set(config_filtered.keys())),
                stacklevel=1,
            )
        for key, val in config_filtered.items():
            if isinstance(val, dict):
                if "@classdef" in val:  # nested Configurable object
                    args[key] = Configurable.from_config(val)
                elif "@category" in val and "@name" in val:  # nested registry object
                    args[key] = Configurable.from_config(val)
                else:
                    args[key] = val
            elif isinstance(val, (list, tuple, set)):
                args[key] = []
                for v in val:
                    if isinstance(v, dict) and "@classdef" in v:
                        args[key].append(Configurable.from_config(v))
                    else:
                        args[key].append(v)
                args[key] = type(val)(args[key])
            else:
                args[key] = val

        # if inspect._empty in args.values():
        #     missing_args = [k for k, v in args.items() if v == inspect._empty]
        #     raise ValueError("Some arguments are not provided in the config file: {}".format(missing_args))
        return cls(**args)
