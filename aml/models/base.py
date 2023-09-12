from abc import ABC, abstractmethod
from copy import deepcopy

import torch

from aml.common.registry import registry
from aml.common.utils import Configurable


@registry.register_model("base_model")
class BaseModel(torch.nn.Module, Configurable, ABC):
    """Base class for models.
    All models should inherit from this class.
    """
    @property
    @abstractmethod
    def output_keys(self) -> tuple[str, ...]:
        """Output keys of the model.
        It would vary depending on the state of the model.
        """

    def get_config(self):
        config = {}
        config["@name"] = self.__class__.name
        config.update(super().get_config())
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = deepcopy(config)
        name = config.pop("@name", None)
        if cls.__name__ == "BaseModel":
            if name is None:
                raise ValueError("Cannot initialize BaseModel from config. Please specify the name of the model.")
            model_class = registry.get_model_class(name)
        else:
            if name is not None and hasattr(cls, "name") and cls.name != name:
                raise ValueError("The name in the config is different from the class name.")
            model_class = cls
        return super().from_config(config, actual_cls=model_class)
