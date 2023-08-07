from abc import ABC, abstractmethod

import torch


class BaseModel(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def output_keys(self) -> tuple[str, ...]:
        """Get the output keys of the model.
        It would vary depending on the state of the model.
        """
