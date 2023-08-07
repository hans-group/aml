from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F

from aml.data import keys as K
from aml.typing import DataDict, OutputDict


class Metric(torch.nn.Module, ABC):
    def __init__(self, key: str, per_atom: bool = False):
        super().__init__()
        self.key = key
        self.per_atom = per_atom
        cls_name_lower = self.__class__.__name__.lower()
        self.name = f"{key}_{cls_name_lower}"
        if per_atom:
            self.name = f"per_atom_{self.name}"

    @abstractmethod
    def forward(self, data: DataDict, outputs: OutputDict) -> torch.Tensor:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(key={self.key}, per_atom={self.per_atom})"

    def __str__(self):
        return self.__repr__()


class RMSE(Metric):
    def forward(self, data: DataDict, outputs: OutputDict) -> torch.Tensor:
        target = data[self.key]
        pred = outputs[self.key]
        if self.per_atom:
            target = target / data[K.n_atoms]
            pred = pred / data[K.n_atoms]
        return F.mse_loss(pred, target).sqrt()


class MAE(Metric):
    def forward(self, data: DataDict, outputs: OutputDict) -> torch.Tensor:
        target = data[self.key]
        pred = outputs[self.key]
        if self.per_atom:
            target = target / data[K.n_atoms]
            pred = pred / data[K.n_atoms]
        return F.l1_loss(pred, target)


def get_metric_fn(name: str) -> Metric:
    def parse_name(s: str) -> tuple[str, str, bool]:
        if s.startswith("per_atom_"):
            return parse_name(s[9:])[:2] + (True,)
        else:
            parts = s.split("_")
            assert len(parts) == 2, f"Invalid metric name: {s}"
            key, metric_name = parts
            return metric_name, key, False

    metric_name, key, per_atom = parse_name(name)
    if key not in ("energy", "force", "stress"):
        raise ValueError(f"Invalid metric key: {key}")

    # Getattr from this module
    metric_cls = globals()[metric_name.upper()]
    return metric_cls(key, per_atom)
