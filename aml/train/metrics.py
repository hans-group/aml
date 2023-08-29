import re
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


def _find_matching_key(dict_, key):
    for key_ in dict_:
        if key.strip().lower() == key_.strip().lower():
            return dict_[key_]


available_metrics = [
    x.lower() for x in globals() if isinstance(globals()[x], type) and issubclass(globals()[x], Metric)
]


def get_metric_fn(name: str) -> Metric:
    regex = re.compile(r"(per_atom_)?([a-zA-Z0-9_]+)_({})?".format("|".join(available_metrics)))
    matches = regex.findall(name)
    if len(matches) > 1:
        raise ValueError("Invalid format of metric")
    assert len(matches[0]) == 3
    per_atom, key, metric_name = matches[0]
    per_atom = bool(per_atom)
    # Getattr from this module
    metric_cls = _find_matching_key(globals(), metric_name)
    return metric_cls(key, per_atom=per_atom)
