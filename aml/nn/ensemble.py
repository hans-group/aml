import torch

from aml.models.base import BaseModel
from aml.typing import DataDict, OutputDict


class EnsembleModel(BaseModel):
    def __init__(
        self,
        models: list[BaseModel],
        return_only_mean: bool = False,
    ):
        super().__init__()
        if any(sorted(m.output_keys) != sorted(models[0].output_keys) for m in models):
            raise ValueError("All models must have the same output keys.")
        self.models = torch.nn.ModuleList(models)
        self.return_only_mean = return_only_mean

    @property
    def output_keys(self) -> tuple[str, ...]:
        return self.models[0].output_keys

    def forward(self, data: DataDict) -> OutputDict:
        # Initialize output dict
        outputs = {key: [] for key in self.output_keys}
        output_mean, output_std = {}, {}
        # Compute outputs
        for _, model in enumerate(self.models):
            output = model(data)
            for key in self.output_keys:
                outputs[key].append(output[key])
        # Compute mean and std
        for key in self.output_keys:
            outputs[key] = torch.stack(outputs[key], dim=0)
            output_mean[key] = outputs[key].mean(dim=0)
            output_std[key] = outputs[key].std(dim=0)
        if self.return_only_mean:
            return output_mean
        return output_mean, output_std
