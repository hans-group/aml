from typing import List, Optional

import torch

from aml.typing import DataDict, OutputDict, Tensor


class ComputeGradient(torch.nn.Module):
    def __init__(self, input_keys, output_key, second_order_required=False):
        super().__init__()
        self.input_keys = input_keys
        self.output_key = output_key
        self.second_order_required = second_order_required

    def forward(self, data: DataDict, outputs: OutputDict) -> OutputDict:
        out = outputs[self.output_key]
        outputs: List[Tensor] = [out]
        inputs: List[Tensor] = [data[k] for k in self.input_keys]
        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(out)]
        if not self.input_keys:
            return {}
        grad_vals = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            create_graph=self.training or self.second_order_required,
            retain_graph=self.training or self.second_order_required,
        )
        results = {}
        for k, v in zip(self.input_keys, grad_vals, strict=True):
            if v is None:
                results[k] = torch.zeros_like(data[k])
            else:
                results[k] = v
        return results
