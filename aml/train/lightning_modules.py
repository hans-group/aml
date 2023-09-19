from copy import deepcopy
from typing import IO, Any, Literal, TypeAlias

import lightning as L
import torch
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import unbatch
from tqdm import tqdm

from aml.common.registry import registry
from aml.common.utils import get_batch_size
from aml.data import keys as K
from aml.models.iap import InterAtomicPotential
from aml.train.loss import WeightedSumLoss
from aml.train.metrics import get_metric_fn
from aml.typing import OutputDict

Config: TypeAlias = dict[str, Any]


def resolve_model(model: InterAtomicPotential | Config) -> tuple[InterAtomicPotential, Config]:
    if isinstance(model, dict):
        model_spec = deepcopy(model)
        model = InterAtomicPotential.from_config(model_spec)
    elif isinstance(model, InterAtomicPotential):
        model_spec = model.get_config()
    else:
        raise ValueError(f"Unsupported type {type(model)} for model.")
    return model, model_spec


class PotentialTrainingModule(L.LightningModule):
    def __init__(
        self,
        model: InterAtomicPotential | dict[str, Any],
        loss_keys: tuple[str, ...] = ("energy", "force"),
        loss_weights: tuple[str, ...] = (1.0, 1.0),
        per_atom_loss_keys: tuple[str, ...] = ("energy",),
        loss_type: str = "mse_loss",
        metrics: tuple[str, ...] = ("energy_mae", "force_mae"),
        train_metric_log_freqeuency: int = 50,
        optimizer: str = "adam",
        optimizer_config: Config | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_config: Config | None = None,
        energy_shift_mode: Literal["mean", "atomic_energies"] = "atomic_energies",
        energy_scale_mode: Literal["energy_mean", "force_rms"] = "force_rms",
        energy_mean: Literal["auto"] | float | None = None,  # Must be per atom
        atomic_energies: Literal["auto"] | dict[str, float] | None = "auto",
        energy_scale: Literal["auto"] | float | dict[str, float] | None = "auto",
        trainable_scales: bool = True,
        autoscale_subset_size: int | float | None = None,
    ):
        super().__init__()
        self.model, self.model_config = resolve_model(model)
        self.loss_keys = loss_keys
        self.loss_weights = loss_weights
        self.per_atom_loss_keys = per_atom_loss_keys
        self.loss_type = loss_type
        self.metrics = metrics
        self.train_metric_log_freqeuency = train_metric_log_freqeuency
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.energy_shift_mode = energy_shift_mode
        self.energy_scale_mode = energy_scale_mode
        self.energy_mean = energy_mean
        self.atomic_energies = atomic_energies
        self.energy_scale = energy_scale
        self.trainable_scales = trainable_scales
        self.autoscale_subset_size = autoscale_subset_size

        if "force" in self.loss_keys:
            self.model.compute_force = True
        else:
            self.model.compute_force = False

        if "stress" in self.loss_keys:
            self.model.compute_stress = True
        else:
            self.model.compute_stress = False

        self.save_hyperparameters(ignore="model")
        # Trick to save the model config in the checkpoint
        # - always config will be fed into __init__ when loading the checkpoint
        self.hparams.model = self.model_config

        # Additional attributes
        self.loss_fn = WeightedSumLoss(self.loss_keys, self.loss_weights, self.loss_type, self.per_atom_loss_keys)
        self.metric_fns = {metric: get_metric_fn(metric) for metric in self.metrics}

        self._initialized = False

    def configure_optimizers(self):
        optimizer = registry.construct_from_config(
            self.optimizer_config, self.optimizer, "optimizer", params=self.parameters()
        )
        config = {"optimizer": optimizer}
        if self.lr_scheduler is not None:
            scheduler = registry.construct_from_config(
                self.lr_scheduler_config, self.lr_scheduler, "scheduler", optimizer=optimizer
            )
            config["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
            }
        return config

    def initialize(self, dataset: InMemoryDataset | None = None):
        energy_model = self.model.energy_model
        energy_model.initialize(
            self.energy_shift_mode,
            self.energy_scale_mode,
            self.energy_mean,
            self.atomic_energies,
            self.energy_scale,
            self.trainable_scales,
            dataset,
            self.autoscale_subset_size,
        )
        self._initialized = True

    def forward(self, *args, **kwargs):
        if not self._initialized:
            raise RuntimeError("Model is not initialized. Call initialize() first.")
        return self.model(*args, **kwargs)

    def _compute_output_and_loss(self, batch: Batch) -> OutputDict:
        output = self(batch)
        loss = self.loss_fn(batch, output)
        output["loss"] = loss
        return output

    def training_step(self, batch, batch_idx):
        output = self._compute_output_and_loss(batch)
        self.log("train_loss", output["loss"].item(), prog_bar=True, batch_size=get_batch_size(batch))
        if self.train_metric_log_freqeuency > 0 and batch_idx % self.train_metric_log_freqeuency == 0:
            self.eval()
            for metric, metric_fn in self.metric_fns.items():
                name = f"train_{metric}"
                self.log(name, metric_fn(batch, output).item(), prog_bar=True, batch_size=get_batch_size(batch))
            self.train()
        return output

    def _common_inference_step(self, batch, batch_idx, mode="val"):
        with torch.inference_mode(False):
            output = self._compute_output_and_loss(batch)
        self.log("val_loss", output["loss"].item(), prog_bar=True, batch_size=get_batch_size(batch))

        for metric, metric_fn in self.metric_fns.items():
            name = f"{mode}_{metric}"
            self.log(
                name,
                metric_fn(batch, output).item(),
                prog_bar=True,
                batch_size=get_batch_size(batch),
                on_step=True,
                on_epoch=False,
            )

        return output

    def validation_step(self, batch, batch_idx):
        output = self._common_inference_step(batch, batch_idx, mode="val")
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, batch_size=get_batch_size(batch))
        return output

    def test_step(self, batch, batch_idx):
        output = self._common_inference_step(batch, batch_idx, mode="test")
        return output

    def predict(self, dataloader, return_as_batch=True, verbose=False):
        self.eval()
        predictions = []
        if verbose:
            data_iter = tqdm(dataloader, desc="Making predictions")
        else:
            data_iter = dataloader
        for batch in data_iter:
            batch = batch.to(self.device)
            output = self(batch)
            output_keys = list(output.keys())
            for key in output_keys:
                true_key = f"{key}_true"
                if key in batch:
                    output[true_key] = batch[key].detach()
            output = unbatch_predictions(batch, output)
            predictions.extend(output)
        if not return_as_batch:
            return predictions
        return Batch.from_data_list(predictions)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: _PATH | IO,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: _PATH | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> "PotentialTrainingModule":
        module = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, **kwargs)
        module._initialized = True
        return module


def unbatch_predictions(batch: Data, output: OutputDict) -> list[Data]:
    unbatched = {}
    batch_idx = batch[K.batch]
    batch_size = batch_idx.max().item() + 1
    for key in output.keys():
        out = output[key].detach()
        if out.ndim <= 1:
            out = out.view(-1, 1)
        if out.size(0) == batch_size:
            unbatched[key] = out
        else:
            unbatched[key] = unbatch(out, batch_idx)

    output_datalist = [Data(**{key: unbatched[key][i] for key in unbatched.keys()}) for i in range(batch_size)]
    return output_datalist
