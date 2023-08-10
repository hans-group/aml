from copy import deepcopy
from typing import IO, Any, TypeAlias

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from torch_geometric.data import Batch, Data
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


class PotentialTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: InterAtomicPotential | dict[str, Any],
        loss_keys: tuple[str, ...] = ("energy", "force"),
        loss_weights: tuple[str, ...] = (1.0, 1.0),
        per_atom_loss_keys: tuple[str, ...] = ("energy",),
        loss_type: str = "mse_loss",
        metrics: tuple[str, ...] = ("energy_mae", "force_mae"),
        optimizer: str = "adam",
        optimizer_config: Config | None = None,
        lr_scheduler: str | None = None,
        lr_scheduler_config: Config | None = None,
        shift_by_avg_atomic_energy: bool = True,  # If True, shift energy by average atomic energy (or per_atom energy)
        scale_by_force_rms: bool = True,  # If True, scale energy by force RMS (or energy std)
        dataset_scale_stride: int | None = None,  # If not None, stride for dataset scaling
        trainable_scales: bool = True,
    ):
        super().__init__()
        self.model, self.model_config = resolve_model(model)
        self.loss_keys = loss_keys
        self.loss_weights = loss_weights
        self.per_atom_loss_keys = per_atom_loss_keys
        self.loss_type = loss_type
        self.metrics = metrics
        self.optimizer = optimizer
        self.optimizer_config = optimizer_config or {}
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.shift_by_avg_atomic_energy = shift_by_avg_atomic_energy
        self.scale_by_force_rms = scale_by_force_rms
        self.dataset_scale_stride = dataset_scale_stride
        self.trainable_scales = trainable_scales

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

    def initialize(self, dataset):
        energy_model = self.model.energy_model
        energy_model.initialize(
            dataset, self.dataset_scale_stride, self.shift_by_avg_atomic_energy, self.scale_by_force_rms
        )
        if hasattr(energy_model, "species_energy_scale"):
            if self.trainable_scales:
                energy_model.species_energy_scale.trainable = True
            else:
                energy_model.species_energy_scale.trainable = False
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
        return output

    def _common_inference_step(self, batch, batch_idx, mode="val"):
        with torch.inference_mode(False):
            output = self._compute_output_and_loss(batch)
        self.log("val_loss", output["loss"].item(), prog_bar=True, batch_size=get_batch_size(batch))

        for metric, metric_fn in self.metric_fns.items():
            name = f"{mode}_{metric}"
            self.log(name, metric_fn(batch, output).item(), prog_bar=True, batch_size=get_batch_size(batch))

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
