import shutil
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Any, Literal

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch_geometric.loader import DataLoader

from aml.data.dataset import ASEDataset
from aml.models.iap import InterAtomicPotential
from aml.train.callbacks import ExponentialMovingAverage
from aml.train.lightning_modules import PotentialTrainingModule

ConfigDict = dict[str, Any]


class PotentialTrainer:
    def __init__(
        self,
        # Model
        model: ConfigDict | InterAtomicPotential,
        # Dataset
        train_dataset: ConfigDict | ASEDataset,
        test_dataset: ConfigDict | ASEDataset | None = None,
        val_size: float = 0.1,
        batch_size: int = 4,
        dataset_cache_dir: str | None = "data",
        energy_shift_mode: Literal["mean", "atomic_energies"] = "atomic_energies",
        energy_scale_mode: Literal["energy_mean", "force_rms"] = "force_rms",
        energy_mean: Literal["auto"] | float | None = None,  # Must be per atom
        atomic_energies: Literal["auto"] | dict[str, float] | None = "auto",
        energy_scale: Literal["auto"] | float | dict[str, float] | None = "auto",
        autoscale_dataset_stride: int | None = None,
        trainable_scales: bool = True,
        # Hyperparameters for LightningModule
        train_force: bool = True,
        train_stress: bool = False,
        loss_weights: dict[str, float] = {"energy": 1.0, "force": 1.0},  # noqa
        loss_type: str = "mse_loss",
        per_atom_energy_loss: bool = True,
        metrics: tuple[str, ...] | None = None,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        optimizer_kwargs: ConfigDict | None = None,
        lr_scheduler: str | None = None,  # Default: reduce_lr_on_plateau
        lr_scheduler_kwargs: ConfigDict | None = None,
        # Hyperparameters for Trainer
        project_name: str = "mypotential",
        experiment_name: str = "version_0",
        max_epochs: int = 100,
        device: str = "cpu",
        logger: str = "tensorboard",
        log_every_n_steps: int = 50,
        early_stopping: bool = True,
        early_stopping_monitor: str = "lr",  # or val_loss
        early_stopping_mode: str = "min",
        early_stopping_patience: int = 1e1000,
        early_stopping_threshold: float = 1e-6,
        checkpoint_monitor: str = "val_loss",
        checkpoint_mode: str = "min",
        checkpoint_save_last: bool = True,
        restart_from_checkpoint: str | None = None,
        gradient_clip_val: float = 1.0,
        ema_decay: float | None = 0.999,
    ):
        self._maybe_model = model
        self._maybe_train_dataset = train_dataset
        self._maybe_test_dataset = test_dataset
        self.val_size = val_size
        self.batch_size = batch_size
        self.dataset_cache_dir = Path(dataset_cache_dir) if dataset_cache_dir is not None else None

        self.energy_shift_mode = energy_shift_mode
        self.energy_scale_mode = energy_scale_mode
        self.energy_mean = energy_mean
        self.atomic_energies = atomic_energies
        self.energy_scale = energy_scale
        self.autoscale_dataset_stride = autoscale_dataset_stride
        self.trainable_scales = trainable_scales

        self.train_force = train_force
        self.train_stress = train_stress
        self.loss_weights = loss_weights
        self.loss_type = loss_type
        self.per_atom_energy_loss = per_atom_energy_loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.device = device
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.ema_decay = ema_decay
        self.early_stopping = early_stopping
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.checkpoint_monitor = checkpoint_monitor
        self.checkpoint_mode = checkpoint_mode
        self.checkpoint_save_last = checkpoint_save_last
        self.restart_from_checkpoint = restart_from_checkpoint

        self._datasets = None

        # Process input arguments
        self.model = None
        self.loss_keys = ("energy",)
        if train_force:
            self.loss_keys += ("force",)
        if train_stress:
            self.loss_keys += ("stress",)
        if any(key not in loss_weights for key in self.loss_keys):
            raise ValueError(f"Keys {self.loss_keys} must be present in loss_weights.")
        self.loss_weights = [loss_weights[key] for key in self.loss_keys]

        self.per_atom_loss_keys = ("energy",) if per_atom_energy_loss else ()
        self.loss_type = loss_type
        self.metrics = metrics
        if self.metrics is None:
            self.metrics = ("energy_mae",)
            if train_force:
                self.metrics += ("force_mae",)
            if train_stress:
                self.metrics += ("stress_mae",)
        else:
            if train_force and "force_mae" not in self.metrics:
                self.metrics += ("force_mae",)
            if train_stress and "stress_mae" not in self.metrics:
                self.metrics += ("stress_mae",)
        self.optimizer = optimizer
        self.optimizer_config = {"lr": lr, "weight_decay": weight_decay}
        if optimizer_kwargs is not None:
            self.optimizer_config.update(optimizer_kwargs)
        self.lr_scheduler = lr_scheduler or "reduce_lr_on_plateau"
        if self.lr_scheduler == "reduce_lr_on_plateau" and lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {"mode": "min", "factor": 0.5, "patience": 15, "min_lr": 1e-6}
        self.lr_scheduler_config = lr_scheduler_kwargs

        self.training_module = None

        # Setup training directory
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_dir = Path("experiments") / self.experiment_name
        self.dataset_cache_dir = Path(dataset_cache_dir) if dataset_cache_dir is not None else None

        self.trainer = None

    def _build_model(self) -> InterAtomicPotential:
        if isinstance(self._maybe_model, dict):
            return InterAtomicPotential.from_config(self._maybe_model)
        elif isinstance(self._maybe_model, InterAtomicPotential):
            return self._maybe_model
        else:
            raise ValueError("The argument model must be either a dict or an InterAtomicPotential instance.")

    def _build_datasets(self) -> tuple[ASEDataset, ASEDataset, ASEDataset | None]:
        def _build_dataset(maybe_dataset: ASEDataset | dict, cache=None) -> ASEDataset:
            if maybe_dataset is None:
                return None
            if cache:
                if Path(cache).exists():
                    return ASEDataset.load(cache)
                else:
                    dataset = _build_dataset(maybe_dataset)
                    dataset.save(cache)
                    return dataset
            if isinstance(maybe_dataset, ASEDataset):
                return maybe_dataset
            if isinstance(maybe_dataset, dict):
                dataset = ASEDataset.from_config(maybe_dataset)
            else:
                raise ValueError("The argument dataset must be either a dict or a Dataset instance.")
            return dataset

        train_cache = self.dataset_cache_dir / "train_dataset.pt" if self.dataset_cache_dir is not None else None
        test_cache = self.dataset_cache_dir / "test_dataset.pt" if self.dataset_cache_dir is not None else None
        train_dataset = _build_dataset(self._maybe_train_dataset, cache=train_cache)
        test_dataset = _build_dataset(self._maybe_test_dataset, cache=test_cache)
        n_val = int(len(train_dataset) * self.val_size)
        n_train = len(train_dataset) - n_val
        train_dataset, val_datset = train_dataset.split(n_train)
        return train_dataset, val_datset, test_dataset

    @property
    def datasets(self) -> tuple[ASEDataset, ASEDataset, ASEDataset | None]:
        if self._datasets is None:
            print("Building datasets...")
            self._datasets = self._build_datasets()
        return self._datasets

    def _build_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader | None]:
        train_dataset, val_dataset, test_dataset = self.datasets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1)
        else:
            test_loader = None
        return train_loader, val_loader, test_loader

    def _build_callbacks(self) -> list[pl.Callback]:
        callbacks = [
            RichProgressBar(leave=True),
        ]
        if self.ema_decay is not None:
            callbacks.append(ExponentialMovingAverage(decay=self.ema_decay))
        if self.early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=self.early_stopping_monitor,
                    mode=self.early_stopping_mode,
                    patience=self.early_stopping_patience,
                    stopping_threshold=self.early_stopping_threshold,
                )
            )
        ckpt_callback = ModelCheckpoint(
            dirpath=self.experiment_dir / "checkpoints",
            monitor=self.checkpoint_monitor,
            mode=self.checkpoint_mode,
            save_last=self.checkpoint_save_last,
            filename=f"{self.experiment_name}_best",
        )
        callbacks.append(ckpt_callback)
        return callbacks

    def _build_logger(self):
        if self.logger == "tensorboard":
            logger = TensorBoardLogger(
                "tensorboard",
                name=self.experiment_name,
            )
        elif self.logger == "wandb":
            logger = WandbLogger(
                name=self.experiment_name,
                project=self.project_name,
                save_dir="wandb",
            )
        elif self.logger is None or not self.logger:
            logger = False
        else:
            raise ValueError(f"Unsupported logger {self.logger}.")
        return logger

    def train(self) -> None:
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        if self.dataset_cache_dir is not None:
            self.dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Training {self.experiment_name}...")
        print(f"Experiment directory: {self.experiment_dir}")
        print("Building model...")
        if isinstance(self._maybe_model, dict):
            equiv_models = ("mace", "nequip", "allegro")
            model_name = self._maybe_model["energy_model"]["@name"]
            avg_num_neighbors = self._maybe_model["energy_model"].get("avg_num_neighbors", None)
            if model_name in equiv_models and avg_num_neighbors is None:
                print("Average number of neighbors are being computed...")
                train_dataset = self.datasets[0]
                avg_num_neighbors = train_dataset.avg_num_neighbors
                print(f"Average number of neighbors: {avg_num_neighbors}")
                self._maybe_model["energy_model"]["avg_num_neighbors"] = avg_num_neighbors
        self.model = self._build_model()
        print("Model info:")
        pprint(self.model.get_config())
        if self.train_force:
            self.model.compute_force = True
        else:
            self.model.compute_force = False
        if self.train_stress:
            self.model.compute_stress = True
        else:
            self.model.compute_stress = False
        self.training_module = PotentialTrainingModule(
            model=self.model,
            loss_keys=self.loss_keys,
            loss_weights=self.loss_weights,
            per_atom_loss_keys=self.per_atom_loss_keys,
            loss_type=self.loss_type,
            metrics=self.metrics,
            optimizer=self.optimizer,
            optimizer_config=self.optimizer_config,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_config=self.lr_scheduler_config,
            energy_shift_mode=self.energy_shift_mode,
            energy_scale_mode=self.energy_scale_mode,
            energy_mean=self.energy_mean,
            atomic_energies=self.atomic_energies,
            energy_scale=self.energy_scale,
            trainable_scales=self.trainable_scales,
            autoscale_dataset_stride=self.autoscale_dataset_stride,
        )
        self.training_module.initialize(self.datasets[0])
        train_loader, val_loader, _ = self._build_dataloaders()
        ckpt_path = self.restart_from_checkpoint
        if ckpt_path is not None:
            if not Path(ckpt_path).exists():
                print("Checkpoint does not exist. Ignoring.")
                ckpt_path = None
        self.trainer = pl.Trainer(
            accelerator=self.device,
            devices=1,  # TODO: Add support for multiple GPUs
            gradient_clip_val=self.gradient_clip_val,
            max_epochs=self.max_epochs,
            callbacks=self._build_callbacks(),
            logger=self._build_logger(),
            log_every_n_steps=self.log_every_n_steps,
        )
        self.trainer.fit(
            self.training_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
        best_model = self.trainer.checkpoint_callback.best_model_path
        if best_model is not None:
            model_dir = Path("model")
            model_dir.mkdir(exist_ok=True, parents=True)
            saved_path = shutil.copy(best_model, model_dir)
            print(f"Best model saved to {saved_path}")

    def test(self) -> None:
        if self.trainer is None:
            raise RuntimeError("Model has not been trained yet.")
        train_loader, val_loader, test_loader = self._build_dataloaders()
        print("Evaluating model")
        predictions = {
            "train": self.training_module.predict(train_loader),
            "val": self.training_module.predict(val_loader),
        }
        if test_loader is not None:
            predictions["test"] = self.training_module.predict(test_loader)
        torch.save(predictions, self.experiment_dir / "predictions.pt")
        print(f"Predictions saved to {self.experiment_dir / 'predictions.pt'}")

    @classmethod
    def from_config(cls, config: dict) -> "PotentialTrainer":
        _config = defaultdict(lambda: None)
        _config.update(config)
        return cls(**_config)
