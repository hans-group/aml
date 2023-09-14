from typing import Any, Dict

import lightning as L
from lightning.pytorch import callbacks
from lightning.pytorch.callbacks.progress.rich_progress import CustomProgress, RichProgressBarTheme
from rich import get_console, reconfigure
from torch_ema import ExponentialMovingAverage as EMA

from aml.common.registry import registry
from aml.train.rich_utils import MetricsTextColumn


@registry.register_callback("exponential_moving_average")
class ExponentialMovingAverage(callbacks.Callback):
    def __init__(self, decay, *args, **kwargs):
        self.decay = decay
        self.ema = None
        self._to_load = None

    def on_fit_start(self, trainer, pl_module):
        if self.ema is None:
            self.ema = EMA(pl_module.parameters(), decay=self.decay)
        if self._to_load is not None:
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()
        self.ema.copy_to()

    def on_train_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.ema.restore()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        self.ema.update()

    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module, *args, **kwargs):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):
        if "ema" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self):
        return {"ema": self.ema.state_dict()}


default_theme = RichProgressBarTheme()


@registry.register_callback("rich_progress_bar")
class RichProgressBar(callbacks.RichProgressBar):
    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        precision: int = 6,
        theme: RichProgressBarTheme = default_theme,
        console_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(refresh_rate, leave, theme, console_kwargs)
        self._precision = precision

    def _init_progress(self, trainer: "L.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer, self.theme.metrics, precision=self._precision)
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False
