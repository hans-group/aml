import lightning as L
from lightning.pytorch import callbacks
from torch_ema import ExponentialMovingAverage as EMA

from aml.common.registry import registry


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
