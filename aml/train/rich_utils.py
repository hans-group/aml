from typing import Any, Dict, Union, cast

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from rich.progress import ProgressColumn, Task, TaskID
from rich.style import Style
from rich.text import Text


class MetricsTextColumn(ProgressColumn):
    """MetricsTextColumn with arbitrary float precision"""

    def __init__(self, trainer: "Trainer", style: Union[str, "Style"], precision: int = 3):
        self._trainer = trainer
        self._tasks: Dict[Union[int, TaskID], Any] = {}
        self._current_task_id = 0
        self._metrics: Dict[Union[str, "Style"], Any] = {}
        self._style = style
        self._precision = precision
        super().__init__()

    def update(self, metrics: Dict[Any, Any]) -> None:
        # Called when metrics are ready to be rendered.
        # This is to prevent render from causing deadlock issues by requesting metrics
        # in separate threads.
        self._metrics = metrics

    def render(self, task: "Task") -> Text:
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Text()
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._current_task_id = cast(TaskID, self._current_task_id)
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        text = ""
        for k, v in self._metrics.items():
            text += f"{k}: {round(v, self._precision) if isinstance(v, float) else v} "
        return Text(text, justify="left", style=self._style)
