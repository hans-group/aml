from abc import ABC, abstractmethod
from typing import List, Optional


class StepOverError(Exception):
    pass


class TemperatureStrategy(ABC):
    """Abstract class for temperature strategy.
    This class is used to control the temperature of the thermostat.

    Args:
        n_steps (Optional[int], optional): Number of steps. Defaults to None.
        if single strategy is used, n_steps will be determined in runner when given None.
    """

    def __init__(self, n_steps: Optional[int] = None):
        self.n_steps = n_steps
        self.step = 1

    def __call__(self):
        """Get temperature at current step and increase step by 1."""
        T = self.get_temperature(self.step)
        if self.n_steps is not None and self.step > self.n_steps:
            raise StepOverError("Step over the maximum number of steps.")
        self.step += 1
        return T

    def curr_temperature(self):
        """Get temperature at current step, without increasing step."""
        T = self.get_temperature(self.step)
        return T

    @abstractmethod
    def get_temperature(self, step: int) -> float:
        """Get temperature at a given step.

        Args:
            step (int): Current step.
        """

    def __repr__(self):
        return f"{self.__class__.__name__}(n_steps={self.n_steps})"

    def __str__(self):
        return self.__repr__()


class ConstantTemperature(TemperatureStrategy):
    """Constant temperature throughout the simulation.

    Args:
        temperature (float): Temperature in K.
        n_steps (Optional[int], optional): Number of steps. Defaults to None.
    """

    def __init__(self, temperature: float, n_steps: Optional[float] = None):
        super().__init__(n_steps)
        self.temperature = temperature

    def get_temperature(self, step: int) -> float:
        T = self.temperature
        return T

    def __repr__(self):
        return f"{self.__class__.__name__}(temperature={self.temperature:.2f}, n_steps={self.n_steps})"

    def __str__(self):
        return self.__repr__()


class LinearTemperature(TemperatureStrategy):
    """Linearly modify temperature from initial_temperature to final_temperature.

    Args:
        initial_temperature (float): Initial temperature in K.
        final_temperature (float): Final temperature in K.
        n_steps (Optional[int], optional): Number of steps. Defaults to None.
    """

    def __init__(self, initial_temperature: float, final_temperature: float, n_steps=None):
        super().__init__(n_steps)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

    def get_temperature(self, step: int) -> float:
        T = self.initial_temperature - (self.initial_temperature - self.final_temperature) * (step - 1) / (
            self.n_steps - 1
        )
        return T

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(initial_temperature={self.initial_temperature:.2f}, "
            f"final_temperature={self.final_temperature:.2f}, n_steps={self.n_steps})"
        )


def combine_strategies(*strategies: List[TemperatureStrategy]):
    """Combine multiple temperature strategies.

    Raises:
        ValueError: You must specify n_steps for all strategies.

    Returns:
        TemperatureStrategy: Combined temperature strategy.
    """
    n_steps_per_strategy = [strategy.n_steps for strategy in strategies]
    if any([n_steps is None for n_steps in n_steps_per_strategy]):
        raise ValueError("You must specify n_steps for all strategies when combining.")
    n_steps = sum([strategy.n_steps for strategy in strategies])

    class CombinedTemperature(TemperatureStrategy):
        def __init__(self):
            super().__init__(n_steps)
            self.strategies = strategies
            self.n_steps_per_strategy = n_steps_per_strategy
            self.current_strategy = 0
            self.current_offset = 0

        def get_temperature(self, step):
            if (
                step > sum(self.n_steps_per_strategy[: self.current_strategy + 1])
                and self.current_strategy < len(self.strategies) - 1
            ):
                self.current_strategy += 1
                self.current_offset = sum(self.n_steps_per_strategy[: self.current_strategy])
            return self.strategies[self.current_strategy].get_temperature(step - self.current_offset)

        def __repr__(self):
            s = f"{self.__class__.__name__}(\n"
            for strategy in self.strategies:
                s += f"    {strategy}\n"
            s += f"    total_steps={self.n_steps}\n"
            s += ")"
            return s

    return CombinedTemperature()
