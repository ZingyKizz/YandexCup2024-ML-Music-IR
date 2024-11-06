from abc import ABC, abstractmethod


class WeightStrategy(ABC):
    @abstractmethod
    def __call__(self, iteration: int) -> float:
        pass


class LinearWeightStrategy(WeightStrategy):
    def __init__(self, start: float, end: float, max_iter: int):
        self.start = start
        self.end = end
        self.max_iter = max_iter

    def __call__(self, iteration: int) -> float:
        if iteration >= self.max_iter:
            return self.end
        return self.start + (self.end - self.start) * (iteration / self.max_iter)


class ConstantWeightStrategy(WeightStrategy):
    def __init__(self, weight: float):
        self.weight = weight

    def __call__(self, iteration: int) -> float:
        return self.weight


class SwitchingWeightStrategy(WeightStrategy):
    def __init__(self, strategies: list[WeightStrategy], switch_iterations: list[int]):
        assert len(strategies) == len(switch_iterations) + 1
        self.strategies = strategies
        self.switch_iterations = switch_iterations

    def __call__(self, iteration: int) -> float:
        for i, switch_iter in enumerate(self.switch_iterations):
            if iteration < switch_iter:
                return self.strategies[i](iteration)
        return self.strategies[-1](iteration)
