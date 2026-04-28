import numpy as np
from algorithms.benchmarks import Benchmarks

class Optimiser:
    def __init__(self, max_iterations=1000, trading_bot=None, val_min=-1, val_max=1):
        self.best_solution = list()
        self.trading_bot = trading_bot

        self.val_min = val_min
        self.val_max = val_max

        self.iteration = 0
        self.max_iterations = max_iterations

    # do a tick
    def update(self) -> None:
        ... # these are so cool who needs pass

    def objective_function(self, values) -> float:
        return self.trading_bot.evaluate_parameters(values)
    
    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations

    def run(self) -> None:
        while not self.termination_criteria_reached():
            self.update()