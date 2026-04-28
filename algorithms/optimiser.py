import numpy as np
from algorithms.benchmarks import Benchmarks

class Optimiser:
    def __init__(self, trading_bot=None, val_min=-1, val_max=1):
        self.best_solution = list()
        self.trading_bot = trading_bot

        self.val_min = val_min
        self.val_max = val_max

    # do a tick
    def update(self) -> None:
        ... # these are so cool who needs pass

    # e.g. error of prediction of price at end of day
    def objective_function(self, values) -> float: 
        return 0
    
    def termination_criteria_reached(self) -> bool:
        return False

    def run(self):
        while not self.termination_criteria_reached():
            self.update()