from algorithms.optimiser import Optimiser
import numpy as np

class BruteForce(Optimiser):
    def __init__(self, dimensions, max_iterations=1, trading_bot=None, val_min=-1, val_max=1, seed=None, decimal_places=2):
        super().__init__(max_iterations=max_iterations, trading_bot=trading_bot, val_min=val_min, val_max=val_max)

        self.best_solution = [0] * dimensions
        self.dimensions = dimensions
        self.decimal_places = decimal_places

    # TODO: Make it do objective function less
    def update(self):
        self.iteration += 1

        best = np.inf

        # Brute force all combinations of decimals from -1 to 1 with 2 decimal places
        # Takes about a minute with 2 dimensions
        for i in range(10 ** self.decimal_places ** self.dimensions):
            # Convert i to a list of decimals
            solution = []
            for _ in range(self.dimensions):
                solution.append((i % (10 ** self.decimal_places)) / (10 ** self.decimal_places) * (self.val_max - self.val_min) + self.val_min)
                i //= (10 ** self.decimal_places)

            obj_value = self.objective_function(solution)

            if obj_value < best:
                best = obj_value
                self.best_solution = solution
