from algorithms.optimiser import Optimiser
import numpy as np


# Super basic SSO optimiser
class GradientDescentOptimiser(Optimiser):
    def __init__(self, dimensions, step_size=0.01, max_iterations=1000, sample_count=10, trading_bot=None, val_min=-1, val_max=1, seed=None):
        super().__init__(max_iterations=max_iterations, trading_bot=trading_bot, val_min=val_min, val_max=val_max)

        if seed is not None:
            np.random.seed(seed)

        self.step_size = step_size
        self.sample_count = sample_count

        self.pos = np.random.uniform(self.val_min, self.val_max, dimensions)
        self.best_solution = self.pos.copy()

    # TODO: Make it do objective function less
    def update(self):
        self.iteration += 1

        best = np.inf
        best_pos = None

        # Take a sample of random points and move in the best direction
        for i in range(self.sample_count):
            newPos = self.pos + self.step_size * np.random.uniform(-1, 1, len(self.pos))
            newPos = np.clip(newPos, self.val_min, self.val_max)

            if self.objective_function(newPos) < best:
                best = self.objective_function(newPos)
                best_pos = newPos

            #if self.iteration < 4:
               #print(f"Sample {i+1}/{self.sample_count} - Position: {newPos} - Objective value: {self.objective_function(newPos)}")

        #print(f"\rIteration {self.iteration}/{self.max_iterations} - Best solution: {best_pos} - Objective value: {best}", end="")

        # Move there if better
        if best < self.objective_function(self.pos):
            self.pos = best_pos

        if best < self.objective_function(self.best_solution):
            self.best_solution = best_pos
