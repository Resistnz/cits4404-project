from algorithms.optimiser import Optimiser
import numpy as np

# Super basic SSO optimiser
class GradientDescentOptimiser(Optimiser):
    def __init__(self, dimensions, step_size=0.01, max_iterations=1000, sample_count=10, trading_bot=None, val_min=-1, val_max=1):
        super().__init__(trading_bot=trading_bot, val_min=val_min, val_max=val_max)

        self.step_size = step_size
        self.iteration = 0
        self.max_iterations = max_iterations
        self.sample_count = sample_count

        self.pos = np.random.uniform(self.val_min, self.val_max, dimensions)
        self.best_solution = self.pos.copy()

    def update(self):
        self.iteration += 1

        best = np.inf
        best_pos = None

        # Take a sample of random points and move in the best direction
        for i in range(self.sample_count):
            newPos = self.pos + self.step_size * np.random.uniform(self.val_min, self.val_max, len(self.pos))
            newPos = np.clip(newPos, self.val_min, self.val_max)

            if self.objective_function(newPos) < best:
                best = self.objective_function(newPos)
                best_pos = newPos

        # Move there if better
        if best < self.objective_function(self.pos):
            self.pos = best_pos

        if best < self.objective_function(self.best_solution):
            self.best_solution = best_pos

        #print(f"\rIteration: {self.iteration}/{self.max_iterations}            ", end="")

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations

    def objective_function(self, values):
        return self.trading_bot.evaluate_parameters(values)