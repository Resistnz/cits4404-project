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

        # Generate all sample positions at once
        perturbations = self.step_size * np.random.uniform(-1, 1, (self.sample_count, len(self.pos)))
        samples = np.clip(self.pos + perturbations, self.val_min, self.val_max)

        # Evaluate all samples in parallel
        fitnesses = self.parallel_evaluate(samples)

        best_idx = np.argmin(fitnesses)
        best = fitnesses[best_idx]
        best_pos = samples[best_idx]

        # Move there if better
        current_fitness = self.objective_function(self.pos)
        if best < current_fitness:
            self.pos = best_pos

        best_solution_fitness = self.objective_function(self.best_solution)
        if best < best_solution_fitness:
            self.best_solution = best_pos
