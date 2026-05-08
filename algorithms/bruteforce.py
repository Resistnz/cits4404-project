import numpy as np
import itertools
from algorithms.optimiser import Optimiser

class BruteForce(Optimiser):
    def __init__(self, dimensions, max_iterations=1, trading_bot=None, val_min=-1, val_max=1, seed=None, step_size=0.1):
        super().__init__(max_iterations=1, trading_bot=trading_bot, val_min=val_min, val_max=val_max)

        self.best_solution = [0] * dimensions
        self.dimensions = dimensions
        self.step_size = step_size

    def update(self):
        self.iteration += 1

        best = np.inf

        # Generate the grid of values for a single dimension.
        axis_values = np.arange(self.val_min, self.val_max + (self.step_size / 2), self.step_size)
        
        # Calculate total iterations: (points per dimension) ^ (number of dimensions)
        points_per_dim = len(axis_values)
        total_combinations = points_per_dim ** self.dimensions
        
        # Update roughly 100 times total (every 1%), ensuring it doesn't try to mod by 0
        update_interval = max(1, total_combinations // 100)

        print(f"Starting Brute Force: {total_combinations} combinations to evaluate.")

        for i, solution_tuple in enumerate(itertools.product(axis_values, repeat=self.dimensions)):
            solution = list(solution_tuple)
            
            obj_value = self.objective_function(solution)

            if obj_value < best:
                best = obj_value
                self.best_solution = solution
            
            # Print update on the same line
            if i % update_interval == 0 or i == total_combinations - 1:
                percent = (i + 1) / total_combinations * 100
                print(f"Progress: {i + 1}/{total_combinations} ({percent:.1f}%) | Current Best: {best:.4f}    ", end='\r', flush=True)
                
        # Move to a new line once the loop finishes so the next terminal output doesn't overwrite it
        print()