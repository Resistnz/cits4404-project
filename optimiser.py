import numpy as np
from benchmarks import Benchmarks

class Optimiser:
    def __init__(self):
        self.best_solution = list()

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
    
# Super basic SSO optimiser
class GradientDescentOptimiser(Optimiser):
    def __init__(self, dimensions, step_size=0.01, max_iterations=1000, sample_count=10):
        super().__init__()

        self.step_size = step_size
        self.iteration = 0
        self.max_iterations = max_iterations
        self.sample_count = sample_count

        self.pos = np.random.uniform(-1, 1, dimensions)
        self.best_solution = self.pos.copy()

    def update(self):
        self.iteration += 1

        best = np.inf
        best_pos = None

        # Take a sample of random points and move in the best direction
        for i in range(self.sample_count):
            newPos = self.pos + self.step_size * np.random.uniform(-1, 1, len(self.pos))
            newPos = np.clip(newPos, -1, 1)

            if self.objective_function(newPos) < best:
                best = self.objective_function(newPos)
                best_pos = newPos

        # Move there if better
        if best < self.objective_function(self.pos):
            self.pos = best_pos

        if best < self.objective_function(self.best_solution):
            self.best_solution = best_pos

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations

    def objective_function(self, values):
        return Benchmarks.f12_shifted_sphere(values)
    
g = GradientDescentOptimiser(10)
g.run()
print(g.best_solution)
print(g.objective_function(g.best_solution))