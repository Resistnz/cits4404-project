import numpy as np
from algorithms.optimiser import Optimiser

class GWOOptimiser(Optimiser):
    def __init__(self, num_wolves, dimensions, max_iterations=100,
                 trading_bot=None, val_min=5, val_max=100):

        super().__init__(max_iterations=max_iterations,
                         trading_bot=trading_bot,
                         val_min=val_min,
                         val_max=val_max)

        self.N = num_wolves   # population size
        self.D = dimensions   # number of parameters

        # initialise population of wolves randomly
        self.population = np.random.uniform(val_min, val_max, (self.N, self.D))


    def update(self):
        self.iteration += 1

        # Step 1: Evaluate the fitness 
        fitness = np.array([
            self.objective_function(x) for x in self.population
        ])

        # Step 2: Identify alpha, beta, delta wolf positions
        sorted_idx = np.argsort(fitness)

        alpha = self.population[sorted_idx[0]]
        beta  = self.population[sorted_idx[1]]
        delta = self.population[sorted_idx[2]]

        # Step 3: Update control parameter a (linearly decreasing)
        a = 2 - (2 * self.iteration / self.max_iterations)

        # Step 4: Update positions. This is the remaining omega wolves of the pack
        new_population = np.zeros_like(self.population)

        for i in range(self.N):
            X = self.population[i]

            # Alpha influence
            r1 = np.random.rand(self.D)
            r2 = np.random.rand(self.D)

            A1 = 2 * a * r1 - a
            C1 = 2 * r2

            D_alpha = np.abs(C1 * alpha - X)
            X1 = alpha - A1 * D_alpha

            # Beta influence
            r1 = np.random.rand(self.D)
            r2 = np.random.rand(self.D)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta - X)
            X2 = beta - A2 * D_beta

            # Delta influence
            r1 = np.random.rand(self.D)
            r2 = np.random.rand(self.D)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta - X)
            X3 = delta - A3 * D_delta

            # Final update (average of three leaders)
            new_population[i] = (X1 + X2 + X3) / 3

        # Step 5: Clip to bounds
        self.population = np.clip(new_population, self.val_min, self.val_max)

        # Step 6: Save best solution
        self.best_solution = alpha