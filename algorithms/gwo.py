import numpy as np

from algorithms.optimiser import Optimiser
from algorithms.benchmarks import functions


class GWOOptimiser(Optimiser):
    """
    This implementation follows the original paper:
    - alpha/beta/delta hierarchy
    - encircling mechanism
    - hunting mechanism
    - attacking/searching behaviour
    - update equations
    """

    def __init__(
        self,
        num_wolves,
        dimensions,
        max_iterations=100,
        function_key="f2",
        trading_bot=None,
        val_min=-1,
        val_max=1,
        seed=None,
    ):

        super().__init__(
            max_iterations=max_iterations,
            trading_bot=trading_bot,
            val_min=val_min,
            val_max=val_max,
        )

        # Reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.N = num_wolves
        self.D = dimensions

        # Initialize wolf population
        self.population = np.random.uniform(
            self.val_min,
            self.val_max,
            (self.N, self.D),
        )

        # Benchmark function compatibility
        if trading_bot is None:
            self.obj_func = functions[function_key]
        else:
            self.obj_func = None

        # Best solution tracking
        self.best_solution = None
        self.best_fitness = np.inf

    def objective_function(self, weights) -> float:
        """
        Evaluate:
        - benchmark function
        OR
        - trading bot objective
        """

        if self.trading_bot is not None:
            return super().objective_function(weights)

        return self.obj_func(weights)

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations

    def update(self):
        """
        Perform one iteration of GWO.
        """
        self.iteration += 1

        # Evaluate all wolves
        fitness = self.parallel_evaluate(self.population)

        # Sort wolves by fitness (minimization)
        sorted_indices = np.argsort(fitness)

        alpha = self.population[sorted_indices[0]].copy()
        beta = self.population[sorted_indices[1]].copy()
        delta = self.population[sorted_indices[2]].copy()

        alpha_fitness = fitness[sorted_indices[0]]

        # Store global best
        if alpha_fitness < self.best_fitness:
            self.best_fitness = alpha_fitness
            self.best_solution = alpha.copy()

        # Linearly decrease a from 2 to 0
        a = 2 - (2 * self.iteration / self.max_iterations)

        new_population = np.zeros_like(self.population)

        # Update ALL wolves
        for i in range(self.N):

            for j in range(self.D):
                # Alpha influence
                r1 = np.random.rand()
                r2 = np.random.rand()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * alpha[j] - self.population[i, j])
                X1 = alpha[j] - A1 * D_alpha

                # Beta influence
                r1 = np.random.rand()
                r2 = np.random.rand()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * beta[j] - self.population[i, j])
                X2 = beta[j] - A2 * D_beta

                # Delta influence
                r1 = np.random.rand()
                r2 = np.random.rand()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * delta[j] - self.population[i, j])
                X3 = delta[j] - A3 * D_delta

                # Equation (3.7)
                new_population[i, j] = (X1 + X2 + X3) / 3

        # Boundary handling
        self.population = np.clip(
            new_population,
            self.val_min,
            self.val_max,
        )