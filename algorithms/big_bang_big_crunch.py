"""From paper 'A new optimization method: Big Bang-Big Crunch'"""

from algorithms.optimiser import Optimiser
import numpy as np
from algorithms.benchmarks import functions


class BigBangBigCrunchOptimiser(Optimiser):
    """The Big Bang-Big Crunch optimisation algorithm"""

    def __init__(
        self,
        dimensions,
        population_size=30,
        calc_center_of_mass=True,
        deviation_fixed=False,
        function_key="f2",
        max_iterations=1000,
        trading_bot=None,
        val_min=-1,
        val_max=1,
    ):
        super().__init__(
            max_iterations=max_iterations,
            trading_bot=trading_bot,
            val_min=val_min,
            val_max=val_max,
        )

        self.dimensions = dimensions
        self.population_size = population_size
        self.calc_center_of_mass = calc_center_of_mass
        self.deviation_fixed = deviation_fixed
        self.population = np.random.uniform(
            val_min, val_max, (population_size, dimensions)
        )

        self.obj_func = functions[function_key]

    def update(self):
        self.iteration += 1

        # quality of each firefly solution
        intensities = np.apply_along_axis(self.objective_function, 1, self.population)

        new_fireflies = self.population.copy()

        # check all firefly pairs? this is O(n^2) idk why the paper says its not
        for i in range(self.population_size):
            for j in range(self.population_size):
                # bros better maybe we get over there
                if intensities[j] < intensities[i]:
                    d = np.random.uniform(-1, 1, self.dimensions)

                    new_fireflies[i] += (
                        (self.population[j] - new_fireflies[i]) * d
                    )  # Eq. 4
                    new_fireflies[i] = np.clip(
                        new_fireflies[i], self.val_min, self.val_max
                    )

        self.population = new_fireflies

        best = np.argmin(intensities)
        self.best_solution = self.population[best]

        print(f"\rIteration: {self.iteration}/{self.max_iterations} ", end="")
