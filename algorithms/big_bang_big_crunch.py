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

        self.big_bang()
        self.big_crunch()

        self.obj_func = functions[function_key]

    def get_new_individual(self):
        return np.clip(
            self.center_of_mass
            + (self.val_max * np.random.normal(size=self.dimensions)) / self.iteration,
            self.val_min,
            self.val_max,
        )

    def big_bang(self):
        self.population = (
            [self.get_new_individual() for _ in range(self.population_size)]
            if self.iteration > 0
            else np.random.uniform(
                self.val_min, self.val_max, (self.population_size, self.dimensions)
            )
        )

    def get_mass(self, index):
        return 1 / self.fitnesses[index]

    def big_crunch(self):
        self.fitnesses = np.apply_along_axis(
            self.objective_function, 1, self.population
        )

        sum_of_mass = 0
        sum_of_mass_individuals = 0

        for index, individual in enumerate(self.population):
            mass = self.get_mass(index)
            sum_of_mass += mass
            sum_of_mass_individuals += mass * individual

        self.center_of_mass = sum_of_mass_individuals / sum_of_mass

    def update(self):
        self.iteration += 1

        self.big_bang()
        self.big_crunch()

        best = np.argmin(self.fitnesses)
        self.best_solution = self.population[best]

        print(f"\rIteration: {self.iteration}/{self.max_iterations} ", end="")
