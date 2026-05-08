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

    def get_new_point(self):
        """
        Generates a new point in the search space using a modified
        Gaussian distribution centered around the current best location.

        Returns:
            numpy.ndarray: The coordinates of the new point.
        """
        return np.clip(
            self.center_of_mass
            + (self.val_max * np.random.normal(size=self.dimensions)) / self.iteration,
            self.val_min,
            self.val_max,
        )

    def big_bang(self):
        """
        Initialises or recalculates the entire population.
        If called after the first iteration, it calculates the new
        points around the current best location.
        Otherwise, it initialises the population randomly across the
        defined search space.
        """
        self.population = (
            [self.get_new_point() for _ in range(self.population_size)]
            if self.iteration > 0
            else np.random.uniform(
                self.val_min, self.val_max, (self.population_size, self.dimensions)
            )
        )

    def get_mass(self, index):
        """
        Calculates the 'mass' associated with a point using the inverse of its fitness score.

        Args:
            index (int): The index of the point whose mass is to be calculated.

        Returns:
            float: The mass of the specified point.
        """
        return 1 / self.fitnesses[index]

    def big_crunch(self):
        """
        Recalculates the center of mass for the current population.
        It first evaluates the fitness of every point, then weights each point's
        position by its inverse fitness (mass) to find the weighted average
        location of the entire population.
        """
        self.fitnesses = np.apply_along_axis(
            self.objective_function, 1, self.population
        )

        best = np.argmin(self.fitnesses)
        self.best_solution = self.population[best]

        if not self.calc_center_of_mass:
            self.center_of_mass = self.best_solution
            return

        sum_of_mass = 0
        sum_of_mass_points = 0

        for index, point in enumerate(self.population):
            mass = self.get_mass(index)
            sum_of_mass += mass
            sum_of_mass_points += mass * point

        self.center_of_mass = sum_of_mass_points / sum_of_mass

    def update(self):
        self.iteration += 1

        self.big_bang()
        self.big_crunch()

        print(f"\rIteration: {self.iteration}/{self.max_iterations} ", end="")
