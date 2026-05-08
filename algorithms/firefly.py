# From paper "An improved firefly algorithm with dynamic self-adaptive adjustment"

from algorithms.optimiser import Optimiser
import numpy as np
from algorithms.benchmarks import functions


class FireflyOptimiser(Optimiser):
    def __init__(self, num_fireflies, dimensions, light_absorption=0.1, step_size=0.01, max_iterations=1000, function_key="f2", trading_bot=None, val_min=-1, val_max=1, seed=None):
        super().__init__(max_iterations=max_iterations, trading_bot=trading_bot, val_min=val_min, val_max=val_max)

        if seed is not None:
            np.random.seed(seed)

        self.n = num_fireflies
        self.D = dimensions
        self.y = light_absorption
        self.a = step_size

        self.fireflies = np.random.uniform(self.val_min, self.val_max, (self.n, self.D))

        self.obj_func = functions[function_key]

    # Get the brightness between 2 given fireflies
    def brightness(self, i, j) -> float:
        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])  # Eq. 2 (distance)

        return np.exp(-self.y * r * r)  # Eq. 3

    def update(self):
        self.iteration += 1

        # quality of each firefly solution (evaluated in parallel)
        intensities = self.parallel_evaluate(self.fireflies)

        new_fireflies = self.fireflies.copy()

        # check all firefly pairs? this is O(n^2) idk why the paper says its not
        for i in range(self.n):
            for j in range(self.n):
                # bros better maybe we get over there
                if intensities[j] < intensities[i]:
                    b = self.brightness(i, j)
                    d = np.random.uniform(-1, 1, self.D)

                    new_fireflies[i] += (
                        b * (self.fireflies[j] - new_fireflies[i]) + self.a * d
                    )  # Eq. 4
                    new_fireflies[i] = np.clip(
                        new_fireflies[i], self.val_min, self.val_max
                    )

        self.fireflies = new_fireflies

        best = np.argmin(intensities)
        self.best_solution = self.fireflies[best]

        # print(f"\rIteration: {self.iteration}/{self.max_iterations}            ", end="")


class ImprovedFireflyOptimiser(FireflyOptimiser):
    def __init__(self, num_fireflies, dimensions, light_absorption=0.1, step_size=0.01, max_iterations=1000, 
                       min_brightness=0.1, w_start=0.9, w_end=0.4, theta=0.1, trading_bot=None, val_min=-1, val_max=1, seed=None):
        
        super().__init__(num_fireflies=num_fireflies, dimensions=dimensions, light_absorption=light_absorption, 
                         step_size=step_size, max_iterations=max_iterations, trading_bot=trading_bot, val_min=val_min, val_max=val_max, seed=seed)

        self.b_min = min_brightness

        self.w_start = w_start
        self.w_end = w_end

        self.theta = theta

    # Clamped brightness
    def brightness(self, i, j) -> float:
        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])  # Eq. 2 (distance)

        return self.b_min + (1 - self.b_min) * np.exp(-self.y * r * r)  # Eq. 5

    def update(self):
        self.iteration += 1

        # quality of each firefly solution (evaluated in parallel)
        intensities = self.parallel_evaluate(self.fireflies)

        new_fireflies = self.fireflies.copy()

        # logarithmic interial weight
        w = self.w_start - (self.w_start - self.w_end) * np.emath.logn(
            self.max_iterations, self.iteration
        )  # Eq. 6

        # dynamic step size
        c = (
            self.theta**self.D
            * self.max_iterations
            * np.exp(-self.iteration / self.max_iterations)
        )  # Eq. 7
        # c = 1

        # check all firefly pairs? this is O(n^2) idk why the paper says its not
        for i in range(self.n):
            move = np.zeros(self.D)
            better_count = 0

            for j in range(self.n):
                # bros better get over there
                if intensities[j] < intensities[i]:
                    b = self.brightness(i, j)

                    move += b * (self.fireflies[j] - self.fireflies[i])

                    better_count += 1

            d = np.random.uniform(-1, 1, self.D)
            random_step = self.a * c * d

            if better_count > 0:
                move /= better_count

            new_fireflies[i] = w * new_fireflies[i] + move + random_step
            new_fireflies[i] = np.clip(new_fireflies[i], self.val_min, self.val_max)

        self.fireflies = new_fireflies

        best = np.argmin(intensities)
        self.best_solution = self.fireflies[best]

        # print(f"\rIteration: {self.iteration}/{self.max_iterations}            ", end="")
