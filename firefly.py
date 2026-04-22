# From paper "An improved firefly algorithm with dynamic self-adaptive adjustment"

from bot import Optimiser
import numpy as np
from benchmarks import functions

class FireflyOptimiser(Optimiser):
    def __init__(self, num_fireflies, dimensions, light_absorption=0.1, step_size=0.01, max_iterations=1000, function_key="f2", tolerance=0.01):
        super().__init__()

        self.n = num_fireflies
        self.D = dimensions
        self.y = light_absorption
        self.a = step_size

        self.fireflies = np.random.uniform(-5.12, 5.12, (self.n, self.D))

        self.iteration = 0
        self.max_iterations = max_iterations

        self.obj_func = functions[function_key]

        self.tolerance = tolerance

    # Get the brightness between 2 given fireflies
    def brightness(self, i, j) -> float:
        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j]) # Eq. 2 (distance)

        return np.exp(-self.y * r*r) # Eq. 3
    
    # try get values closest to all zeroes 
    def objective_function(self, weights) -> float:
        #return np.sum(np.square(weights)) # Sphere 
        #return 10 * self.D + np.sum(weights**2 - 10 * np.cos(2 * np.pi * weights)) # Rastrigin

        return self.obj_func(weights)

    def update(self):
        self.iteration += 1

        # quality of each firefly solution 
        intensities = np.apply_along_axis(self.objective_function, 1, self.fireflies)

        new_fireflies = self.fireflies.copy()

        # check all firefly pairs? this is O(n^2) idk why the paper says its not
        for i in range(self.n):
            for j in range(self.n):
                # bros better maybe we get over there
                if intensities[j] < intensities[i]:
                    b = self.brightness(i, j)
                    d = np.random.uniform(-1, 1, self.D)

                    new_fireflies[i] += b * (self.fireflies[j] - new_fireflies[i]) + self.a * d # Eq. 4
                    new_fireflies[i] = np.clip(new_fireflies[i], -5.12, 5.12)

        self.fireflies = new_fireflies

        best = np.argmin(intensities)
        self.best_solution = self.fireflies[best]

        print(f"\rIteration: {self.iteration}/{self.max_iterations}            ", end="")

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations or (len(self.best_solution) > 0 and self.objective_function(self.best_solution) <= self.tolerance)
    
    def run(self):
        super().run()

        #print(f"Completed {self.iteration} iterations")
        #print(f"Best solution: {self.best_solution}")
        print(f"Objective value: {self.objective_function(self.best_solution)}")

class ImprovedFireflyOptimiser(FireflyOptimiser):
    def __init__(self, num_fireflies, dimensions, light_absorption=0.1, step_size=0.01, max_iterations=1000, function_key="f2", 
                       tolerance=0.01, min_brightness=0.1, w_start=0.9, w_end=0.4, theta=0.1):
        
        super().__init__(num_fireflies, dimensions, light_absorption, step_size, max_iterations, function_key, tolerance)

        self.b_min = min_brightness

        self.w_start = w_start
        self.w_end = w_end

        self.theta = theta

    # Clamped brightness
    def brightness(self, i, j) -> float:
        r = np.linalg.norm(self.fireflies[i] - self.fireflies[j]) # Eq. 2 (distance)

        return self.b_min + (1 - self.b_min) * np.exp(-self.y * r*r) # Eq. 5

    def update(self):
        self.iteration += 1

        # quality of each firefly solution 
        intensities = np.apply_along_axis(self.objective_function, 1, self.fireflies)

        new_fireflies = self.fireflies.copy()

        # logarithmic interial weight
        w = self.w_start - (self.w_start - self.w_end) * np.emath.logn(self.max_iterations, self.iteration) # Eq. 6

        # dynamic step size
        c = np.pow(self.theta, self.D) * self.max_iterations * np.exp(-self.iteration/self.max_iterations) # Eq. 7

        # check all firefly pairs? this is O(n^2) idk why the paper says its not
        for i in range(self.n):
            move = np.zeros(self.D)
            better_count = 0

            for j in range(self.n):
                # bros better get over there
                if intensities[j] < intensities[i]:
                    b = self.brightness(i, j)
                    
                    move += b * (self.fireflies[j] - new_fireflies[i])
                    better_count += 1

                    #new_fireflies[i] = new_fireflies[i] * w + b * (self.fireflies[j] - new_fireflies[i]) + self.a * c * d # Eq. 8
                    #new_fireflies[i] = np.clip(new_fireflies[i], -5.12, 5.12)

            d = np.random.uniform(-1, 1, self.D)

            #if better_count > 0:
            new_fireflies[i] = w * new_fireflies[i] + move + self.a * c * d

        self.fireflies = new_fireflies

        best = np.argmin(intensities)
        self.best_solution = self.fireflies[best]

        print(f"\rIteration: {self.iteration}/{self.max_iterations}            ", end="")

# hyper parameters
LIGHT_ABSORPTION = 0.1
STEP_SIZE = 0.1
MAX_ITERATIONS = 1000
MIN_BRIGHTNESS = 0.1
TOLERANCE = 1e-10

FUNCTION_KEY = "f12"

#print("Original firefly algorithm:")
#firefly = FireflyOptimiser(30, 30, light_absorption=LIGHT_ABSORPTION, step_size=STEP_SIZE, max_iterations=MAX_ITERATIONS, function_key=FUNCTION_KEY, tolerance=TOLERANCE)
#firefly.run()

print("Improved firefly algorithm:")
improved_firefly = ImprovedFireflyOptimiser(30, 30, light_absorption=LIGHT_ABSORPTION, step_size=STEP_SIZE, max_iterations=MAX_ITERATIONS, min_brightness=MIN_BRIGHTNESS, function_key=FUNCTION_KEY, tolerance=TOLERANCE)
improved_firefly.run()