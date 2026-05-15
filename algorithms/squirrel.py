from algorithms.optimiser import Optimiser
import numpy as np
import math
from algorithms.benchmarks import functions

class SquirrelOptimiser(Optimiser):
    def __init__(self, num_squirrels, dimensions, p_predator = 0.1, val_min=None, val_max=None, max_iterations=1000, function_key="f2", trading_bot=None, seed=None):
        super().__init__(max_iterations=max_iterations, trading_bot=trading_bot, val_min=val_min, val_max=val_max)
        
        if seed is not None:
            np.random.seed(seed)

        self.n = num_squirrels
        self.d = dimensions
        self.p_predator = p_predator
        self.hickory_ratio = 0.25
        self.val_min = val_min
        self.val_max = val_max

        self.iteration = 0
        self.max_iterations = max_iterations

        self.squirrels = np.random.uniform(self.val_min, self.val_max, (self.n, self.d))

        if trading_bot is None:
            self.obj_func = functions[function_key]
        else:
            self.obj_func = None

        

    def objective_function(self, weights) -> float:
        if self.trading_bot is not None:
            return super().objective_function(weights)
        return self.obj_func(weights)
    
    
    def update(self):
        self.iteration += 1

        # quality of each squirrel solution
        fitness = self.parallel_evaluate(self.squirrels)
        fitness_sorted_indices = np.argsort(fitness)

        # sort squirrels by fitness (ascending -> best first)
        self.squirrels = self.squirrels[fitness_sorted_indices]

        # best and next-best groups
        hickory_nut = self.squirrels[0]  # best squirrel (hickory)
        acorn_nut = self.squirrels[1:4]  # next 3 best squirrels (acorns)

        # split remaining squirrels into those that will move towards hickory
        # and those that will move towards acorns
        cutoff = int(self.hickory_ratio * self.n)
        if cutoff < 5:
            cutoff = 5

        # indices for groups
        # keep first 4 as elite (do not move them here)
        normal_to_hickory_idx = list(range(4, min(cutoff, self.n)))
        normal_to_acorn_idx = list(range(min(cutoff, self.n), self.n))

        # update squirrels moving toward the hickory nut (best)
        for i in normal_to_hickory_idx:
            self.squirrels[i] = self.update_squirrel(self.squirrels[i], hickory_nut)

        # update squirrels moving toward one of the acorn nuts (choose randomly per squirrel)
        if len(acorn_nut) == 0:
            # fallback: all move toward hickory
            target_choices = [hickory_nut]
        else:
            target_choices = acorn_nut

        for i in normal_to_acorn_idx:
            target = target_choices[np.random.randint(0, len(target_choices))]
            self.squirrels[i] = self.update_squirrel(self.squirrels[i], target)

        # clip to allowed range
        self.squirrels = np.clip(self.squirrels, self.val_min, self.val_max)
        self.best_solution = self.squirrels[0]
        print(f"\rSquirrel Iteration: {self.iteration}/{self.max_iterations}            ", end="")


    def update_squirrel(self, squirrel, target):
        glide_constant = 1.9
        
        r = np.random.rand()
        if r < self.p_predator:
            # run away from predator
            # relocate randomly within the search range
            return np.random.uniform(self.val_min, self.val_max, self.d)
        else:
            # move towards target
            # stochastic glide towards the chosen target
            step = glide_constant * np.random.rand(self.d) * (target - squirrel)
            return squirrel + step
        
    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations
    
    def run(self):
        super().run()
        self.best_solution = self.squirrels[0]
        return self.objective_function(self.squirrels[0])


class CorrectSquirrelOptimiser(SquirrelOptimiser):
    def __init__(self, num_squirrels, dimensions, p_predator = 0.1, val_min=None, val_max=None, max_iterations=1000, function_key="f2", trading_bot=None, seed=None):
        super().__init__(
            num_squirrels=num_squirrels,
            dimensions=dimensions,
            p_predator=p_predator,
            val_min=val_min,
            val_max=val_max,
            max_iterations=max_iterations,
            function_key=function_key,
            trading_bot=trading_bot,
            seed=seed
        )

        # SSA Specific Constants [cite: 1]
        self.hickory_ratio = 0.25
        self.h_g = 8.0          # Loss in height after gliding
        self.C_D = 0.60         # Frictional drag coefficient
        self.sf = 18.0          # Scaling factor
        self.G_c = 1.9          # Gliding constant
        
        # Precompute Levy flight sigma value
        self.beta = 1.5
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        self.sigma = (num / den) ** (1 / self.beta)

    def objective_function(self, weights) -> float:
        if self.trading_bot is not None:
            return super().objective_function(weights)
        return self.obj_func(weights)

    def _get_levy(self):
        """Generates a Levy flight step array for d dimensions."""
        ra = np.random.normal(0, 1, self.d)
        rb = np.random.normal(0, 1, self.d)
        levy = 0.01 * (ra * self.sigma) / (np.abs(rb) ** (1 / self.beta))
        return levy

    def update(self):
        self.iteration += 1

        # Evaluate fitness
        fitness = self.parallel_evaluate(self.squirrels)
        fitness_sorted_indices = np.argsort(fitness)
        self.squirrels = self.squirrels[fitness_sorted_indices]

        # Identify groups
        hickory_nut = self.squirrels[0]
        acorn_nuts = self.squirrels[1:4]
        
        cutoff = max(int(self.hickory_ratio * self.n), 4)

        # 1. Update squirrels on acorn trees moving towards the hickory tree
        for i in range(1, 4):
            self.squirrels[i] = self.update_squirrel(self.squirrels[i], hickory_nut)

        # 2. Update normal squirrels moving towards the hickory tree
        for i in range(4, min(cutoff, self.n)):
            self.squirrels[i] = self.update_squirrel(self.squirrels[i], hickory_nut)

        # 3. Update normal squirrels moving towards acorn trees
        for i in range(min(cutoff, self.n), self.n):
            target = acorn_nuts[np.random.randint(0, len(acorn_nuts))] if len(acorn_nuts) > 0 else hickory_nut
            self.squirrels[i] = self.update_squirrel(self.squirrels[i], target)

        # Seasonal Monitoring Condition
        # Calculate seasonal constant (Sc) based on acorn squirrels' distance to hickory squirrel
        Sc = np.sum(np.sqrt(np.sum((self.squirrels[1:4] - self.squirrels[0]) ** 2, axis=1)))
        
        # Calculate minimum seasonal constant (Smin) which decays over time
        Smin = 1e-6 / (365 ** (self.iteration / (self.max_iterations / 2.5)))

        # Relocate squirrels using Levy flights if winter is over
        if Sc < Smin:
            for i in range(1, self.n):
                levy = self._get_levy()
                # Relocate using Levy distribution bounds
                self.squirrels[i] = self.val_min + levy * (self.val_max - self.val_min)

        # Clip all squirrels to allowed bounds
        self.squirrels = np.clip(self.squirrels, self.val_min, self.val_max)
        self.best_solution = self.squirrels[0]
        
        print(f"\rCorrect Squirrel Iteration: {self.iteration}/{self.max_iterations}            ", end="")

    def update_squirrel(self, squirrel, target):
        r = np.random.rand()
        
        # Predator presence check
        if r >= self.p_predator:
            # Calculate Aerodynamic Gliding Distance (d_g)
            C_L = np.random.uniform(0.675, 1.5)
            # L/D = C_L / C_D; d_g = h_g / tan(phi) = h_g * (C_L / C_D)
            d_g = (self.h_g * C_L / self.C_D) / self.sf 
            
            step = d_g * self.G_c * (target - squirrel)
            return squirrel + step
        else:
            # Random location due to predator evasion
            return np.random.uniform(self.val_min, self.val_max, self.d)

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations
    
    def run(self):
        super().run()
        self.best_solution = self.squirrels[0]
        return self.objective_function(self.squirrels[0])
    
class ImprovedSquirrelOptimiser(CorrectSquirrelOptimiser):
    def __init__(self, num_squirrels, dimensions, p_predator=0.1, val_min=None, val_max=None, max_iterations=1000, function_key="f2", trading_bot=None, seed=None):
        super().__init__(
            num_squirrels=num_squirrels,
            dimensions=dimensions,
            p_predator=p_predator,
            val_min=val_min,
            val_max=val_max,
            max_iterations=max_iterations,
            function_key=function_key,
            trading_bot=trading_bot,
            seed=seed
        )

        # SSA Specific Constants
        self.hickory_ratio = 0.25
        self.h_g = 8.0          # Loss in height after gliding
        self.C_D = 0.60         # Frictional drag coefficient
        self.sf = 18.0          # Scaling factor
        self.G_c = 1.9          # Gliding constant
        
        # ISSA Specific Constants
        self.w = 0.8            # Inertia weight
        self.c1 = 2.0           # Learning factor 1
        self.c2 = 2.0           # Learning factor 2
        self.m = self.max_iterations // 2  # "Early algebra" threshold (t < m)
        
        self.Sc = float('inf')  # Initialize Seasonal constant
        self.Smin = 1e-6        # Initialize Minimum seasonal constant
        
        # Precompute Levy flight sigma value
        self.beta = 1.5
        num = math.gamma(1 + self.beta) * math.sin(math.pi * self.beta / 2)
        den = math.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        self.sigma = (num / den) ** (1 / self.beta)

    def _get_dg(self):
        """Calculates aerodynamic gliding distance (d_g)"""
        C_L = np.random.uniform(0.675, 1.5)
        return (self.h_g * C_L / self.C_D) / self.sf 

    def update(self):
        self.iteration += 1

        # Evaluate fitness
        fitness = self.parallel_evaluate(self.squirrels)
        fitness_sorted_indices = np.argsort(fitness)
        self.squirrels = self.squirrels[fitness_sorted_indices]

        # Identify groups
        hickory_nut = self.squirrels[0]
        acorn_nuts = self.squirrels[1:4]
        
        cutoff = max(int(self.hickory_ratio * self.n), 4)

        # Check if the previous generation's seasonal condition was met
        condition_met = (self.Sc < self.Smin)

        # 1. Update squirrels on acorn trees moving towards the hickory tree
        for i in range(1, 4):
            r = np.random.rand()
            if self.iteration < self.m:
                # Early Stage (Eq 3)
                if r >= self.p_predator:
                    self.squirrels[i] = self.squirrels[i] + self._get_dg() * self.G_c * (hickory_nut - self.squirrels[i])
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)
            else:
                # Late Stage - New Update Condition (Eq 11)
                if r >= self.p_predator or condition_met:
                    self.squirrels[i] = self.squirrels[i] + self._get_dg() * self.G_c * (hickory_nut - self.squirrels[i])
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)

        # 2. Update normal squirrels moving towards the hickory tree
        for i in range(4, min(cutoff, self.n)):
            r = np.random.rand()
            target_acorn = acorn_nuts[np.random.randint(0, len(acorn_nuts))] if len(acorn_nuts) > 0 else hickory_nut
            
            if self.iteration < self.m:
                # Early Stage - PSO Strategy (Eq 10)
                if r >= self.p_predator:
                    term1 = self.w * self.squirrels[i]
                    term2 = self.c2 * np.random.rand() * self._get_dg() * self.G_c * (target_acorn - self.squirrels[i])
                    term3 = self.c1 * np.random.rand() * self._get_dg() * self.G_c * (hickory_nut - self.squirrels[i])
                    self.squirrels[i] = term1 + term2 + term3
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)
            else:
                # Late Stage - New Update Condition (Eq 13)
                if r >= self.p_predator or condition_met:
                    self.squirrels[i] = self.squirrels[i] + self._get_dg() * self.G_c * (hickory_nut - self.squirrels[i])
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)

        # 3. Update normal squirrels moving towards acorn trees
        for i in range(min(cutoff, self.n), self.n):
            target = acorn_nuts[np.random.randint(0, len(acorn_nuts))] if len(acorn_nuts) > 0 else hickory_nut
            r = np.random.rand()
            
            if self.iteration < self.m:
                # Early Stage - PSO Strategy (Eq 9)
                if r >= self.p_predator:
                    self.squirrels[i] = self.w * self.squirrels[i] + self.c1 * np.random.rand() * self._get_dg() * self.G_c * (target - self.squirrels[i])
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)
            else:
                # Late Stage - New Update Condition (Eq 12)
                if r >= self.p_predator or condition_met:
                    self.squirrels[i] = self.squirrels[i] + self._get_dg() * self.G_c * (target - self.squirrels[i])
                else:
                    self.squirrels[i] = np.random.uniform(self.val_min, self.val_max, self.d)

        # Seasonal Monitoring Condition (Eq 6)
        self.Sc = np.sum(np.sqrt(np.sum((self.squirrels[1:4] - self.squirrels[0]) ** 2, axis=1)))
        
        # Relocate using Levy flights if winter is over
        if self.Sc < self.Smin:
            for i in range(1, self.n):
                levy = self._get_levy()
                self.squirrels[i] = self.val_min + levy * (self.val_max - self.val_min)

        # Adaptive Scmin Strategy (Eq 14 & 15)
        if self.iteration < 100:
            self.Smin = 1e-6 / (365 ** (self.iteration / (self.max_iterations / 2.5)))
        else:
            II = self.iteration % 100
            if II == 0: II = 100  # Prevent division by zero boundary issues
            self.Smin = 1e-6 / (365 ** (II / (self.max_iterations / 2.5)))

        # Clip all squirrels to allowed bounds
        self.squirrels = np.clip(self.squirrels, self.val_min, self.val_max)
        self.best_solution = self.squirrels[0]
        
        print(f"\rImproved Squirrel Iteration: {self.iteration}/{self.max_iterations}            ", end="")

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations
    
    def run(self):
        super().run()
        self.best_solution = self.squirrels[0]
        return self.objective_function(self.squirrels[0])