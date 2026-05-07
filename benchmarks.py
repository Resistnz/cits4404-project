import numpy as np

class Benchmarks:
    @staticmethod
    def _u(x, a, k, m):
        """
        Helper piecewise function u(x_i, a, k, m) used in F11 and F12.
        """
        res = np.zeros_like(x, dtype=float)
        
        # x_i > a
        mask_gt = x > a
        res[mask_gt] = k * (x[mask_gt] - a)**m
        
        # x_i < -a
        mask_lt = x < -a
        res[mask_lt] = k * (-x[mask_lt] - a)**m
        
        # -a <= x_i <= a remains 0
        return res

    @staticmethod
    def f7(weights):
        # Range: [-500, 500]
        return np.sum(-weights * np.sin(np.sqrt(np.abs(weights))))

    @staticmethod
    def f8(weights):
        # Range: [-5.12, 5.12]
        return np.sum(weights**2 - 10 * np.cos(2 * np.pi * weights) + 10)

    @staticmethod
    def f9(weights):
        # Range: [-32, 32]
        n = len(weights)
        term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(weights**2) / n))
        term2 = -np.exp(np.sum(np.cos(2 * np.pi * weights)) / n)
        return term1 + term2 + 20 + np.e

    @staticmethod
    def f10(weights):
        # Range: [-600, 600]
        indices = np.arange(1, len(weights) + 1)
        sum_part = np.sum(weights**2) / 4000.0
        prod_part = np.prod(np.cos(weights / np.sqrt(indices)))
        return sum_part - prod_part + 1

    @staticmethod
    def f11(weights):
        # Range: [-50, 50]
        n = len(weights)
        y = 1 + (weights + 1) / 4.0
        
        # Note: Transcribed exactly from the image image which shows 10*sin(\pi*y_1) 
        # without a square on the first term.
        term1 = 10 * np.sin(np.pi * y[0]) 
        term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
        term3 = (y[-1] - 1)**2
        
        main_part = (np.pi / n) * (term1 + term2 + term3)
        penalty_part = np.sum(Benchmarks._u(weights, 10, 100, 4))
        
        return main_part + penalty_part

    @staticmethod
    def f12(weights):
        # Range: [-50, 50]
        n = len(weights)
        x = weights
        
        term1 = np.sin(3 * np.pi * x[0])**2
        term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
        term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
        
        main_part = 0.1 * (term1 + term2 + term3)
        penalty_part = np.sum(Benchmarks._u(x, 5, 100, 4))
        
        return main_part + penalty_part


FUNCTION_BOUNDS = {
    "f7": (-500, 500),
    "f8": (-5.12, 5.12),
    "f9": (-32, 32),
    "f10": (-600, 600),
    "f11": (-50, 50),
    "f12": (-50, 50),
}

# Helper dictionary to pick functions by name
functions = {
    "f7": Benchmarks.f7,
    "f8": Benchmarks.f8,
    "f9": Benchmarks.f9,
    "f10": Benchmarks.f10,
    "f11": Benchmarks.f11,
    "f12": Benchmarks.f12
}


def get_function_bounds(function_key):
    return FUNCTION_BOUNDS[function_key]