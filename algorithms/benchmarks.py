# This was AI generated to match the 10 functions used in the paper

import numpy as np


class Benchmarks:
    @staticmethod
    def f1_schaffer(weights):
        # Interval: [-10, 10]
        sum_sq = np.sum(np.square(weights))
        numerator = np.square(np.sin(np.sqrt(sum_sq))) - 0.5
        denominator = np.square(1 + 0.001 * sum_sq)
        return 0.5 + numerator / denominator

    @staticmethod
    def f2_sphere(weights):
        # Interval: [-100, 100]
        return np.sum(np.square(weights))

    @staticmethod
    def f3_rastrigin(weights):
        # Interval: [-5.12, 5.12]
        d = len(weights)
        return 10 * d + np.sum(weights**2 - 10 * np.cos(2 * np.pi * weights))

    @staticmethod
    def f4_griewank(weights):
        # Interval: [-100, 100]
        indices = np.arange(1, len(weights) + 1)
        sum_part = np.sum(np.square(weights)) / 4000
        prod_part = np.prod(np.cos(weights / np.sqrt(indices)))
        return sum_part - prod_part + 1

    @staticmethod
    def f5_ackley(weights):
        # Interval: [-35, 35]
        d = len(weights)
        sum_sq = np.sum(np.square(weights))
        sum_cos = np.sum(np.cos(2 * np.pi * weights))
        term1 = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return term1 + term2 + 20 + np.e

    @staticmethod
    def f6_sum_squares(weights):
        # Interval: [-10, 10]
        indices = np.arange(1, len(weights) + 1)
        return np.sum(indices * np.square(weights))

    @staticmethod
    def f7_zakharov(weights):
        # Interval: [-5, 10]
        indices = np.arange(1, len(weights) + 1)
        sum1 = np.sum(np.square(weights))
        sum2 = np.sum(0.5 * indices * weights)
        return sum1 + sum2**2 + sum2**4

    @staticmethod
    def f8_schwefel_1_2(weights):
        # Interval: [-10, 10]
        # This is the sum of squared cumulative sums
        return np.sum([np.sum(weights[: i + 1]) ** 2 for i in range(len(weights))])

    @staticmethod
    def f9_schwefel_2_21(weights):
        # Interval: [-100, 100]
        return np.max(np.abs(weights))

    @staticmethod
    def f10_schwefel_2_22(weights):
        # Interval: [-10, 10]
        abs_w = np.abs(weights)
        return np.sum(abs_w) + np.prod(abs_w)

    @staticmethod
    def f11_rosenbrock(weights):
        # Search Range: [-5, 5] (usually)
        # Global Optimum at (1, 1, ..., 1)
        term1 = 100.0 * np.square(weights[1:] - np.square(weights[:-1]))
        term2 = np.square(1.0 - weights[:-1])
        return np.sum(term1 + term2)

    @staticmethod
    def f12_shifted_sphere(weights):
        # Target is now (0.5, 0.5, ..., 0.5)
        return np.sum(np.square(weights - 0.5))


# Helper dictionary to pick functions by name/index
functions = {
    "f1": Benchmarks.f1_schaffer,
    "f2": Benchmarks.f2_sphere,
    "f3": Benchmarks.f3_rastrigin,
    "f4": Benchmarks.f4_griewank,
    "f5": Benchmarks.f5_ackley,
    "f6": Benchmarks.f6_sum_squares,
    "f7": Benchmarks.f7_zakharov,
    "f8": Benchmarks.f8_schwefel_1_2,
    "f9": Benchmarks.f9_schwefel_2_21,
    "f10": Benchmarks.f10_schwefel_2_22,
    "f11": Benchmarks.f11_rosenbrock,
    "f12": Benchmarks.f12_shifted_sphere,
}