# Import all Optimiser classes from algorithms/
from algorithms.optimiser import Optimiser
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser

optimisers = [Optimiser, GradientDescentOptimiser, FireflyOptimiser, ImprovedFireflyOptimiser]

