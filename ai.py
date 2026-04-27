# gemini made these to test it 
# confirmed that LWFA is ass and doesn't work

import numpy as np
import time

# ==========================================
# 1. Benchmark Functions (Yu Li et al. 2021)
# ==========================================

def sphere(x):
    """Unimodal. Optimum: 0 at x=(0,...,0)"""
    return np.sum(x**2)

def shifted_sphere(x):
    """Unimodal. Optimum: 0 at x=(0.5,...,0.5)"""
    return np.sum((x - 20)**2)

def rastrigin(x):
    """Multimodal with regular local optima. Optimum: 0 at x=(0,...,0)"""
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

def ackley(x):
    """Multimodal with a large flat outer region. Optimum: 0 at x=(0,...,0)"""
    d = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e

def griewank(x):
    """Multimodal with interlinked variables. Optimum: 0 at x=(0,...,0)"""
    sum_sq = np.sum(x**2) / 4000.0
    i_vals = np.arange(1, len(x) + 1)
    prod_cos = np.prod(np.cos(x / np.sqrt(i_vals)))
    return sum_sq - prod_cos + 1.0

# ==========================================
# 2. Initialization & Core Routines
# ==========================================

def init_firefly_population(n_fireflies, dim, bounds, obj_func):
    lb, ub = bounds[0], bounds[1]
    positions = np.random.uniform(lb, ub, (n_fireflies, dim))
    fitness = np.apply_along_axis(obj_func, 1, positions)
    return positions, fitness

def run_standard_fa(obj_func, dim, bounds, n_fireflies=40, max_iter=200, alpha=0.2, beta0=1.0, gamma=1.0):
    positions, fitness = init_firefly_population(n_fireflies, dim, bounds, obj_func)
    lb, ub = bounds[0], bounds[1]
    scale = ub - lb
    
    for _ in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(positions[i] - positions[j])
                    beta = beta0 * np.exp(-gamma * r**2)
                    rand_step = alpha * (np.random.rand(dim) - 0.5) * scale
                    
                    positions[i] = positions[i] + beta * (positions[j] - positions[i]) + rand_step
                    positions[i] = np.clip(positions[i], lb, ub)
                    fitness[i] = obj_func(positions[i])
                    
        # Global best moves randomly
        best_idx = np.argmin(fitness)
        rand_step = alpha * (np.random.rand(dim) - 0.5) * scale
        positions[best_idx] = positions[best_idx] + rand_step
        positions[best_idx] = np.clip(positions[best_idx], lb, ub)
        fitness[best_idx] = obj_func(positions[best_idx])
        
    return np.min(fitness)

def run_lwfa(obj_func, dim, bounds, n_fireflies=40, max_iter=200, 
             alpha0=0.5, delta=0.97, beta0=1.0, beta_min=0.2, gamma=1.0, w_max=0.9, w_min=0.4):
    positions, fitness = init_firefly_population(n_fireflies, dim, bounds, obj_func)
    lb, ub = bounds[0], bounds[1]
    scale = ub - lb
    
    for t in range(max_iter):
        # Dynamic step-size
        alpha_t = alpha0 * (delta ** t)
        
        # Self-adaptive logarithmic inertia weight
        progress = t / max_iter
        w_t = w_max - (w_max - w_min) * np.log(1 + (np.e - 1) * progress)
        
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] < fitness[i]:
                    r = np.linalg.norm(positions[i] - positions[j])
                    # Minimum attractiveness limit
                    beta_t = beta_min + (beta0 - beta_min) * np.exp(-gamma * r**2)
                    
                    rand_step = alpha_t * (np.random.rand(dim) - 0.5) * scale
                    positions[i] = w_t * positions[i] + beta_t * (positions[j] - positions[i]) + rand_step
                    positions[i] = np.clip(positions[i], lb, ub)
                    fitness[i] = obj_func(positions[i])
                    
        # Global best moves randomly with dynamic step-size
        best_idx = np.argmin(fitness)
        rand_step = alpha_t * (np.random.rand(dim) - 0.5) * scale
        positions[best_idx] = positions[best_idx] + rand_step
        positions[best_idx] = np.clip(positions[best_idx], lb, ub)
        fitness[best_idx] = obj_func(positions[best_idx])
        
    return np.min(fitness)

# ==========================================
# 3. Execution & Benchmarking Suite
# ==========================================

def execute_benchmark():
    # Common hyperparameters matching metaheuristic literature benchmarks
    dim = 100
    n_fireflies = 30
    max_iter = 1000
    independent_runs = 3
    
    test_suite = [
        # {"name": "Sphere", "func": sphere, "bounds": (-100, 100)},
        # {"name": "Rastrigin", "func": rastrigin, "bounds": (-5.12, 5.12)},
        # {"name": "Ackley", "func": ackley, "bounds": (-32, 32)},
        # {"name": "Griewank", "func": griewank, "bounds": (-600, 600)},
        {"name": "Shifted Sphere", "func": shifted_sphere, "bounds": (-100, 100)}
    ]
    
    print(f"Starting Benchmark: {independent_runs} runs, {dim} Dimensions, {max_iter} Iterations\n")
    print("-" * 75)
    print(f"{'Function':<12} | {'Algorithm':<10} | {'Mean Best Fitness':<20} | {'Std Dev':<15}")
    print("-" * 75)
    
    for test in test_suite:
        func = test["func"]
        bounds = test["bounds"]
        name = test["name"]
        
        # Array to hold the best fitness found in each independent run
        fa_results = np.zeros(independent_runs)
        lwfa_results = np.zeros(independent_runs)
        
        for run in range(independent_runs):
            # Print progress dynamically on the same line
            print(f"\rEvaluating {name}: Trial {run + 1}/{independent_runs}...", end="", flush=True)
            
            # Run Standard FA
            fa_results[run] = run_standard_fa(
                func, dim, bounds, n_fireflies, max_iter
            )
            # Run Improved LWFA
            lwfa_results[run] = run_lwfa(
                func, dim, bounds, n_fireflies, max_iter
            )
            
        # Clear the progress line before printing the final results
        print("\r" + " " * 60 + "\r", end="", flush=True)
            
        # Standard FA stats
        fa_mean = np.mean(fa_results)
        fa_std = np.std(fa_results)
        fa_best = np.min(fa_results)
        
        # LWFA stats
        lwfa_mean = np.mean(lwfa_results)
        lwfa_std = np.std(lwfa_results)
        lwfa_best = np.min(lwfa_results)
        
        print(f"{name:<12} | {'Std FA':<10} | {fa_mean:<20.4e} | {fa_std:<15.4e}")
        print(f"{'':<12} | {'LWFA':<10} | {lwfa_mean:<20.4e} | {lwfa_std:<15.4e}")
        print(f"{'':<12} | {'Best':<10} | {fa_best:<20.4e} | {lwfa_best:<15.4e}")
        print("-" * 75)

if __name__ == "__main__":
    # Ensure reproducibility for benchmark comparisons
    np.random.seed(42) 
    start_time = time.time()
    execute_benchmark()
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f} seconds")