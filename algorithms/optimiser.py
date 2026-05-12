import numpy as np
from algorithms.benchmarks import Benchmarks
from concurrent.futures import ProcessPoolExecutor
import os

# --- Module-level worker state for multiprocessing ---
_worker_bot = None

def _init_worker(bot_class_name, bot_module, price_history, train_end, eval_mode):
    """Called once per worker process to create a local bot instance."""
    global _worker_bot
    import importlib
    mod = importlib.import_module(bot_module)
    bot_class = getattr(mod, bot_class_name)
    
    # Properly instantiate the bot so __init__ is called
    _worker_bot = bot_class(eval_mode=eval_mode)
    
    # Overwrite price history with the shared data to be safe
    _worker_bot.price_history = price_history
    _worker_bot.P = price_history[:train_end]

def _evaluate_worker(weights):
    """Evaluate a single weight vector using the worker-local bot."""
    return _worker_bot.evaluate_parameters(weights)


class Optimiser:
    def __init__(self, max_iterations=1000, trading_bot=None, val_min=-1, val_max=1):
        self.best_solution = list()
        self.trading_bot = trading_bot

        self.val_min = val_min
        self.val_max = val_max

        self.iteration = 0
        self.max_iterations = max_iterations

        self._pool = None

    def _get_pool(self):
        """Lazily create a process pool, reused across iterations."""
        if self._pool is None and self.trading_bot is not None:
            bot = self.trading_bot
            bot_class_name = type(bot).__name__
            bot_module = type(bot).__module__
            num_workers = max(1, os.cpu_count() - 1)
            self._pool = ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(bot_class_name, bot_module, bot.price_history, 1858, bot.eval_mode),
            )
        return self._pool

    def parallel_evaluate(self, population):
        """Evaluate a batch of weight vectors in parallel across worker processes."""
        pool = self._get_pool()
        if pool is not None:
            results = list(pool.map(_evaluate_worker, [w for w in population]))
            return np.array(results)
        # Fallback to sequential if no pool available
        return np.array([self.objective_function(x) for x in population])

    def shutdown_pool(self):
        """Clean up the process pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    # do a tick
    def update(self) -> None: ...  # these are so cool who needs pass

    def objective_function(self, values) -> float:
        return self.trading_bot.evaluate_parameters(values)

    def termination_criteria_reached(self) -> bool:
        return self.iteration >= self.max_iterations

    def run(self) -> None:
        try:
            while not self.termination_criteria_reached():
                self.update()

                print(f"Iteration {self.iteration} - Objective value: {self.objective_function(self.best_solution)}", end='\r')
        finally:
            self.shutdown_pool()
