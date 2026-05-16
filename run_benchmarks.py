import argparse
import time
import numpy as np
import csv
from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from bots.macd_bot import MACDBot
from bots.breakout import BreakoutBot
from bots.triple_sma_bot import TripleSMABot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser
from algorithms.gwo import GWOOptimiser
from algorithms.big_bang_big_crunch import BigBangBigCrunchOptimiser
from algorithms.squirrel import SquirrelOptimiser, CorrectSquirrelOptimiser, ImprovedSquirrelOptimiser

STARTING_CAPITAL = 1000.0
HOLDOUT_START_INDEX = 1858
MAX_ITERATIONS = 50
POPULATION_SIZE = 60
SAMPLE_COUNT = 20
SEED = 8008135

BOT_CONFIGS = [
    {
        "name": "BasicBot",
        "class": BasicBot,
        "dimensions": 2,
    },
    {
        "name": "BetterBot",
        "class": BetterBot,
        "dimensions": 14,
    },
    {
        "name": "MACDBot",
        "class": MACDBot,
        "dimensions": 3,
    },
    {
        "name": "BreakoutBot",
        "class": BreakoutBot,
        "dimensions": 2,
    },
    {
        "name": "TripleSMABot",
        "class": TripleSMABot,
        "dimensions": 3,
    },
]

OPTIMISER_CONFIGS = [
    {
        "name": "GradientDescentOptimiser",
        "class": GradientDescentOptimiser,
        "factory": lambda bot, dims, seed: GradientDescentOptimiser(
            dimensions=dims,
            step_size=0.1,
            sample_count=SAMPLE_COUNT,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "FireflyOptimiser",
        "class": FireflyOptimiser,
        "factory": lambda bot, dims, seed: FireflyOptimiser(
            num_fireflies=POPULATION_SIZE,
            dimensions=dims,
            light_absorption=0.2,
            step_size=0.2,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "ImprovedFireflyOptimiser",
        "class": ImprovedFireflyOptimiser,
        "factory": lambda bot, dims, seed: ImprovedFireflyOptimiser(
            num_fireflies=POPULATION_SIZE,
            dimensions=dims,
            light_absorption=0.2,
            step_size=0.2,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "GWOOptimiser",
        "class": GWOOptimiser,
        "factory": lambda bot, dims, seed: GWOOptimiser(
            num_wolves=POPULATION_SIZE,
            dimensions=dims,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "BigBangBigCrunchOptimiser",
        "class": BigBangBigCrunchOptimiser,
        "factory": lambda bot, dims, seed: BigBangBigCrunchOptimiser(
            dimensions=dims,
            population_size=POPULATION_SIZE,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
        ),
    },
    {
        "name": "SquirrelOptimiser",
        "class": SquirrelOptimiser,
        "factory": lambda bot, dims, seed: SquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "CorrectSquirrelOptimiser",
        "class": CorrectSquirrelOptimiser,
        "factory": lambda bot, dims, seed: CorrectSquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
    {
        "name": "ImprovedSquirrelOptimiser",
        "class": ImprovedSquirrelOptimiser,
        "factory": lambda bot, dims, seed: ImprovedSquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=seed,
        ),
    },
]


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


def calculate_returns_variance(portfolio_values: np.ndarray) -> float:
    if len(portfolio_values) < 2:
        return 0.0
    returns = np.diff(portfolio_values) / np.maximum(portfolio_values[:-1], 1e-8)
    return float(np.var(returns))


def evaluate_holdout_performance(bot, weights):
    final_balance, portfolio_values, _ = bot.run_on_period(weights, bot.price_history[HOLDOUT_START_INDEX:])
    profit = final_balance - STARTING_CAPITAL
    variance = calculate_returns_variance(np.asarray(portfolio_values, dtype=float))
    return {
        "final_balance": float(final_balance),
        "profit": float(profit),
        "variance": variance,
        "return_percentage": float(profit / STARTING_CAPITAL * 100.0),
    }


def run_optimizer_for_bot(bot, optimizer_config, seed):
    np.random.seed(seed)
    optimiser = optimizer_config["factory"](bot, bot.dimensions, seed)
    start_time = time.perf_counter()
    optimiser.run()
    elapsed = time.perf_counter() - start_time

    best_solution = np.asarray(optimiser.best_solution, dtype=float)
    transformed_solution = bot.transform_weights(best_solution)
    objective_value = float(optimiser.objective_function(best_solution))
    holdout = evaluate_holdout_performance(bot, transformed_solution)

    return {
        "optimiser": optimizer_config["name"],
        "iterations": MAX_ITERATIONS,
        "runtime_seconds": elapsed,
        "objective_value": objective_value,
        "raw_weights": str(list(best_solution)).replace(',', ';'),
        **holdout,
    }


def run_all_benchmarks(num_runs=3, output_file="benchmark_results.csv"):
    results = []

    csv_columns = [
        "bot", "eval_mode", "optimiser", "run_index", "iterations", 
        "runtime_seconds", "objective_value", "final_balance", 
        "profit", "variance", "return_percentage", "weights"
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

        for bot_config in BOT_CONFIGS:
            for eval_mode in ["profit", "log_excess", "drawdown"]:
                bot_name = bot_config["name"]

                print("\n" + "=" * 110)
                print(f"Benchmarking TradingBot: {bot_name} | Eval Mode: {eval_mode}")
                print("=" * 110)
                print(
                    f"Shared iteration budget: {MAX_ITERATIONS} | "
                    f"Holdout start index: {HOLDOUT_START_INDEX} | "
                    f"Starting capital: ${STARTING_CAPITAL:.2f} | "
                    f"Runs: {num_runs}"
                )
                print(
                    "Optimiser                        | Run | Runtime   | Objective   | Profit    | Return %  | Variance"
                    "\n" + "-" * 110
                )

                for optimizer_config in OPTIMISER_CONFIGS:
                    for run_idx in range(num_runs):
                        bot = bot_config["class"](eval_mode=eval_mode)
                        bot.dimensions = bot_config["dimensions"]
                        seed = SEED + run_idx * 1000  # Change seed per run
                        
                        summary = run_optimizer_for_bot(bot, optimizer_config, seed)
                        
                        row = {
                            "bot": bot_name,
                            "eval_mode": eval_mode,
                            "run_index": run_idx + 1,
                            **summary
                        }
                        results.append(row)
                        writer.writerow(row)
                        csvfile.flush() # flush to see progress if stopped
                        
                        print(
                            f"{summary['optimiser']:30} | "
                            f"{run_idx+1:3d} | "
                            f"{format_seconds(summary['runtime_seconds']):8} | "
                            f"{summary['objective_value']:10.4f} | "
                            f"{summary['profit']:9.2f} | "
                            f"{summary['return_percentage']:8.2f}% | "
                            f"{summary['variance']:9.6f}"
                        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run all optimisers against all TradingBot implementations.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Number of optimisation iterations for every optimiser.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs per configuration for statistical accuracy.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.csv",
        help="Output CSV file for the results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global MAX_ITERATIONS
    MAX_ITERATIONS = args.iterations

    run_all_benchmarks(num_runs=args.runs, output_file=args.output)


if __name__ == "__main__":
    main()

