import argparse
import time
import numpy as np

from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser
from algorithms.gwo import GWOOptimiser
from algorithms.big_bang_big_crunch import BigBangBigCrunchOptimiser

STARTING_CAPITAL = 1000.0
HOLDOUT_START_INDEX = 1858
MAX_ITERATIONS = 50
POPULATION_SIZE = 60
SAMPLE_COUNT = 20
SEED = 80085

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
]

OPTIMISER_CONFIGS = [
    {
        "name": "GradientDescentOptimiser",
        "class": GradientDescentOptimiser,
        "factory": lambda bot, dims: GradientDescentOptimiser(
            dimensions=dims,
            step_size=0.1,
            sample_count=SAMPLE_COUNT,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=SEED,
        ),
    },
    {
        "name": "FireflyOptimiser",
        "class": FireflyOptimiser,
        "factory": lambda bot, dims: FireflyOptimiser(
            num_fireflies=POPULATION_SIZE,
            dimensions=dims,
            light_absorption=0.2,
            step_size=0.2,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
            seed=SEED,
        ),
    },
    # {
    #     "name": "ImprovedFireflyOptimiser",
    #     "class": ImprovedFireflyOptimiser,
    #     "factory": lambda bot, dims: ImprovedFireflyOptimiser(
    #         num_fireflies=POPULATION_SIZE,
    #         dimensions=dims,
    #         light_absorption=0.2,
    #         step_size=0.2,
    #         max_iterations=MAX_ITERATIONS,
    #         trading_bot=bot,
    #         val_min=-1,
    #         val_max=1,
    #         seed=SEED,
    #     ),
    # },
    {
        "name": "GWOOptimiser",
        "class": GWOOptimiser,
        "factory": lambda bot, dims: GWOOptimiser(
            num_wolves=POPULATION_SIZE,
            dimensions=dims,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
        ),
    },
    {
        "name": "BigBangBigCrunchOptimiser",
        "class": BigBangBigCrunchOptimiser,
        "factory": lambda bot, dims: BigBangBigCrunchOptimiser(
            dimensions=dims,
            population_size=POPULATION_SIZE,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
            val_min=-1,
            val_max=1,
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


def run_optimizer_for_bot(bot, optimizer_config):
    optimiser = optimizer_config["factory"](bot, bot.dimensions)
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
        **holdout,
    }


def run_all_benchmarks():
    results = []

    for bot_config in BOT_CONFIGS:
        bot = bot_config["class"]()
        bot.dimensions = bot_config["dimensions"]
        bot_name = bot_config["name"]

        print("\n" + "=" * 110)
        print(f"Benchmarking TradingBot: {bot_name}")
        print("=" * 110)
        print(
            f"Shared iteration budget: {MAX_ITERATIONS} | "
            f"Holdout start index: {HOLDOUT_START_INDEX} | "
            f"Starting capital: ${STARTING_CAPITAL:.2f}"
        )
        print(
            "Optimiser                        | Iter | Runtime   | Objective   | Profit    | Return %  | Variance"
            "\n" + "-" * 110
        )

        for optimizer_config in OPTIMISER_CONFIGS:
            summary = run_optimizer_for_bot(bot, optimizer_config)
            results.append({"bot": bot_name, **summary})
            print(
                f"{summary['optimiser']:30} | "
                f"{summary['iterations']:4d} | "
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
    return parser.parse_args()


def main():
    args = parse_args()
    global MAX_ITERATIONS
    MAX_ITERATIONS = args.iterations

    run_all_benchmarks()


if __name__ == "__main__":
    main()

