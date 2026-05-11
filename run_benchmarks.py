import argparse
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
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
SEED = 80085

BOT_CONFIGS = [
    #{
    #    "name": "BasicBot",
    #    "class": BasicBot,
    #    "dimensions": 2,
    #},
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
    {
        "name": "SquirrelOptimiser",
        "class": SquirrelOptimiser,
        "factory": lambda bot, dims: SquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            p_predator=0.1,
            range_min=-1,
            range_max=1,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
        ),
    },
    {
        "name": "CorrectSquirrelOptimiser",
        "class": CorrectSquirrelOptimiser,
        "factory": lambda bot, dims: CorrectSquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            p_predator=0.1,
            range_min=-1,
            range_max=1,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
        ),
    },
    {
        "name": "ImprovedSquirrelOptimiser",
        "class": ImprovedSquirrelOptimiser,
        "factory": lambda bot, dims: ImprovedSquirrelOptimiser(
            num_squirrels=POPULATION_SIZE,
            dimensions=dims,
            p_predator=0.1,
            range_min=-1,
            range_max=1,
            max_iterations=MAX_ITERATIONS,
            trading_bot=bot,
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


def run_all_benchmarks(run_count=1, base_seed=SEED):
    results = []
    if run_count < 1:
        return results

    if run_count == 1:
        seeds = [base_seed]
    else:
        rng = np.random.default_rng(base_seed)
        seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=run_count)]

    for run_index, seed in enumerate(seeds, start=1):
        global SEED
        SEED = int(seed)
        np.random.seed(SEED)

        for bot_config in BOT_CONFIGS:
            bot = bot_config["class"]()
            bot.dimensions = bot_config["dimensions"]
            bot_name = bot_config["name"]

            print("\n" + "=" * 110)
            if run_count == 1:
                print(f"Benchmarking TradingBot: {bot_name}")
            else:
                print(f"Benchmarking TradingBot: {bot_name} (Run {run_index}/{run_count}, Seed {SEED})")
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
                results.append({"bot": bot_name, "run": run_index, "seed": SEED, **summary})
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


def plot_box_whisker(results, metric, ylabel, title):
    """Draw a grouped box-and-whisker plot of `metric` per optimiser per bot."""
    bots = sorted(set(r["bot"] for r in results))
    bot_count = len(bots)
    optimisers = sorted(set(r["optimiser"] for r in results))

    fig, axes = plt.subplots(1, bot_count, figsize=(7 * bot_count, 6), sharey=False)
    if bot_count == 1:
        axes = [axes]

    for ax, bot_name in zip(axes, bots):
        bot_results = [r for r in results if r["bot"] == bot_name]
        data = [[r[metric] for r in bot_results if r["optimiser"] == o] for o in optimisers]

        bp = ax.boxplot(data, labels=optimisers, patch_artist=True)
        ax.set_title(f"{bot_name}")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)

        # colour boxes
        colors = plt.cm.tab10(np.linspace(0, 1, len(optimisers)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


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
        default=1,
        help="Number of random-seed runs to execute. Use 1 for a single run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base seed used for a single run or to generate multiple sample seeds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global MAX_ITERATIONS
    MAX_ITERATIONS = args.iterations

    results = run_all_benchmarks(run_count=args.runs, base_seed=args.seed)

    if args.runs > 1 and results:
        # Write CSV
        csv_path = "benchmark_reports/benchmark_results.csv"
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n  Saved {csv_path} ({len(results)} rows)")

        # Plot
        print("Generating box-and-whisker plot...")

        fig = plot_box_whisker(
            results,
            metric="profit",
            ylabel="Profit ($)",
            title="Profit Distribution Across Runs",
        )
        fig.savefig("benchmark_profit_boxplot.png", dpi=150)
        print("  Saved benchmark_profit_boxplot.png")
        plt.show()


if __name__ == "__main__":
    main()

