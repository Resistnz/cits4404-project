from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser

def main():
    bot = BasicBot() # This can be any TradingBot child class
    optimiser = GradientDescentOptimiser( # This can be any Optimiser child class
        dimensions=2, 
        trading_bot=bot, 
        max_iterations=100, 
        step_size=0.01,
        val_min=-1,
        val_max=1,
        # num_fireflies=30,
        # light_absorption=0.2
        #seed=8008135
        )

    # Train up the optimiser against some past data
    optimiser.run()

    # Check our best solution
    print()
    print(f"Best solution found: {optimiser.best_solution}")

    usd, _ = bot.run_on_period(optimiser.best_solution, bot.price_history[1858:]) # Run on time after 2020 (holdout testing)
    print(f"We ended with: ${usd}!")

    # Graph it
   # bot.generate_signals(optimiser.best_solution, graph=True)


if __name__ == "__main__":
    main()