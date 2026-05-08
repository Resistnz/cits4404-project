from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser

def main():
    bot = BetterBot() # This can be any TradingBot child class
    optimiser = FireflyOptimiser( # This can be any Optimiser child class
        dimensions=7, 
        trading_bot=bot, 
        max_iterations=50, 
        step_size=0.2,
        val_min=-1,
        val_max=1,
        num_fireflies=60,
        light_absorption=0.2
        #seed=8008135
        )

    # Train up the optimiser against some past data
    optimiser.run()

    # Check our best solution
    print()
    best_transformed = bot.transform_weights(optimiser.best_solution)
    print(f"Best solution found: {best_transformed} with objective value: {optimiser.objective_function(optimiser.best_solution)}")

    usd, _ = bot.run_on_period(best_transformed, bot.price_history[1858:]) # Run on time after 2020 (holdout testing)
    print(f"We ended with: ${usd}!")

    # Graph it
    # This is the graph of the post 2020 data (unseen). If our bot makes money here, it has done very well :)
    bot.generate_signals(best_transformed, graph=True)


if __name__ == "__main__":
    main()