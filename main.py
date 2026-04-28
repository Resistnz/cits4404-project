from bots.basic_bot import BasicBot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser

def main():
    bot = BasicBot() # This can be any TradingBot child class
    optimiser = FireflyOptimiser( # This can be any Optimiser child class
        dimensions=2, 
        trading_bot=bot, 
        max_iterations=500, 
        step_size=5,
        val_min=1,
        val_max=300,
        num_fireflies=30
        )

    # Train up the optimiser against some past data
    optimiser.run()

    # Check our best solution
    print(f"Best solution found: {optimiser.best_solution}")

    usd = bot.run(optimiser.best_solution)
    print(f"We ended with: ${usd}!")

    # Graph it
    bot.generate_signals(optimiser.best_solution, graph=True)

if __name__ == "__main__":
    main()