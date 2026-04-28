from bots.bot import TradingBot, BasicBot
from algorithms.gradient_descent import GradientDescentOptimiser

def main():
    bot = BasicBot()
    optimiser = GradientDescentOptimiser(
        dimensions=2, 
        trading_bot=bot, 
        max_iterations=1000, 
        step_size=1,
        val_min=3,
        val_max=40
        )

    # train up the optimiser against some past data
    optimiser.run()

    # then we run our best one
    print(f"Best solution found: {optimiser.best_solution}")

    usd = bot.run(optimiser.best_solution)
    print(f"We ended with: ${usd}!")

if __name__ == "__main__":
    main()