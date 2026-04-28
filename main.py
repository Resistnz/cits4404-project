from bots.bot import BasicBot
from algorithms.optimiser import GradientDescentOptimiser

def main():
    bot = BasicBot()
    #optimiser = GradientDescentOptimiser(dimensions=10, trading_bot=bot)

    # train up the optimiser against some past data
    #optimiser.run()
    bot.evaluate_parameters([])

    # then we check it against new data

if __name__ == "__main__":
    main()