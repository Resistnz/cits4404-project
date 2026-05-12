from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from bots.macd_bot import MACDBot
from bots.breakout import BreakoutBot
from bots.mean_reversion import ZScoreBot
from bots.reverse_macd_bot import ReverseMACDBot
from bots.triple_sma_bot import TripleSMABot

from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser
from algorithms.gwo import GWOOptimiser
from algorithms.big_bang_big_crunch import BigBangBigCrunchOptimiser
from algorithms.bruteforce import BruteForce
import numpy as np

def main():
    bot = ReverseMACDBot(eval_mode="profit") # This can be any TradingBot child class
    optimiser = GWOOptimiser    ( # This can be any Optimiser child class
        dimensions=6,
        trading_bot=bot, 
        max_iterations=50, 
        val_min=-1,
        val_max=1,
        #step_size=0.1,
        #step_size=(1/50)
        num_wolves=30
        # step_size=0.4,
        # num_fireflies=30,
        # light_absorption=0.4
        #seed=8008135
        )

    # Train up the optimiser against some past data
    optimiser.run()

    # Check our best solution
    test_particular_solution(bot, optimiser.best_solution)

def test_particular_solution(bot, solution):
    print()
    best_transformed = bot.transform_weights(solution)

    print(f"Best solution found: {[float(round(x, 6)) for x in best_transformed]} with objective value: {bot.evaluate_parameters(solution)}")
    print(f"Raw best solution: {[float(round(x, 6)) for x in solution]}")

    usd, _, _ = bot.run_on_period(best_transformed, bot.price_history[1858:]) # Run on time after 2020 (holdout testing)
    print(f"We ended with: ${usd} on post 2020 data!")

    # Graph it
    # This is the graph of the post 2020 data (unseen). If our bot makes money here, it has done very well :)
    bot.generate_signals(best_transformed, graph=True)

if __name__ == "__main__":
    main()

"""
For Basic Bot:
Best solution found: [11.0, 39.0] with objective value: -888.7426911231121
Raw best solution: [-0.56, 0.56] ([5.0, 0, 0, 0.0526, 0, 0, 0, 5.0, 0, 0, 0.3714, 0, 0, 0] for BetterBot)
We ended with: $4513.577454938214! (on post 2020 data)

    with different eval function:
    Best solution found: [13.0, 45.0] with objective value: -11103.943853149773
    Raw best solution: [-0.5, 0.8]
    We ended with: $5596.672507662535 on post 2020 data!

For Better Bot:
Best solution found: [-0.67944, -0.687918, -0.521329, 12.0, 14.0, 11.0, 0.670732, -0.604257, 0.212866, -0.492989, 44.0, 24.0, 43.0, 0.324713] with objective value: -0.020392201829341015
Raw best solution: [-0.828073, -0.843993, -0.578162, 0.179095, 0.415353, 0.062571, 0.670732, -0.699826, 0.216172, -0.540002, 0.679143, -0.453247, 0.643049, -0.732196]
We ended with: $5628.710705540577 on post 2020 data!

buying only twice
Best solution found: [-0.45871, 0.249038, 0.196701, 9.0, 9.0, 15.0, 0.134395, -0.029276, -0.27782, -0.004221, 32.0, 33.0, 42.0, 0.646578] with objective value: -29820.42906921856
Raw best solution: [-0.495676, 0.254387, 0.199299, -0.113238, -0.054222, 0.474071, 0.134395, -0.029284, -0.285318, -0.004221, 0.020811, 0.08231, 0.582781, 0.604032]
We ended with: $6989.8795200536715 on post 2020 data!

crazy overfit:
[0.586068, 0.167405, 0.061332, 0.756782, -0.855786, 0.076329, 0.888863, -0.727061, 0.438095, -0.035268, -0.639324, -0.101507, -0.006211, -0.534552] (Objective value: -0.0613)
"""