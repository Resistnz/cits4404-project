from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from algorithms.gradient_descent import GradientDescentOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser
from algorithms.gwo import GWOOptimiser
from algorithms.big_bang_big_crunch import BigBangBigCrunchOptimiser
from algorithms.bruteforce import BruteForce
import numpy as np

def main():
    bot = BetterBot() # This can be any TradingBot child class
    optimiser = FireflyOptimiser( # This can be any Optimiser child class
        dimensions=14, 
        trading_bot=bot, 
        max_iterations=50, 
        val_min=-1,
        val_max=1,
        #step_size=(1/50)
        #num_wolves=30
        step_size=0.4,
        num_fireflies=200,
        light_absorption=0.2
        #seed=8008135
        )

    # Train up the optimiser against some past data
    optimiser.run()

    # Check our best solution
    test_particular_solution(bot, optimiser.best_solution)

# [np.float64(-0.746707890411129), np.float64(-0.3894777707411653), np.float64(-0.4544590197176597), 2, 3, 5, np.float64(0.49780712473668337)]
def test_particular_solution(bot, solution):
    print()
    best_transformed = bot.transform_weights(solution)

    print(f"Best solution found: {[float(round(x, 6)) for x in best_transformed]} with objective value: {bot.evaluate_parameters(solution)}")
    print(f"Raw best solution: {[float(round(x, 6)) for x in solution]}")

    usd, _ = bot.run_on_period(best_transformed, bot.price_history[1858:]) # Run on time after 2020 (holdout testing)
    print(f"We ended with: ${usd} on post 2020 data!")

    # Graph it
    # This is the graph of the post 2020 data (unseen). If our bot makes money here, it has done very well :)
    bot.generate_signals(best_transformed, graph=True)

if __name__ == "__main__":
    #test_particular_solution(BetterBot(), [0.979046, -0.980954, 1.0, -0.723163, 0.406437, 0.226748, 0.38846, -0.748922, -0.157238, 0.295489, -0.245239, 1.0, -0.869922, 0.620778])
    main()

"""
Best solution found: [11.0, 39.0] with objective value: -888.7426911231121
Raw best solution: [-0.56, 0.56]
We ended with: $4513.577454938214! (on post 2020 data)
"""