from engine import TradingEngine
from optimiser import GradientDescentOptimiser

def main():
    # idk do stuff
    optimiser = GradientDescentOptimiser()
    engine = TradingEngine()

    while not optimiser.termination_criteria_reached():
        optimiser.update()


if __name__ == "__main__":
    main()