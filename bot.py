from engine import TradingEngine
from optimiser import GradientDescentOptimiser

def main():
    engine = TradingEngine()
    optimiser = GradientDescentOptimiser(engine)

    # train up the optimiser against some past data
    optimiser.run()

    # then we check it against new data

if __name__ == "__main__":
    main()