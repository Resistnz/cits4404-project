from enum import Enum

# should put this in its own file lowkey but u get the idea
class Optimiser:
    def __init__(self):
        self.best_solution = list()

    # do a tick
    def update(self) -> None:
        ... # these are so cool who needs pass

    # e.g. error of prediction of price at end of day
    def objective_function(self, param1, param2) -> float: 
        return 0
    
    def termination_criteria_reached(self) -> bool:
        return False
    
    def run(self):
        while not self.termination_criteria_reached():
            self.update()
    
    # could have a specific Solution subclass that holds all the parameters if you want
    
# override stuff, makes it ez to plug in other ones
class SpecificOptimiser(Optimiser):
    ...


class Signal(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class TradingEngine:
    def __init__(self):
        print(" tell me what to do and ill do it")

    # main engine logic go in here
    # bro knows nothing other than the indicators it gets
    def detect_signal(self, indicator1, indicator2) -> Signal:
        # e.g. if indicator1 > indicator 2 then BUY pls

        return Signal.BUY
    
def main():

    # idk do stuff
    optimiser = SpecificOptimiser()
    engine = TradingEngine()

    while not optimiser.termination_criteria_reached():
        optimiser.update()

    # we would have a good solution here now, this is after its been trained and we tryna make money
    while True:
        signal = engine.detect_signal(*optimiser.best_solution)

        if signal == Signal.BUY:
            # simulate a buy
            ...
        elif signal == Signal.SELL:
            # simulate a sell
            ...

if __name__ == "__main__":
    main()