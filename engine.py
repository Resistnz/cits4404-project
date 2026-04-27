from enum import Enum

class Signal(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class TradingEngine:
    def __init__(self):
        print(" tell me what to do and ill do it")

    # Use the given weights to predict the price at the end of a day
    @staticmethod
    def predict_end_of_day_price(weights):
        return 0

    # main engine logic go in here
    # bro knows nothing other than the indicators it gets
    def run(self, indicator1, indicator2) -> Signal:
        # e.g. if indicator1 > indicator 2 then BUY pls

        return Signal.BUY