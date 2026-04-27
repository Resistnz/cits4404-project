from enum import Enum
import numpy as np

class Signal(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class TradingEngine:
    def __init__(self):
        print(" tell me what to do and ill do it")

        self.price_history = []

    def sma(self, window=10):
        return np.mean(self.price_history[-window:]) if len(self.price_history) >= window else np.mean(self.price_history) if self.price_history else 0

    def ema(self, window=10):
        return 0

    # Use the given weights to predict the price at the end of a day
    def predict_end_of_day_price(self, weights):
        return 0
    
    # Use the weights to predict the price at the end of the day, and return the error
    # Lower error is better, i.e. minimiser
    def evaluate_parameters(self, weights):
        predicted_price = self.predict_end_of_day_price(weights)

        actual_price = 0 # idk

        return abs(predicted_price - actual_price)