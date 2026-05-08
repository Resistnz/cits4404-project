import numpy as np
from bots.bot import TradingBot, Signal

# The basic bot from p. 11 in the project outline
class BasicBot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)

        MIN_WINDOW_SIZE = 1
        MAX_WINDOW_SIZE = 50

        new_weights[0] = int((weights[0] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d1 from -1,1 to 1,50
        new_weights[1] = int((weights[1] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d2

        return new_weights

    # [d1, d2]
    def generate_signals(self, weights, graph=False):
        weights = self.transform_weights(weights)
        smaA = self.wma(self.P, int(weights[0]), self.sma_filter(int(weights[0])))
        smaB = self.wma(self.P, int(weights[1]), self.sma_filter(int(weights[1])))

        sma_diff = smaA - smaB
        sign_diff = np.sign(sma_diff)

        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(sign_diff, kernel, mode='valid')

        buy_signal_aligned = np.sign(buy_signal)  # normalize to -1,0,1

        # Prepend a HOLD so signals align one-for-one with self.P
        signals = [Signal.HOLD] + [Signal(int(x)) for x in buy_signal_aligned]

        # Graph it if you want (convert signals to numeric array)
        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals