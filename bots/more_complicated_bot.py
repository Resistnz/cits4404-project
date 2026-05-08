import numpy as np
from bots.bot import TradingBot, Signal

# The better bot from p. 13 in the project outline
class BetterBot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)

        # Transform weights to [-1,1] (allow negative for more flexibility)
        new_weights[0] = np.tanh(weights[0])  # w1
        new_weights[1] = np.tanh(weights[1])  # w2  
        new_weights[2] = np.tanh(weights[2])  # w3
        new_weights[7] = np.tanh(weights[7])  # w4
        new_weights[8] = np.tanh(weights[8])  # w5  
        new_weights[9] = np.tanh(weights[9])  # w6

        # High frequency window
        MIN_WINDOW_SIZE = 1
        MAX_WINDOW_SIZE = 20

        new_weights[3] = int((weights[3] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d1 from -1,1 to 1,50
        new_weights[4] = int((weights[4] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d2
        new_weights[5] = int((weights[5] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d3

        # Low frequency window
        MIN_WINDOW_SIZE = 15
        MAX_WINDOW_SIZE = 50

        new_weights[10] = int((weights[10] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d4 from -1,1 to 1,50
        new_weights[11] = int((weights[11] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d5
        new_weights[12] = int((weights[12] + 1) * (MAX_WINDOW_SIZE - MIN_WINDOW_SIZE) / 2 + MIN_WINDOW_SIZE)  # d6

        # Alpha for EMA: [0,1]
        new_weights[13] = 1 / (1 + np.exp(-weights[13]))  # a

        return new_weights

    # [w1, w2, w3, d1, d2, d3, a1, w4, w5, w6, d4, d5, d6, a2]
    def generate_signals(self, weights, graph=False):
        # Eq. 7 from the project description
        sma = self.wma(self.P, int(weights[3]), self.sma_filter(int(weights[3])))
        lma = self.lma(self.P, int(weights[4]))
        ema = self.ema(self.P, int(weights[5]), weights[6])
        high = (weights[0]*sma + weights[1]*lma + weights[2]*ema)/np.sum(weights[:3])

        sma = self.wma(self.P, int(weights[10]), self.sma_filter(int(weights[10])))
        lma = self.lma(self.P, int(weights[11]))
        ema = self.ema(self.P, int(weights[12]), weights[13])
        low = (weights[7]*sma + weights[8]*lma + weights[9]*ema)/np.sum(weights[7:10])

        sma_diff = high - low
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