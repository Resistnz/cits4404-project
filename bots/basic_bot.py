import numpy as np
from bots.bot import TradingBot, Signal

# The basic bot from p. 11 in the project outline
class BasicBot(TradingBot):
    def generate_signals(self, weights, graph=False):
        smaA = self.wma(self.P, int(weights[0]), self.sma_filter(int(weights[0])))
        smaB = self.wma(self.P, int(weights[1]), self.sma_filter(int(weights[1])))

        sma_diff = smaA - smaB
        sign_diff = np.sign(sma_diff)

        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(sign_diff, kernel, mode='valid')

        # Convert e.g [0, 0, 1, 0, -1] to Signal.BUY and Signal.SELL
        signals = [Signal(x) for x in buy_signal]

        # Graph it if you want
        if graph:
            TradingBot.graph_price(self.P, buy_signal)

        return signals