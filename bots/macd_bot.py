import numpy as np
from bots.bot import TradingBot, Signal

class MACDBot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)

        new_weights[0] = int((weights[0] + 1) * 10 + 5)   # Fast EMA window
        new_weights[1] = int((weights[1] + 1) * 20 + 20)  # Slow EMA window
        new_weights[2] = int((weights[2] + 1) * 10 + 5)   # Signal EMA window

        return new_weights

    # [w1, w2, w3]
    def generate_signals(self, weights, graph=False):
        fast_window = int(weights[0])
        slow_window = int(weights[1])
        sig_window = int(weights[2])

        fast_alpha = 2.0 / (fast_window + 1)
        slow_alpha = 2.0 / (slow_window + 1)
        sig_alpha = 2.0 / (sig_window + 1)

        fast_ema = self.ema(self.P, fast_window, fast_alpha)
        slow_ema = self.ema(self.P, slow_window, slow_alpha)

        macd_line = fast_ema - slow_ema
        
        signal_line = self.ema(macd_line, sig_window, sig_alpha)

        macd_hist = macd_line - signal_line
        hist_sign = np.sign(macd_hist)

        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(hist_sign, kernel, mode='valid')

        buy_signal_aligned = np.sign(buy_signal)

        signals = [Signal.HOLD] + [Signal(int(x)) for x in buy_signal_aligned]

        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals