import numpy as np
from bots.bot import TradingBot, Signal


class TripleSMABot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)

        new_weights[0] = int((weights[0] + 1) * 9.5 + 1)   # d_fast: 1–20
        new_weights[1] = int((weights[1] + 1) * 15 + 10)   # d_mid:  10–40
        new_weights[2] = int((weights[2] + 1) * 27.5 + 25) # d_slow: 25–80

        return new_weights

    # [d_fast, d_mid, d_slow]
    def generate_signals(self, weights, graph=False):
        d_fast = max(1, int(weights[0]))
        d_mid  = max(d_fast + 1, int(weights[1]))
        d_slow = max(d_mid + 1, int(weights[2]))

        fast = self.wma(self.P, d_fast, self.sma_filter(d_fast))
        mid  = self.wma(self.P, d_mid,  self.sma_filter(d_mid))
        slow = self.wma(self.P, d_slow, self.sma_filter(d_slow))

        # Trend condition: fast > mid > slow  (+1) or fast < mid < slow (-1)
        trend = np.sign(fast - mid) * np.sign(mid - slow)

        # Crossover detector on (fast - mid) difference
        diff_fm = fast - mid
        sign_fm = np.sign(diff_fm)
        kernel  = np.array([0.5, -0.5])
        cross_fm = np.convolve(sign_fm, kernel, mode='valid')

        # Only fire when a fast/mid crossover occurs AND the slow confirms the trend
        # Align trend (length N) with cross_fm (length N-1): drop the first element
        trend_aligned = trend[1:]
        buy_signal = np.sign(cross_fm) * (trend_aligned > 0).astype(float)
        # For sells: crossover is negative AND trend < 0
        sell_signal = np.sign(cross_fm) * (trend_aligned < 0).astype(float)
        combined = buy_signal + sell_signal
        combined = np.clip(combined, -1, 1)

        signals = [Signal.HOLD] + [Signal(int(x)) for x in combined]

        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals
