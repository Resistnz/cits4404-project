import numpy as np
from bots.bot import TradingBot, Signal

class BreakoutBot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)
        
        # Lookback window for Highs (Buy trigger)
        new_weights[0] = int((weights[0] + 1) * 20 + 10) 
        # Lookback window for Lows (Sell trigger)
        new_weights[1] = int((weights[1] + 1) * 20 + 10)
        
        return new_weights

    # [w1, w2]
    def generate_signals(self, weights, graph=False):
        high_window = int(weights[0])
        low_window = int(weights[1])
        prices = self.P
        
        raw_signals = np.zeros_like(prices)
        
        for i in range(1, len(prices)):
            high_start = max(0, i - high_window)
            low_start = max(0, i - low_window)
            
            # Buy if price breaks the recent ceiling
            if prices[i] > np.max(prices[high_start:i]):
                raw_signals[i] = 1
            # Sell if price breaks the recent floor
            elif prices[i] < np.min(prices[low_start:i]):
                raw_signals[i] = -1
            else:
                # Hold previous state
                raw_signals[i] = raw_signals[i-1] 

        # Detect crossover state changes for entry/exit triggers
        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(raw_signals, kernel, mode='valid')
        buy_signal_aligned = np.sign(buy_signal)

        signals = [Signal.HOLD] + [Signal(int(x)) for x in buy_signal_aligned]

        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals