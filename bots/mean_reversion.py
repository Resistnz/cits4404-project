import numpy as np
from bots.bot import TradingBot, Signal

class ZScoreBot(TradingBot):
    def transform_weights(self, weights):
        new_weights = list(weights)
        
        # Window size for the rolling mean and std (e.g., 10 to 50)
        new_weights[0] = int((weights[0] + 1) * 20 + 10) 
        # Z-score entry threshold (e.g., 1.0 to 3.0 standard deviations)
        new_weights[1] = (weights[1] + 1) * 1.0 + 1.0
        
        return new_weights

    # [w1, w2]
    def generate_signals(self, weights, graph=False):
        window = int(weights[0])
        threshold = weights[1]

        prices = self.P
        rolling_mean = np.zeros_like(prices)
        rolling_std = np.zeros_like(prices)
        
        # Calculate rolling statistics
        for i in range(len(prices)):
            start = max(0, i - window + 1)
            window_slice = prices[start:i+1]
            rolling_mean[i] = np.mean(window_slice)
            rolling_std[i] = np.std(window_slice) if len(window_slice) > 1 else 1.0
            
        # Prevent division by zero on flat data
        rolling_std[rolling_std == 0] = 1e-8 
        z_score = (prices - rolling_mean) / rolling_std

        # 1 for oversold (buy), -1 for overbought (sell)
        raw_signals = np.zeros_like(prices)
        raw_signals[z_score < -threshold] = 1   
        raw_signals[z_score > threshold] = -1   

        # Detect crossover state changes for entry/exit triggers
        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(raw_signals, kernel, mode='valid')
        buy_signal_aligned = np.sign(buy_signal)

        signals = [Signal.HOLD] + [Signal(int(x)) for x in buy_signal_aligned]

        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals