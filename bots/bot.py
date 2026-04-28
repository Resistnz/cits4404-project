from enum import IntEnum
import numpy as np

class Signal(IntEnum):
    BUY = 1
    SELL = -1
    HOLD = 0

class TradingBot:
    def __init__(self):
        self.P = self.load_price_history(start_time=1614556800, end_time=1646092800)

    @staticmethod
    def load_price_history(start_time, end_time):
        filepath = "data/BTC-Daily.csv"

        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.size == 0:
            P = np.array([])
        elif data.ndim > 1:
            unix_times = data[:, 0]
            mask = (unix_times >= start_time) & (unix_times <= end_time)
            P = data[mask, 6]  # column 6 is 'close'
        else:
            P = np.array([])

        # Reverse it cause its backwards for some reason
        P = P[::-1]

        return P

    @staticmethod
    def pad(P, N):
        padding = -np.flip(P[1:N])
        return np.append(padding, P)
    
    @staticmethod
    def sma_filter(N):
        return np.ones(N)/N
    
    @staticmethod
    def wma(P, N, kernel):
        return np.convolve(TradingBot.pad(P,N), kernel, 'valid')    
    
    @staticmethod
    def graph_price(P, sma10, sma20, sma_diff, buy_signal):
        import matplotlib.pyplot as plt

        # Create a figure with a wide aspect ratio similar to the image
        plt.figure(figsize=(12, 6))

        # 1. Plot Price (P) with black line and '+' markers
        plt.plot(P, color='black', marker='+', linestyle='-', linewidth=1, markersize=5, label='P')

        # 2. Plot the 10-day and 20-day Simple Moving Averages
        plt.plot(sma10, color='#1f77b4', label='10 day SMA') # standard matplotlib blue
        plt.plot(sma20, color='#ff7f0e', label='20 day SMA') # standard matplotlib orange

        # 3. Plot the SMA difference as a dashed grey line
        # Note: Using the label exactly as it appears in your image's legend
        plt.plot(sma_diff, color='grey', linestyle='--', label='SMA10-SMA40')

        # 4. Plot the buy signal spikes as a solid grey line
        plt.plot(buy_signal, color='darkgrey', linestyle='-', linewidth=1.5, label='buy signal')

        # 5. Add the horizontal baseline at y=0
        plt.axhline(0, color='grey', linewidth=1)

        # 6. Isolate the indices for buy and sell signals
        buy_indices = np.where(buy_signal == 1)[0]
        sell_indices = np.where(buy_signal == -1)[0]

        # 7. Plot the Buy (green ^) and Sell (red v) markers
        # In your reference image, the markers are fixed at a y-value of roughly 2.
        # We generate arrays of 2s that match the length of the signal indices.
        plt.scatter(buy_indices, np.full(len(buy_indices), 2), color='green', marker='^', s=100, label='buy', zorder=5)
        plt.scatter(sell_indices, np.full(len(sell_indices), 2), color='red', marker='v', s=100, label='sell', zorder=5)

        ymin, ymax = plt.ylim()
        plt.vlines(buy_indices, ymin, ymax, color='green', linestyles='dashed', alpha=0.4)
        plt.vlines(sell_indices, ymin, ymax, color='red', linestyles='dashed', alpha=0.4)

        # Add the legend in the top right corner
        plt.legend(loc='upper right')

        # Display the plot
        plt.show()
    
    # Use the weights to run the bot, and then see if it makes money or not
    # Can override this with different scaling functions
    # Currently this is a minimiser
    def evaluate_parameters(self, weights):
        final_balance = self.run(weights) # One year

        return 1000 - final_balance
    
    # Override this
    def generate_signals(self, weights):
        signals = [Signal.HOLD] * len(self.P) # A signal for each time, either 

        return signals
    
    # Simulate a whole run of the bot 
    def run(self, weights):
        #print("Starting with $1000 USD")

        usd = 1000
        bitcoin = 0

        signals = self.generate_signals(weights)

        # Move across all signals
        for i in range(len(signals)):
            signal = signals[i]

            if signal == Signal.BUY and usd > 0:
                # convert all USD to BTC
                bitcoin = usd / self.P[i] * 0.97
                usd = 0

                #print(f"Buying BTC at day {i}! We now have {bitcoin} BTC")

            elif signal == Signal.SELL and bitcoin > 0:
                # convert all BTC to USD
                usd = bitcoin * self.P[i] * 0.97
                bitcoin = 0

                #print(f"Sellin BTC at day {i}! We now have {usd} USD")

        # convert any remaining BTC to USD
        if bitcoin > 0:
            usd = bitcoin * self.P[i] * 0.97

        #print(f"Ending with ${usd}")

        return usd
    
# The basic bot from p. 11 in the project outline
class BasicBot(TradingBot):
    def generate_signals(self, weights):
        smaA = self.wma(self.P, int(weights[0]), self.sma_filter(int(weights[0])))
        smaB = self.wma(self.P, int(weights[1]), self.sma_filter(int(weights[1])))

        sma_diff = smaA - smaB
        sign_diff = np.sign(sma_diff)

        kernel = np.array([0.5, -0.5])
        buy_signal = np.convolve(sign_diff, kernel, mode='valid')

        # Convert e.g [0, 0, 1, 0, -1] to Signal.BUY and Signal.SELL
        signals = [Signal(x) for x in buy_signal]

        # Graph it if you want
        #TradingBot.graph_price(P, sma10, sma20, sma_diff, buy_signal)

        return signals