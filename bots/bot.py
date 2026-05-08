from enum import IntEnum
import numpy as np

class Signal(IntEnum):
    BUY = 1
    SELL = -1
    HOLD = 0

class TradingBot:
    def __init__(self):
        self.load_price_history()

        # Everything up until 2020 for training
        self.P = self.price_history[:1858]

    def load_price_history(self):
        filepath = "data/BTC-Daily.csv"

        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        if data.size == 0:
            self.price_history = np.array([])
        elif data.ndim > 1:
            self.price_history = data[:, 6]  # column 6 is 'close'
        else:
            self.price_history = np.array([])

        # Reverse it cause its backwards for some reason
        self.price_history = self.price_history[::-1]

    @staticmethod
    def pad(P, N):
        # initial windows use only available (past/current) data.
        if N <= 1:
            return P
        
        padding = np.full(N - 1, P[0])
        return np.append(padding, P)
    
    @staticmethod
    def sma_filter(N):
        return np.ones(N)/N
    
    @staticmethod
    def wma(P, N, kernel):
        return np.convolve(TradingBot.pad(P,N), kernel, 'valid')    
    
    @staticmethod
    def lma_filter(N):
        """Generates a linear-weighted (triangular) filter."""
        if N <= 0:
            return np.array([])
            
        k = np.arange(N)
        return (2 / (N + 1)) * (1 - k / N)

    @staticmethod
    def lma(P, N):
        """Calculates the Linear-Weighted Moving Average (LMA)."""
        return TradingBot.wma(P, N, TradingBot.lma_filter(N))

    @staticmethod
    def ema_filter(N, alpha):
        """Generates an exponential decay filter."""
        if N <= 0:
            return np.array([])
            
        k = np.arange(N)
        return alpha * ((1 - alpha) ** k)

    @staticmethod
    def ema(P, N, alpha):
        """Calculates the Exponential Moving Average (EMA)."""
        return TradingBot.wma(P, N, TradingBot.ema_filter(N, alpha))
    
    @staticmethod
    def graph_price(P, buy_signal):
        import matplotlib.pyplot as plt

        # Create a figure with a wide aspect ratio similar to the image
        plt.figure(figsize=(12, 6))

        # 1. Plot Price (P) with black line and '+' markers
        plt.plot(P, color='black', marker='+', linestyle='-', linewidth=1, markersize=5, label='P')

        # 2. Plot the 10-day and 20-day Simple Moving Averages
        # plt.plot(sma10, color='#1f77b4', label='1 day SMA') # standard matplotlib blue
        # plt.plot(sma20, color='#ff7f0e', label='5 day SMA') # standard matplotlib orange

        # # 3. Plot the SMA difference as a dashed grey line
        # # Note: Using the label exactly as it appears in your image's legend
        # plt.plot(sma_diff, color='grey', linestyle='--', label='SMA1-SMA5')

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

    # Override this too if needed
    def transform_weights(self, weights):
        # This is where you can transform the raw weights from the optimiser (-1,1) into something usable for the bot
        new_weights = list(weights)

        return new_weights
    
    # Use the weights to run the bot, and then see if it makes money or not
    # Can override this with different scaling functions
    # Currently this is a minimiser
    def evaluate_parameters(self, weights, num_trials=5):
        # Scale the weights into something usable for the bot
        transformed_weights = self.transform_weights(weights)

        # Run on different periods
        total_balance = 0
        for i in range(num_trials):
            # set self.P to a random length (between 100 to 500) day period from day 0 to 1,858
            start_index = np.random.randint(0, 1858 - 500)
            end_index = start_index + np.random.randint(100, 500)
            self.P = self.price_history[start_index:end_index]

            total_balance += self.run(transformed_weights)

        average_balance = total_balance / num_trials
        return 1000 - average_balance # We minimise this
    
    # Override this
    def generate_signals(self, weights):
        signals = [Signal.HOLD] * len(self.P) # A signal for each time, either 

        return signals
    
    # Simulate a whole run of the bot 
    def run(self, weights):
        #print("Starting with $1000 USD")

        usd = 1000
        bitcoin = 0
        transaction_fee = 0.97

        signals = self.generate_signals(weights)

        # Move across all signals
        for i in range(len(signals)):
            signal = signals[i]

            if signal == Signal.BUY and usd > 0:
                # convert all USD to BTC
                bitcoin = usd / self.P[i] * transaction_fee
                usd = 0

                #print(f"Buying BTC at day {i}! We now have {bitcoin} BTC")

            elif signal == Signal.SELL and bitcoin > 0:
                # convert all BTC to USD
                usd = bitcoin * self.P[i] * transaction_fee
                bitcoin = 0

                #print(f"Sellin BTC at day {i}! We now have {usd} USD")

        # convert any remaining BTC to USD
        if bitcoin > 0:
            usd = bitcoin * self.P[i] * transaction_fee

        #print(f"Ending with ${usd}")

        return usd