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

        data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
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
        return np.ones(N) / N

    @staticmethod
    def wma(P, N, kernel):
        return np.convolve(TradingBot.pad(P, N), kernel, "valid")

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
        plt.plot(
            P,
            color="black",
            marker="+",
            linestyle="-",
            linewidth=1,
            markersize=5,
            label="P",
        )

        # 2. Plot the 10-day and 20-day Simple Moving Averages
        # plt.plot(sma10, color='#1f77b4', label='1 day SMA') # standard matplotlib blue
        # plt.plot(sma20, color='#ff7f0e', label='5 day SMA') # standard matplotlib orange

        # # 3. Plot the SMA difference as a dashed grey line
        # # Note: Using the label exactly as it appears in your image's legend
        # plt.plot(sma_diff, color='grey', linestyle='--', label='SMA1-SMA5')

        # 4. Plot the buy signal spikes as a solid grey line
        plt.plot(
            buy_signal,
            color="darkgrey",
            linestyle="-",
            linewidth=1.5,
            label="buy signal",
        )

        # 5. Add the horizontal baseline at y=0
        plt.axhline(0, color="grey", linewidth=1)

        # 6. Isolate the indices for buy and sell signals
        buy_indices = np.where(buy_signal == 1)[0]
        sell_indices = np.where(buy_signal == -1)[0]

        # 7. Plot the Buy (green ^) and Sell (red v) markers
        # In your reference image, the markers are fixed at a y-value of roughly 2.
        # We generate arrays of 2s that match the length of the signal indices.
        plt.scatter(
            buy_indices,
            np.full(len(buy_indices), 2),
            color="green",
            marker="^",
            s=100,
            label="buy",
            zorder=5,
        )
        plt.scatter(
            sell_indices,
            np.full(len(sell_indices), 2),
            color="red",
            marker="v",
            s=100,
            label="sell",
            zorder=5,
        )

        ymin, ymax = plt.ylim()
        plt.vlines(
            buy_indices, ymin, ymax, color="green", linestyles="dashed", alpha=0.4
        )
        plt.vlines(
            sell_indices, ymin, ymax, color="red", linestyles="dashed", alpha=0.4
        )

        # Add the legend in the top right corner
        plt.legend(loc="upper right")

        # Display the plot
        plt.show()

    # Override this too if needed
    def transform_weights(self, weights):
        # This is where you can transform the raw weights from the optimiser (-1,1) into something usable for the bot
        new_weights = list(weights)

        return new_weights
    
    # Use the weights to run the bot, and then see if it makes money or not
    # Currently this is a minimiser
    def evaluate_parameters(self, weights, num_trials=5):
        transformed_weights = self.transform_weights(weights)

        total_days = 1858  # Before 2020
        # Leave the first 200 days as warm-up for the moving average windows,
        # then slide a 120-day evaluation window every 30 days
        window_size = 120
        step_size = 30
        warm_up = 200

        sliding_folds = [
            (start, start + window_size)
            for start in range(warm_up, total_days - window_size, step_size)
        ]

        fold_scores = []
        starting_capital = 1000.0

        # Reward per completed BUY/SELL round trip
        round_trip_incentive = 0.05
        round_trip_cap = 5

        # discourage churning (buying/selling quickly)
        churn_penalty = 0.005

        originalP = self.P

        try:
            for fold_start, fold_end in sliding_folds:
                validation_data = self.price_history[fold_start:fold_end]

                balance, _, signals = self.run_on_period(transformed_weights, validation_data)

                signal_count = sum(1 for s in signals if s != Signal.HOLD)
                round_trips = self.count_round_trips(signals)

                # Buy-and-hold benchmark on this window (same 3% fees)
                bh_btc = (starting_capital / validation_data[0]) * 0.97
                bh_usd = max(bh_btc * validation_data[-1] * 0.97, 1e-8)

                # Log excess return: zero for buy-and-hold, positive if we beat it
                log_excess = np.log(max(balance, 1e-8) / bh_usd)

                rt_bonus = round_trip_incentive * min(round_trips, round_trip_cap)
                score = log_excess + rt_bonus - churn_penalty * signal_count

                fold_scores.append(score)

            return -np.mean(fold_scores)

        finally:
            self.P = originalP
    
    # Override this
    def generate_signals(self, weights):
        signals = [Signal.HOLD] * len(self.P)

        return signals
    
    def count_round_trips(self, signals):
        in_position = False
        round_trips = 0
        for s in signals:
            if s == Signal.BUY and not in_position:
                in_position = True
            elif s == Signal.SELL and in_position:
                in_position = False
                round_trips += 1
        return round_trips
    
    # Simulate a whole run of the bot
    def run(self, weights):
        usd = 1000
        bitcoin = 0
        transaction_fee = 0.97

        signals = self.generate_signals(weights)

        portfolio_values = []  # Track daily portfolio value

        # Move across all signals
        for i in range(len(signals)):
            signal = signals[i]

            if signal == Signal.BUY and usd > 0:
                # convert all USD to BTC
                bitcoin = usd / self.P[i] * transaction_fee
                usd = 0

            elif signal == Signal.SELL and bitcoin > 0:
                # convert all BTC to USD
                usd = bitcoin * self.P[i] * transaction_fee
                bitcoin = 0

            # Calculate current portfolio value
            current_value = usd + bitcoin * self.P[i]
            portfolio_values.append(current_value)

        # convert any remaining BTC to USD
        if bitcoin > 0:
            usd = bitcoin * self.P[len(signals)-1] * transaction_fee
            portfolio_values[-1] = usd  # Update last value

        return usd, portfolio_values, signals
    
    # Run on a specific period
    def run_on_period(self, weights, period_data):
        self.P = period_data
        balance, portfolio_values, signals = self.run(weights)

        return balance, portfolio_values, signals
