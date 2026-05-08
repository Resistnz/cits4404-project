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
        # Scale the weights into something usable for the bot
        transformed_weights = self.transform_weights(weights)

        # Walk-forward validation setup
        total_days = 1858 # Before 2020
        initial_train_days = 600
        validation_fold_days = (total_days - initial_train_days) // 4  # about 300 days per fold

        folds = [(50, initial_train_days)] # Make sure we populate the window before we start
        start = initial_train_days
        for i in range(4):
            end = min(start + validation_fold_days, total_days)
            folds.append((start, end))
            start = end

        # Evaluate on each validation fold
        fold_scores = []
        starting_capital = 1000.0
        drawdown_penalty = 0.0
        trade_penalty = 1.0
        holding_penalty = 0.1  # Penalise long holding periods

        originalP = self.P

        try:
            for fold_start, fold_end in folds:
                validation_data = self.price_history[fold_start:fold_end]

                balance, portfolio_values = self.run_on_period(transformed_weights, validation_data)                
                max_dd = self.compute_max_drawdown(portfolio_values)

                # Profit after fees is the primary metric, with risk penalty for drawdown
                profit = balance - starting_capital
                score = profit - drawdown_penalty * max_dd * starting_capital

                # Penalise excessive trading under fixed fees
                signals = self.generate_signals(transformed_weights)

                trades = np.count_nonzero(signals) / 2


                trade_count = int(np.count_nonzero(signals))
                score -= trade_penalty * trade_count

                # Penalise long holding periods
                holding_streaks = self.compute_holding_streaks(signals)
                if holding_streaks:
                    avg_holding = np.mean(holding_streaks)
                    score -= holding_penalty * avg_holding

                fold_scores.append(score)
            
            return -np.mean(fold_scores)
        
        finally:
            self.P = originalP

    def compute_daily_log_returns(self, values):
        if len(values) < 2:
            return np.array([0.0])
        
        # Ensure no non-positive values for log
        safe_values = np.maximum(values, 1e-8)
        returns = np.log(safe_values[1:] / safe_values[:-1])

        return returns

    def compute_sharpe_ratio(self, returns, risk_free_rate=0.0):
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 365 

        return np.mean(excess_returns) / np.std(excess_returns)

    def compute_max_drawdown(self, portfolio_values):
        if len(portfolio_values) < 2:
            return 0.0
        
        peak = portfolio_values[0]
        max_dd = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value

            dd = (peak - value) / peak

            if dd > max_dd:
                max_dd = dd

        return max_dd
    
    # Override this
    def generate_signals(self, weights):
        signals = [Signal.HOLD] * len(self.P)  # A signal for each time, either

        return signals
    
    def compute_holding_streaks(self, signals):
        """Compute the lengths of holding periods after each BUY until SELL."""
        streaks = []
        holding = False
        streak = 0
        for signal in signals:
            if signal == Signal.BUY:
                if holding:
                    streaks.append(streak)
                holding = True
                streak = 0
            elif signal == Signal.SELL:
                if holding:
                    streaks.append(streak)
                holding = False
                streak = 0
            elif signal == Signal.HOLD and holding:
                streak += 1
        # If still holding at the end, count it
        if holding:
            streaks.append(streak)
        return streaks
    
    # Simulate a whole run of the bot 
    def run(self, weights):
        # print("Starting with $1000 USD")

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

        return usd, portfolio_values
    
    # Run on a specific period
    def run_on_period(self, weights, period_data):
        self.P = period_data
        balance, portfolio_values = self.run(weights)
        
        return balance, portfolio_values
