import numpy as np
from bots.bot import TradingBot, Signal


class ReverseMACDBot(TradingBot):
    """
    Reverse MACD / Divergence Bot.

    Standard MACD fires *after* the histogram crosses zero.
    This bot fires *earlier*, at the histogram's local peak or trough —
    i.e. when the histogram's *slope* (momentum) reverses direction.

    The key fix over a naive derivative approach: the raw histogram momentum
    (np.diff) is extremely noisy and would flip sign every tick, causing
    thousands of trades. We smooth it with an EMA before taking the sign,
    which gives a clean signal of sustained momentum reversals.

    Additionally we gate each trade on the *current direction of the histogram*:
      - We BUY only when the smoothed momentum turns positive AND the histogram
        is still negative (bullish divergence — approaching zero from below).
      - We SELL only when the smoothed momentum turns negative AND the histogram
        is still positive (bearish divergence — approaching zero from above).

    This is the true "early entry" concept: we enter in anticipation of the
    histogram crossing zero, not after it already has.

    Weights: [d_fast, d_slow, d_signal, alpha_fast, alpha_slow, alpha_sig]
      d_fast     : fast EMA window  (5–25 days)
      d_slow     : slow EMA window  (15–60 days)
      d_signal   : signal EMA window (3–20 days), also used to smooth momentum
      alpha_fast : EMA smoothing factor for fast line
      alpha_slow : EMA smoothing factor for slow line
      alpha_sig  : EMA smoothing factor for signal & momentum smoother
    """

    def transform_weights(self, weights):
        new_weights = list(weights)

        new_weights[0] = int((weights[0] + 1) * 10 + 5)    # d_fast:   5–25
        new_weights[1] = int((weights[1] + 1) * 22.5 + 15) # d_slow:   15–60
        new_weights[2] = int((weights[2] + 1) * 8.5 + 3)   # d_signal: 3–20

        # EMA alphas mapped to (0, 1) via sigmoid
        new_weights[3] = 1 / (1 + np.exp(-weights[3]))  # alpha_fast
        new_weights[4] = 1 / (1 + np.exp(-weights[4]))  # alpha_slow
        new_weights[5] = 1 / (1 + np.exp(-weights[5]))  # alpha_sig

        return new_weights

    # [d_fast, d_slow, d_signal, alpha_fast, alpha_slow, alpha_sig]
    def generate_signals(self, weights, graph=False):
        d_fast   = max(2, int(weights[0]))
        d_slow   = max(d_fast + 1, int(weights[1]))
        d_signal = max(2, int(weights[2]))
        a_fast   = float(weights[3])
        a_slow   = float(weights[4])
        a_sig    = float(weights[5])

        # Step 1: MACD line = fast EMA - slow EMA
        fast_ema  = self.ema(self.P, d_fast, a_fast)
        slow_ema  = self.ema(self.P, d_slow, a_slow)
        macd_line = fast_ema - slow_ema

        # Step 2: Signal line = EMA of MACD
        signal_line = self.ema(macd_line, d_signal, a_sig)

        # Step 3: Histogram
        histogram = macd_line - signal_line

        # Step 4: Raw momentum (first difference of histogram)
        # This is very noisy on its own — sign flips almost every tick
        hist_momentum_raw = np.diff(histogram)

        # Step 5: SMOOTH the momentum with an EMA to suppress noise.
        # d_signal // 2 (min 2) keeps it shorter than the signal window so
        # we still lead the histogram crossover, but noise is filtered out.
        smooth_window = max(2, d_signal // 2)
        hist_momentum = self.ema(hist_momentum_raw, smooth_window, a_sig)

        # Step 6: Detect when smoothed momentum reverses sign
        sign_momentum = np.sign(hist_momentum)
        kernel = np.array([0.5, -0.5])
        momentum_flip = np.convolve(sign_momentum, kernel, mode='valid')

        # Step 7: Gate on histogram direction — only enter on genuine divergence:
        #   BUY  when momentum flips positive AND histogram still below zero
        #         (bullish divergence: approaching crossover from below)
        #   SELL when momentum flips negative AND histogram still above zero
        #         (bearish divergence: approaching crossover from above)
        #
        # Alignment: diff eats 1, convolve eats 1 → momentum_flip is 2 shorter than histogram
        hist_gating = histogram[2:]  # align with momentum_flip
        buy_gated  =  (momentum_flip > 0) & (hist_gating < 0)  # bullish divergence
        sell_gated =  (momentum_flip < 0) & (hist_gating > 0)  # bearish divergence

        combined = buy_gated.astype(float) - sell_gated.astype(float)

        # Two prepended HOLDs (diff + convolve each eat one element)
        signals = [Signal.HOLD, Signal.HOLD] + [Signal(int(x)) for x in combined]

        if graph:
            TradingBot.graph_price(self.P, np.array([int(s) for s in signals]))

        return signals

