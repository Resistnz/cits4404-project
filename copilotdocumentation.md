# CITS4404 Project Documentation

## Project Overview

This repository contains a trading simulation framework for building and evaluating cryptocurrency trading bots and optimisation algorithms. It is designed so team members can add new trading strategies in `bots/` and new optimisation routines in `algorithms/`, then measure how well those approaches perform on historical Bitcoin price data.

## Repository Structure

- `main.py`
  - Entry point for running simulations and experiments.
- `run_benchmarks.py`
  - Script for running benchmark comparisons across bots and optimisers.
- `bots/`
  - `bot.py`
    - Base bot class and common utilities for trading bots.
  - `basic_bot.py`
    - Example simple trading bot implementation.
  - `more_complicated_bot.py`
    - Example of a more advanced trading bot structure.
- `algorithms/`
  - `optimiser.py`
    - Base optimiser class and interfaces for optimisation algorithms.
  - `gradient_descent.py`
    - Example optimiser using gradient descent.
  - `firefly.py`
    - Example optimiser using the firefly metaheuristic.
  - `benchmarks.py`
    - Tools for measuring and comparing optimiser performance.
- `data/`
  - `BTC-Daily.csv`
  - `BTC-Hourly.csv`
  - Historical Bitcoin price datasets used for training and evaluation.

## Adding Your Own Trading Bot

1. Create a new file in `bots/`, for example `my_custom_bot.py`.
2. Import the base class from `bots/bot.py`.
3. Implement the required methods:
   - transform_weights 
        - convert weights from [-1,1] to a range your bot can use
   - generate_signals:
        - generate a list of buy/sell signals however you want
        - our current examples scale indicators by weights
        - but you can lowkey do anything

        - output in the form:
            [Signal.HOLD, Signal.BUY, Signal]
4. Follow the patterns used by `basic_bot.py` and `more_complicated_bot.py`.

### Example Bot Class Pattern

```python
from bots.bot import BaseBot

class MyCustomBot(BaseBot):
    def transform_weights(self, weights):
        # turn weights from [-1,1] to meaningful numbers
        ...

    def generate_signals(self, weights, graph=False):
        # generate some buy/sell signals
        ...
```

## Adding Your Own Optimisation Algorithm

1. Create a new file in `algorithms/`, for example `my_optimizer.py`.
2. Import the base optimiser interface from `algorithms/optimiser.py`.
3. Implement the algorithm-specific methods:
   - initialise your variables
   - update
4. Use `gradient_descent.py` or `firefly.py` as templates for algorithm structure.

### Example Optimiser Class Pattern

```python
from algorithms.optimiser import BaseOptimiser

class MyOptimiser(BaseOptimiser):
    def __init__(self, config):
        super().__init__(config)
        # configure optimiser-specific settings

    def update(self):
        # search parameter space and return best solution
        pass
```

### Important Optimiser Concepts

- `objective_function`: this calls TradingBot.evaluate_parameters(). You can do something different here if you would like, but this is how its set upt now

## Performance Evaluation

We evaluate bot and optimisation performance through benchmark comparisons on historical price data.

### Bot Performance Measurement

Bot performance is measured using the `evaluate_parameters` method in `bot.py`. This method employs walk-forward validation to assess trading strategies on historical Bitcoin price data up to 2020. The evaluation process includes:

- **Walk-Forward Validation**: The data is divided into an initial training period (600 days) followed by 4 validation folds of approximately 300 days each. This simulates real-world trading by validating on future data not seen during training.

- **Simulation and Scoring**: For each validation fold, the bot generates trading signals based on the provided weights, simulates trading with a starting capital of $1000, and accounts for transaction fees (3% per trade). The performance score is calculated as:

  ```
  score = profit - (drawdown_penalty * max_drawdown * starting_capital) - (trade_penalty * trade_count) - (holding_penalty * average_holding_streak)
  ```

  Where:
  - `profit` is the final balance minus starting capital.
  - `max_drawdown` is the maximum percentage decline from the peak portfolio value.
  - `trade_count` is the number of buy/sell transactions.
  - `average_holding_streak` is the mean length of periods holding positions without trading.

- **Penalties**: The scoring system penalizes excessive risk (via drawdown), frequent trading (via trade count), and prolonged holding periods to encourage balanced strategies.

- **Final Metric**: The average score across all 4 folds is negated and returned, as the optimization algorithms minimize this value to find optimal parameters.

This approach ensures robust evaluation by testing strategies on unseen data and balancing profitability with risk management.

1. Read `main.py` to understand how the system loads data and runs a simulation.
2. Review `bots/basic_bot.py` and `algorithms/gradient_descent.py` to learn the coding patterns.
3. Add a new bot or optimiser, then run `run_benchmarks.py` to compare it against existing methods.

## Tips for Collaborators

- Keep bot strategy code isolated in `bots/` and optimiser logic in `algorithms/`.
- Document new strategy parameters clearly in your source file.
- When adding a new benchmark, make sure it is repeatable and uses the same dataset splits as other experiments.