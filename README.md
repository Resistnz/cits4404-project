## How does it work

                                optimisation alg.

                                    |
                                    |  (weightings)
                                    V
    
      trading engine   <----    weighted indicators  < ---- indicators (1-day-ema, 3-day-ema, max, min etc)

            |
            |
            V

        buy/sell

## Structurally:

Our TradingBot provides functions for calculating indicators on historical data, such as an n-day-SMA. In making our own TradingBot, we just need to override the `generate_signals` function. In here, we use some combination of indicators, scaled or otherwise defined by some weights, to generate a list of `Signals`, such as:

[Signal.HOLD, Signal.HOLD, Signal.BUY, Signal.HOLD, Signal.SELL, Signal.HOLD, Signal.BUY]

Each element of this list corresponds to a day of data. When the program reaches a Signal.BUY day, it will buy, and the same for Signal.SELL.

See `bots/basic_bot.py` for a simple example from the project description.

-----

Our Optimiser class is how we find the good weights for the indicators that our TradingBot uses. Override the `init`, `update` and `termination_criteria_reached` functions and implement your own optimisation algorithm. See `algorithms/gradient_descent.py` for a simple example.

Each iteration, the Optimiser will simulate trading over a period (e.g. 1 year) in the TradingBot using its current hypothesis, and the amount of money it makes becomes the objective function. 
The Optimiser uses this to **minimise** this objective function (as currently more profit == lower objective value), but you can change this if you want by multiplying the objective function value by -1 to gain a **maximiser**. 

The language of our Optimser is just some floats, and we are simply optimising for a vector of real values of size `dimensions`. Our hypothesis space is defined by the `val_min` and `val_max` parameters, which clip each value during optimisation. 

-----

Our main function is simple - we pick our trading bot and our optimiser, train the optimiser to get good weights for the indicators (using the amount of money made as the objective function), and then we get our `best_solution`, which we can use to trade on future data.

## TLDR

### Making an Optimiser Algorithm
