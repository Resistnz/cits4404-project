# cits4404-project
best trading bot of all time

current setup:

                                optimisation alg.

                                    |
                                    |  (weightings)
                                    V
    
      trading engine   <----    weighted indicators  < ---- indicators (1-day-ema, 3-day-ema, max, min etc)

            |
            |
            V

        buy/sell


- trading engine is something we do research on and make e.g. SVR
- our language is just a bunch of weights (numbers)
- dimension will be however many indicators we pick

pros:
 - probably relatively smooth solution space
 - ez pz to figure out our hypothesis space/language cause its just all numbers

cons:
 - we gotta figure out some trading engine (proabbly still not hard)


