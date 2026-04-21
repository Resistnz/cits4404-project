# cits4404-project
best trading bot of all time

this looks so gpt but i promise i did write it myself

im thinkin we have like:

                                optimisation alg.

                                    |
                                    |  (weightings)
                                    V
    
      trading engine   <----    weighted indicators  < ---- indicators (1-day-ema, 3-day-ema, max, min etc)

            |
            |
            V

        buy/sell


where trading engine is something we do research on and make e.g. SVR
our language is just a bunch of weights (numbers)
dimension will be however many indicators we pick

pros:
 - probably relatively smooth solution space
 - ez pz to figure out our hypothesis space/language cause its just all numbers

cons:
 - we gotta figure out some trading engine (proabbly still not hard)

_______________________________________________ OR:

we full send it and get the optimisation alg. to build a bot entirely
our language becomes more complicated where we actually build a bot, e.g. practice test Q1

pros:
 - more focused on optimisation
 - could be cool 
cons:
 - solution space gets all fucked up and discontinuous
 - probably wouldn't even work that well BUT it might
