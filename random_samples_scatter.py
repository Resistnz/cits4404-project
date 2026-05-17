import numpy as np
import matplotlib.pyplot as plt
from bots.basic_bot import BasicBot
from bots.more_complicated_bot import BetterBot
from bots.macd_bot import MACDBot
from bots.breakout import BreakoutBot
from bots.triple_sma_bot import TripleSMABot

BOT_CLASSES = {
    "BasicBot": (BasicBot, 2),
    "BetterBot": (BetterBot, 14),
    "MACDBot": (MACDBot, 3),
    "BreakoutBot": (BreakoutBot, 2),
    "TripleSMABot": (TripleSMABot, 3)
}

EVAL_MODES = ["profit", "log_excess", "drawdown"]
NUM_SAMPLES = 500
STARTING_CAPITAL = 1000.0
HOLDOUT_START_INDEX = 1858

def main():
    for mode in EVAL_MODES:
        plt.figure(figsize=(10, 6))
        
        all_obj_vals = []
        all_profits = []
        
        for bot_name, (bot_class, dims) in BOT_CLASSES.items():
            bot = bot_class(eval_mode=mode)
            
            obj_vals = []
            profits = []
            
            for _ in range(NUM_SAMPLES):
                sample = np.random.uniform(-1, 1, dims)
                
                # Evaluate objective function
                obj_val = bot.evaluate_parameters(sample)
                
                # Calculate final profit using logic from test_particular_solution
                transformed = bot.transform_weights(sample)
                final_balance, _, _ = bot.run_on_period(transformed, bot.price_history[HOLDOUT_START_INDEX:])
                profit = final_balance - STARTING_CAPITAL
                
                if obj_val < 100000: # Filter out outliers
                    obj_vals.append(obj_val)
                    profits.append(profit)
                    
                    all_obj_vals.append(obj_val)
                    all_profits.append(profit)
                    
            plt.scatter(obj_vals, profits, label=bot_name, alpha=0.6, edgecolors='w', linewidth=0.5)

        # Calculate and plot line of best fit for all points combined
        if len(all_obj_vals) > 1:
            all_obj_vals = np.array(all_obj_vals)
            all_profits = np.array(all_profits)
            
            corr_coef = np.corrcoef(all_obj_vals, all_profits)[0, 1]
            m, b = np.polyfit(all_obj_vals, all_profits, 1)
            
            x_seq = np.linspace(all_obj_vals.min(), all_obj_vals.max(), 100)
            plt.plot(x_seq, m * x_seq + b, color='red', alpha=0.8, 
                     label=f'Line of Best Fit\nOverall Correlation: {corr_coef:.3f}', linestyle='-')
        
        plt.title(f'Random Samples: Objective Value vs Final Profit ({mode})')
        plt.xlabel('Objective Value')
        plt.ylabel('Final Profit')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'random_scatter_{mode}.png')

    plt.show()

if __name__ == '__main__':
    main()
