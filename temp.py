import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the benchmark results
df = pd.read_csv('benchmark_results.csv')

# Get unique eval_modes
eval_modes = df['eval_mode'].unique()

for mode in eval_modes:
    mode_df = df[df['eval_mode'] == mode]
    
    # Filter out outliers
    mode_df = mode_df[mode_df['objective_value'] < 100000]
    
    # Extract the columns
    objective_value = mode_df['objective_value']
    final_balance = mode_df['final_balance']
    
    # Calculate correlation coefficient
    if len(objective_value) > 1:
        corr_coef = np.corrcoef(objective_value, final_balance)[0, 1]
    else:
        corr_coef = 0.0
        
    # Create the scatterplot
    plt.figure(figsize=(10, 6))
    plt.scatter(objective_value, final_balance, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Calculate and plot line of best fit
    if len(objective_value) > 1:
        m, b = np.polyfit(objective_value, final_balance, 1)
        # Create a line of best fit over the range of data
        x_seq = np.linspace(objective_value.min(), objective_value.max(), 100)
        plt.plot(x_seq, m * x_seq + b, color='red', alpha=0.8, 
                 label=f'Line of Best Fit\nCorrelation: {corr_coef:.3f}')
        plt.legend()
    
    # Add title and labels
    plt.title(f'Scatterplot of Objective Value vs Final Balance ({mode})')
    plt.xlabel('Objective Value')
    plt.ylabel('Final Balance')
    
    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'scatterplot_{mode}.png')

# Display all plots
plt.show()
