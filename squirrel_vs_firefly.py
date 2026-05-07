import numpy as np
from squirrel import SquirrelOptimiser, CorrectSquirrelOptimiser, ImprovedSquirrelOptimiser
from firefly import FireflyOptimiser
from benchmarks import functions, get_function_bounds

def main():
    function_list = list(functions.keys())
    
    # Statistical analysis requires multiple independent runs (30 is standard)
    num_runs = 5
    
    algorithms = ['Squirrel', 'Correct_Squirrel', 'Improved_Squirrel', 'Firefly'] # Add 'Firefly' here if uncommented
    
    # Dictionary to store the results of all independent runs
    results = {f: {algo: [] for algo in algorithms} for f in function_list}

    for f in function_list:
        print(f"Running optimisers on {f} for {num_runs} independent runs...")
        range_min, range_max = get_function_bounds(f)
        for run in range(num_runs):
            # Print progress indicator
            print(f"\r  Run {run+1}/{num_runs}", end="", flush=True)
            
            squirrel = SquirrelOptimiser(num_squirrels=50, dimensions=30, max_iterations=1000, function_key=f, range_min=range_min, range_max=range_max)
            correct_squirrel = CorrectSquirrelOptimiser(num_squirrels=50, dimensions=30, max_iterations=1000, function_key=f, range_min=range_min, range_max=range_max)
            improved_squirrel = ImprovedSquirrelOptimiser(num_squirrels=50, dimensions=30, max_iterations=1000, function_key=f, range_min=range_min, range_max=range_max)
            firefly = FireflyOptimiser(num_fireflies=50, dimensions=30, max_iterations=1000, function_key=f, range_min=range_min, range_max=range_max)
            
            results[f]['Squirrel'].append(squirrel.run())
            results[f]['Correct_Squirrel'].append(correct_squirrel.run())
            results[f]['Improved_Squirrel'].append(improved_squirrel.run())
            results[f]['Firefly'].append(firefly.run())
        
        print() # New line after the runs for the current function finish

    print("\nTable 5: Statistical results of multimodal function")
    
    # Print the Header
    header = f"{'Function':<10}" + "".join([f"{algo:>18}" for algo in algorithms])
    print(header)

    # Print the calculated statistics for each function
    for f in function_list:
        print(f)
        
        # Calculate stats
        stats = {
            'Best': {},
            'Worst': {},
            'Mean': {},
            'SD': {}
        }
        
        for algo in algorithms:
            runs_data = results[f][algo]
            stats['Best'][algo] = np.min(runs_data)
            stats['Worst'][algo] = np.max(runs_data)
            stats['Mean'][algo] = np.mean(runs_data)
            stats['SD'][algo] = np.std(runs_data, ddof=0) # Population standard deviation

        # Print rows for Best, Worst, Mean, and SD
        for stat_name in ['Best', 'Worst', 'Mean', 'SD']:
            row_str = f"{stat_name:<10}"
            for algo in algorithms:
                # Use uppercase scientific notation with 4 decimal places (e.g., -1.2569E+04)
                row_str += f"{stats[stat_name][algo]:>18.4E}"
            print(row_str)

if __name__ == "__main__":
    main()