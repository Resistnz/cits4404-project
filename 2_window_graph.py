from bots.basic_bot import BasicBot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# plot all combinations of 2 parameters and their objective function value, in range 1 to 50
def plot_2d_window_space(bot, val_min=1, val_max=50, step=1):
    """
    Plot objective function value as a 3D surface across 2-parameter space.
    Parameters are window sizes ranging from val_min to val_max.
    """
    x = np.arange(val_min, val_max + step, step)
    y = np.arange(val_min, val_max + step, step)

    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X, dtype=float)

    total = X.shape[0] * X.shape[1]
    count = 0

    best = np.inf
    best_obj = []
    
    print(f"Evaluating {total} parameter combinations...")
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Convert the range 1 to 50 into -1 to 1
            converted_x = (X[i, j] - 25) / 25
            converted_y = (Y[i, j] - 25) / 25

            Z[i, j] = bot.evaluate_parameters([converted_x, converted_y])

            count += 1
            if count % 50 == 0:
                print(f"  Progress: {count}/{total}", end="\r", flush=True)

            if Z[i, j] < best:
                best = Z[i, j]
                best_obj = [X[i, j], Y[i, j]]

    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=1, edgecolor='none')
    
    ax.set_xlabel('SMA Window Size 1')
    ax.set_ylabel('SMA Window Size 2')
    ax.set_zlabel('Objective Function Value')
    ax.set_title('BasicBot Objective Function Landscape (1-50 Window Range)')

    # print the best config with best objective score
    print(f"\nBest configuration: Window1={best_obj[0]}, Window2={best_obj[1]} with score: {best}")
    
    fig.colorbar(surf, ax=ax, label='Objective Value')
    plt.show()


if __name__ == "__main__":
    bot = BasicBot()
    plot_2d_window_space(bot, val_min=1, val_max=50, step=1)
