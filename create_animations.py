import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — much faster for saving GIFs
import matplotlib.pyplot as plt
import os
import sys
import argparse
from mpl_toolkits.mplot3d import Axes3D
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import io

from bots.basic_bot import BasicBot
from algorithms.gwo import GWOOptimiser
from algorithms.firefly import FireflyOptimiser, ImprovedFireflyOptimiser
from algorithms.big_bang_big_crunch import BigBangBigCrunchOptimiser
from algorithms.gradient_descent import GradientDescentOptimiser
#from algorithms.squirrel import SquirrelOptimiser, ImprovedSquirrelOptimiser

worker_bot = None

def init_worker():
    global worker_bot
    worker_bot = BasicBot(eval_mode="log_excess")

def eval_point(args):
    x, y = args
    return worker_bot.evaluate_parameters([x, y])

def get_z_landscape():
    cache_file = "z_landscape_log_excess.npy"
    if os.path.exists(cache_file):
        print("Loading Z landscape from cache...")
        return np.load(cache_file)
        
    print("Computing Z landscape... (this may take a minute)")
    val_min, val_max, step = 1, 50, 1
    x = np.arange(val_min, val_max + step, step)
    y = np.arange(val_min, val_max + step, step)
    X, Y = np.meshgrid(x, y)
    tasks = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            converted_x = (X[i, j] - 25.5) / 24.5
            converted_y = (Y[i, j] - 25.5) / 24.5
            tasks.append((converted_x, converted_y))
    with ProcessPoolExecutor(initializer=init_worker) as pool:
        results = list(pool.map(eval_point, tasks))
    Z = np.array(results).reshape(X.shape)
    np.save(cache_file, Z)
    return Z

def weight_to_window(w):
    w = np.array(w)
    return (w + 1) * 24.5 + 1

def get_pop(opt, algo_name):
    if algo_name == "GWO":
        return np.copy(opt.population)
    elif "Firefly" in algo_name:
        return np.copy(opt.fireflies)
    elif algo_name == "BBBC":
        return np.copy(opt.population)
    elif "Squirrel" in algo_name:
        return np.copy(opt.squirrels)
    elif algo_name == "GradientDescent":
        return np.array([opt.pos])

def render_frame_to_image(fig):
    """Render a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img

def main():
    parser = argparse.ArgumentParser(description="Animate optimization algorithm exploration in 3D and 2D.")
    parser.add_argument("algorithm", type=str, nargs='?', default="GWO",
                        help="The algorithm to animate: GWO, Firefly, ImprovedFirefly, BBBC, GradientDescent")
    args = parser.parse_args()
    algo_name = args.algorithm

    val_min, val_max, step = 1, 50, 1
    x = np.arange(val_min, val_max + step, step)
    y = np.arange(val_min, val_max + step, step)
    X, Y = np.meshgrid(x, y)
    Z = get_z_landscape()

    bot = BasicBot(eval_mode="log_excess")
    dimensions = 2
    max_iterations = 30

    if algo_name == "GWO":
        opt = GWOOptimiser(num_wolves=30, dimensions=dimensions, max_iterations=max_iterations, trading_bot=bot, val_min=-1, val_max=1)
    elif algo_name == "Firefly":
        opt = FireflyOptimiser(num_fireflies=30, dimensions=dimensions, max_iterations=max_iterations, trading_bot=bot)
    elif algo_name == "ImprovedFirefly":
        opt = ImprovedFireflyOptimiser(num_fireflies=30, dimensions=dimensions, max_iterations=max_iterations, trading_bot=bot)
    elif algo_name == "BBBC":
        opt = BigBangBigCrunchOptimiser(population_size=30, dimensions=dimensions, max_iterations=max_iterations, trading_bot=bot)
    elif algo_name == "GradientDescent":
        opt = GradientDescentOptimiser(dimensions=dimensions, max_iterations=max_iterations, sample_count=30, trading_bot=bot)
    else:
        print(f"Unknown algorithm: {algo_name}")
        sys.exit(1)

    history = []
    print(f"Running {algo_name}...")
    for _ in range(max_iterations):
        history.append(weight_to_window(get_pop(opt, algo_name)))
        opt.update()

    if hasattr(opt, "shutdown_pool"):
        opt.shutdown_pool()

    # --- Shared pre-processing ---
    vmax_val = np.percentile(Z, 90)
    vmin_val = np.min(Z)
    Z_clipped = np.clip(Z, vmin_val, vmax_val)

    # Global minimum position in window-space
    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    global_min_x = float(X[min_idx])
    global_min_y = float(Y[min_idx])
    global_min_z = float(Z_clipped[min_idx])

    def get_z_for_points(pop):
        """Look up Z for each population member and float them slightly above the surface."""
        z_vals = []
        for p in pop:
            xi = np.clip(int(round(p[0] - 1)), 0, 49)
            yi = np.clip(int(round(p[1] - 1)), 0, 49)
            z_vals.append(Z_clipped[yi, xi] + (vmax_val - vmin_val) * 0.05)
        return z_vals

    os.makedirs("animations", exist_ok=True)

    # =========================================================================
    # 3D ANIMATION — render each frame with matplotlib using the Agg backend
    # =========================================================================
    print("Generating 3D GIF frames...")
    frames_3d = []

    fig3d = plt.figure(figsize=(12, 9), dpi=100)
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.view_init(elev=30, azim=120)

    ax3d.plot_surface(X, Y, Z_clipped, cmap='viridis', alpha=0.6, edgecolor='none')
    ax3d.set_xlabel('SMA Window Size 1')
    ax3d.set_ylabel('SMA Window Size 2')
    ax3d.set_zlabel('Objective Score (Lower is Better)')
    ax3d.set_zlim(vmin_val, vmax_val)

    # Mark global minimum with a yellow star
    ax3d.plot([global_min_x], [global_min_y], [global_min_z + (vmax_val - vmin_val) * 0.08],
              marker='*', color='yellow', markersize=20, markeredgecolor='black',
              markeredgewidth=1.2, linestyle='', zorder=10, label='Global Min')
    ax3d.legend(loc='upper right')
    fig3d.suptitle(f'{algo_name} Exploring BasicBot Search Space (log_excess)', fontsize=14)

    # Hot pink agents for maximum contrast against viridis
    scat3d, = ax3d.plot([], [], [], marker='o', linestyle='', color='#FF00FF',
                        markersize=12, markeredgecolor='white', markeredgewidth=2.0)

    for frame in range(max_iterations):
        pop = history[frame]
        scat3d.set_data(pop[:, 0], pop[:, 1])
        scat3d.set_3d_properties(get_z_for_points(pop))
        ax3d.set_title(f"Iteration {frame+1}/{max_iterations}")
        frames_3d.append(render_frame_to_image(fig3d))
        print(f"  3D frame {frame+1}/{max_iterations}", end='\r')

    plt.close(fig3d)
    out3d = f"animations/optimisation_{algo_name}_3d.gif"
    print(f"\nSaving 3D animation to {out3d}...")
    frames_3d[0].save(out3d, save_all=True, append_images=frames_3d[1:], duration=250, loop=0)

    # =========================================================================
    # 2D TOP-DOWN ANIMATION
    # =========================================================================
    print("Generating 2D GIF frames...")
    frames_2d = []

    fig2d, ax2d = plt.subplots(figsize=(9, 8), dpi=100)
    fig2d.suptitle(f'{algo_name} — Top-Down View (log_excess)', fontsize=14)

    cf = ax2d.contourf(X, Y, Z_clipped, levels=60, cmap='viridis')
    fig2d.colorbar(cf, ax=ax2d, label='Objective Value')
    ax2d.set_xlabel('SMA Window Size 1')
    ax2d.set_ylabel('SMA Window Size 2')

    # Mark global minimum with a yellow star
    ax2d.plot(global_min_x, global_min_y, marker='*', color='yellow', markersize=20,
              markeredgecolor='black', markeredgewidth=1.2, linestyle='', zorder=10, label='Global Min')
    ax2d.legend(loc='upper right')

    # Hot pink agents for maximum contrast against viridis
    scat2d = ax2d.scatter([], [], c='#FF00FF', s=100, edgecolors='white', linewidths=2.0, zorder=5)
    title2d = ax2d.set_title("Iteration 1")

    for frame in range(max_iterations):
        pop = history[frame]
        scat2d.set_offsets(pop[:, :2])
        title2d.set_text(f"Iteration {frame+1}/{max_iterations}")
        fig2d.canvas.draw()
        frames_2d.append(render_frame_to_image(fig2d))
        print(f"  2D frame {frame+1}/{max_iterations}", end='\r')

    plt.close(fig2d)
    out2d = f"animations/optimisation_{algo_name}_2d.gif"
    print(f"\nSaving 2D animation to {out2d}...")
    frames_2d[0].save(out2d, save_all=True, append_images=frames_2d[1:], duration=250, loop=0)

    print(f"\nDone! Saved:\n  {out3d}\n  {out2d}")

if __name__ == "__main__":
    main()
