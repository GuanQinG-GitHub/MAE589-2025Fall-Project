import os
import json
import numpy as np
import matplotlib.pyplot as plt 

def _plot_and_save_bo_results(result, explored, avg_mos_vals, param_bounds, out_dir):
        # Scatter plot: param1 vs param2 colored by avg_mos
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(explored[:, 0], explored[:, 1], c=avg_mos_vals, cmap="viridis", s=80, edgecolors="k")
        plt.colorbar(sc, label="Average cost")
        plt.scatter([result.x[0]], [result.x[1]], marker="*", color="red", s=200, label="Best")
        plt.plot([b[0] for b in param_bounds], [b[1] for b in param_bounds], 'r--', alpha=0.5)
        plt.xlabel("Ankle stiffness pitch")
        plt.ylabel("Ankle damping roll")
        plt.title("BO explored points: params vs Average cost")
        plt.legend()
        plt.grid(alpha=0.3)
        fig_path = os.path.join(out_dir, "bo_param_vs_cost.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        print(f"Plot saved to: {fig_path}")
        try:
            plt.show()
        except Exception:
            # In headless environments, showing may fail; continue silently.
            pass

        print("Best stiffness:", result.x)
        print("Best MoS achieved:", -result.fun)
        return result
