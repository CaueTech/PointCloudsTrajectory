import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectories(
    estimated_traj: np.ndarray, gt_traj: np.ndarray
) -> None:
    """Plota a trajetória estimada vs Ground Truth em 3D."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Trajetória estimada (ICP)
    ax.plot(
        estimated_traj[:, 0],
        estimated_traj[:, 1],
        estimated_traj[:, 2],
        label="Estimated Trajectory (ICP)",
        color="blue",
        linewidth=2,
    )

    # Trajetória real (Ground Truth)
    ax.plot(
        gt_traj[:, 0],
        gt_traj[:, 1],
        gt_traj[:, 2],
        label="Ground Truth Trajectory",
        color="red",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title("Estimated vs Ground Truth Vehicle Trajectory")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.grid(True)
    plt.show()