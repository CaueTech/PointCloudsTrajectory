import numpy as np


def compute_rmse(estimated_path: np.ndarray, gt_path: np.ndarray) -> float:
    """Calcula o Root Mean Square Error (RMSE) entre as trajetórias."""
    return float(
        np.sqrt(np.mean(np.sum((estimated_path - gt_path) ** 2, axis=1)))
    )