from typing import Tuple
import numpy as np
from scipy.spatial import KDTree


def icp(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    outlier_percentile: float = 95.0,
) -> Tuple[np.ndarray, float]:
    """Alinha source_points com target_points via algoritmo ICP.

    Aplica rejeição de outliers baseada em percentil de distância.
    """
    T = np.eye(4)
    prev_error = None
    src = np.copy(source_points)
    mean_error = 0.0

    for _ in range(max_iterations):
        tree = KDTree(target_points)
        distances, indices = tree.query(src)
        correspondences = target_points[indices]

        # Rejeição de outliers
        threshold = np.percentile(distances, outlier_percentile)
        mask = distances < threshold
        src_filtered = src[mask]
        corr_filtered = correspondences[mask]

        if len(src_filtered) < 3:
            print("Poucos pontos após filtragem. Encerrando ICP.")
            break

        # Centroides
        centroid_src = np.mean(src_filtered, axis=0)
        centroid_corr = np.mean(corr_filtered, axis=0)

        centered_src = src_filtered - centroid_src
        centered_corr = corr_filtered - centroid_corr

        # SVD para achar a rotação R
        H = np.dot(centered_src.T, centered_corr)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Correção ortogonal
        U_r, _, Vt_r = np.linalg.svd(R)
        R = np.dot(U_r, Vt_r)

        # Translação
        t = centroid_corr - np.dot(R, centroid_src)
        src = np.dot(src, R.T) + t

        # Acumula transformação
        t_update = np.eye(4)
        t_update[:3, :3] = R
        t_update[:3, 3] = t
        T = np.dot(T, t_update)

        mean_error = np.mean(np.linalg.norm(src_filtered - corr_filtered, axis=1))
        if prev_error is not None and abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return T, mean_error