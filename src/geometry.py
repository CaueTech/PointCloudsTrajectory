import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.5) -> np.ndarray:
    """Realiza o downsample da nuvem de pontos agrupando por grade voxel."""
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    return points[unique_indices]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Aplica uma matriz de transformação homogênea 4x4 numa nuvem Nx3."""
    n = points.shape[0]
    homo_points = np.hstack((points, np.ones((n, 1))))  # Nx4
    transformed = (transform @ homo_points.T).T  # Nx4
    return transformed[:, :3]