from pathlib import Path
from typing import List, Tuple
import numpy as np
import trimesh

from src.geometry import transform_points, voxel_downsample


def load_point_cloud(file_path: str, voxel_size: float = 0.5) -> np.ndarray:
    """Carrega vértices de um arquivo .obj e aplica o voxel downsampling."""
    try:
        mesh = trimesh.load(file_path, process=False)
        return voxel_downsample(np.array(mesh.vertices), voxel_size=voxel_size)
    except Exception as e:
        print(f"Erro ao carregar o arquivo {file_path}: {e}")
        return None


def load_kitti_sequence(
    base_dir: str = "./assets", num_scans: int = 30
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Carrega as nuvens locais e as poses relativas de Ground Truth."""
    base_path = Path(base_dir)

    # 1. Carrega ground truth
    gt_poses = np.load(base_path / "ground_truth.npy")

    # 2. Carrega nuvens locais
    clouds_local = []
    for i in range(num_scans):
        scan_file = (
            base_path
            / "KITTI-Sequence"
            / f"{i:06d}"
            / f"{i:06d}_points.obj"
        )
        cloud = load_point_cloud(str(scan_file))
        if cloud is not None:
            clouds_local.append(cloud)

    # 3. Constrói poses globais do Ground Truth
    global_poses = [np.eye(4)]
    for i in range(1, len(gt_poses)):
        pose_global = global_poses[-1] @ gt_poses[i]
        global_poses.append(pose_global)

    # 4. Transforma nuvens locais para o referencial global inicial
    point_clouds = [
        transform_points(cloud, global_poses[i])
        for i, cloud in enumerate(clouds_local)
    ]

    return point_clouds, gt_poses