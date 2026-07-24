import numpy as np
from src.data_loader import load_kitti_sequence
from src.icp import icp
from src.metrics import compute_rmse
from src.visualization import plot_trajectories


def run_pipeline():
    print("1. Carregando dados do KITTI...")
    point_clouds, gt_poses = load_kitti_sequence(base_dir="./assets", num_scans=30)

    # Inicializa trajetórias
    general_position = np.eye(4)
    expected_trajectory = [general_position]

    print("2. Executando alinhamento via ICP...")
    for i in range(len(point_clouds) - 1):
        print(f"   Processando scan nº {i}...")

        source_cloud = point_clouds[i]
        target_cloud = point_clouds[i + 1]

        relative_transform, error = icp(
            source_cloud, target_cloud, max_iterations=100, tolerance=1e-6
        )

        print(f"   Erro ICP para scan {i}: {error:.4f}")

        general_position = general_position @ relative_transform
        expected_trajectory.append(general_position)

    estimated_trajectory = np.array([pose[:3, 3] for pose in expected_trajectory])

    # Reconstrói trajetória global de referência (Ground Truth)
    gt_global_path = [np.eye(4)]
    current_gt_pose = np.eye(4)
    for rel_gt in gt_poses[1:]:
        current_gt_pose = current_gt_pose @ rel_gt
        gt_global_path.append(current_gt_pose)
    gt_trajectory = np.array([pose[:3, 3] for pose in gt_global_path])

    # 3. Avaliação de Métricas
    rmse = compute_rmse(estimated_trajectory, gt_trajectory)
    print(f"\nRMSE (Estimado vs Real): {rmse:.4f} m")

    # 4. Visualização
    print("3. Exibindo visualização 3D...")
    plot_trajectories(estimated_trajectory, gt_trajectory)


if __name__ == "__main__":
    run_pipeline()