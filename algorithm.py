import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

class Routine:
    def __init__(self):
        self.pointClouds = []
        self.groundTruthPoses = None
        self.expectedTrajectory = []
        self.generalPosition = None

    def load_obj(self, filePath):
        # Carrega os vértices de um arquivo .obj e aplica downsample voxel para otimizar
        try:
            mesh = trimesh.load(filePath, process=False)
            return self.voxel_downsample(np.array(mesh.vertices), voxelSize=0.5)
        except Exception as e:
            print(f"Unable to load file {filePath}: {e}")
            return None

    def voxel_downsample(self, points, voxelSize):
        """
        Realiza um downsample da nuvem de pontos usando grade voxel.
        Reduz o número de pontos mantendo a estrutura geral.
        """
        coords = np.floor(points / voxelSize).astype(np.int32)
        _, unique_indices = np.unique(coords, axis=0, return_index=True)
        return points[unique_indices]

    def transform_points(self, points, transform):
        """
        Aplica uma transformação 4x4 (matriz homogênea) numa nuvem Nx3.
        """
        n = points.shape[0]
        homo_points = np.hstack((points, np.ones((n, 1))))  # Nx4
        transformed = (transform @ homo_points.T).T  # Nx4
        return transformed[:, :3]

    def icp(self, sourcePoints, targetPoints, maxIterations, tolerance):
        """
        Alinha sourcePoints com targetPoints usando o algoritmo ICP.
        Aplica rejeição de outliers baseada em distância (95% mais próximos).
        """
        T = np.eye(4)
        prevError = None
        src = np.copy(sourcePoints)

        for i in range(maxIterations):
            tree = KDTree(targetPoints)
            distances, indices = tree.query(src)
            correspondences = targetPoints[indices]

            # Rejeição de outliers: mantém apenas os 95% mais próximos
            threshold = np.percentile(distances, 95)
            mask = distances < threshold
            src_filtered = src[mask]
            corr_filtered = correspondences[mask]

            if len(src_filtered) < 3:
                print("Poucos pontos após filtragem, encerrando ICP.")
                break

            centroidSource = np.mean(src_filtered, axis=0)
            centroidTarget = np.mean(corr_filtered, axis=0)

            centeredSource = src_filtered - centroidSource
            centeredTarget = corr_filtered - centroidTarget

            H = np.dot(centeredSource.T, centeredTarget)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(Vt.T, U.T)

            # Correção numérica para evitar acúmulo de erro
            U_r, _, Vt_r = np.linalg.svd(R)
            R = np.dot(U_r, Vt_r)

            t = centroidTarget - np.dot(R, centroidSource)
            src = np.dot(src, R.T) + t

            TUpdate = np.eye(4)
            TUpdate[:3, :3] = R
            TUpdate[:3, 3] = t
            T = np.dot(T, TUpdate)

            meanError = np.mean(np.linalg.norm(src_filtered - corr_filtered, axis=1))
            if prevError is not None and abs(prevError - meanError) < tolerance:
                break
            prevError = meanError

        finalError = meanError
        return T, finalError

    def run_ICP(self):
        # Carrega as nuvens locais e poses da ground truth
        fileNames = [f'./assets/KITTI-Sequence/{i:06d}/{i:06d}_points.obj' for i in range(30)]
        clouds_local = [self.load_obj(name) for name in fileNames]
        self.groundTruthPoses = np.load('./assets/ground_truth.npy')

        # Constrói as poses globais acumuladas da ground truth
        global_poses = [np.eye(4)]
        for i in range(1, len(self.groundTruthPoses)):
            pose_global = global_poses[-1] @ self.groundTruthPoses[i]
            global_poses.append(pose_global)
        global_poses = np.array(global_poses)

        # Aplica as poses globais para transformar as nuvens locais para o espaço global
        self.pointClouds = []
        for i, cloud in enumerate(clouds_local):
            transformed_cloud = self.transform_points(cloud, global_poses[i])
            self.pointClouds.append(transformed_cloud)

        # Inicialização para estimar trajetória
        initial_pose = np.eye(4)
        self.expectedTrajectory = [initial_pose]
        self.generalPosition = initial_pose

        # Roda ICP entre nuvens já posicionadas globalmente
        for i in range(len(self.pointClouds) - 1):
            print(f"Processing scan nº{i}...")

            sourceCloud = self.pointClouds[i]
            targetCloud = self.pointClouds[i + 1]

            relativeTransform, error = self.icp(sourceCloud, targetCloud, 100, 1e-6)

            # Corrige ground truth relativa para comparação
            gt_relative = np.linalg.inv(self.groundTruthPoses[i]) @ self.groundTruthPoses[i + 1]

            print(f"ICP error for scan {i}: {error:.4f}")

            self.generalPosition = self.generalPosition @ relativeTransform
            self.expectedTrajectory.append(self.generalPosition)

        print("Estimated trajectory complete.")
        self.estimatedTrajectory = np.array([pose[:3, 3] for pose in self.expectedTrajectory])

        # Reconstrói o caminho global a partir das poses da ground truth
        ground_truth_global_path = [np.eye(4)]
        current_gt_pose = np.eye(4)
        for relative_gt_transform in self.groundTruthPoses[1:]:
            current_gt_pose = current_gt_pose @ relative_gt_transform
            ground_truth_global_path.append(current_gt_pose)
        self.groundTruthTrajectory = np.array([pose[:3, 3] for pose in ground_truth_global_path])

    def check_RMSE(self):
        """
        Calcula o RMSE entre a trajetória estimada pelo ICP e a trajetória real (ground truth).
        """
        rmse = np.sqrt(np.mean(np.sum((self.estimatedTrajectory - self.groundTruthTrajectory) ** 2, axis=1)))
        print(f"RMSE (estimative vs real): {rmse:.4f}m")

    def plot(self):
        """
        Plota a trajetória estimada pelo ICP junto com a trajetória real (ground truth).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Trajetória estimada (ICP)
        ax.plot(self.estimatedTrajectory[:, 0], self.estimatedTrajectory[:, 1], self.estimatedTrajectory[:, 2],
                label='Estimated Trajectory (ICP)', color='blue', linewidth=2)

        # Trajetória real (Ground Truth)
        ax.plot(self.groundTruthTrajectory[:, 0], self.groundTruthTrajectory[:, 1], self.groundTruthTrajectory[:, 2],
                label='Ground Truth Trajectory', color='red', linewidth=2, linestyle='--')

        ax.set_title('Estimated vs Ground Truth Vehicle Trajectory')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.grid(True)
        plt.show()


# Execução do pipeline completo
firstRun = Routine()
firstRun.run_ICP()
firstRun.check_RMSE()
firstRun.plot()