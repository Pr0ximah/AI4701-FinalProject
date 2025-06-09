"""
运行Bundle Adjustment
"""

import numpy as np
from scipy.optimize import least_squares


def perform_BA(
    points2D_list, points3D_list, camera_poses_list, camera_intrinsic
):
    """
    使用OpenCV的全局优化函数进行Bundle Adjustment。

    参数:
        points2D_list (list): 每个相机的2D点列表。
        points3D_list (list): 3D点列表。
        camera_poses_list (list): 每个相机的位姿列表(R, t)。
        camera_intrinsic (numpy.ndarray): 相机内参矩阵。

    返回:
        优化后的相机位姿和3D点。
    """
    # 将相机位姿转换为优化所需的格式
    camera_params = []
    for R, t in camera_poses_list:
        camera_params.append(R.flatten())
        camera_params.append(t.flatten())

    # 将3D点转换为优化所需的格式
    points3D_flat = np.concatenate([p.flatten() for p in points3D_list])

    # 准备优化问题
    def residuals(params):
        residuals = []
        idx = 0
        for points2D in points2D_list:
            R = params[idx : idx + 9].reshape(3, 3)
            t = params[idx + 9 : idx + 12].reshape(3, 1)
            idx += 12

            for j, point2D in enumerate(points2D):
                point3D = points3D_flat[j * 3 : (j + 1) * 3]
                projected_point = camera_intrinsic @ (R @ point3D + t)
                projected_point /= projected_point[2]  # 齐次坐标归一化

                residuals.append(point2D - projected_point[:2])

        return np.array(residuals).flatten()

    result = least_squares(residuals, np.concatenate(camera_params))

    # 提取优化后的相机位姿和3D点
    optimized_camera_poses = []
    idx = 0
    for i in range(len(camera_poses_list)):
        R = result.x[idx : idx + 9].reshape(3, 3)
        t = result.x[idx + 9 : idx + 12].reshape(3, 1)
        optimized_camera_poses.append((R, t))
        idx += 12

    optimized_points3D = result.x

    return optimized_camera_poses, optimized_points3D.reshape(-1, 3)
