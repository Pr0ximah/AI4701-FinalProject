"""
运行Bundle Adjustment
"""

import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm


def perform_BA(points2D_list, points3D_list, camera_poses_list, camera_intrinsic):
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

    # 记录全局迭代次数
    iteration_count = 0
    pbar = None  # 初始化 pbar

    def residuals(params):
        """
        计算残差函数，返回每个2D点与投影点之间的差异。
        """
        nonlocal iteration_count, pbar
        residuals = []
        idx = 0
        for points2D in points2D_list:
            R = params[idx : idx + 9].reshape(3, 3)
            t = params[idx + 9 : idx + 12].reshape(3, 1)
            idx += 12

            for j, point2D in enumerate(points2D):
                point3D = points3D_flat[j * 3 : (j + 1) * 3].reshape(3, 1)
                projected_point = camera_intrinsic @ (R @ point3D + t)
                projected_point /= projected_point[2]  # 齐次坐标归一化

                residuals.append(point2D - projected_point[:2])

        pbar.update(1)
        if pbar.n >= pbar.total:
            pbar.reset(total=params_count)
            iteration_count += 1
            pbar.set_description(f"BA Round {iteration_count}")

        return np.array(residuals).flatten()

    # 初始参数
    initial_params = np.concatenate(camera_params)
    params_count = initial_params.shape[0]
    print(f"优化参数数量: {params_count}")

    # 创建全局进度条
    pbar = tqdm(total=params_count, desc=f"BA Round {iteration_count}")

    try:
        # 运行优化
        result = least_squares(residuals, initial_params, verbose=1)
    finally:
        if pbar:
            pbar.close()  # 确保进度条在优化完成或发生异常时关闭

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
