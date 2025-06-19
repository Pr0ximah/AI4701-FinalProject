import numpy as np
from scipy.optimize import least_squares
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from tqdm import tqdm


def _rotate(points, rvecs):
    theta = np.linalg.norm(rvecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rvecs / theta
        v = np.nan_to_num(v)  # 替换NaN为0
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return (
        cos_theta * points + sin_theta * np.cross(v, points) + (1 - cos_theta) * dot * v
    )


def _project(points, camera_params, camera_intrinsic):
    points_proj = _rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = camera_intrinsic @ points_proj.T
    points_proj = points_proj[:2, :] / points_proj[2, :]
    return points_proj.T


def _residuals(points3D_params, camera_params, cameras):
    _camera_params = camera_params.reshape(-1, 6)
    _points3D_params = points3D_params.reshape(-1, 3)

    all_residuals = []
    for i, camera in enumerate(cameras):
        # 获取当前相机的参数
        camera_params_i = _camera_params[i, :].reshape(6)

        # 提取旋转和平移向量
        rvec = camera_params_i[:3]
        t = camera_params_i[3:6]

        # 获取当前相机匹配的3D点
        points3D_filtered = _points3D_params[camera.matched_indices_3D]

        # 投影
        projected = _project(
            points3D_filtered,
            np.tile(camera_params_i, (len(points3D_filtered), 1)),
            camera.camera_intrinsic,
        )

        # 计算残差
        points2D = camera.get_filtered_keypoints()
        residuals = points2D - projected

        all_residuals.append(residuals.ravel())

    return np.concatenate(all_residuals).ravel()


def _bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A


def perform_BA(points3D, cameras, colors):
    points3D_copy = points3D.copy()
    cameras_copy = cameras.copy()[1:]  # 排除第一个相机
    colors_copy = colors.copy()
    # 初始化优化变量
    camera_params = []
    for camera in cameras_copy:
        rvec = cv2.Rodrigues(camera.R)[0].flatten()
        t = camera.t.flatten()
        camera_params.append(np.hstack((rvec, t)))
    camera_params = np.hstack(camera_params)
    points3D_params = points3D_copy.flatten()
    camera_params_cnt = len(camera_params)
    points3D_params_cnt = len(points3D_params)
    total_params = camera_params_cnt + points3D_params_cnt
    print(f"优化参数数量: {camera_params_cnt} + {points3D_params_cnt} = {total_params}")

    # 初始参数
    initial_params = np.hstack((camera_params, points3D_params))

    camera_indices = []
    points_indices = []
    for camera in cameras_copy:
        points_indices.extend(camera.matched_indices_3D)
        camera_indices.extend(
            [cameras_copy.index(camera)] * len(camera.matched_indices_3D)
        )
    camera_indices = np.array(camera_indices, dtype=int)
    points_indices = np.array(points_indices, dtype=int)

    A = _bundle_adjustment_sparsity(
        len(cameras_copy),
        points3D_copy.shape[0],
        camera_indices,
        points_indices,
    )

    # 执行最小二乘优化
    result = least_squares(
        _residuals,
        initial_params,
        args=(camera_params, cameras_copy),
        jac_sparsity=A,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        loss="cauchy",
        f_scale=1.0,
        max_nfev=50,
        verbose=2,
    )

    # 仅更新有效的点云和相机参数
    # 过滤掉过远或过近的点
    optimized_points3D = result.x[camera_params_cnt:].reshape(-1, 3)
    valid_points_mask = (
        np.all(np.abs(optimized_points3D) < 100, axis=1)
        & (np.linalg.norm(optimized_points3D, axis=1) < 100)
        & (np.linalg.norm(optimized_points3D, axis=1) > 0.1)
    )

    optimized_points3D = optimized_points3D[valid_points_mask]

    # 更新相机参数时应用一些约束
    idx = 0
    for i, camera in enumerate(cameras_copy):
        R = cv2.Rodrigues(result.x[idx : idx + 3])[0]
        t = result.x[idx + 3 : idx + 6].reshape(3, 1)

        # 约束: 保持相机的上向量
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt

        camera.R = R
        camera.t = t
        idx += 6

    # 将第一个相机添加到相机列表
    cameras_copy.insert(0, cameras[0])

    return optimized_points3D, cameras_copy, colors_copy[valid_points_mask]
