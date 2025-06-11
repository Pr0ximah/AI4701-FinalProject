import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm


def perform_BA(points3D, cameras):
    # 初始化优化变量
    camera_params = []
    for camera in cameras:
        R = camera.R.flatten()
        t = camera.t.flatten()
        camera_params.append(np.hstack((R, t)))
    camera_params = np.hstack(camera_params)
    points3D_params = points3D.flatten()
    camera_params_cnt = len(camera_params)
    points3D_params_cnt = len(points3D_params)
    total_params = camera_params_cnt + points3D_params_cnt
    print(f"优化参数数量: {camera_params_cnt} + {points3D_params_cnt} = {total_params}")

    # 初始参数
    initial_params = np.hstack((camera_params, points3D_params))

    def residuals_vectorized(params):
        nonlocal total_params

        # 解析参数
        camera_params = params[:camera_params_cnt].reshape(-1, 12)
        points3D_restore = params[camera_params_cnt:].reshape(-1, 3)

        all_residuals = []

        for i in range(len(cameras)):
            # 获取当前相机的参数
            camera_params_i = camera_params[i, :].reshape(12)

            # 提取旋转和平移向量
            R_vec = camera_params_i[:9]
            t_vec = camera_params_i[9:12]

            R = R_vec.reshape(3, 3)
            t = t_vec.reshape(3, 1)

            # 获取当前相机匹配的3D点
            points3D_filtered = points3D_restore[cameras[i].matched_indices_3D]

            # 向量化投影
            projected = (camera.camera_intrinsic @ (R @ points3D_filtered.T + t)).T
            projected /= projected[:, 2:3]

            # 计算残差
            points2D = cameras[i].get_filtered_keypoints()
            residuals = points2D - projected[:, :2]
            all_residuals.append(residuals)

        return np.concatenate(all_residuals).flatten()

    # 运行优化
    result = least_squares(residuals_vectorized, initial_params, verbose=2)

    # 更新优化后的相机位姿和3D点
    idx = 0
    for i in range(len(cameras)):
        R = result.x[idx : idx + 9].reshape(3, 3)
        t = result.x[idx + 9 : idx + 12].reshape(3, 1)
        cameras[i].R = R
        cameras[i].t = t
        idx += 12

    optimized_points3D = result.x[camera_params_cnt:]

    return optimized_points3D.reshape(-1, 3)


def perform_BA_on_all(points3D, cameras):
    cameras_split = np.array_split(cameras, 20)
    points3D_sum = np.zeros_like(points3D)
    for i, cameras_batch in enumerate(cameras_split):
        print(f"执行批量BA: {i + 1}/{len(cameras_split)}")
        # 执行批量BA
        points3D_sum += perform_BA(points3D, cameras_batch)
    # 合并所有批次的结果
    return points3D_sum / len(points3D_sum)  # 平均化结果
