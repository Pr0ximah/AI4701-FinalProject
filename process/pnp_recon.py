"""
使用 PnP 算法估计相机姿态
"""

import numpy as np
import cv2
from tqdm import tqdm


def perform_PnP(
    featrues1, features2, points3D_ori, matches, camera_intrinsic, recon_valid_points
):
    index1 = [m.queryIdx for m in matches]
    index2 = [m.trainIdx for m in matches]
    points2D = np.float32([features2[0][i].pt for i in index2])

    # 只选择有效点
    valid_index_mask = np.isin(index1, recon_valid_points)
    valid_index_mask2 = np.isin(recon_valid_points, index1)
    points2D = points2D[valid_index_mask]
    points3D = points3D_ori[valid_index_mask2]

    distCoeffs = np.zeros((4, 1), dtype=np.float32)  # 假设无畸变

    # 使用 solvePnP 求解相机姿态
    success, r, t = cv2.solvePnP(points3D, points2D, camera_intrinsic, distCoeffs)

    if not success:
        raise RuntimeError("PnP estimation failed")

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(r)

    return R, t


def perform_PnP_on_all(
    points3D_ori, features, matches, camera_intrinsic, recon_valid_points, camera_pose
):
    """
    对所有图像点对执行 PnP。
    """
    results = [camera_pose]  # 初始化结果为已有的相机位姿
    for i, feature_cam in tqdm(enumerate(features[2:])):
        R, t = perform_PnP(
            features[0],
            feature_cam,
            points3D_ori,
            matches[(0, i + 2)],
            camera_intrinsic,
            recon_valid_points,
        )
        results.append((R, t))
    return results
