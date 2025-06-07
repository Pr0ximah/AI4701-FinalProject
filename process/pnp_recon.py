"""
使用 PnP 算法估计相机姿态
"""

import numpy as np
import cv2


def perform_PnP(featrues1, features2, matches, camera_intrinsic):
    index1 = [m.queryIdx for m in matches]
    index2 = [m.trainIdx for m in matches]
    points1 = np.float32([featrues1[0][i].pt for i in index1])
    points2 = np.float32([features2[0][i].pt for i in index2])

    distCoeffs = np.zeros((4, 1), dtype=np.float32)  # 假设无畸变

    # 使用 solvePnP 求解相机姿态
    success, r, t = cv2.solvePnP(points3D, points2D, cameraMatrix, distCoeffs)

    if not success:
        raise RuntimeError("PnP estimation failed")

    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(r)

    return R, t


def perform_PnP_on_all(points3D, keypoints_list, cameraMatrix):
    """
    对所有图像点对执行 PnP。

    参数:
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
        points2D_list (list): 包含多个 2D 图像点的列表，每个元素是一个形状为(N, 2)的 numpy 数组。
        cameraMatrix (numpy.ndarray): 相机内参矩阵，形状为(3, 3)。

    返回:
        list: 每个元素是一个元组 (R, t)，表示相机的旋转矩阵和平移向量。
    """
    results = []
    for points2D in keypoints_list:
        R, t = perform_PnP(points3D, points2D[0], cameraMatrix)
        results.append((R, t))
    return results
