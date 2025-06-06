# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import cv2
import numpy as np
import open3d as o3d


def init_recon(features1, features2, camera_intrinsic, matches):
    """
    初始化重建过程，计算基础矩阵和本质矩阵。

    参数:
        features1 (list): 第一个图像的特征(keypoints, descriptors)。
        features2 (list): 第二个图像的特征(keypoints, descriptors)。
        camera_intrinsic (numpy.ndarray): 相机内参矩阵。
        matches (list): 匹配结果列表。

    返回:
        3D点云和相机位姿。
    """
    # 提取匹配点
    index1 = [m.queryIdx for m in matches]
    index2 = [m.trainIdx for m in matches]
    points1 = np.float32([features1[0][i] for i in index1].pt).reshape(-1, 1, 2)
    points2 = np.float32([features2[0][i] for i in index2].pt).reshape(-1, 1, 2)

    # 计算基础矩阵
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    print(f"基础矩阵:\n{F}\n")

    # 计算本质矩阵
    E = camera_intrinsic.T @ F @ camera_intrinsic
    print(f"本质矩阵:\n{E}\n")

    # 得到旋转/平移矩阵
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_intrinsic)
    print(f"旋转矩阵:\n{R}\n")
    print(f"平移向量:\n{t}\n")

    # 三角化生成点云
    points4D = cv2.triangulatePoints(
        np.hstack((np.eye(3, 4), np.zeros((3, 1)))),  # 第一个相机的投影矩阵
        np.hstack((R, t)),  # 第二个相机的投影矩阵
        points1.T,  # 第一个图像的点
        points2.T,  # 第二个图像的点
    )
    points3D = points4D[:3] / points4D[3]  # 转换为非齐次坐标
    points3D = points3D.T  # 转置为 Nx3 的格式

    return points3D, R, t


def visualize_point_cloud(points3D):
    """
    可视化3D点云。

    参数:
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)

    # 可视化点云
    o3d.visualization.draw_geometries([pcd], window_name="3D Point Cloud")
