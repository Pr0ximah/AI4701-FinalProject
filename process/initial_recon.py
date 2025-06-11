import cv2
import numpy as np


def init_recon(features1, features2, camera1, camera2, match):
    # 提取匹配点
    index1 = np.array([m.queryIdx for m in match])
    index2 = np.array([m.trainIdx for m in match])
    points1 = np.float32([features1[0][i].pt for i in index1])
    points2 = np.float32([features2[0][i].pt for i in index2])

    # 基础矩阵
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    print(f"基础矩阵:\n{F}\n")

    # 本质矩阵
    E = camera1.camera_intrinsic.T @ F @ camera1.camera_intrinsic
    print(f"本质矩阵:\n{E}\n")

    # 得到旋转/平移矩阵
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera1.camera_intrinsic)
    print(f"旋转矩阵:\n{R}\n")
    print(f"平移向量:\n{t}\n")

    # 更新相机位姿
    camera2.R = R
    camera2.t = t

    # 三角化生成点云
    points4D = cv2.triangulatePoints(
        camera1.camera_intrinsic @ camera1.get_extrinsic(),
        camera1.camera_intrinsic @ camera2.get_extrinsic(),
        points1.T,
        points2.T,
    )
    points3D = points4D[:3] / points4D[3]  # 转换为非齐次坐标

    # 过滤掉无效的点
    valid_indices = np.where(((points3D[2] > 0) & (points3D[2] < 60)))[0]
    points3D = points3D.T  # 转置为 Nx3 的格式
    points3D = points3D[valid_indices]
    index1 = index1[valid_indices]
    index2 = index2[valid_indices]

    # 记录场景中有效点的索引
    camera1.matched_indices_3D = np.arange(len(points3D))
    camera2.matched_indices_3D = np.arange(len(points3D))
    camera1.matched_indices_2D = index1
    camera2.matched_indices_2D = index2

    return points3D
