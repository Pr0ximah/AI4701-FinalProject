import cv2
from copy import deepcopy
import numpy as np


def init_recon(features1, features2, camera1, camera2, match, img1, img2, max_depth):
    camera1_mod = deepcopy(camera1)
    camera2_mod = deepcopy(camera2)

    # 提取匹配点
    index1 = np.array([m.queryIdx for m in match])
    index2 = np.array([m.trainIdx for m in match])
    points1 = np.float32([features1[0][i].pt for i in index1])
    points2 = np.float32([features2[0][i].pt for i in index2])

    # 记录特征点
    camera1_mod.keypoints = np.array([p.pt for p in features1[0]])
    camera2_mod.keypoints = np.array([p.pt for p in features2[0]])

    # 基础矩阵
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
    print(f"基础矩阵:\n{F}\n")

    # 本质矩阵
    E = camera1_mod.camera_intrinsic.T @ F @ camera1_mod.camera_intrinsic
    print(f"本质矩阵:\n{E}\n")

    # 得到旋转/平移矩阵
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera1_mod.camera_intrinsic)
    print(f"旋转矩阵:\n{R}\n")
    print(f"平移向量:\n{t}\n")

    # 更新相机位姿
    camera2_mod.R = R
    camera2_mod.t = t

    # 三角化生成点云
    points4D = cv2.triangulatePoints(
        camera1_mod.camera_intrinsic @ camera1_mod.get_extrinsic(),
        camera1_mod.camera_intrinsic @ camera2_mod.get_extrinsic(),
        points1.T,
        points2.T,
    )
    points3D = points4D[:3] / points4D[3]  # 转换为非齐次坐标

    # 过滤掉无效的点
    valid_indices = np.where(((points3D[2] > 0) & (points3D[2] < max_depth)))[0]
    points3D = points3D.T  # 转置为 Nx3 的格式
    points3D = points3D[valid_indices]
    index1 = index1[valid_indices]
    index2 = index2[valid_indices]

    # 从图像中提取颜色
    colors = None
    if img1 is not None:
        valid_points1 = points1[valid_indices]
        colors = np.zeros((len(valid_points1), 3))
        for i, pt in enumerate(valid_points1):
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                # 获取BGR颜色并转换为RGB
                color = img1[y, x][::-1] / 255.0  # BGR->RGB, 归一化到[0,1]
                colors[i] = color
            else:
                colors[i] = [0.5, 0.5, 0.5]  # 默认灰色

    # 记录场景中有效点的索引
    camera1_mod.matched_indices_3D = np.arange(len(points3D))
    camera2_mod.matched_indices_3D = np.arange(len(points3D))
    camera1_mod.matched_indices_2D = index1
    camera2_mod.matched_indices_2D = index2

    return points3D, colors, camera1_mod, camera2_mod
