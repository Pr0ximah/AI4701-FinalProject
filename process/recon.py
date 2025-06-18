from copy import deepcopy
import cv2
from tqdm import tqdm
import numpy as np


def perform_PnP(points3D, features, cameras, matches):
    cameras_mod = deepcopy(cameras)
    for i in tqdm(range(len(cameras_mod) - 2)):
        camera1_indx = 0
        camera2_indx = i + 2

        camera1 = cameras_mod[camera1_indx]
        camera2 = cameras_mod[camera2_indx]
        feature1 = features[camera1_indx]
        feature2 = features[camera2_indx]

        # 只考虑相机02/12之间的匹配
        match12 = matches[(camera1_indx, camera2_indx)]

        # 先考虑PnP的匹配
        index1 = np.array([m.queryIdx for m in match12])
        index2 = np.array([m.trainIdx for m in match12])
        points2 = np.float32([feature2[0][i].pt for i in index2])

        # 记录特征点
        camera2.keypoints = np.array([p.pt for p in feature2[0]])

        # 寻找匹配点
        valid_index_mask_2D_pnp = np.isin(index1, camera1.matched_indices_2D)
        valid_index_mask_3D_for_camera1 = np.isin(
            camera1.matched_indices_2D, index1[valid_index_mask_2D_pnp]
        )
        valid_index_mask_3D_pnp = np.isin(
            np.arange(points3D.shape[0]),
            camera1.matched_indices_3D[valid_index_mask_3D_for_camera1],
        )
        points2_filtered_pnp = points2[valid_index_mask_2D_pnp]
        points3D_filtered_pnp = points3D[valid_index_mask_3D_pnp]

        # 更新相机的匹配索引
        camera2.matched_indices_2D = index2[valid_index_mask_2D_pnp]
        camera2.matched_indices_3D = camera1.matched_indices_3D[
            valid_index_mask_3D_for_camera1
        ]

        # 计算PnP
        distCoeffs = np.zeros((4, 1), dtype=np.float32)  # 假设无畸变
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3D_filtered_pnp,
            points2_filtered_pnp,
            camera2.camera_intrinsic,
            distCoeffs,
        )

        if not success:
            raise RuntimeError(
                f"PnP failed for camera pair {camera1_indx} and {camera2_indx}."
            )

        R, _ = cv2.Rodrigues(rvec)
        camera2.R = R
        camera2.t = tvec

    return cameras_mod


def extend_points_cloud(points3D, cameras, features, matches, images=None):
    points3D_extended = deepcopy(points3D)
    colors_extended = None

    # 初始化颜色数组
    if images is not None:
        colors_extended = np.ones((len(points3D_extended), 3)) * 0.5  # 默认灰色

    for i in tqdm(range(len(cameras) - 2)):
        camera1_indx = i + 1
        camera2_indx = i + 2
        camera1 = cameras[camera1_indx]
        camera2 = cameras[camera2_indx]
        feature1 = features[camera1_indx]
        feature2 = features[camera2_indx]
        match12 = matches[(camera1_indx, camera2_indx)]
        index1 = np.array([m.queryIdx for m in match12])
        index2 = np.array([m.trainIdx for m in match12])
        points1 = np.float32([feature1[0][i].pt for i in index1])
        points2 = np.float32([feature2[0][i].pt for i in index2])

        # 基线角度过滤（剔除视线夹角过小的匹配）
        invK1 = np.linalg.inv(camera1.camera_intrinsic)
        invK2 = np.linalg.inv(camera2.camera_intrinsic)
        thresh_deg = 1.0  # 最小角度阈值（度）
        mask = []
        for p1, p2 in zip(points1, points2):
            # 在相机坐标系下的 bearing vector
            b1 = invK1 @ np.array([p1[0], p1[1], 1.0])
            b2 = invK2 @ np.array([p2[0], p2[1], 1.0])
            # 转到世界坐标系
            d1 = camera1.R.T @ b1
            d2 = camera2.R.T @ b2
            # 计算夹角
            cosang = np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))
            ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            mask.append(ang >= thresh_deg)
        mask = np.array(mask, dtype=bool)
        points1 = points1[mask]
        points2 = points2[mask]
        index1 = index1[mask]
        index2 = index2[mask]

        # 生成新的三维点
        points4D = cv2.triangulatePoints(
            camera1.camera_intrinsic @ camera1.get_extrinsic(),
            camera2.camera_intrinsic @ camera2.get_extrinsic(),
            points1.T,
            points2.T,
        )
        points3D_new = points4D[:3] / points4D[3]

        # 过滤掉无效的点
        valid_indices = np.where((points3D_new[2] > 0) & (points3D_new[2] < 60))[0]
        points3D_new = points3D_new.T
        points3D_new = points3D_new[valid_indices]
        index1 = index1[valid_indices]
        index2 = index2[valid_indices]

        # 更新相机的匹配索引
        camera1.matched_indices_2D = np.concatenate(
            (camera1.matched_indices_2D, index1), axis=0
        )
        camera1.matched_indices_3D = np.concatenate(
            (
                camera1.matched_indices_3D,
                np.arange(len(points3D_new)) + len(points3D_extended),
            ),
            axis=0,
        )
        camera2.matched_indices_2D = np.concatenate(
            (camera2.matched_indices_2D, index2), axis=0
        )
        camera2.matched_indices_3D = np.concatenate(
            (
                camera2.matched_indices_3D,
                np.arange(len(points3D_new)) + len(points3D_extended),
            ),
            axis=0,
        )

        # 提取颜色
        colors_new = None
        if images is not None and len(points3D_new) > 0:
            img1 = images[camera1_indx]
            valid_points1 = points1[valid_indices]
            colors_new = np.zeros((len(valid_points1), 3))
            for j, pt in enumerate(valid_points1):
                x, y = int(pt[0]), int(pt[1])
                if 0 <= y < img1.shape[0] and 0 <= x < img1.shape[1]:
                    # 获取BGR颜色并转换为RGB
                    color = img1[y, x][::-1] / 255.0  # BGR->RGB, 归一化到[0,1]
                    colors_new[j] = color
                else:
                    colors_new[j] = [0.5, 0.5, 0.5]  # 默认灰色

        # 合并新点云
        points3D_extended = np.vstack((points3D_extended, points3D_new))

        # 合并颜色
        if colors_new is not None and colors_extended is not None:
            colors_extended = np.vstack((colors_extended, colors_new))

    return points3D_extended, colors_extended


def pnp_recon(points3D, features, cameras, matches, images=None):
    # 执行PnP算法来估计相机姿态
    cameras_pnp = perform_PnP(points3D, features, cameras, matches)
    points3D_pnp, colors3D_pnp = extend_points_cloud(
        points3D, cameras_pnp, features, matches, images
    )
    return cameras_pnp, points3D_pnp, colors3D_pnp
