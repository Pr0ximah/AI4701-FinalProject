import numpy as np
from copy import deepcopy
import cv2
from tqdm import tqdm


def perform_PnP(points3D, features, cameras, matches, added_camera_num):
    points3D_mod = deepcopy(points3D)
    cameras_mod = deepcopy(cameras)
    for i in tqdm(range(added_camera_num)):
        # camera0 = cameras[i]
        camera1 = cameras_mod[i + 1]
        camera2 = cameras_mod[i + 2]
        # feature0 = features[i]
        feature1 = features[i + 1]
        feature2 = features[i + 2]

        # 只考虑相机02/12之间的匹配
        match12 = matches[(i + 1, i + 2)]
        # match02 = matches[(i, i + 2)]

        # 先考虑PnP的匹配
        index1 = np.array([m.queryIdx for m in match12])
        index2 = np.array([m.trainIdx for m in match12])
        points1 = np.float32([feature1[0][i].pt for i in index1])
        points2 = np.float32([feature2[0][i].pt for i in index2])
        valid_index_mask_2D_pnp = np.isin(index1, camera1.matched_indices_2D)
        valid_index_mask_3D_for_camera1 = np.isin(
            camera1.matched_indices_2D, index1[valid_index_mask_2D_pnp]
        )
        valid_index_mask_3D_pnp = np.isin(
            np.arange(points3D_mod.shape[0]),
            camera1.matched_indices_3D[valid_index_mask_3D_for_camera1],
        )
        points2_filtered_pnp = points2[valid_index_mask_2D_pnp]
        points3D_filtered_pnp = points3D_mod[valid_index_mask_3D_pnp]

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
        R, _ = cv2.Rodrigues(rvec)
        camera2.R = R
        camera2.t = tvec

        # 更新点云
        points1_filtered_pnp = points1[~valid_index_mask_2D_pnp]
        points2_filtered_pcd = points2[~valid_index_mask_2D_pnp]
        points4D = cv2.triangulatePoints(
            camera1.camera_intrinsic @ camera1.get_extrinsic(),
            camera2.camera_intrinsic @ camera2.get_extrinsic(),
            points1_filtered_pnp.T,
            points2_filtered_pcd.T,
        )
        points3D_new = points4D[:3] / points4D[3]

        # 过滤掉无效的点
        valid_indices = np.where(((points3D_new[2] > 0) & (points3D_new[2] < 60)))[0]
        points3D_new = points3D_new.T
        points3D_new = points3D_new[valid_indices]

        # 更新相机的匹配索引
        camera1.matched_indices_3D = np.concatenate(
            (
                camera1.matched_indices_3D,
                np.arange(len(points3D_new)) + len(points3D_mod),
            )
        )
        camera2.matched_indices_3D = np.concatenate(
            (
                camera2.matched_indices_3D,
                np.arange(len(points3D_new)) + len(points3D_mod),
            )
        )
        camera1.matched_indices_2D = np.concatenate(
            (
                camera1.matched_indices_2D,
                index1[~valid_index_mask_2D_pnp][valid_indices],
            )
        )
        camera2.matched_indices_2D = np.concatenate(
            (
                camera2.matched_indices_2D,
                index2[~valid_index_mask_2D_pnp][valid_indices],
            )
        )

        # 合并新点云
        points3D_mod = np.vstack((points3D_mod, points3D_new))

    return points3D_mod, cameras_mod
