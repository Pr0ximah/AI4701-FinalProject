import numpy as np
from copy import deepcopy
import cv2
from tqdm import tqdm


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


def extend_points_cloud(points3D, cameras, features, matches):
    points3D_extended = deepcopy(points3D)
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

        points3D_extended = np.vstack((points3D_extended, points3D_new))

    return points3D_extended
