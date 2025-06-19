from copy import deepcopy
import cv2
from tqdm import tqdm
import numpy as np
from .bundle_adjustment import perform_BA


def perform_pnp_recon(
    cameras_recon_all, features, matches, points3D, camera1_indx, camera2_indx
):
    camera1 = cameras_recon_all[camera1_indx]
    camera2 = cameras_recon_all[camera2_indx]
    feature2 = features[camera2_indx]

    match = matches[(camera1_indx, camera2_indx)]
    index1 = np.array([m.queryIdx for m in match])
    index2 = np.array([m.trainIdx for m in match])
    points2 = np.float32([feature2[0][i].pt for i in index2])

    # 记录特征点
    camera2.keypoints = np.array([p.pt for p in feature2[0]])

    # 寻找匹配点
    valid_index_mask_2D_recon = np.isin(index1, camera1.matched_indices_2D)
    valid_index_mask_3D_for_camera1 = np.isin(
        camera1.matched_indices_2D, index1[valid_index_mask_2D_recon]
    )
    valid_index_mask_3D_recon = np.isin(
        np.arange(points3D.shape[0]),
        camera1.matched_indices_3D[valid_index_mask_3D_for_camera1],
    )
    points2_filtered_recon = points2[valid_index_mask_2D_recon]
    points3D_filtered_recon = points3D[valid_index_mask_3D_recon]

    # 更新相机的匹配索引
    camera2.matched_indices_2D = index2[valid_index_mask_2D_recon]
    camera2.matched_indices_3D = camera1.matched_indices_3D[
        valid_index_mask_3D_for_camera1
    ]

    # 计算PnP
    distCoeffs = np.zeros((4, 1), dtype=np.float32)  # 假设无畸变
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3D_filtered_recon,
        points2_filtered_recon,
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

    return cameras_recon_all


def perform_epipolar_recon(
    cameras_recon_all, features, matches, _, camera1_indx, camera2_indx
):
    camera1 = cameras_recon_all[camera1_indx]
    camera2 = cameras_recon_all[camera2_indx]
    feature1 = features[camera1_indx]
    feature2 = features[camera2_indx]

    match = matches[(camera1_indx, camera2_indx)]
    index1 = np.array([m.queryIdx for m in match])
    index2 = np.array([m.trainIdx for m in match])
    points1 = np.float32([feature1[0][i].pt for i in index1])
    points2 = np.float32([feature2[0][i].pt for i in index2])

    # 记录特征点
    camera2.keypoints = np.array([p.pt for p in feature2[0]])

    # 寻找匹配点
    _, idx1, idx_cam1 = np.intersect1d(
        index1, camera1.matched_indices_2D, return_indices=True
    )

    # 更新相机的匹配索引
    camera2.matched_indices_2D = index2[idx1]
    camera2.matched_indices_3D = camera1.matched_indices_3D[idx_cam1]

    # 计算对极几何
    # 基础矩阵
    F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    # 本质矩阵
    E = camera1.camera_intrinsic.T @ F @ camera1.camera_intrinsic

    # 得到旋转/平移矩阵
    _, R, t, _ = cv2.recoverPose(E, points1, points2, camera1.camera_intrinsic)

    original_transform_matrix = np.vstack(
        (np.hstack((R, t.reshape(3, 1))), np.array([[0, 0, 0, 1]]))
    )
    camera1_extrinsic = camera1.get_extrinsic()
    camera1_transform_matrix = np.vstack((camera1_extrinsic, np.array([[0, 0, 0, 1]])))

    # 计算相机2的外参
    camera2_extrinsic = camera1_transform_matrix @ original_transform_matrix

    # 更新相机位姿
    camera2.R = camera2_extrinsic[:3, :3]
    camera2.t = camera2_extrinsic[:3, 3].reshape(3, 1)

    return cameras_recon_all


def extend_pcd(
    cameras_recon_all,
    features,
    matches,
    points3D,
    images,
    colors,
    camera1_indx,
    camera2_indx,
    max_depth,
):
    camera1 = cameras_recon_all[camera1_indx]
    camera2 = cameras_recon_all[camera2_indx]
    feature1 = features[camera1_indx]
    feature2 = features[camera2_indx]

    match = matches[(camera1_indx, camera2_indx)]
    index1 = np.array([m.queryIdx for m in match])
    index2 = np.array([m.trainIdx for m in match])
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
    valid_indices = np.where((points3D_new[2] > 0) & (points3D_new[2] < max_depth))[0]
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
            np.arange(len(points3D_new)) + len(points3D),
        ),
        axis=0,
    )
    camera2.matched_indices_2D = np.concatenate(
        (camera2.matched_indices_2D, index2), axis=0
    )
    camera2.matched_indices_3D = np.concatenate(
        (
            camera2.matched_indices_3D,
            np.arange(len(points3D_new)) + len(points3D),
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
    points3D = np.vstack((points3D, points3D_new))

    # 合并颜色
    if colors_new is not None and colors is not None:
        colors = np.vstack((colors, colors_new))

    return cameras_recon_all, points3D, colors


def recon_all(
    recon_method,
    points3D,
    features,
    cameras,
    matches,
    max_depth,
    images=None,
    skip_BA=False,
):
    cameras_recon_all = deepcopy(cameras)
    points3D_recon_all = deepcopy(points3D)
    colors_recon_all = np.ones((len(points3D_recon_all), 3)) * 0.5  # 默认灰色

    for i in tqdm(range(len(cameras_recon_all) - 2)):
        # Step 1: 相机0与相机i+2对比，重建相机位姿
        # ============================================================
        if recon_method == "epipolar":
            # 使用对极几何重建相机位姿
            camera1_indx = i + 1
            camera2_indx = i + 2
            # 重建相机位姿
            cameras_recon_all = perform_epipolar_recon(
                cameras_recon_all,
                features,
                matches,
                points3D_recon_all,
                camera1_indx,
                camera2_indx,
            )
        elif recon_method == "pnp":
            # 使用PnP重建相机位姿
            camera1_indx = 0
            camera2_indx = i + 2
            # 重建相机位姿
            cameras_recon_all = perform_pnp_recon(
                cameras_recon_all,
                features,
                matches,
                points3D,
                camera1_indx,
                camera2_indx,
            )

        # Step 2: 扩展点云
        # ============================================================
        if i != 0:  # 第一次迭代不需要扩展点云
            camera1_indx = i + 1
            camera2_indx = i + 2

            # 扩展点云
            (cameras_recon_all, points3D_recon_all, colors_recon_all) = extend_pcd(
                cameras_recon_all,
                features,
                matches,
                points3D_recon_all,
                images,
                colors_recon_all,
                camera1_indx,
                camera2_indx,
                max_depth,
            )

        # Step 3: 对已有点云+相机进行BA优化
        # ============================================================
        if skip_BA:
            # 跳过BA
            continue

        # 只对当前相机和之前的相机进行BA
        points3D_recon_all, cameras_recon_all[: i + 2], colors_recon_all = perform_BA(
            points3D_recon_all,
            cameras_recon_all[: i + 2],
            colors_recon_all,
        )

    return cameras_recon_all, points3D_recon_all, colors_recon_all
