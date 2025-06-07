# use epipolar geometry to find the fundamental matrix and the essential matrix
# use the essential matrix to find the relative pose between two cameras
# use the relative pose to triangulate 3D points

import cv2
import numpy as np


def init_recon(features1, features2, matches, camera_intrinsic):
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
    points1 = np.float32([features1[0][i].pt for i in index1])
    points2 = np.float32([features2[0][i].pt for i in index2])

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
        camera_intrinsic
        @ np.hstack((np.eye(3, 3), np.zeros((3, 1)))),  # 第一个相机的投影矩阵
        camera_intrinsic @ np.hstack((R, t)),  # 第二个相机的投影矩阵
        points1.T,  # 第一个图像的点
        points2.T,  # 第二个图像的点
    )
    points3D = points4D[:3] / points4D[3]  # 转换为非齐次坐标
    valid_indices = np.where((points3D[2] > 0))[0]
    points3D = points3D.T  # 转置为 Nx3 的格式
    points3D = points3D[valid_indices]

    return points3D, (R, t)


def visualize_camera_pose_and_pcd(camera_poses, points3D):
    """
    可视化相机和点云位姿。

    参数:
        camera_poses (list): 相机位姿列表，每个元素为(R, t)元组。
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
    """
    import open3d as o3d

    # 添加第一个相机的位姿（原点处）
    all_poses = [(np.eye(3), np.zeros((3, 1)))] + camera_poses

    # 创建相机几何体
    cameras = []
    for i, (R, t) in enumerate(all_poses):
        # 创建更大更明显的坐标系
        size = 5
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

        # 创建相机锥体（更直观的相机表示）
        camera_cone = o3d.geometry.TriangleMesh.create_cone(radius=0.3, height=0.8)
        camera_cone.paint_uniform_color(
            [0.8, 0.2, 0.2] if i == 0 else [0.2, 0.8, 0.2]
        )  # 不同颜色区分

        # 旋转锥体使其指向z轴负方向（相机朝向）
        camera_cone.rotate(
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), center=(0, 0, 0)
        )

        # 应用相机的旋转和平移
        coordinate_frame.rotate(R, center=(0, 0, 0))
        coordinate_frame.translate(t.flatten())
        camera_cone.rotate(R, center=(0, 0, 0))
        camera_cone.translate(t.flatten())

        cameras.extend([coordinate_frame, camera_cone])

    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)

    # 为点云着色（使用更淡的颜色，突出相机）
    colors = np.random.uniform(0.3, 0.7, size=(len(points3D), 3))  # 更淡的随机颜色
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 设置可视化参数
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="相机位姿和点云", width=1200, height=800)

    # 添加相机几何体
    for camera in cameras:
        vis.add_geometry(camera)

    # 添加点云
    vis.add_geometry(pcd)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.05, 0.05, 0.05])  # 更深的背景色
    render_option.point_size = 1.5  # 稍小的点云尺寸
    render_option.mesh_show_wireframe = False
    render_option.mesh_show_back_face = True

    # 运行可视化
    vis.run()
    vis.destroy_window()
