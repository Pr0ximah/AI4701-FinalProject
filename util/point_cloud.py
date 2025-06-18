import open3d as o3d
import numpy as np


def convert_to_point_cloud(points3D):
    """
    将3D点云转换为Open3D格式。

    参数:
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。

    返回:
        Open3D点云对象。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    return pcd


def save_point_cloud(pcd, filename):
    """
    保存3D点云到文件。

    参数:
        pcd (open3d.geometry.PointCloud): Open3D点云对象。
        filename (str): 保存的文件名。
    """
    o3d.io.write_point_cloud(filename, pcd)
    print(f"点云已保存到 {filename}")


def load_point_cloud(filename):
    """
    从文件加载3D点云。

    参数:
        filename (str): 点云文件名。

    返回:
        Open3D点云对象。
    """
    pcd = o3d.io.read_point_cloud(filename)
    print(f"点云已从 {filename} 加载")
    return pcd


def visualize_point_cloud(pcd, window_name="3D Point Cloud"):
    """
    可视化3D点云。

    参数:
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
    """
    # 可视化点云
    o3d.visualization.draw_geometries(
        [pcd],
        window_name=window_name,
    )


def create_point_cloud(points3D, colors=None):
    """
    创建Open3D点云对象。

    参数:
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
        colors (numpy.ndarray, optional): 点云颜色数据，形状为(N, 3)，范围[0,1]。

    返回:
        Open3D点云对象。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def visualize_camera_pose_and_pcd(camera_poses, points3D, colors=None):
    """
    可视化相机和点云位姿。

    参数:
        camera_poses (list): 相机位姿列表，每个元素为(R, t)元组。
        points3D (numpy.ndarray): 3D点云数据，形状为(N, 3)。
        colors (numpy.ndarray, optional): 点云颜色数据，形状为(N, 3)，范围[0,1]。
    """
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

    # 为点云着色
    if colors is not None:
        # 使用传入的颜色
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # 使用更淡的随机颜色，突出相机
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

    view_control = vis.get_view_control()
    # # 将视野中心设置为原点
    view_control.set_lookat(np.array([0.0, 0.0, 0.0]))
    # 将视角对齐到相机0的姿态（朝向-Z，up为-Y）
    view_control.set_front(np.array([0.0, 0.0, -1.0]))
    view_control.set_up(np.array([0.0, -1.0, 0.0]))
    view_control.set_lookat(np.array([0.0, 0.0, 0.0]))
    view_control.set_zoom(0.1)

    # 运行可视化
    vis.run()
    vis.destroy_window()
