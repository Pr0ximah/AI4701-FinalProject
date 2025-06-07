"""
3D点云处理工具
"""

import open3d as o3d


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
