import numpy as np


class Camera:
    def __init__(self, camera_intrinsic):
        """
        初始化相机对象。

        参数:
            camera_intrinsic (numpy.ndarray): 相机内参矩阵，形状为(3, 3)。
        """
        self.camera_intrinsic = camera_intrinsic
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.matched_indices_2D = []
        self.matched_indices_3D = []
        self.keypoints = None

    def get_pose(self):
        """
        获取相机的位姿。

        返回:
            tuple: 相机的旋转矩阵R和平移向量t。
        """
        return self.R, self.t

    def get_extrinsic(self):
        """
        获取相机的外参矩阵。

        返回:
            numpy.ndarray: 外参矩阵，形状为(3, 4)。
        """
        extrinsic = np.hstack((self.R, self.t))
        return extrinsic

    def get_filtered_keypoints(self):
        """
        获取当前相机的过滤后的关键点。

        返回:
            numpy.ndarray: 过滤后的关键点，形状为(N, 2), 其中N是关键点的数量。
        """
        return self.keypoints[self.matched_indices_2D]
