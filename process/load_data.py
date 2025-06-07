"""
从文件中加载图像和相机内参
"""

from pathlib import Path
import numpy as np
import cv2


def load_images_and_camera_intrinsic(data_path):
    """
    从指定路径加载图像和相机内参。

    参数:
        data_path (str or Path): 包含图像和相机内参的目录路径。、

    返回:
        images (list): 图像列表。
        camera_intrinsic (numpy.ndarray): 相机内参矩阵，形状为(3, 3)。
    """
    data_path = Path(data_path)
    img_folder = data_path / "images"
    camera_intrinsic_file = data_path / "camera_intrinsic.txt"

    images = []
    camera_intrinsic = None

    # 读取图像
    all_image_files = list(img_folder.glob("*.jpg"))
    all_image_files.sort()  # 确保图像文件按名称排序
    all_image_files.reverse()  # 按要求的顺序排列
    for img_file in all_image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)

    # 读取相机内参
    if camera_intrinsic_file.exists():
        with open(camera_intrinsic_file, "r") as f:
            lines = f.readlines()
            camera_intrinsic = np.array(
                [list(map(float, line.split())) for line in lines]
            )
    else:
        raise FileNotFoundError(f"相机内参文件不存在 {camera_intrinsic_file}")

    return images, camera_intrinsic
