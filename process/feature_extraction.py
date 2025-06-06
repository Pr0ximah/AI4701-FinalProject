"""
特征提取模块
该模块用于从图像中提取特征向量，使用OpenCV的SIFT算法。
"""

import cv2
from tqdm import tqdm


def extract_features(images):
    """
    从图像列表中提取特征。

    参数:
        images (list): 需要提取特征的图像列表。

    返回:
        list: 每个图像的特征向量列表。
    """
    features = []
    for image in tqdm(images):
        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用SIFT检测关键点并计算描述符
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        # 转换KeyPoints为可序列化格式
        keypoints = convert_keypoints(keypoints)
        # 将特征向量添加到列表中
        features.append((keypoints, descriptors))
    return features


def convert_keypoints(keypoints):
    """
    将特征向量中的KeyPoints转换为可序列化的格式。
    """
    serialized_keypoints = []
    for kp in keypoints:
        # 将KeyPoint对象转换为可序列化的元组
        serialized_kp = (
            (kp.pt[0], kp.pt[1]),  # 位置
            kp.size,  # 尺寸
            kp.angle,  # 角度
            kp.response,  # 响应值
            kp.octave,  # 八度数
            kp.class_id,  # 类ID
        )
        serialized_keypoints.append(serialized_kp)
    return serialized_keypoints


def unconvert_keypoints(serialized_keypoints):
    """
    将序列化的KeyPoints转换回OpenCV的KeyPoint对象。

    参数:
        serialized_keypoints (list): 序列化后的KeyPoints列表。

    返回:
        list: OpenCV的KeyPoint对象列表。
    """
    keypoints = []
    for kp in serialized_keypoints:
        # 将元组转换回KeyPoint对象
        keypoint = cv2.KeyPoint(
            x=kp[0][0],
            y=kp[0][1],
            size=kp[1],
            angle=kp[2],
            response=kp[3],
            octave=kp[4],
            class_id=kp[5],
        )
        keypoints.append(keypoint)
    return keypoints


def after_load(features):
    """
    在加载特征后进行处理。

    参数:
        features (list): 特征向量列表。

    返回:
        list: 处理后的特征向量列表。
    """
    for i, (keypoints_serialized, descriptors) in enumerate(features):
        keypoints = unconvert_keypoints(keypoints_serialized)
        features[i] = (keypoints, descriptors)
    return features
