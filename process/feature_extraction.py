import cv2


def extract_features(images):
    """
    从图像列表中提取特征。

    参数:
        images (list): 需要提取特征的图像列表。

    返回:
        list: 每个图像的特征向量列表。
    """
    features = []
    for image in images:
        # 将图像转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 使用SIFT检测关键点并计算描述符
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        features.append((keypoints, descriptors))
    return features
