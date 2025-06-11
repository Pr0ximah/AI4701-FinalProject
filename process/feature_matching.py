import cv2
from itertools import combinations
from tqdm import tqdm


def match_features(features1, features2):
    """
    使用FLANN匹配器对两个特征向量列表进行匹配。

    参数:
        features1 (list): 第一个图像的特征(keypoints, descriptors)。
        features2 (list): 第二个图像的特征(keypoints, descriptors)。

    返回:
        list: 匹配结果列表，每个元素是一个匹配对。
    """
    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 提取描述符
    _, descriptors1 = features1
    _, descriptors2 = features2

    # 执行匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 应用低e比率测试来过滤匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 交叉验证匹配
    matches_rev = flann.knnMatch(descriptors2, descriptors1, k=2)
    good_rev = []
    for m, n in matches_rev:
        if m.distance < 0.7 * n.distance:
            good_rev.append(m)
    rev_pairs = set((m.trainIdx, m.queryIdx) for m in good_rev)
    good_matches = [m for m in good_matches if (m.queryIdx, m.trainIdx) in rev_pairs]

    return good_matches


def match_all_paires(features_list):
    """
    对特征向量列表中的所有图像对进行匹配。

    参数:
        features_list (list): 包含多个图像特征的列表，每个元素是一个元组 (keypoints, descriptors)。

    返回:
        dict: 包含所有图像对的匹配结果，键为图像对的索引元组 (i, j)，值为匹配结果列表。
    """
    # image_pairs = list(combinations(range(len(features_list)), 2))
    # all_matches = {}
    # for i, j in tqdm(image_pairs):
    #     matches = match_features(features_list[i], features_list[j])
    #     matches = convert_matches(matches)  # 转换为可序列化格式
    #     # 存储匹配结果
    #     all_matches[(i, j)] = matches
    # return all_matches
    all_matches = {}
    for i in tqdm(range(len(features_list) - 1)):
        j = i + 1
        matches = match_features(features_list[i], features_list[j])
        matches = convert_matches(matches)  # 转换为可序列化格式
        all_matches[(i, j)] = matches
    return all_matches


def visualize_matches(image1, image2, matches, features1, features2):
    """
    可视化两个图像之间的匹配结果。

    参数:
        image1 (numpy.ndarray): 第一个图像。
        image2 (numpy.ndarray): 第二个图像。
        matches (list): 匹配结果列表。
        features1 (list): 第一个图像的特征(keypoints, descriptors)。
        features2 (list): 第二个图像的特征(keypoints, descriptors)。
    """
    matched_image = cv2.drawMatches(
        image1,
        features1[0],
        image2,
        features2[0],
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return matched_image


def convert_matches(matches):
    """
    将匹配结果转换为可序列化的格式。

    参数:
        matches (list): 匹配结果列表。

    返回:
        list: 序列化后的匹配结果列表。
    """
    return [(m.queryIdx, m.trainIdx, m.distance) for m in matches]


def unconvert_matches(serialized_matches):
    """
    将序列化的匹配结果转换回OpenCV的DMatch对象。

    参数:
        serialized_matches (list): 序列化后的匹配结果列表。

    返回:
        list: OpenCV的DMatch对象列表。
    """
    return [cv2.DMatch(*m) for m in serialized_matches]


def after_load(matches):
    """
    在加载匹配结果后进行处理。

    参数:
        matches (dict): 匹配结果字典，键为图像对的索引元组 (i, j)，值为匹配结果列表。

    返回:
        dict: 处理后的匹配结果字典。
    """
    for key in matches.keys():
        matches[key] = unconvert_matches(matches[key])
    return matches
