"""
使用plt展示图像的辅助函数
"""

import matplotlib.pyplot as plt


def plt_show(img, title=None):
    """使用matplotlib显示cv2图片"""
    plt.figure(figsize=(5, 5))
    if title is not None:
        plt.title(title)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
