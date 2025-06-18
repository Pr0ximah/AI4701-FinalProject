import matplotlib.pyplot as plt
import cv2


def visualize_img(img, title=None, show=True, save=False, save_dir=None, dpi=500):
    """使用matplotlib显示或保存cv2图片，默认高分辨率输出"""
    plt.figure(dpi=dpi)
    if title is not None:
        plt.title(title)
    img_convert = img.copy()
    if img_convert.ndim == 3 and img_convert.shape[2] == 3:
        # 如果是彩色图像，转换为RGB格式
        img_convert = cv2.cvtColor(img_convert, cv2.COLOR_BGR2RGB)
    elif img_convert.ndim == 2:
        img_convert = cv2.cvtColor(img_convert, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError("Unsupported image format")
    plt.imshow(img_convert)
    plt.axis("off")
    if save and save_dir is not None:
        plt.savefig(
            save_dir / f"{title or 'image'}.png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
    if show:
        plt.show()
    plt.close()
