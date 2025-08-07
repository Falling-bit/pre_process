import cv2
import numpy as np
from skimage import morphology, filters
import matplotlib.pyplot as plt


def specialized_preprocessing(img):
    """
    专门针对纸板线条的预处理
    1. 保留线条的同时平滑噪点
    2. 增强线性特征
    """
    # 第一步：使用各向异性扩散滤波保留边缘
    img_float = img.astype(np.float32) / 255.0
    smoothed = np.uint8(255 * filters.rank.mean_bilateral(img, morphology.disk(2)))

    # 第二步：方向感知滤波
    kernel_size = 5
    kernels = []
    for angle in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), 2.0, angle, 5.0, 0.5, 0, ktype=cv2.CV_32F)
        kernels.append(kernel)

    filtered = np.zeros_like(img_float)
    for kernel in kernels:
        filtered = np.maximum(filtered, cv2.filter2D(img_float, cv2.CV_32F, kernel))

    # 归一化并增强对比度
    filtered = np.uint8(255 * (filtered - filtered.min()) / (filtered.max() - filtered.min()))
    filtered = cv2.addWeighted(smoothed, 0.7, filtered, 0.3, 0)

    return filtered


def enhanced_line_detection(img):
    """
    改进的线条检测方法
    1. 使用相位一致性检测线条
    2. 多尺度线状特征增强
    """
    # 相位一致性边缘检测（对光照变化不敏感）
    edges_pc = filters.frangi(img, sigmas=range(1, 4, 1), black_ridges=False)
    edges_pc = np.uint8(255 * (edges_pc - edges_pc.min()) / (edges_pc.max() - edges_pc.min()))

    # 多尺度线状检测
    lines_multi = np.zeros_like(img)
    for scale in [1, 2, 3]:
        lines = filters.meijering(img, sigmas=[scale], black_ridges=False)
        lines = (lines - lines.min()) / (lines.max() - lines.min())
        lines_multi = np.maximum(lines_multi, lines)

    lines_multi = np.uint8(255 * lines_multi)

    # 合并检测结果
    combined = cv2.addWeighted(edges_pc, 0.6, lines_multi, 0.4, 0)

    # 非极大值抑制细化线条
    sobel_x = cv2.Sobel(combined, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(combined, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(sobel_y, sobel_x)
    thinned = np.zeros_like(combined)

    for i in range(1, combined.shape[0] - 1):
        for j in range(1, combined.shape[1] - 1):
            angle = direction[i, j]
            # 找到梯度方向上的相邻像素
            if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle <= 2 * np.pi):
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8):
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8):
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            elif (5 * np.pi / 8 <= angle < 7 * np.pi / 8):
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]
            else:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]

            if magnitude[i, j] >= max(neighbors):
                thinned[i, j] = combined[i, j]

    return thinned


def mark_detected_lines(img, lines, threshold=30):
    """
    标记检测到的线条
    :param img: 原始图像
    :param lines: 检测到的线条图
    :param threshold: 线条强度阈值
    :return: 标记结果图像
    """
    # 二值化线条图
    _, binary = cv2.threshold(lines, threshold, 255, cv2.THRESH_BINARY)

    # 形态学处理连接断线
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 标记线条
    marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    marked[connected > 0] = [0, 0, 255]  # 用红色标记线条

    return marked, connected


def visualize_results(original, preprocessed, lines, marked):
    """可视化各步骤结果"""
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(lines, cmap='gray')
    plt.title('Detected Lines')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(marked)
    plt.title('Marked Result')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def process_image(img_path):
    """处理单张图像"""
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None

    # 专用预处理
    preprocessed = specialized_preprocessing(img)

    # 增强线条检测
    lines = enhanced_line_detection(preprocessed)

    # 标记检测结果
    marked, binary_lines = mark_detected_lines(img, lines)

    # 可视化
    visualize_results(img, preprocessed, lines, marked)

    return {
        'original': img,
        'preprocessed': preprocessed,
        'lines': lines,
        'marked': marked,
        'binary_lines': binary_lines
    }


# 使用示例
if __name__ == "__main__":
    img_path = "./pics/Pic_20250721132121728.bmp"  # 替换为你的图像路径
    result = process_image(img_path)

    # 保存结果
    if result is not None:
        cv2.imwrite("preprocessed.bmp", result['preprocessed'])
        cv2.imwrite("detected_lines.bmp", result['lines'])
        cv2.imwrite("marked_result.bmp", result['marked'])