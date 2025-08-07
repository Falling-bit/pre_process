import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob


def load_images(folder_path):
    """加载文件夹中的所有bmp图像"""
    image_paths = sorted(glob(os.path.join(folder_path, '*.bmp')))
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]
    return images, image_paths


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """应用对比度受限的自适应直方图均衡化(CLAHE)"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def contrast_stretching(image, lower_percent=2, upper_percent=98):
    """对比度拉伸"""
    # 计算上下百分位值
    lower_val = np.percentile(image, lower_percent)
    upper_val = np.percentile(image, upper_percent)

    # 线性拉伸
    stretched = np.clip((image - lower_val) * (255.0 / (upper_val - lower_val)), 0, 255)
    return stretched.astype(np.uint8)


def edge_enhancement(image):
    """边缘增强"""
    # 高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # 非锐化掩蔽
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def morphological_operations(image):
    """形态学操作增强"""
    # 使用顶帽变换增强亮特征
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)

    # 增强后的图像
    enhanced = cv2.add(image, tophat)
    return enhanced


def process_images(images):
    """应用所有预处理方法"""
    processed = []

    for img in images:
        # 原始图像
        original = img.copy()

        # 方法1: CLAHE
        clahe_img = apply_clahe(original)

        # 方法2: 对比度拉伸
        stretched_img = contrast_stretching(original)

        # 方法3: 边缘增强
        edge_img = edge_enhancement(original)

        # 方法4: 形态学操作
        morph_img = morphological_operations(original)

        # 方法5: CLAHE + 边缘增强
        combined1 = edge_enhancement(apply_clahe(original))

        # 方法6: 对比度拉伸 + 形态学
        combined2 = morphological_operations(contrast_stretching(original))

        processed.append({
            'original': original,
            'clahe': clahe_img,
            'stretched': stretched_img,
            'edge': edge_img,
            'morph': morph_img,
            'combined1': combined1,
            'combined2': combined2
        })

    return processed


def visualize_results(processed_images, image_paths, save_folder='results'):
    """可视化预处理结果"""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    methods = ['original', 'clahe', 'stretched', 'edge', 'morph', 'combined1', 'combined2']
    titles = ['Original', 'CLAHE', 'Contrast Stretched', 'Edge Enhanced',
              'Morphological', 'CLAHE+Edge', 'Stretched+Morph']

    for idx, result in enumerate(processed_images):
        plt.figure(figsize=(20, 10))

        # 获取图像名称
        img_name = os.path.basename(image_paths[idx])
        base_name = os.path.splitext(img_name)[0]

        # 显示所有方法的结果
        for i, method in enumerate(methods):
            plt.subplot(2, 4, i + 1)
            plt.imshow(result[method], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.suptitle(f'Preprocessing Results for {img_name}', fontsize=16)
        plt.tight_layout()

        # 保存结果为PNG格式
        save_path = os.path.join(save_folder, f'result_{base_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 同时保存每种处理方法的单独图像为PNG
        method_save_folder = os.path.join(save_folder, 'individual_results')
        if not os.path.exists(method_save_folder):
            os.makedirs(method_save_folder)

        for method in methods:
            method_save_path = os.path.join(method_save_folder, f'{base_name}_{method}.png')
            cv2.imwrite(method_save_path, result[method])

        print(f"Saved results for {img_name}")


def main():
    # 设置图像文件夹路径
    folder_path = '.\pics'  # 替换为您的图像文件夹路径

    # 加载图像
    images, image_paths = load_images(folder_path)
    print(f"Loaded {len(images)} images")

    # 检查是否加载成功
    for i, img in enumerate(images):
        if img is None:
            print(f"Failed to load image: {image_paths[i]}")

    # 处理图像
    processed = process_images(images)

    # 可视化并保存结果
    visualize_results(processed, image_paths)


if __name__ == "__main__":
    main()