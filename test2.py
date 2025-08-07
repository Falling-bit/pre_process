import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt


def detect_marks(img):
    """修正版检测逻辑：最黑=切痕(红)，次黑=压痕(蓝)"""
    # 预处理：降噪同时保留边缘
    blurred = cv2.bilateralFilter(img, 9, 75, 75)

    # 第一步：检测最黑区域（切痕）
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cut_marks = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)

    # 第二步：在非切痕区域检测次黑区域（压痕）
    non_cut = blurred[cut_marks == 0]
    if len(non_cut) > 0:
        bg_mean = np.mean(non_cut)
        # 压痕范围：比背景暗但比切痕浅
        press_lower = max(0, int(bg_mean * 0.7))  # 比背景暗30%
        press_upper = max(0, int(bg_mean * 0.9))  # 比背景暗10%
        press_mask = cv2.inRange(blurred, press_lower, press_upper)
        press_marks = cv2.bitwise_and(press_mask, cv2.bitwise_not(cut_marks))
    else:
        press_marks = np.zeros_like(img)

    return cut_marks, press_marks


def visualize(img, cut, press):
    """可视化：切痕红，压痕蓝"""
    marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  #BGR格式
    marked[cut > 0] = [255, 0, 0]  # 切痕蓝色（最黑）
    marked[press > 0] = [0, 0, 255]  # 压痕红色（次黑）

    # 灰度验证图
    verify = img.copy()
    verify[cut > 0] = 50  # 切痕最黑
    verify[press > 0] = 150  # 压痕中等

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(marked,cv2.COLOR_BGR2RGB)), plt.title("Detection Results")
    plt.subplot(1, 3, 3), plt.imshow(verify, cmap='gray', vmin=0, vmax=255)
    plt.title("Gray Value Verification")
    plt.show()


def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading: {img_path}")
        return None

    cut, press = detect_marks(img)

    # 后处理
    cut = morphology.remove_small_holes(cut.astype(bool), 100).astype(np.uint8) * 255
    press = morphology.remove_small_objects(press.astype(bool), 200).astype(np.uint8) * 255

    # 打印灰度信息
    print(f"切痕平均灰度: {np.mean(img[cut > 0]):.1f} (应最小)")
    print(f"压痕平均灰度: {np.mean(img[press > 0]):.1f} (应中等)")
    print(f"背景平均灰度: {np.mean(img[(cut == 0) & (press == 0)]):.1f} (应最大)")

    visualize(img, cut, press)
    return {'cut': cut, 'press': press}


# 使用示例
if __name__ == "__main__":
    result = process_image("./pics/Pic_20250721132335241.bmp")
    if result:
        cv2.imwrite("cut_result.bmp", result['cut'])
        cv2.imwrite("press_result.bmp", result['press'])