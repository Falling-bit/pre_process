import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

''''''
def detect_marks(img):
    """修正版检测逻辑：最黑=切痕(红)，次黑=压痕(蓝)"""
    # 预处理：降噪同时保留边缘
    blurred = cv2.bilateralFilter(img, 9, 75, 75)

    # 第一步：检测最黑区域（切痕）
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cut_marks = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)

    return cut_marks #, press_marks



def refine_detection(img, cut_mask):
    non_cut_area = img.copy()
    non_cut_area[cut_mask > 0] = 0


    # 直接使用自适应阈值
    press_mask = cv2.adaptiveThreshold(
        non_cut_area, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    press_mask = cv2.bitwise_and(press_mask, cv2.bitwise_not(cv2.dilate(cut_mask, np.ones((5, 5)))))
    press_mask = morphology.remove_small_objects(press_mask.astype(bool), min_size=100)
    return press_mask.astype(np.uint8) * 255

def visualize(img, cut, press):
    """可视化：切痕红，压痕蓝"""
    marked = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # BGR格式
    marked[cut > 0] = [0, 0, 255]  # 切痕红色（最黑）
    marked[press > 0] = [255, 0, 0]  # 压痕蓝色（次黑）

    # 灰度验证图
    verify = img.copy()
    verify[cut > 0] = 50  # 切痕最黑
    verify[press > 0] = 150  # 压痕中等

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title("Original")
    plt.subplot(1, 3, 2), plt.imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    plt.title("Detection Results")
    plt.subplot(1, 3, 3), plt.imshow(verify, cmap='gray', vmin=0, vmax=255)
    plt.title("Gray Value Verification")
    plt.show()


def process_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading: {img_path}")
        return None

    # 第一步：粗检测切痕
    cut = detect_marks(img)
    cut = morphology.remove_small_holes(cut.astype(bool), 100).astype(np.uint8) * 255

    # 第二步：在非切痕区域精细化检测压痕
    press = refine_detection(img, cut)
    #_, press = detect_marks(img)

    # 打印灰度信息
    print(f"切痕平均灰度: {np.mean(img[cut > 0]):.1f} (应最小)")
    print(f"压痕平均灰度: {np.mean(img[press > 0]):.1f} (应中等)")
    print(f"背景平均灰度: {np.mean(img[(cut == 0) & (press == 0)]):.1f} (应最大)")

    visualize(img, cut, press)
    return {'cut': cut, 'press': press}


# 使用示例
if __name__ == "__main__":
    result = process_image("./pics/Pic_20250710140316687.bmp")
    if result:
        cv2.imwrite("cut_result.bmp", result['cut'])
        cv2.imwrite("press_result.bmp", result['press'])