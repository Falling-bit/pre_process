import os
import cv2
import numpy as np
import json
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk


class AdvancedImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("工业缺陷标注工具 - 专业版")

        # 参数配置
        self.crop_size = (128, 128)  # 固定裁剪尺寸
        self.current_scale = 1.0  # 图像显示缩放比例

        # 初始化界面
        self.setup_ui()

        # 状态变量
        self.image_paths = []
        self.current_index = 0
        self.original_image = None
        self.display_image = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.annotations = []

        # 默认目录
        self.input_dir = ""
        self.output_dir = ""

    def setup_ui(self):
        """设置增强版用户界面"""
        # 主框架
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        # 图像显示区域
        self.canvas = Canvas(main_frame, bg='gray', cursor="cross")
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)

        # 控制面板
        control_frame = Frame(main_frame, width=200)
        control_frame.pack(side=RIGHT, fill=Y)

        # 目录设置
        dir_frame = LabelFrame(control_frame, text="目录设置")
        dir_frame.pack(pady=5, padx=5, fill=X)

        Button(dir_frame, text="选择输入目录", command=self.select_input_dir).pack(fill=X)
        Button(dir_frame, text="选择输出目录", command=self.select_output_dir).pack(fill=X)

        # 类别选择
        self.class_var = StringVar(value="cut")
        class_frame = LabelFrame(control_frame, text="缺陷类别")
        class_frame.pack(pady=5, padx=5, fill=X)

        classes = ["cut", "press", "background", "other"]
        for cls in classes:
            Radiobutton(class_frame, text=cls, variable=self.class_var, value=cls).pack(anchor=W)

        # 裁剪控制
        crop_frame = LabelFrame(control_frame, text="裁剪控制")
        crop_frame.pack(pady=5, padx=5, fill=X)

        Button(crop_frame, text="保存裁剪", command=self.save_crop).pack(fill=X)
        Button(crop_frame, text="跳过图像", command=self.next_image).pack(fill=X)

        # 导航控制
        nav_frame = LabelFrame(control_frame, text="图像导航")
        nav_frame.pack(pady=5, padx=5, fill=X)

        Button(nav_frame, text="上一张 (←)", command=self.prev_image).pack(fill=X)
        Button(nav_frame, text="下一张 (→)", command=self.next_image).pack(fill=X)

        # 缩放控制
        scale_frame = LabelFrame(control_frame, text="显示缩放")
        scale_frame.pack(pady=5, padx=5, fill=X)

        Button(scale_frame, text="放大 (+)", command=lambda: self.adjust_scale(1.2)).pack(side=LEFT)
        Button(scale_frame, text="缩小 (-)", command=lambda: self.adjust_scale(0.8)).pack(side=RIGHT)

        # 状态栏
        self.status_var = StringVar()
        self.status_bar = Label(self.root, textvariable=self.status_var, bd=1, relief=SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

        # 绑定键盘事件
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<space>", lambda e: self.save_crop())

        # 绑定鼠标事件
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def select_input_dir(self):
        """选择输入目录"""
        self.input_dir = filedialog.askdirectory(title="选择包含BMP文件的目录")
        if self.input_dir:
            self.load_image_list()

    def select_output_dir(self):
        """选择输出目录"""
        self.output_dir = filedialog.askdirectory(title="选择保存裁剪结果的目录")

    def load_image_list(self):
        """加载图像列表"""
        self.image_paths = [os.path.join(self.input_dir, f)
                            for f in os.listdir(self.input_dir)
                            if f.lower().endswith('.bmp')]
        self.current_index = 0
        if self.image_paths:
            self.load_current_image()
        else:
            self.status_var.set("错误: 目录中没有BMP文件")

    def load_current_image(self):
        """加载当前图像"""
        if not self.image_paths:
            return

        self.status_var.set(f"正在加载: {os.path.basename(self.image_paths[self.current_index])}")
        self.root.update()

        try:
            # 读取图像并转为RGB
            self.original_image = cv2.imread(self.image_paths[self.current_index])
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

            # 初始缩放显示
            self.adjust_scale(1.0, reset=True)
            self.update_status()
        except Exception as e:
            self.status_var.set(f"加载失败: {str(e)}")

    def adjust_scale(self, factor, reset=False):
        """调整显示比例"""
        if self.original_image is None:
            return

        if reset:
            self.current_scale = 1.0
        else:
            self.current_scale *= factor
            self.current_scale = max(0.1, min(3.0, self.current_scale))  # 限制缩放范围

        # 计算缩放后的尺寸
        h, w = self.original_image.shape[:2]
        new_w = int(w * self.current_scale)
        new_h = int(h * self.current_scale)

        # 缩放图像
        self.display_image = cv2.resize(self.original_image, (new_w, new_h))

        # 显示图像
        self.show_image()
        self.update_status()

    def show_image(self):
        """在画布上显示当前图像"""
        if self.display_image is None:
            return

        # 转换图像格式
        img_pil = Image.fromarray(self.display_image)
        self.tk_image = ImageTk.PhotoImage(img_pil)

        # 更新画布
        self.canvas.delete("all")
        self.canvas.config(width=self.tk_image.width(), height=self.tk_image.height())
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)

    def on_press(self, event):
        """鼠标按下事件"""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        # 删除旧的选择框
        if self.rect_id:
            self.canvas.delete(self.rect_id)

        # 创建新的选择框（固定大小）
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y,
            self.start_x + self.crop_size[0] * self.current_scale,
            self.start_y + self.crop_size[1] * self.current_scale,
            outline="red", width=2, dash=(4, 4), tags="rect")

    def on_drag(self, event):
        """鼠标拖动事件 - 移动选择框"""
        if not self.rect_id:
            return

        # 计算新位置（保持固定大小）
        new_x = self.canvas.canvasx(event.x)
        new_y = self.canvas.canvasy(event.y)

        # 限制在图像范围内
        img_width = self.tk_image.width()
        img_height = self.tk_image.height()

        crop_w = self.crop_size[0] * self.current_scale
        crop_h = self.crop_size[1] * self.current_scale

        new_x = max(0, min(new_x, img_width - crop_w))
        new_y = max(0, min(new_y, img_height - crop_h))

        # 更新选择框位置
        self.canvas.coords(
            self.rect_id,
            new_x, new_y,
            new_x + crop_w,
            new_y + crop_h)

    def on_release(self, event):
        """鼠标释放事件"""
        pass

    def save_crop(self):
        """保存当前裁剪区域"""
        # 安全检查（推荐方式）
        if (self.rect_id is None or
                not isinstance(self.output_dir, str) or
                not hasattr(self, 'original_image') or
                not isinstance(self.original_image, np.ndarray) or
                self.original_image.size == 0):
            self.status_var.set("错误: 请先选择输出目录并创建选择框")
            return

        # 获取裁剪区域坐标（画布坐标系）
        try:
            x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
        except:
            self.status_var.set("错误: 无效的选择框")
            return

        # 转换到原始图像坐标
        try:
            scale_factor = 1 / self.current_scale
            orig_x = int(x1 * scale_factor)
            orig_y = int(y1 * scale_factor)
            crop = self.original_image[
                   orig_y:orig_y + self.crop_size[1],
                   orig_x:orig_x + self.crop_size[0]]
        except Exception as e:
            self.status_var.set(f"坐标转换错误: {str(e)}")
            return

        # 确保尺寸正确
        if crop.shape[:2] != self.crop_size:
            self.status_var.set(f"错误: 裁剪尺寸无效 {crop.shape[:2]}")
            return

        # 保存文件
        try:
            class_name = self.class_var.get()
            class_dir = os.path.join(self.output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            base_name = os.path.splitext(os.path.basename(
                self.image_paths[self.current_index]))[0]
            save_path = os.path.join(class_dir,
                                     f"{base_name}_{orig_x}_{orig_y}.bmp")

            cv2.imwrite(save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            self.status_var.set(f"已保存: {save_path}")
        except Exception as e:
            self.status_var.set(f"保存失败: {str(e)}")

    def prev_image(self):
        """切换到上一张图像"""
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

    def next_image(self):
        """切换到下一张图像"""
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_current_image()
        else:
            self.status_var.set("已是最后一张图像")

    def update_status(self):
        """更新状态栏信息"""
        if self.image_paths and self.original_image is not None:
            orig_h, orig_w = self.original_image.shape[:2]
            status = (f"图像 {self.current_index + 1}/{len(self.image_paths)} | "
                      f"原始尺寸: {orig_w}x{orig_h} | "
                      f"显示比例: {self.current_scale:.2f}x")
            self.status_var.set(status)

    def save_annotations(self):
        """保存标注信息"""
        if self.output_dir and self.annotations:
            with open(os.path.join(self.output_dir, "annotations.json"), "w") as f:
                json.dump(self.annotations, f, indent=2)
            self.status_var.set(f"标注已保存到 {self.output_dir}")


if __name__ == "__main__":
    root = Tk()
    app = AdvancedImageCropper(root)

    # 窗口关闭时保存标注
    root.protocol("WM_DELETE_WINDOW", lambda: [app.save_annotations(), root.quit()])

    # 启动主循环
    root.mainloop()