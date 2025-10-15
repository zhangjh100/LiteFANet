import cv2
import numpy as np
import os
from glob import glob


def batch_modify_pixels(input_dir, output_dir):
    """
    批量处理文件夹中的图像，修改特定像素值并保存

    参数:
        input_dir: 包含输入图像的文件夹路径
        output_dir: 保存处理后图像的文件夹路径
    """
    # 支持的图像文件扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tif']
    image_paths = []

    # 收集所有图像文件路径
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"在 {input_dir} 中未找到任何图像文件")
        return

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 定义像素值映射关系
    pixel_mapping = {
        171: 1,
        114: 2,
        57: 3
    }

    # 遍历处理每个图像
    for img_path in image_paths:
        # 读取图像
        image = cv2.imread(img_path)

        if image is None:
            print(f"无法读取图像: {img_path}，已跳过")
            continue

        # 创建副本避免修改原图
        modified_image = image.copy()

        # 应用像素值替换
        for original_val, new_val in pixel_mapping.items():
            modified_image[modified_image == original_val] = new_val

        # 获取文件名并构建输出路径
        img_filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_filename)

        # 保存处理后的图像
        if cv2.imwrite(output_path, modified_image):
            print(f"已处理并保存: {output_path}")
        else:
            print(f"保存图像失败: {output_path}")

    print("批量处理完成")


if __name__ == "__main__":
    # 输入文件夹路径 - 请替换为你的图像所在文件夹
    input_directory = "./datasets/acdc/train/annotations"
    # 输出文件夹路径 - 请替换为你想要保存结果的文件夹
    output_directory = "./datasets/acdc/train/annotations_0"

    # 执行批量处理
    batch_modify_pixels(input_directory, output_directory)