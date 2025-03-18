import argparse
import numpy as np
from PIL import Image

def process_image(input_path, output_path, target_size=(448, 448)):
    # 读取图片
    img = Image.open(input_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')  # 统一转换为RGB格式

    # Resize操作（默认拉伸填充整个画布）
    resized_img = img.resize(target_size, Image.BICUBIC)  # 可选的插值方法

    # 转换为numpy数组并归一化到[0,1)
    img_array = np.asarray(resized_img, dtype=np.float32) / 255.0

    # 转换为FP16精度
    fp16_data = img_array.astype(np.float16)

    # 保存为二进制文件（无元数据）
    fp16_data.tofile(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PIL图片处理工具')
    parser.add_argument('input_img', help='输入图片路径')
    parser.add_argument('output_bin', help='输出二进制路径')
    args = parser.parse_args()

    process_image(args.input_img, args.output_bin)
