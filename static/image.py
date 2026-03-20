import os
from PIL import Image, ImageOps

def add_white_border_to_png(input_folder, output_folder, border_size=20):
    """
    给文件夹中的所有 PNG 图片添加白色边框

    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param border_size: 边框大小（像素）
    """

    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # 打开图片
                img = Image.open(input_path)

                # 添加白色边框
                bordered_img = ImageOps.expand(
                    img,
                    border=border_size,
                    fill="white"
                )

                # 保存图片
                bordered_img.save(output_path)

                print(f"处理完成: {filename}")

            except Exception as e:
                print(f"处理失败 {filename}: {e}")


if __name__ == "__main__":
    input_folder = "images"   # 替换为你的输入文件夹路径
    output_folder = "output_images" # 输出文件夹

    add_white_border_to_png(input_folder, output_folder, border_size=20)