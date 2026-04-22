import cv2
import os


def images_to_video(image_folder, output_video_path, fps=30, size=None):
    """
    将指定文件夹内的图片按顺序合成为视频。
    """
    # 获取并排序图片
    images = [img for img in os.listdir(image_folder) if img.lower().endswith((".png", ".jpg", ".jpeg"))]
    images.sort()

    if not images:
        print(f"错误：在文件夹 '{image_folder}' 中没有找到任何图片！")
        return

    # 读取第一张图片以确定尺寸
    first_image_path = os.path.join(image_folder, images[0])
    print(f"正在读取第一张图片以确定尺寸: {first_image_path}")

    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"致命错误：无法读取图片 '{first_image_path}'。")
        print("可能原因：1. 文件路径错误；2. 文件已损坏；3. OpenCV不支持此图片格式。")
        return

    if size is None:
        height, width, _ = frame.shape
        size = (width, height)
    print(f"视频尺寸将设为: {size}")

    # 创建VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    if not video_writer.isOpened():
        print(f"错误：无法创建视频文件 '{output_video_path}'。请检查路径和写入权限。")
        return

    # 写入所有帧
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"警告：跳过无法读取的图片 '{image_path}'")
            continue

        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)

        video_writer.write(frame)

    video_writer.release()
    print(f"✅ 视频已成功生成: {os.path.abspath(output_video_path)}")


# --- 使用示例 ---
if __name__ == "__main__":
    # 请务必替换为您的实际路径！建议使用绝对路径。
    image_folder = r"E:\web\ir_v"# 注意前面的 'r'
    output_video = r"E:\output_vide.mp4"

    images_to_video(image_folder, output_video, fps=25)