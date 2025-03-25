import os
import cv2

def extract_frames_with_interval(video_path, save_folder_path, interval, video_index):
    """
    从指定视频文件按指定间隔提取帧并保存为 JPEG 图像，并裁剪掉黑色边缘

    :param video_path: 视频文件的路径
    :param save_folder_path: 保存提取帧的文件夹路径
    :param interval: 提取帧的间隔
    :param video_index: 视频文件的序号
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件：{video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"视频 '{video_path}' 无法读取帧数，跳过。")
        cap.release()
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            # 裁剪中间区域
            height, width, _ = frame.shape
            left = int(width * 0.25)
            right = int(width * 0.75)
            cropped_frame = frame[:, left:right]

            frame_filename = f"video_{video_index}_frame_{frame_count:04d}.jpg"
            frame_filepath = os.path.join(save_folder_path, frame_filename)
            cv2.imwrite(frame_filepath, cropped_frame)

        frame_count += 1

    cap.release()
    print(f"已处理视频文件：{video_path}")


def process_folder(folder_path, interval):
    """
    遍历指定文件夹及其子文件夹，对每个视频文件执行帧提取操作。

    :param folder_path: 要处理的根文件夹路径
    :param interval: 提取帧的间隔
    """
    save_folder_path = os.path.join(folder_path, 'frames')
    os.makedirs(save_folder_path, exist_ok=True)

    video_index = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(root, file)
                extract_frames_with_interval(video_path, save_folder_path, interval, video_index)
                video_index += 1


if __name__ == "__main__":
    try:
        # 设定要开始处理的根文件夹路径
        root_folder = r"F:\dlproj\face anti-spoofing\archive\archive (2)\samples"
        interval = 30  # 设置提取帧的间隔，例如每30帧提取一帧
        process_folder(root_folder, interval)
    except KeyboardInterrupt:
        print("处理被中断。")