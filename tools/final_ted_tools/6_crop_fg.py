import cv2
import numpy as np
import pdb
def calculate_global_bbox_with_margin(bboxes, frame_width, frame_height, margin_ratio=0.1):
    """
    计算所有帧中人的 bbox 的最小包围矩形，并在裁剪框上预留 margin。
    bboxes: 一个包含每帧 bbox 的列表，每个 bbox 是 (x1, y1, x2, y2)
    frame_width: 视频帧的宽度
    frame_height: 视频帧的高度
    margin_ratio: 预留裁剪区域的比例（默认 15%）
    """
    x1_min = min([bbox[0] for bbox in bboxes])
    y1_min = min([bbox[1] for bbox in bboxes])
    x2_max = max([bbox[2] for bbox in bboxes])
    y2_max = max([bbox[3] for bbox in bboxes])
    
    # 计算裁剪宽度和高度
    cropped_width = x2_max - x1_min
    cropped_height = y2_max - y1_min
    
    # 计算预留的 margin (15%)
    margin_width = int(cropped_width * margin_ratio / 2)
    margin_height = int(cropped_height * margin_ratio / 2)
    
    # 扩展 bbox，确保不会超出视频的边界
    x1_min = max(0, x1_min - margin_width)
    y1_min = max(0, y1_min - margin_height)
    x2_max = min(frame_width, x2_max + margin_width)
    y2_max = min(frame_height, y2_max + margin_height)
    
    return int(x1_min), int(y1_min), int(x2_max), int(y2_max)

def crop_video(input_video_path, output_video_path, bboxes):
    """
    裁剪视频以减少背景部分
    input_video_path: 输入视频路径
    output_video_path: 输出裁剪后的视频路径
    bboxes: 每一帧的人的 bbox 列表
    """
    # 打开视频
    cap = cv2.VideoCapture(input_video_path)
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算全局裁剪区域
    x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bboxes,frame_width,frame_height)
    
    # 裁剪后的视频宽度和高度
    cropped_width = x2_max - x1_min
    cropped_height = y2_max - y1_min
    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (cropped_width, cropped_height))
    
    # 逐帧处理视频
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 根据全局 bbox 裁剪每帧
        cropped_frame = frame[y1_min:y2_max, x1_min:x2_max]
        
        # 写入裁剪后的视频
        out.write(cropped_frame)
        
        frame_idx += 1
        if frame_idx >= total_frames:
            break
    
    # 释放资源
    cap.release()
    out.release()

# 示例使用
# 假设你有每帧的bbox，格式为 (x1, y1, x2, y2)
input_video_path = '/projects/p32321/fating/dataset/final_ted_dataset_v2/videos/000305/000305_10_scene_1.mp4'

bboxes = np.load(input_video_path.replace('videos','bboxes').replace('mp4','npy')).squeeze(1)
output_video_path = 'output_cropped_video.mp4'

crop_video(input_video_path, output_video_path, bboxes)
