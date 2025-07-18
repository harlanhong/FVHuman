import os
import glob
import numpy as np
import subprocess
import cv2
from tqdm import tqdm
def calculate_global_bbox_with_margin(bboxes, frame_width, frame_height, margin_ratio=0.15):
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

ted_fp = '/projects/p32321/fating/dataset/final_ted_dataset_v2'
output_fp = '/projects/p32321/fating/dataset/final_ted_dataset_crop'

videos = glob.glob(os.path.join(ted_fp, 'videos', '*/*.mp4'))
for video in tqdm(videos[::-1]):
    try:
        bbox_fp = video.replace('videos', 'bboxes').replace('.mp4', '.npy')
        bboxes = np.load(bbox_fp).squeeze(1)
        
        # 使用 OpenCV 读取视频
        cap = cv2.VideoCapture(video)
        mask_cap = cv2.VideoCapture(video.replace('videos', 'masks'))
        
        # 获取视频的宽度和高度
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算全局边界框
        x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bboxes, img_width, img_height)
        
        # 计算裁剪区域
        width = int(x2_max - x1_min)
        height = int(y2_max - y1_min)
        
        # 准备输出路径
        rel_path = os.path.relpath(video, os.path.join(ted_fp, 'videos'))
        out_video = os.path.join(output_fp, 'videos', rel_path)
        out_dwpose = video.replace('videos', 'dwpose').replace('.mp4', '_weighted_heatmap.mp4')
        out_dwpose = os.path.join(output_fp, os.path.relpath(out_dwpose, ted_fp))
        if os.path.exists(out_video) and os.path.exists(out_dwpose):
            print(f"已存在: {out_video}")
            continue
        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        os.makedirs(os.path.dirname(out_dwpose), exist_ok=True)
        
        # 定义视频编码格式
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
        
        while cap.isOpened() and mask_cap.isOpened():
            ret, frame = cap.read()
            ret_mask, mask_frame = mask_cap.read()
            if not ret or not ret_mask:
                break
            
            # 裁剪帧
            cropped_frame = frame[y1_min:y2_max, x1_min:x2_max]
            cropped_mask = mask_frame[y1_min:y2_max, x1_min:x2_max]
            
            # 转换掩码为灰度并应用
            # gray_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_BGR2GRAY)
            blended_frame = cropped_frame* (cropped_mask / 255.0)
            
            # 写入输出视频
            out.write(blended_frame.astype('uint8'))
        
        cap.release()
        mask_cap.release()
        out.release()
        print(out_video)
        # 裁剪DWPose视频
        # 裁剪DWPose视频
        dwpose_fp = video.replace('videos', 'dwpose').replace('.mp4', '_weighted_heatmap.mp4')
        dwpose_cap = cv2.VideoCapture(dwpose_fp)
        out_dwpose_writer = cv2.VideoWriter(out_dwpose, fourcc, dwpose_cap.get(cv2.CAP_PROP_FPS), (width, height))
        
        while dwpose_cap.isOpened():
            ret, frame = dwpose_cap.read()
            if not ret:
                break
            
            # 裁剪DWPose帧
            cropped_dwpose_frame = frame[y1_min:y2_max, x1_min:x2_max]
            
            # 写入裁剪后的DWPose视频
            out_dwpose_writer.write(cropped_dwpose_frame)
        
        dwpose_cap.release()
        out_dwpose_writer.release()

        print(f"处理完成: {video}")
    except Exception as e:
        print(e)
