import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pdb
from tqdm import tqdm
import os
import random
import glob
import imageio
from segment_anything import sam_model_registry, SamPredictor

import decord
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

def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = (mask.reshape(h, w, 1) * color[:3]).astype(np.uint8)
    return mask_image

def show_box(box, image):
    x0, y0, x1, y1 = box
    cv2.rectangle(image, (int(x0),int(y0)), (int(x1), int(y1)), color=(0, 255, 0), thickness=2)

def load_s3_video(s3_url, s3_client=None):
    bytesio = _read_s3_to_bytesio(s3_url, s3_client)
    # Reset the BytesIO object to start reading from the beginning
    vr = decord.VideoReader(bytesio, ctx=decord.cpu(0))
    return vr
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt") 
# model = torch.hub.load('ultralytics/yolov8', 'yolov8')
def is_size_too_small(detection, frame_size, small_threshold=0.1):
    x1, y1, x2, y2 = detection[:4]
    bbox_size = (x2 - x1) * (y2 - y1)
    frame_area = frame_size[0] * frame_size[1]
    
    if bbox_size / frame_area < small_threshold:
        return True
    return False
def detect_humans(frame):
    # Run YOLOv8 inference on the frame
    results = model(frame)
    # Filter detections to include only humans (class label 0 in COCO dataset)
    human_detections = []
    
    # 将三个列表打包在一起
    combined = list(zip(results[0].boxes.cls.cpu().tolist(), results[0].boxes.conf.cpu().tolist(), results[0].boxes.xyxy.cpu().tolist()))

    # 过滤出 class 为 0 的元素
    filtered = [item for item in combined if item[0] == 0]

    max_confidence_item = max(filtered, key=lambda x: x[1])
    highest_conf_bbox = max_confidence_item[2]
    print("Confidence 最高的 class 为 0 的 bbox:", highest_conf_bbox)
    human_detections.append(highest_conf_bbox)
    return human_detections
def check_similar_sizes(detections, threshold=0.3):
    if len(detections) < 2:
        return False
    # Calculate bounding box sizes
    sizes = [(det[2] - det[0]) * (det[3] - det[1]) for det in detections]
    mean_size = np.mean(sizes)
    # Check if all sizes are similar within the threshold
    for size in sizes:
        if abs(size - mean_size) / mean_size > threshold:
            return False
    return True
def get_largest_bbox(detections):
    if not detections:
        return None
    # Find the largest bounding box
    largest_bbox = max(detections, key=lambda det: (det[2] - det[0]) * (det[3] - det[1]))
    return largest_bbox
def process_frame(frame):
    # Detect humans in the frame
    human_detections = detect_humans(frame)
    # Check if there are multiple persons with similar bounding box sizes
    return human_detections
# Example usage with a video feed

def process_video(video_path):
    vr =decord.VideoReader(video_path, ctx=decord.cpu(0))
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    rst = True
    video_results = []
    videos = []
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rst = process_frame(frame)
        video_results.append(rst)
        if len(rst)>1:
            print(i)
        for [x1, y1, x2, y2] in rst:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        videos.append(frame)
    return videos, video_results

def sam_process_video(video_path, bboxes, output_path):
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    # 读取视频
    vr =decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    # 准备保存视频
    writer = imageio.get_writer(output_path, fps=fps)

    frame_idx = 0
    for frame in tqdm(vr):
        frame = frame.asnumpy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame_idx < len(bboxes):
            input_box = bboxes[frame_idx]
        else:
            break  # 如果 bboxes 列表中的数据不足以覆盖所有帧，则停止处理
        predictor.set_image(frame_rgb)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(input_box)[None, :],
            multimask_output=False,
        )
        mask_image = show_mask(masks[0])
        # image_with_mask = cv2.addWeighted(frame_rgb, 1, mask_image*255, 1, 0)
        image_with_mask = frame_rgb*mask_image[...,2][...,None]
        image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR)
        writer.append_data(image_with_mask)
        frame_idx += 1
    writer.close()

def convert_bgr2rgb(video_path,save_path):
    basename = os.path.basename(video_path)
    save_name = os.path.join(save_path, basename)
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    fps = vr.get_avg_fps()
    writer = imageio.get_writer(save_name, fps=fps)
    for frame in vr:
        frame = frame.asnumpy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(frame)
    writer.close()


def step1_scene_detection(video_path, output_folder):
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    from scenedetect.video_splitter import split_video_ffmpeg

    # 创建 VideoManager 和 SceneManager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # 添加内容检测器（你可以调整 threshold 参数）
    scene_manager.add_detector(ContentDetector(threshold=30.0))

    # 启动 VideoManager
    video_manager.set_downscale_factor()
    video_manager.start()

    # 检测场景变化
    scene_manager.detect_scenes(frame_source=video_manager)

    # 获取检测到的场景时间戳
    scene_list = scene_manager.get_scene_list()
    print(f"Detected {len(scene_list)} scenes!")

    # 打印每个场景的开始和结束时间
    for i, scene in enumerate(scene_list):
        print(f"Scene {i + 1}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")

    # 使用 FFMpeg 将每个场景切割为独立视频
    split_video_ffmpeg(video_path, scene_list, output_folder, arg_override='-c:v libx264 -crf 23 -preset veryfast')
    # 释放资源
    video_manager.release()

def step2_sam_process(input_folder, save_folder):
    videos = glob.glob(os.path.join(input_folder, '*.mp4'))
    for video in tqdm(videos):
        try:
            _, bbox = process_video(video)
            sam_process_video(video, bbox, os.path.join(save_folder, os.path.basename(video)))
            print(f"Processed {video} videos")
        except Exception as e:
            print(f"Error processing {video}: {e}")

def step3_crop_video(input_folder, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    videos = glob.glob(os.path.join(input_folder, '*.mp4'))
    for video in tqdm(videos):  
        try:
            _, bbox = process_video(video)
            bbox = np.array(bbox)[:,0]
            crop_video(video, os.path.join(save_folder, os.path.basename(video)), bbox)
        except Exception as e:
            print(f"Error processing {video}: {e}")
if __name__ == "__main__":
    # step1_scene_detection('/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/user_test/457223175-1-208.mp4', '/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips')
    # step1_scene_detection('/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/user_test/457223175-1-208.mp4', '/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips')
    
    step2_sam_process('/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips', '/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips_sam')
    
    step3_crop_video('/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips_sam', '/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/demo2_output_clips_sam_crop')
  