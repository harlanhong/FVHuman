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
# Load the YOLOv8 model (assuming you have a pretrained YOLOv8 model)
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3, _read_s3_to_bytesio
import decord
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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Num workers")
    parser.add_argument("--job_idx", type=int, default=0, help="Num workers")
    args = parser.parse_args()
    job_idx= args.job_idx
    num_workers= args.num_workers
    
    rst = []

    # video_list = list_s3_dir('s3://zhanxu-public/fating/final_ted_v2/')
    video_list = glob.glob('/mnt/localssd/final_ted_dataset/videos/*/*.mp4')
    # split into chunks,
    random.shuffle(video_list)
    batch_size = (len(video_list) + num_workers - 1) // num_workers 
    print(f"Num videos: {len(video_list)} {batch_size = }")
    video_chunks = [
        video_list[i : i + batch_size]
        for i in range(0, len(video_list), batch_size)
    ]
    with open('useful.txt', 'a+') as f:
        for vid in tqdm(video_chunks[job_idx]):
            try:
                bbox_fp = vid.replace('videos','bboxes').replace('mp4','npy')
                if os.path.exists(bbox_fp):
                    continue
                vr =decord.VideoReader(vid, ctx=decord.cpu(0))
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
                
                dirname = os.path.dirname(bbox_fp)
                os.makedirs(dirname, exist_ok=True)
                np.save(bbox_fp, video_results)
            except Exception as e:
                print(vid)
                f.write(vid)






