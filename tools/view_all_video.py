import cv2
import numpy as np
import glob
import os
from moviepy.editor import VideoFileClip
from PIL import Image

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Failed to extract frame from {video_path}")
    return frame

def concatenate_images(image_list):
    # 获取每张图像的高度和宽度
    heights, widths = zip(*(i.shape[:2] for i in image_list))
    total_width = sum(widths)
    max_height = max(heights)
    
    # 创建一个新的图像，宽度是所有图像宽度的总和，高度是最大高度
    concatenated_image = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    x_offset = 0
    for img in image_list:
        h, w = img.shape[:2]
        concatenated_image[:, x_offset:x_offset+w] = img
        x_offset += w
    
    return concatenated_image
fp = '/projects/p32321/fating/dataset/final_ted_dataset_v2/videos'
vids = glob.glob(f'{fp}/*')
save_fp = '/projects/p32296/fating/src/novel-view-human-master/outputs/vis_dataset'
os.makedirs(save_fp,exist_ok=True)
for vid in vids:
    print(vid)
    file_name = os.path.basename(vid)
    clips = sorted(glob.glob(f'{vid}/*.mp4'))
    images = []
    names = []
    for vfp in clips:
        first_frame = extract_first_frame(vfp)
        images.append(first_frame)
        names.append(os.path.basename(vfp)[:-4])
    if images:
        name = '-'.join(names)
        concatenated_image = concatenate_images(images)
        cv2.imwrite(os.path.join(save_fp,f'{file_name}-{name}.jpg'), concatenated_image)
        print(f"Concatenated image saved to {os.path.join(save_fp,f'{file_name}.jpg')}")
    else:
        print("No video files found in the folder.")