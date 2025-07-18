import json
import pdb
import os
import numpy as np
import cv2
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
vid_meta = json.load(open('/projects/p32296/fating/src/novel-view-human-master/data/final_ted_meta.json', "r"))
# self.vid_meta = vid_meta
vid_meta = [item for item in vid_meta if item.get("mode") == 'test']
root = '/projects/p32321/fating/dataset/final_ted_dataset_v2'
save_fp = 'test_all_view'
for meta in vid_meta:
    data = meta['data']
    id = meta['id']
    names=[]
    videos_fp_list = []
    images = []
    for token in data:
        video_fp = token['video_path']
        videos_fp_list.append(video_fp)
    videos_fp_list = sorted(videos_fp_list)
    for video_fp in videos_fp_list:
        original_fp = os.path.join(root, video_fp)
        first_frame = extract_first_frame(original_fp)
        images.append(first_frame)
        names.append(os.path.basename(video_fp)[:-4])
    if images:
        name = '-'.join(names)
        concatenated_image = concatenate_images(images)
        if id=='000150' or id == '000406':
            pdb.set_trace()
            print('a')
        cv2.imwrite(os.path.join(save_fp,f'{id}.jpg'), concatenated_image)
        print(f"Concatenated image saved to {os.path.join(save_fp,f'{id}.jpg')}")