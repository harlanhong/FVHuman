import numpy as np
import torch
import cv2
import imageio
import pdb
from decord import VideoReader,cpu
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

# 加载视频和bbox
video_path = '/mnt/localssd/final_ted_v2/000373/10.mp4'
bboxes = np.load('/mnt/localssd/final_ted_v2_bbox/000373/10.npy')
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# 读取视频
vr = VideoReader(video_path, ctx=cpu(0))
fps = vr.get_avg_fps()
# 准备保存视频
output_path = 'sam1.mp4'
writer = imageio.get_writer(output_path, fps=fps)

frame_idx = 0
for frame in vr:
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
    image_with_mask = cv2.addWeighted(frame, 1, mask_image*255, 0.6, 0)
    # image_with_mask = mask_image + frame_bgr

    writer.append_data(cv2.cvtColor(image_with_mask, cv2.COLOR_RGB2BGR))
    frame_idx += 1
writer.close()

print(f"Video saved to {output_path}")
