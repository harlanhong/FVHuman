import cv2
import numpy as np
import os


def split_image(image_path,num_split, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    width_split = w // num_split
    for i in range(num_split):
        cv2.imwrite(os.path.join(save_dir, f"{i}.jpg"), image[:, i*width_split:(i+1)*width_split, :])

if __name__ == "__main__":
    split_image("/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/attn_vis/result_25.png", 18, "/apdcephfs_cq8/share_1367250/harlanhong/src/fating-novel-view-human/attn_vis/25")

