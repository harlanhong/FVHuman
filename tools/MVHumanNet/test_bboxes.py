import json
import cv2
import os

def crop_images_from_annots(bbox, image, mask, output_dir):
    x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # 裁剪bbox对应的区域
    cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

# 示例调用
annots = data['annots']
img_fp = '/path/to/images_lr/CC32871A004/0005_img.jpg'
mask_fp = '/path/to/fmask_lr/CC32871A004/0005_img_fmask.png'
output_dir = '/path/to/output_directory'

crop_images_from_annots(annots, img_fp, mask_fp, output_dir)
