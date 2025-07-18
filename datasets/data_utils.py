import os
import json
import random
from typing import List
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
import cv2
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

def make_bbox_square_and_crop_with_padding(image: np.array, bbox: tuple, padding_factor: float = 0.15) -> np.array:
    x1, y1, x2, y2 = bbox
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1
    
    # 计算需要的填充以使边界框成为正方形
    if width > height:
        pad = (width - height) // 2
        y1 = max(0, y1 - pad)
        y2 = min(image.shape[0], y2 + pad)
    else:
        pad = (height - width) // 2
        x1 = max(0, x1 - pad)
        x2 = min(image.shape[1], x2 + pad)
    
    # 计算新的宽度和高度，并添加缓冲区
    new_width = x2 - x1
    new_height = y2 - y1
    buffer_width = int(new_width * (1 + padding_factor))
    buffer_height = int(new_height * (1 + padding_factor))
    
    # 调整边界框以包含缓冲区
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    x1 = max(0, center_x - buffer_width // 2)
    x2 = min(image.shape[1], center_x + buffer_width // 2)
    y1 = max(0, center_y - buffer_height // 2)
    y2 = min(image.shape[0], center_y + buffer_height // 2)
    
    # 计算最终宽度和高度
    final_width = x2 - x1
    final_height = y2 - y1
    
    # 裁剪图像
    cropped_image = image[y1:y2, x1:x2]
    
    # 检查裁剪后的图像是否为正方形，如果不是，则进行填充
    if final_width != final_height:
        size = max(final_width, final_height)
        top = (size - final_height) // 2
        bottom = size - final_height - top
        left = (size - final_width) // 2
        right = size - final_width - left
        
        # 使用填充值 0 进行填充
        cropped_image = cv2.copyMakeBorder(cropped_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return cropped_image

def process_bbox(bbox, H, W, scale=1.):
    # transform a bbox(xmin, ymin, xmax, ymax) to (H, W) square
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    
    side_length = max(width, height)
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    scaled_side_length = side_length * scale
    scaled_xmin = center_x - scaled_side_length / 2
    scaled_xmax = center_x + scaled_side_length / 2
    scaled_ymin = center_y - scaled_side_length / 2
    scaled_ymax = center_y + scaled_side_length / 2

    scaled_xmin = int(max(0, scaled_xmin))
    scaled_xmax = int(min(W, scaled_xmax))
    scaled_ymin = int(max(0, scaled_ymin))
    scaled_ymax = int(min(H, scaled_ymax))
    
    return scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax

def crop_bbox(img, bbox, do_resize=False, size=512):
    
    if isinstance(img, (Path, str)):
        img = Image.open(img)
    cropped_img = img.crop(bbox)
    if do_resize:
        cropped_W, cropped_H = cropped_img.size
        ratio = size / max(cropped_W, cropped_H)
        new_W = cropped_W * ratio
        new_H = cropped_H * ratio
        cropped_img = cropped_img.resize((new_W, new_H))
    
    return cropped_img

def mask_to_bbox(mask_path):
    mask = np.array(Image.open(mask_path))[..., 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax

def mask_to_bkgd(img_path, mask_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    
    mask = Image.open(mask_path).convert("RGB")
    mask_array = np.array(mask)
    
    img_array = np.where(mask_array > 0, img_array, 0)
    return Image.fromarray(img_array)
    