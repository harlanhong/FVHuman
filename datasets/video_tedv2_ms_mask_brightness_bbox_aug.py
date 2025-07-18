import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
from decord import VideoReader
# from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd
import os
import pdb
from .data_utils import calculate_global_bbox_with_margin
from utils.util import calculate_brightness,calculate_contrast
def sinusoidal_encode(value, d_model):
    """
    使用正弦和余弦函数将一个标量值编码成多维向量
    :param value: 需要编码的标量值
    :param d_model: 编码的维度
    :return: 编码后的多维向量
    """
    # 创建一个位置索引数组
    position = np.arange(d_model // 2)
    
    # 计算编码
    angles = value / (10000 ** (position / (d_model // 2)))
    
    # 计算正弦和余弦
    encoded_vector = np.zeros(d_model)
    encoded_vector[0::2] = np.sin(angles)
    encoded_vector[1::2] = np.cos(angles)
    
    return encoded_vector

class PairedRandomMaxSquareCropResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pose, img):
        width, height = img.size
        resize_pose = transforms.functional.resize(pose,(height, width))
        min_dim = min(width, height)
        min_dim = random.randint(int(min_dim*0.7), min_dim)
        x = random.randint(0, width - min_dim)
        y = random.randint(0, height - min_dim)
       
        img = img.crop((x, y, x + min_dim, y + min_dim))
        
        img = transforms.functional.resize(img, self.size)
        
        pose = resize_pose.crop((x, y, x + min_dim, y + min_dim))
            
        pose = transforms.functional.resize(pose, self.size)
        return pose, img
class ListPairedRandomMaxSquareCropResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, pose_list, img_list):
        width, height = img_list[0].size
        resize_pose = []
        for pose in pose_list:
            p = transforms.functional.resize(pose,(height, width))
            resize_pose.append(p)
        min_dim = min(width, height)
        min_dim = random.randint(int(min_dim*0.7), min_dim)
        x = random.randint(0, width - min_dim)
        y = random.randint(0, height - min_dim)
        new_list = []
        for img in img_list:
            # Crop both images to a square of size min_dim x min_dim
            img = img.crop((x, y, x + min_dim, y + min_dim))
            
            img = transforms.functional.resize(img, self.size)
            new_list.append(img)
        new_pose = []
        for pose in resize_pose:
            # Crop both images to a square of size min_dim x min_dim
            pose = pose.crop((x, y, x + min_dim, y + min_dim))
            
            pose = transforms.functional.resize(pose, self.size)
            new_pose.append(pose)
        return new_pose, new_list 
class VideoDataset(Dataset):
    def __init__(
        self,
        root,
        image_size: int = 512,
        sample_frames: int = 16,
        sample_rate: int = 1,
        aug_type: str = "Resize",
        data_meta_paths=["./data/fashion_meta.json"],
        mode="train",
        seed_frames = 4,
        ref_num = 2,
        generated_frames = 8,
    ):
        super().__init__()
        self.num_ref = ref_num
        self.root = root
        self.img_size_single = image_size
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.aug_type = aug_type  
        self.seed_frames = seed_frames
        self.generated_frames = generated_frames
        
        if 2 * self.seed_frames + self.generated_frames != self.sample_frames:
            print(
                f"error: generated {self.generated_frames} sample {self.sample_frames} seed {self.seed_frames}" 
            )
        
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        # self.vid_meta = vid_meta
        self.vid_meta = [item for item in vid_meta if item.get("mode") == mode]
        
        self.clip_image_processor = CLIPImageProcessor()
        if self.aug_type == "Resize":
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size_single, self.img_size_single)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.Resize((self.img_size_single, self.img_size_single)),
                transforms.ToTensor(),
            ])

        elif self.aug_type == "Padding":
            self.transform = transforms.Compose([
                transforms.Lambda(self.resize_long_edge),
                transforms.Lambda(self.padding_short_edge),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.Lambda(self.resize_long_edge),
                transforms.Lambda(self.padding_short_edge),
                transforms.ToTensor(),
            ])

        elif self.aug_type == "RandomResizeCrop": # center crop
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size_single, self.img_size_single), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            self.cond_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(self.img_size_single, self.img_size_single), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        elif self.aug_type == "RandomCrop":
            self.paired_transform = ListPairedRandomMaxSquareCropResize((self.img_size_single, self.img_size_single))
            self.transform = transforms.Compose(
                [   
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.cond_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
        
    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img
    
    def set_clip_idx(self, video_length):
        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int
        ).tolist()
        return clip_idxes
        
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            ret_lst = []
            for img in images:
                if isinstance(img, list):
                    transformed_sub_images = [transform(sub_img) for sub_img in img]
                    sub_ret_tensor = torch.cat(transformed_sub_images, dim=0)  # (c*n, h, w)
                    ret_lst.append(sub_ret_tensor)
                else:
                    transformed_images = transform(img)
                    ret_lst.append(transformed_images)  # (c*1, h, w)
            ret_tensor = torch.stack(ret_lst, dim=0)  # (f, c*n, h, w)     
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def __len__(self):
        return len(self.vid_meta)
    
    def _get_data(self, idx):
        video_meta = self.vid_meta[idx]['data']
        video_clip_num = len(video_meta)

        selected_index = random.choices(range(video_clip_num), k=self.num_ref+1)
        ref_img_pil_list = []
        ref_pose_pil_list = []
        for ind in selected_index[:-1]:
            ref_meta = video_meta[ind]
            ref_video_path = os.path.join(self.root,ref_meta["video_path"])
            ref_kps_path = os.path.join(self.root,ref_meta["kps_path"])
            if 'mask_path' in ref_meta:
                ref_mask_path = os.path.join(self.root,ref_meta["mask_path"])
            else:
                ref_mask_path = ref_video_path.replace('final_ted_v2','final_ted_v2_mask2')
            ref_bbox_path = ref_video_path.replace('videos','bboxes').replace('mp4','npy')
            
            ref_video_reader = VideoReader(ref_video_path)
            ref_kps_reader = VideoReader(ref_kps_path)
            ref_mask_reader = VideoReader(ref_mask_path)
            
            ref_video_length = len(ref_video_reader)
            if len(ref_video_reader) != len(ref_kps_reader): ref_video_length = min(len(ref_kps_reader),len(ref_video_reader))
            ref_img_idx = random.randint(0, ref_video_length - 1)
            ref_img = ref_video_reader[ref_img_idx]
            ref_mask = ref_mask_reader[ref_img_idx].asnumpy()/255

           
            ref_img_pil = Image.fromarray(ref_img.asnumpy()*ref_mask.astype(np.uint8))
            ref_pose_img = ref_kps_reader[ref_img_idx]
            ref_pose_pil = Image.fromarray(ref_pose_img.asnumpy())
            bboxes = np.load(ref_bbox_path).squeeze(1)
            x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bboxes,ref_img_pil.size[0],ref_img_pil.size[1])
            ref_img_pil = ref_img_pil.crop((x1_min, y1_min, x2_max, y2_max))
            ref_pose_pil = ref_pose_pil.crop((x1_min, y1_min, x2_max, y2_max))
            ref_img_pil_list.append(ref_img_pil)
            ref_pose_pil_list.append(ref_pose_pil)
            
            
        target_index = selected_index[-1]
        target_meta = video_meta[target_index]
        target_video_path = os.path.join(self.root,target_meta["video_path"])
        target_kps_path = os.path.join(self.root,target_meta["kps_path"])
        if 'mask_path' in target_meta:
            tgt_mask_path = os.path.join(self.root,target_meta["mask_path"])
        else:
            tgt_mask_path = target_video_path.replace('final_ted_v2','final_ted_v2_mask2')
        target_bbox_path = target_video_path.replace('videos','bboxes').replace('mp4','npy')
        target_video_reader = VideoReader(target_video_path)
        target_kps_reader = VideoReader(target_kps_path)
        target_mask_reader = VideoReader(tgt_mask_path)
        
        target_video_length = len(target_video_reader)
        
        if len(target_video_reader) != len(target_kps_reader): target_video_length = min(len(target_kps_reader),len(target_video_reader))
        
        # tgt frames indexes
        clip_idxes = self.set_clip_idx(target_video_length)
        
        # read frames and kps
        tgt_vidpil_lst = []
        tgt_guid_vid_list = []
        brightness_list = []
        contrast_list = []
        bboxes = np.load(target_bbox_path).squeeze(1)
        tgt_template = Image.fromarray(target_video_reader[0].asnumpy())
        
        x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bboxes,tgt_template.size[0],tgt_template.size[1])
        for index in clip_idxes:
            img = target_video_reader[index]
            mask = target_mask_reader[index].asnumpy()/255
            pil_img = Image.fromarray(img.asnumpy()*mask.astype(np.uint8))
            
            tgt_guid_img = target_kps_reader[index]
            tgt_guid_pil = Image.fromarray(tgt_guid_img.asnumpy())
            pil_img = pil_img.crop((x1_min, y1_min, x2_max, y2_max))
            tgt_guid_pil = tgt_guid_pil.crop((x1_min, y1_min, x2_max, y2_max))
            tgt_vidpil_lst.append(pil_img)
            tgt_guid_vid_list.append(tgt_guid_pil)
            brightness_list.append(calculate_brightness(pil_img))
            contrast_list.append(calculate_contrast(pil_img))
        brightness = np.mean(brightness_list)
        contrast = np.mean(contrast_list)
        brightness = sinusoidal_encode(brightness,128)
        contrast = sinusoidal_encode(contrast,128)
        env_code = np.concatenate((brightness,contrast),0)
        
        if self.aug_type == "RandomCrop":
            pose_pil_list = []
            img_pil_list = []
            for pose, img in zip(ref_pose_pil_list,ref_img_pil_list):
                pose, img = self.paired_transform([pose], [img])
                pose_pil_list+=pose
                img_pil_list+=img
            ref_pose_pil_list = pose_pil_list
            ref_img_pil_list = img_pil_list
            pose_pil_list = []
            img_pil_list = []
            tgt_guid_vid_list, tgt_vidpil_lst = self.paired_transform(tgt_guid_vid_list, tgt_vidpil_lst)
            
        # transform
        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.transform, state)
        tgt_guid_vid = self.augmentation(tgt_guid_vid_list, self.cond_transform, state)
        
        ref_pose_img_list = [self.augmentation(ref_pose, self.cond_transform, state) for ref_pose in ref_pose_pil_list]
        ref_img_vae_list = [self.augmentation(ref_img, self.transform, state) for ref_img in ref_img_pil_list]
        clip_image_list = [self.clip_image_processor(images=img_pil, return_tensors="pt").pixel_values[0] for img_pil in ref_img_pil_list]
        
        
        sample = dict(
            tgt_vid=tgt_vid,
            tgt_guid_vid=tgt_guid_vid,
            ref_img=ref_img_vae_list,
            clip_img=clip_image_list,
            ref_pose_img = ref_pose_img_list,
            env_code=env_code,
        )
        return sample
    def __getitem__(self, index):
        while 1:
            try:
                return self._get_data(index)
            except Exception as e:
                index = random.randint(0, len(self.vid_meta)-1)