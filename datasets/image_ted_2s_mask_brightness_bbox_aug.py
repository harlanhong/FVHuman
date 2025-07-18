# no bbox crop
import json
import random
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
import numpy as np
import cv2
import torchvision.transforms.functional as F
import pdb
import os
from .data_utils import calculate_global_bbox_with_margin
from utils.util import calculate_brightness,calculate_contrast
from flash_s3_dataloader.s3_io import \
    load_s3_image, save_s3_image, \
    load_s3_text, save_s3_text, \
    load_s3_json, save_s3_json, \
    check_s3_exists, list_s3_dir, \
    parallel_upload_folder_to_s3, parallel_download_folder_from_s3, \
    upload_file, download_file, \
    get_s3_filesize, load_s3_exr, \
    save_ckpt_to_s3, load_ckpt_from_s3,_read_s3_to_bytesio
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
        if width > height:
            x = random.randint(0, width - min_dim)
            y = 0
        else:
            x = 0
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
class ImageDataset(Dataset):
    def __init__(
        self,
        root,
        img_size,
        data_meta_paths=["./data/fahsion_meta.json"],
        sample_margin=30,
        aug_type="Resize",
        mode="train",
        ref_num = 2
    ):
        super().__init__()
        self.num_ref = ref_num
        self.root = root
        self.img_size = img_size
        self.img_size_single = img_size[0]
        self.sample_margin = sample_margin
        vid_meta = []
        for data_meta_path in data_meta_paths:
            vid_meta.extend(json.load(open(data_meta_path, "r")))
        # self.vid_meta = vid_meta
        self.vid_meta = [item for item in vid_meta if item.get("mode") == mode]
        self.clip_image_processor = CLIPImageProcessor()
        self.aug_type = aug_type
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
            self.paired_transform = PairedRandomMaxSquareCropResize((self.img_size_single, self.img_size_single))
            self.transform = transforms.Compose(
                [   
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            self.cond_transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size_single, self.img_size_single)),
                    transforms.ToTensor(),
                ]
            )

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.img_size_single / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.img_size_single, self.img_size_single
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img

    def _get_data(self, index):
        video_meta = self.vid_meta[index]['data']
        video_clip_num = len(video_meta)

        selected_index = random.choices(range(video_clip_num), k=self.num_ref+1,replace=True)
        ref_img_pil_list = []
        ref_pose_pil_list = []
        for ind in selected_index[:-1]:
            ref_meta = video_meta[ind]
            ref_video_path = os.path.join(self.root,ref_meta["video_path"])
            ref_kps_path = os.path.join(self.root,ref_meta["kps_path"])
            ref_bbox_path = ref_video_path.replace('videos','bboxes').replace('mp4','npy')
            
            if 'mask_path' in ref_meta:
                ref_mask_path = os.path.join(self.root,ref_meta["mask_path"])
            else:
                ref_mask_path = ref_video_path.replace('final_ted_v2','final_ted_v2_mask2')
            
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
        if 'mask_path' in target_meta:
            tgt_mask_path = os.path.join(self.root,target_meta["mask_path"])
        else:
            tgt_mask_path = target_video_path.replace('final_ted_v2','final_ted_v2_mask2')
        target_kps_path = os.path.join(self.root,target_meta["kps_path"])
        target_bbox_path = target_video_path.replace('videos','bboxes').replace('mp4','npy')
        target_video_reader = VideoReader(target_video_path)
        target_kps_reader = VideoReader(target_kps_path)
        tgt_mask_reader = VideoReader(tgt_mask_path)
        
        target_video_length = len(target_video_reader)
        
        if len(target_video_reader) != len(target_kps_reader): target_video_length = min(len(target_kps_reader),len(target_video_reader))
        target_img_idx = random.randint(0, target_video_length - 1)
        target_img = target_video_reader[target_img_idx]
        target_mask =  tgt_mask_reader[target_img_idx].asnumpy()/255
        tgt_img_pil = Image.fromarray(target_img.asnumpy()*target_mask.astype(np.uint8))
        
        bboxes = np.load(target_bbox_path).squeeze(1)
        x1_min, y1_min, x2_max, y2_max = calculate_global_bbox_with_margin(bboxes,tgt_img_pil.size[0],tgt_img_pil.size[1])
        tgt_img_pil = tgt_img_pil.crop((x1_min, y1_min, x2_max, y2_max))
        
        brightness = calculate_brightness(tgt_img_pil)
        contrast = calculate_contrast(tgt_img_pil)
        brightness = sinusoidal_encode(brightness,128)
        contrast = sinusoidal_encode(contrast,128)
        env_code = np.concatenate((brightness,contrast),0)
        target_pose_img = target_kps_reader[target_img_idx]
        tgt_pose_pil = Image.fromarray(target_pose_img.asnumpy())
        tgt_pose_pil = tgt_pose_pil.crop((x1_min, y1_min, x2_max, y2_max))
        
        tgt_mask_pil = Image.fromarray(tgt_mask_reader[target_img_idx].asnumpy())
        state = torch.get_rng_state()
        if self.aug_type == "RandomCrop":
            pose_pil_list = []
            img_pil_list = []
            for pose, img in zip(ref_pose_pil_list+[tgt_pose_pil],ref_img_pil_list+[tgt_img_pil]):
                pose, img = self.paired_transform(pose, img)
                pose_pil_list.append(pose)
                img_pil_list.append(img)
                
        tgt_img = self.augmentation(img_pil_list[-1], self.transform, state)
        tgt_pose_img = self.augmentation(pose_pil_list[-1], self.cond_transform, state)
        ref_pose_img_list = [self.augmentation(ref_pose, self.cond_transform, state) for ref_pose in pose_pil_list[:-1]]
        ref_img_vae_list = [self.augmentation(ref_img, self.transform, state) for ref_img in img_pil_list[:-1]]
        tgt_mask_img = self.augmentation(tgt_mask_pil, self.cond_transform, state)
        
        clip_image_list = [self.clip_image_processor(images=img_pil, return_tensors="pt").pixel_values[0] for img_pil in ref_img_pil_list]
        sample = dict(
            id = self.vid_meta[index]['id'],
            ref_pose_img=ref_pose_img_list,
            tgt_img=tgt_img,
            tgt_guid=tgt_pose_img,
            env_code = env_code,
            ref_img=ref_img_vae_list,
            clip_img=clip_image_list,
            tgt_mask_img=tgt_mask_img,
        )
        return sample
    
    def _get_data2(self, index):
        video_meta = self.vid_meta[index]['data']
        video_clip_num = len(video_meta)

        selected_index = random.choices(range(video_clip_num), k=self.num_ref+1)
        ref_img_pil_list = []
        ref_pose_pil_list = []
        for ind in selected_index[:-1]:
            ref_meta = video_meta[ind]
            ref_video_path = os.path.join(self.root,ref_meta["video_path"])
            ref_kps_path = os.path.join(self.root,ref_meta["kps_path"])
            
            
            ref_video_reader = VideoReader(ref_video_path)
            ref_kps_reader = VideoReader(ref_kps_path)
            
            ref_video_length = len(ref_video_reader)
            if len(ref_video_reader) != len(ref_kps_reader): ref_video_length = min(len(ref_kps_reader),len(ref_video_reader))
            ref_img_idx = random.randint(0, ref_video_length - 1)
            ref_img = ref_video_reader[ref_img_idx]
            ref_img_pil = Image.fromarray(ref_img.asnumpy())
            ref_pose_img = ref_kps_reader[ref_img_idx]
            ref_pose_pil = Image.fromarray(ref_pose_img.asnumpy())
            ref_img_pil_list.append(ref_img_pil)
            ref_pose_pil_list.append(ref_pose_pil)
            
        target_index = selected_index[-1]
        target_meta = video_meta[target_index]
        target_video_path = os.path.join(self.root,target_meta["video_path"])
        target_kps_path = os.path.join(self.root,target_meta["kps_path"])
        target_video_reader = VideoReader(target_video_path)
        target_kps_reader = VideoReader(target_kps_path)
        
        target_video_length = len(target_video_reader)
        
        if len(target_video_reader) != len(target_kps_reader): target_video_length = min(len(target_kps_reader),len(target_video_reader))
        target_img_idx = random.randint(0, target_video_length - 1)
        target_img = target_video_reader[target_img_idx]
        tgt_img_pil = Image.fromarray(target_img.asnumpy())
        
        brightness = calculate_brightness(tgt_img_pil)
        contrast = calculate_contrast(tgt_img_pil)
        brightness = sinusoidal_encode(brightness,128)
        contrast = sinusoidal_encode(contrast,128)
        env_code = np.concatenate((brightness,contrast),0)
        target_pose_img = target_kps_reader[target_img_idx]
        tgt_pose_pil = Image.fromarray(target_pose_img.asnumpy())
        
        state = torch.get_rng_state()
        if self.aug_type == "RandomCrop":
            pose_pil_list = []
            img_pil_list = []
            for pose, img in zip(ref_pose_pil_list+[tgt_pose_pil],ref_img_pil_list+[tgt_img_pil]):
                pose, img = self.paired_transform(pose, img)
                pose_pil_list.append(pose)
                img_pil_list.append(img)
                
        tgt_img = self.augmentation(img_pil_list[-1], self.transform, state)
        tgt_pose_img = self.augmentation(pose_pil_list[-1], self.cond_transform, state)
        ref_pose_img_list = [self.augmentation(ref_pose, self.cond_transform, state) for ref_pose in pose_pil_list[:-1]]
        ref_img_vae_list = [self.augmentation(ref_img, self.transform, state) for ref_img in img_pil_list[:-1]]
        
        clip_image_list = [self.clip_image_processor(images=img_pil, return_tensors="pt").pixel_values[0] for img_pil in ref_img_pil_list]
        sample = dict(
            id = self.vid_meta[index]['id'],
            ref_pose_img=ref_pose_img_list,
            tgt_img=tgt_img,
            tgt_guid=tgt_pose_img,
            env_code = env_code,
            ref_img=ref_img_vae_list,
            clip_img=clip_image_list,
        )
        return sample
    
    def __getitem__(self, index):
        while 1:
            try:
                return self._get_data(index)
            except Exception as e:
                index = random.randint(0, len(self.vid_meta)-1)

    def __len__(self):
        return len(self.vid_meta)
