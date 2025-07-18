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

class PairedRandomMaxSquareCropResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img1, img2, img3):
        width, height = img1.size
        img3 = transforms.functional.resize(img3,(height, width))
        min_dim = min(width, height)
        if width > height:
            x = random.randint(0, width - min_dim)
            y = 0
        else:
            x = 0
            y = random.randint(0, height - min_dim)
        
        # Crop both images to a square of size min_dim x min_dim
        img1 = img1.crop((x, y, x + min_dim, y + min_dim))
        img2 = img2.crop((x, y, x + min_dim, y + min_dim))
        img3 = img3.crop((x, y, x + min_dim, y + min_dim))
        # print("3", img1.size, img2.size, img3.size)
        
        img1 = transforms.functional.resize(img1, self.size)
        img2 = transforms.functional.resize(img2, self.size)
        img3 = transforms.functional.resize(img3, self.size)
        # print("4", img1.size, img2.size, img3.size)
        return img1, img2, img3
    
class VideoDataset(Dataset):
    def __init__(
        self,
        image_size: int = 512,
        sample_frames: int = 16,
        sample_rate: int = 1,
        aug_type: str = "Resize",
        data_meta_paths=["./data/fashion_meta.json"],
        mode="train",
        seed_frames = 4,
        generated_frames = 8,
    ):
        super().__init__()
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
            self.paired_transform = PairedRandomMaxSquareCropResize((self.img_size_single, self.img_size_single))
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
    
    def __getitem__(self, idx):
        video_meta = self.vid_meta[idx]
        video_path = video_meta["video_path"]
        kps_path = video_meta["kps_path"]

        video_reader = VideoReader(video_path)
        kps_reader = VideoReader(kps_path)
        video_length = len(video_reader)
        if len(video_reader) != len(kps_reader): video_length = min(len(video_reader),len(kps_reader))
        
        # tgt frames indexes
        clip_idxes = self.set_clip_idx(video_length)
        
        # assign the last start seed as reference
        ref_img_idx = random.randint(0, video_length - 1)
        ref_img_np = video_reader[ref_img_idx].asnumpy()
        ref_img_pil = Image.fromarray(ref_img_np)

        # read frames and kps
        tgt_vidpil_lst = []
        tgt_guid_vid_list = []
        for index in clip_idxes:
            img = video_reader[index]
            pil_img = Image.fromarray(img.asnumpy())
            tgt_guid_img = kps_reader[index]
            tgt_guid_pil = Image.fromarray(tgt_guid_img.asnumpy())
            if self.aug_type == "RandomCrop":
                pil_img, ref_img_pil, tgt_guid_pil = self.paired_transform(pil_img, ref_img_pil, tgt_guid_pil)     
            tgt_vidpil_lst.append(pil_img)
            tgt_guid_vid_list.append(tgt_guid_pil)

        # transform
        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.transform, state)
        tgt_guid_vid = self.augmentation(tgt_guid_vid_list, self.cond_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.transform, state)
        clip_img = self.clip_image_processor(
            images=ref_img_pil, return_tensor="pt"
        ).pixel_values[0]
        
        sample = dict(
            tgt_vid=tgt_vid,
            tgt_guid_vid=tgt_guid_vid,
            ref_img=ref_img_vae,
            clip_img=clip_img,
        )
        return sample